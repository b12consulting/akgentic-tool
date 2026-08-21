"""``MetadataTool``: the team's business context, rendered into the frozen prefix.

The orchestrator already holds a team's business context — an opaque
``SerializableBaseModel`` the deployment wrote with ``set_metadata`` — and until
this card nothing surfaced it to the model, so deployments copied the same facts
into every role's backstory. This card renders one operator-written template
against that single authoritative copy and contributes it on ``SYSTEM_PROMPT``:
content that never changes, in the region that is never re-rendered (ADR-038).

Three properties are load-bearing, and each is a review rejection if weakened:

* **The block is a snapshot, taken at the first *successful* render.** A later
  ``set_metadata`` is not reflected, by design — re-reading per turn would be a
  volatile system prompt, and one write would invalidate every agent's prefix
  cache. A *degraded* render caches nothing, so a ``set_metadata`` that lands
  just after start-up still produces its block on a later turn.
* **Loud at wiring, visible at render.** A malformed or unresolvable placeholder
  raises in :meth:`MetadataTool.observer`, next to the mistake. Nothing here may
  raise inside context construction: a prompt callable that throws kills the
  turn, far from the cause.
* **Bare identifiers only.** ``str.format`` on an operator-supplied template is
  an attribute walk — ``{a.__class__}`` is legal Python formatting. Restricting
  the grammar to plain field names removes that surface, and it is also what
  keeps validation honest: a name that cannot be checked is one that must not be
  accepted.

The metadata model itself stays opaque. This module reads
``type(metadata).model_fields`` and ``getattr``, and names no field of any
deployment's business shape.
"""

from __future__ import annotations

import logging
import string
from collections.abc import Callable
from typing import Annotated, Any, cast

from pydantic import PrivateAttr, StringConstraints

from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import (
    COMMAND,
    SYSTEM_PROMPT,
    BaseToolParam,
    Channels,
    ToolCard,
    _resolve,
)
from akgentic.tool.core.observer import ActorToolObserver, ToolObserver

logger = logging.getLogger(__name__)

NonEmptyTemplate = Annotated[str, StringConstraints(min_length=1)]
"""A template that says nothing is a misconfiguration, not an empty block."""


def _placeholder_names(template: str) -> list[str]:
    """Return the placeholder names in *template*, in first-appearance order.

    Only a bare field name is accepted. ``string.Formatter().parse`` yields
    ``(literal_text, field_name, format_spec, conversion)`` per segment, with
    ``field_name`` set to ``None`` for trailing literal text and never produced at
    all for the escaped braces ``{{`` and ``}}``. A dotted path, an index, an
    auto-numbered ``{}`` and a positional ``{0}`` all fail ``str.isidentifier``,
    which is what makes one rule cover every rejected form.

    Args:
        template: The operator-written template.

    Returns:
        The distinct placeholder names, in the order they first appear.

    Raises:
        ValueError: If a placeholder is anything but a bare field name. The
            message names the offending placeholder.
    """
    names: list[str] = []
    for _literal, field_name, format_spec, conversion in string.Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name.isidentifier():
            raise ValueError(
                f"MetadataTool template placeholder '{{{field_name}}}' is not a plain "
                "field name. Placeholders must be bare identifiers — no dotted paths, "
                "indices, or positional fields."
            )
        if conversion is not None:
            raise ValueError(
                f"MetadataTool template placeholder '{{{field_name}!{conversion}}}' uses a "
                "conversion. Placeholders must be bare identifiers."
            )
        if format_spec:
            raise ValueError(
                f"MetadataTool template placeholder '{{{field_name}:{format_spec}}}' uses a "
                "format spec. Placeholders must be bare identifiers."
            )
        if field_name not in names:
            names.append(field_name)
    return names


def _check_names(names: list[str], metadata: SerializableBaseModel) -> None:
    """Check every placeholder name against the metadata model's own fields.

    Args:
        names: Placeholder names, already known to be bare identifiers.
        metadata: The model the orchestrator handed back. Read only — it is the
            orchestrator's own state, returned by reference.

    Raises:
        ValueError: If a name is not a field of *metadata*. The message names the
            offending placeholder, the model, and the fields it does declare.
    """
    fields = type(metadata).model_fields
    for name in names:
        if name not in fields:
            raise ValueError(
                f"MetadataTool template placeholder '{{{name}}}' is not a field of "
                f"{type(metadata).__name__}. Its fields are: {sorted(fields)}."
            )


def _render(params: RenderMetadata, names: list[str], metadata: SerializableBaseModel) -> str:
    """Substitute *names* out of *metadata* into the template, header included.

    Safe precisely because validation ran first: the identifier-only grammar is
    what removes ``str.format``'s attribute-walking surface.

    Args:
        params: The capability configuration carrying the template and header.
        names: Placeholder names, already checked against *metadata*.
        metadata: The model to read the values from.

    Returns:
        The rendered block, with a bold header line when one is configured.
    """
    values = {name: str(getattr(metadata, name)) for name in names}
    rendered = params.template.format_map(values)
    return f"**{params.header}**\n{rendered}" if params.header else rendered


class RenderMetadata(BaseToolParam):
    """Render the team's business metadata from a template.

    Attributes:
        template: The text to render, carrying ``{field}`` placeholders that name
            fields of the team's metadata model. Bare identifiers only. **The
            result is a snapshot:** it is rendered once, at the first render that
            succeeds, and a later ``set_metadata`` is not reflected. A deployment
            whose business context genuinely changes mid-life does not want this
            capability.
        header: An optional header line, rendered bold above the text.
        expose: Both non-LLM channels by default — the prompt the agent reads and
            the command a human can call to see exactly what it was given.
    """

    template: NonEmptyTemplate
    header: str | None = None
    expose: set[Channels] = {SYSTEM_PROMPT, COMMAND}


class MetadataTool(ToolCard):
    """The team's business context, rendered once into every agent's prefix.

    Attributes:
        render_metadata: The one capability, and the one capability in this
            package that ships **off**. Every other capability here defaults on,
            because every other one is meaningful with no configuration at all —
            a roster renders itself, a graph summarises itself. This one cannot:
            its entire content is an operator-written ``template``, and there is
            no template a framework could supply. That makes ``True`` here not
            merely unhelpful but *unsatisfiable*: it means "enabled with
            defaults", ``template`` is required, and so it raises at
            :meth:`observer` — which is where a missing template should be
            noticed, and why ``True`` stays a legal value. The card is turned on
            by handing it the template it exists to render: a ``RenderMetadata``
            instance supplies that and narrows the channels. ``False``, the
            default, removes the capability entirely, so a card nobody has
            configured contributes nothing and raises nothing.
    """

    render_metadata: RenderMetadata | bool = False

    # Runtime handles: an actor proxy is not serializable and never a field, and
    # neither is the snapshot — a restored card renders afresh against its own team.
    _orchestrator_proxy: Orchestrator | None = PrivateAttr(default=None)
    _rendered: str | None = PrivateAttr(default=None)
    _names: list[str] = PrivateAttr(default_factory=list)

    def observer(self, observer: ToolObserver) -> MetadataTool:
        """Attach the observer, bind the orchestrator, and validate the template.

        The parameter keeps the base ``ToolObserver`` type — ``ToolFactory``
        attaches one observer to every card uniformly, so narrowing it would break
        substitutability; :meth:`_actor_observer` applies the narrower type.

        Grammar is validated unconditionally. Names are validated only when the
        team already has metadata: ``set_metadata`` may legitimately run after the
        agents start, and a name that cannot be checked yet is checked at the
        first render instead.

        Args:
            observer: Observer implementing the ``ActorToolObserver`` protocol.

        Returns:
            Self, enabling method chaining.

        Raises:
            ValueError: If the capability is enabled without a usable template,
                if ``observer.orchestrator`` is None, or if the template carries a
                placeholder that is not a bare identifier or not a field of the
                team's metadata model.
        """
        super().observer(observer)  # store the observer weakly via the base setter
        # A card can be wired more than once — ``ToolFactory`` attaches an observer
        # to every card it is handed — and the snapshot belongs to the team it was
        # rendered for. Drop it, or a second binding would serve the first team's
        # business context to the second team's agents.
        self._orchestrator_proxy = None
        self._rendered = None
        self._names = []

        params = _resolve(self.render_metadata, RenderMetadata)
        if params is None:
            return self  # nothing configured: no proxy to bind, no template to check

        actor_observer = self._actor_observer()
        if actor_observer.orchestrator is None:
            raise ValueError("MetadataTool requires access to the orchestrator.")
        self._orchestrator_proxy = actor_observer.proxy_ask(
            actor_observer.orchestrator, Orchestrator
        )

        self._names = _placeholder_names(params.template)
        metadata = self._orchestrator_proxy.get_metadata()
        if metadata is not None:
            _check_names(self._names, metadata)
        return self

    def _actor_observer(self) -> ActorToolObserver:
        """Live observer typed as the actor protocol. Raises once the agent stops."""
        return cast(ActorToolObserver, self._observer)

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return no tools, ever.

        Handing the model a way to fetch metadata costs a round trip for content
        that never changes and requires the model to know it should ask. The block
        is simply there, in the prefix, from the first token.
        """
        return []

    def get_system_prompts(self) -> list[Callable[..., Any]]:
        """Return the metadata block callable when the prompt channel is exposed."""
        params = _resolve(self.render_metadata, RenderMetadata)
        if params and SYSTEM_PROMPT in params.expose:
            return [self._team_metadata_factory(params)]
        return []

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return ``team_metadata`` when the command channel is exposed."""
        params = _resolve(self.render_metadata, RenderMetadata)
        if params and COMMAND in params.expose:
            return {RenderMetadata: self._team_metadata_factory(params)}
        return {}

    def _team_metadata_factory(self, params: RenderMetadata) -> Callable[..., Any]:
        """Create the ``team_metadata`` callable for one channel.

        Each channel gets its own closure, and both read the one snapshot held on
        the card — which is what makes the command show exactly what the agent was
        given. The closure captures the *card*, never a dereferenced observer: the
        ``tool → agent`` edge is weak, so holding the tool never roots the agent.

        Args:
            params: Configuration for the render capability.

        Returns:
            A zero-argument callable named ``team_metadata``.
        """
        card = self

        def team_metadata() -> str:
            """Get the team's business context, exactly as the agents received it.

            Returns:
                The rendered metadata block, or an empty string when the team
                declares no metadata.
            """
            return card._render_block(params)

        team_metadata.__doc__ = params.format_docstring(team_metadata.__doc__)
        return team_metadata

    def _render_block(self, params: RenderMetadata) -> str:
        """Return the snapshot, rendering it on the first successful attempt.

        Never raises: a block that cannot be produced contributes nothing and says
        so in the log. Nothing is cached until a render succeeds, so metadata set
        after the agents start still reaches the prompt on a later turn.

        Args:
            params: Configuration for the render capability.

        Returns:
            The rendered block, or ``""`` when it cannot be produced.
        """
        if self._rendered is not None:
            return self._rendered
        try:
            metadata = self._fetch_metadata()
            if metadata is None:
                logger.warning(
                    "MetadataTool: the team declares no metadata; the metadata block "
                    "contributes nothing this turn."
                )
                return ""
            _check_names(self._names, metadata)
            rendered = _render(params, self._names, metadata)
        except ValueError as exc:
            # Expected and self-describing: an unresolvable placeholder, or a card
            # that was never wired. The message carries everything worth knowing.
            logger.error(f"MetadataTool: cannot render the team metadata block. {exc}")
            return ""
        except Exception:
            # Anything else is a genuine surprise, and a prompt callable is the one
            # place a traceback cannot be recovered afterwards — keep it.
            logger.exception("MetadataTool: cannot render the team metadata block.")
            return ""
        self._rendered = rendered
        return rendered

    def _fetch_metadata(self) -> SerializableBaseModel | None:
        """Ask the orchestrator for the team's metadata.

        Returns:
            The team's metadata model, or ``None`` when the team declares none.
            The model is the orchestrator's own state, by reference: read only.

        Raises:
            ValueError: If the card was never wired.
        """
        if self._orchestrator_proxy is None:
            raise ValueError("observer() must run before the metadata block can be rendered.")
        return self._orchestrator_proxy.get_metadata()
