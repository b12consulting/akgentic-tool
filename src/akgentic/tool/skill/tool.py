"""``SkillTool``: the menu in the frozen prefix, the bodies on demand.

Domain guidance — a refund procedure, an escalation policy, a report playbook — has
nowhere to live but an agent's backstory, so **every agent pays for every playbook on
every turn**, and the instructions that matter for the turn compete with seven that do
not. This card splits one library by size and volatility: the **menu** (small,
immutable) goes into the frozen system prefix, the **bodies** (large, optional) arrive
at the tail on demand as ordinary tool returns (ADR-039).

Three properties are load-bearing, and each is a review rejection if weakened:

* **``use_skill`` returns the body as its tool result.** The model asked because it
  needs the body for the answer it is composing; anything that arrives on the next turn
  arrives too late. Returning an acknowledgement — "the refund-policy skill is now
  loaded" — reads correct and is ADR-039's rejected Alternative C.
* **The card is stateless.** No loaded set, no baseline, no re-delivery, and no private
  attribute of its own. A tool return is already durable context and already
  event-sourced by the runtime, so a card tracking a parallel copy could only drift from
  it. Re-calling ``use_skill`` on the same name simply returns the body again — there is
  no "already loaded" case to special-case, and after a compaction the repeat call is
  exactly the intended recovery.
* **The menu lives in the frozen prefix, and that is the recovery path.** After a
  restart or a fold the model can always re-call ``use_skill`` because the menu survives
  everything. A menu at the tail would be one compaction away from an agent that no
  longer knows its skills exist.

Skill bodies are operator-written catalog data, so nothing here runs one through
``str.format``: a body containing a brace would raise, or worse, resolve an attribute.
This card renders no templates.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import cleandoc
from typing import Any

from pydantic import Field, model_validator

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import (
    COMMAND,
    SYSTEM_PROMPT,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    ToolCard,
    _resolve,
)
from akgentic.tool.errors import RetriableError

MENU_HEADER = "**Skills available to you** — call use_skill(name) to load one."
"""The one line that tells the model the library exists and how to open it."""


class SkillEntry(SerializableBaseModel):
    """One skill: a handle, the line that advertises it, and the body itself.

    Attributes:
        name: The handle the model passes to ``use_skill``, e.g. ``"refund-policy"``.
        description: The one line that goes in the menu. It is the only thing the model
            sees until it asks, so a description that undersells its skill means the
            skill is never used and nothing surfaces that.
        content: The body, delivered on demand. It never reaches the prefix, so its size
            costs nothing until a conversation actually needs it.
    """

    name: str = Field(min_length=1, description="Handle the model passes to use_skill.")
    description: str = Field(min_length=1, description="One-line summary shown in the menu.")
    content: str = Field(min_length=1, description="Full instructions, delivered on demand.")


class Skills(BaseToolParam):
    """The library this card advertises, and the channels it reaches.

    Attributes:
        skills: The entries, in the order they should appear in the menu. Names are
            unique within one library — a duplicate raises here, at validation, rather
            than at call time, because the second entry could never be reached.
        expose: All three channels by default. Each carries what it is good at: the menu
            in the cached prefix, the body on demand, and the same menu behind a command
            a human can call.
    """

    skills: list[SkillEntry] = []
    expose: set[Channels] = {SYSTEM_PROMPT, TOOL_CALL, COMMAND}

    @model_validator(mode="after")
    def _reject_duplicate_names(self) -> Skills:
        """Reject a name claimed twice.

        Returns:
            Self, unchanged.

        Raises:
            ValueError: If two entries share a name. The message names the duplicate.
        """
        seen: set[str] = set()
        for entry in self.skills:
            if entry.name in seen:
                raise ValueError(
                    f"Duplicate skill name '{entry.name}'. Skill names must be unique "
                    "within one card — a second entry under the same name could never "
                    "be reached by use_skill."
                )
            seen.add(entry.name)
        return self


def _render_menu(params: Skills) -> str:
    """Render the menu: the header, then one ``name — description`` line per entry.

    Reads ``name`` and ``description`` only, never ``content`` — which is what makes the
    prefix cost O(skills) rather than O(content) by construction rather than by care.

    Args:
        params: The configured library.

    Returns:
        The rendered menu, or ``""`` when the library is empty — an empty library has
        nothing to advertise, and a header alone would advertise nothing.
    """
    if not params.skills:
        return ""
    lines = [MENU_HEADER]
    lines.extend(f"{entry.name} — {entry.description}" for entry in params.skills)
    return "\n".join(lines)


class SkillTool(ToolCard):
    """A library of skills: the menu in the prefix, the bodies on demand.

    Attributes:
        skills: The one capability. ``True`` — the default — enables it with an empty
            library, which contributes an empty menu and a ``use_skill`` that knows no
            names: harmless, and the shape an operator sees before configuring anything.
            A ``Skills`` instance supplies the entries and may narrow the channels.
            ``False`` removes the capability entirely.
    """

    skills: Skills | bool = True

    def get_system_prompts(self) -> list[Callable[..., Any]]:
        """Return the menu callable when the prompt channel is exposed."""
        params = _resolve(self.skills, Skills)
        if params and SYSTEM_PROMPT in params.expose:
            return [self._menu_factory(params)]
        return []

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return ``use_skill`` when the tool channel is exposed."""
        params = _resolve(self.skills, Skills)
        if params and TOOL_CALL in params.expose:
            return [self._use_skill_factory(params)]
        return []

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return ``skills`` when the command channel is exposed."""
        params = _resolve(self.skills, Skills)
        if params and COMMAND in params.expose:
            return {Skills: self._menu_factory(params)}
        return {}

    def _menu_factory(self, params: Skills) -> Callable[..., Any]:
        """Create the menu callable for one channel.

        The prompt and the command share this one renderer, which is what makes the
        command show exactly what the agent was given. Nothing observer-shaped is
        captured: this card reads only its own configuration, so a closure outlives its
        agent without pinning it (ADR-030).

        Args:
            params: The configured library.

        Returns:
            A zero-argument callable named ``skills``.
        """

        def skills() -> str:
            """List the skills available, one line each.

            Returns:
                The menu — a header line, then one ``name — description`` line per
                skill — or an empty string when no skills are configured.
            """
            return _render_menu(params)

        skills.__doc__ = params.format_docstring(skills.__doc__)
        return skills

    def _use_skill_factory(self, params: Skills) -> Callable[..., Any]:
        """Create the ``use_skill`` callable.

        The lookup is built once, at bind time, off entries whose names are already
        unique by validation.

        Args:
            params: The configured library.

        Returns:
            A callable named ``use_skill`` taking the skill's name.
        """
        entries = {entry.name: entry for entry in params.skills}

        def use_skill(name: str) -> str:
            """Load a skill's full instructions.

            Call this when the menu says a skill covers the question in front of you.
            The instructions come back in the result, so you can use them for the answer
            you are composing right now. Calling it again later is fine and is how you
            recover a skill whose text has since dropped out of the conversation.

            Args:
                name: The skill's name, exactly as the menu spells it.

            Returns:
                The skill's name, a blank line, and its full instructions.

            Raises:
                RetriableError: If no skill goes by that name. The message lists the
                    names that do.
            """
            entry = entries.get(name)
            if entry is None:
                available = ", ".join(entries) if entries else "none — no skills are configured"
                raise RetriableError(f"Unknown skill '{name}'. Available skills: {available}.")
            return f"{entry.name}\n\n{entry.content}"

        # ``format_docstring`` appends a flush-left block, and the schema builder parses
        # the result with griffe, which dedents only when every line past the first
        # shares one margin. Appending to a still-indented docstring leaves the ``Args:``
        # section unparsed and ``name`` undescribed — ``cleandoc`` first is what keeps
        # the two compatible.
        use_skill.__doc__ = params.format_docstring(cleandoc(use_skill.__doc__ or ""))
        return use_skill
