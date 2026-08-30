"""``ModelTool``: the model roster on three channels (akgentic-llm ADR-018 §5).

One card, three capabilities. ``list_models`` and ``switch_model`` are served on
both ``TOOL_CALL`` and ``COMMAND`` — the model can escalate itself mid-run, and a
human can type ``/switch_model openai:gpt-5.2`` at the same agent — while
``active_model`` publishes the model in force as an ``LLM_CONTEXT`` delta block
rather than re-rendering it into the frozen system prefix.

The card creates no actor and makes no proxy call: it does not override
``observer()``. Everything it needs is two methods on the observer, plus the
agent's live state object.

**It holds no ``ToolState``.** ``init_state()`` replaces the agent's state object
wholesale on restore, so a held reference — in a field, in a closure, or in a
local computed before the call that will use it — goes silently stale and keeps
writing into an abandoned carrier (ADR-041 §3). The only legal form is the full
``observer.state.tool_state`` chain, dereferenced at the moment of the write.

The card is **not auto-injected**. ``BaseAgent`` auto-adds ``TeamTool`` and
``MailboxTool``; granting every agent the standing power to change its own model
is a cost and governance decision that belongs to whoever writes the card list.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from inspect import cleandoc
from typing import Any, cast

from akgentic.tool.core import (
    COMMAND,
    LLM_CONTEXT,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    ContextState,
    ToolCard,
    _resolve,
)
from akgentic.tool.errors import RetriableError, ToolObserverGone
from akgentic.tool.model.observer import ModelSwitchToolObserver
from akgentic.tool.model.state import ActiveModelState, ModelRow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_EMPTY_ROSTER = "This agent has no model roster, so there is nothing to switch within."


class ListModels(BaseToolParam):
    """Read the roster this agent may switch within."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


class SwitchModel(BaseToolParam):
    """Make one roster entry the model in force."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


class ActiveModel(BaseToolParam):
    """Publish the model in force as structured context state."""

    expose: set[Channels] = {LLM_CONTEXT}


def _render_model_row(row: ModelRow) -> str:
    """One roster entry on one line, key first, the entry in force marked."""
    context = str(row.context_length) if row.context_length is not None else "undeclared"
    marker = " [ACTIVE]" if row.active else ""
    return f"{row.key} (context: {context}){marker}"


def _render_model_rows(rows: list[ModelRow]) -> str:
    """Render *rows* in the order given, one line each.

    Deliberately order-preserving and side-effect free: the roster's order is the
    roster's decision, and re-sorting here would make the listing disagree with
    every other view of the same configuration. Nothing turn-varying enters the
    text, so two calls against an unchanged roster are byte-identical.
    """
    if not rows:
        return _EMPTY_ROSTER
    return "\n".join(_render_model_row(row) for row in rows)


class ModelTool(ToolCard):
    """Runtime model switching: the roster, the switch, and the model in force.

    Three channels, three capabilities:

    - ``list_models`` — ``TOOL_CALL`` and ``COMMAND``. Renders the roster.
    - ``switch_model`` — ``TOOL_CALL`` and ``COMMAND``. Makes one entry current
      and records its key in ``ToolState.active_model``, where a restart finds it.
    - ``active_model`` — ``LLM_CONTEXT``. Publishes the model in force as an
      :class:`ActiveModelState` delta block.

    The card holds no ``ToolState`` and no observer of its own: the durable slot
    is reached live through ``observer.state.tool_state`` at the moment of the
    write, and the observer edge is the weak one every card inherits.

    Not auto-injected — an agent gets this card only because its card list says so.

    Attributes:
        list_models: The roster listing. ``True`` (the default) enables it with
            the param's defaults; a ``ListModels`` instance may narrow the
            channels; ``False`` removes exactly this capability and nothing else.
        switch_model: The switch itself. Same ``Param | bool`` convention.
        active_model: The ``LLM_CONTEXT`` state provider. Same convention.
    """

    list_models: ListModels | bool = True
    switch_model: SwitchModel | bool = True
    active_model: ActiveModel | bool = True

    def _model_observer(self) -> ModelSwitchToolObserver:
        """Live observer typed as the model-switch protocol. Raises once the agent stops.

        Conformance is a documented precondition of :meth:`observer`, not a runtime
        gate — observers are duck-typed, so a non-conforming one fails at first use
        just as it did before.
        """
        return cast(ModelSwitchToolObserver, self._observer)

    def _model_observer_or_none(self) -> ModelSwitchToolObserver | None:
        """Live observer typed as the model-switch protocol; ``None`` once the agent stops."""
        return cast(ModelSwitchToolObserver | None, self._observer_or_none())

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return the capabilities that serve ``TOOL_CALL``."""
        tools: list[Callable[..., Any]] = []

        lm = _resolve(self.list_models, ListModels)
        if lm and TOOL_CALL in lm.expose:
            tools.append(self._list_models_factory(lm))

        sm = _resolve(self.switch_model, SwitchModel)
        if sm and TOOL_CALL in sm.expose:
            tools.append(self._switch_model_factory(sm))

        return tools

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return the capabilities that serve ``COMMAND``.

        ``list_models`` is inserted first: the registry preserves this dict's
        insertion order, and that order is what ``descriptors()`` announces to
        every frontend's command palette.
        """
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}

        lm = _resolve(self.list_models, ListModels)
        if lm and COMMAND in lm.expose:
            commands[ListModels] = self._list_models_factory(lm)

        sm = _resolve(self.switch_model, SwitchModel)
        if sm and COMMAND in sm.expose:
            commands[SwitchModel] = self._switch_model_factory(sm)

        return commands

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        """Return the ``active_model_state`` provider when it serves ``LLM_CONTEXT``."""
        am = _resolve(self.active_model, ActiveModel)
        if am and LLM_CONTEXT in am.expose:
            return [self._active_model_state_factory(am)]
        return []

    def _list_models_factory(self, params: ListModels) -> Callable[..., Any]:
        """Create the ``list_models`` callable.

        The closure captures the bound accessor, never the observer, so it cannot
        pin a stopped agent (ADR-030).

        Args:
            params: Configuration for the listing capability.

        Returns:
            A zero-argument callable named ``list_models``.
        """
        observer_or_none = self._model_observer_or_none  # bound method -> weak edge to agent

        def list_models() -> str:
            """List the models you can switch to, one per line.

            Each line starts with the model's roster KEY — that key, exactly as
            written, is what ``switch_model`` takes. The model you are running on
            now is marked. The listing is the roster's own order, not a ranking.
            """
            observer = observer_or_none()
            if observer is None:
                raise ToolObserverGone("list_models used after its owning agent was stopped")
            return _render_model_rows(observer.list_model_rows())

        list_models.__doc__ = params.format_docstring(cleandoc(list_models.__doc__ or ""))
        return list_models

    def _switch_model_factory(self, params: SwitchModel) -> Callable[..., Any]:
        """Create the ``switch_model`` callable.

        The parameter is named ``model`` because ADR-018 §5 declares the card
        surface as ``switch_model(model: str) -> str``; the observer's own
        parameter is ``key``, from §6. Two contracts, two names, deliberately —
        the command descriptor advertises ``model``.

        Args:
            params: Configuration for the switch capability.

        Returns:
            A one-argument callable named ``switch_model``.
        """
        observer_or_none = self._model_observer_or_none  # bound method -> weak edge to agent

        def switch_model(model: str) -> str:
            """Switch to a different model for the rest of this conversation.

            Args:
                model: The roster KEY of the model to switch to, exactly as
                    ``list_models`` prints it (for example
                    ``openai:gpt-5.2``). Call ``list_models`` first if you are
                    not sure what is available.

            Returns:
                A short confirmation of the outcome.
            """
            observer = observer_or_none()
            if observer is None:
                raise ToolObserverGone("switch_model used after its owning agent was stopped")
            try:
                outcome = observer.switch_model(model)
            except RetriableError:
                raise  # a retry from below is already readable; re-wrapping buries it
            except Exception as exc:
                # No stable exception type to name: the llm layer refuses a switch
                # for unrelated reasons (unknown key, compaction bounds, an entry
                # that will not build) and this package may not import it.
                raise RetriableError(f"Cannot switch to '{model}': {exc}") from exc
            # The live slot, dereferenced HERE and never held (ADR-041 §3). Last,
            # so a refusal leaves it untouched, and so the recorded key is one the
            # llm layer actually accepted. The agent's existing checkpoints
            # persist this in-place mutation — no event, no notification.
            observer.state.tool_state.active_model = model
            return outcome

        switch_model.__doc__ = params.format_docstring(cleandoc(switch_model.__doc__ or ""))
        return switch_model

    def _active_model_state_factory(
        self, params: ActiveModel
    ) -> Callable[[], ContextState | None]:
        """Create the ``active_model_state`` context-state provider.

        The callable's ``__name__`` is load-bearing twice over: ``ToolFactory``
        aggregates providers under it, and it is the key the baseline is persisted
        under in ``ToolState.context_baselines``.

        Args:
            params: Configuration for the active-model capability.

        Returns:
            Zero-arg provider producing the model in force, or ``None`` when that
            is unavailable. Never raises (ADR-037 §3).
        """
        observer_or_none = self._model_observer_or_none  # bound method -> weak edge to agent

        def active_model_state() -> ActiveModelState | None:
            try:
                observer = observer_or_none()
                if observer is None:
                    return None  # agent gone -> state unavailable
                # The roster's own flag, never ``ToolState.active_model``: the slot
                # is a persisted PREFERENCE the agent layer re-applies on restore,
                # so rendering it would show a stale key as though it were live.
                active = next((row for row in observer.list_model_rows() if row.active), None)
                if active is None:
                    return None
                return ActiveModelState(key=active.key)
            except Exception:
                logger.error("Failed to get active model state", exc_info=True)
                return None

        return active_model_state
