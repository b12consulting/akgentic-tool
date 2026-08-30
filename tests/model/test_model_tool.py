"""Tests for ``ModelTool``: the card on three channels (Story 36-2).

The fake observer extends the shape story 36-1 proved at
``tests/model/test_model_row_and_observer.py:99-134`` — ``_ObserverBase`` supplies
the six members ``ActorToolObserver`` declares on this branch base, and ``state``
stays a plain mutable attribute so the staleness test can replace the carrier
wholesale, exactly as ``init_state()`` does on restore.
"""

from __future__ import annotations

import gc
import inspect
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType

from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL, ToolCard, ToolFactory, ToolState
from akgentic.tool.errors import RetriableError, ToolObserverGone
from akgentic.tool.model import ModelSwitchToolObserver
from akgentic.tool.model.state import ActiveModelState, ModelRow
from akgentic.tool.model.tool import ActiveModel, ListModels, ModelTool, SwitchModel

# ---------------------------------------------------------------------------
# The fake observer — recording and controllable
# ---------------------------------------------------------------------------


class _Carrier:
    """Minimal ``ToolStateCarrier``: an object exposing ``tool_state``."""

    def __init__(self) -> None:
        self.tool_state = ToolState()


class _ObserverBase:
    """Every member ``ActorToolObserver`` declares on this base, and nothing more."""

    def __init__(self) -> None:
        self.myAddress = SimpleNamespace(name="@Alice")  # noqa: N815 — protocol member name
        self.orchestrator = None
        self.team_id = uuid.uuid4()
        self.state = _Carrier()

    def notify_event(self, event: object) -> None:
        pass

    def proxy_ask(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
        timeout: int | None = None,
    ) -> Any:
        return None


class _FakeObserver(_ObserverBase):
    """The base plus both model-switch members, recording and controllable.

    ``rows`` is what ``list_model_rows`` returns; ``switch_calls`` records every
    key ``switch_model`` was handed; ``raise_on_switch`` and ``raise_on_list``
    make the refusal and degradation paths reachable.
    """

    def __init__(self, rows: list[ModelRow] | None = None) -> None:
        super().__init__()
        self.rows: list[ModelRow] = rows if rows is not None else []
        self.switch_calls: list[str] = []
        self.raise_on_switch: Exception | None = None
        self.raise_on_list: Exception | None = None

    def list_model_rows(self) -> list[ModelRow]:
        if self.raise_on_list is not None:
            raise self.raise_on_list
        return list(self.rows)

    def switch_model(self, key: str) -> str:
        if self.raise_on_switch is not None:
            raise self.raise_on_switch
        self.switch_calls.append(key)
        return f"Now using {key}."


def _row(key: str, active: bool = False, context_length: int | None = 400_000) -> ModelRow:
    """One roster row, with ``provider``/``model`` split out of *key*."""
    provider, _, model = key.partition(":")
    return ModelRow(
        key=key,
        provider=provider,
        model=model,
        active=active,
        context_length=context_length,
    )


# A deliberately non-alphabetical roster: the card must not re-order it.
def _roster() -> list[ModelRow]:
    return [
        _row("openai:gpt-5.2", active=True),
        _row("anthropic:claude-opus-5"),
        _row("mistral:medium-3", context_length=None),
    ]


def _wired_card(rows: list[ModelRow] | None = None, **fields: object) -> tuple[
    ModelTool, _FakeObserver
]:
    """A card with an attached fake observer; the observer is returned to keep it alive."""
    observer = _FakeObserver(rows)
    assert isinstance(observer, ModelSwitchToolObserver)  # structural conformance
    card = ModelTool.model_validate(fields)
    card.observer(observer)
    return card, observer


def test_the_fake_observer_conforms_to_the_protocol() -> None:
    """A drifting fake must fail loudly rather than quietly testing nothing."""
    assert isinstance(_FakeObserver(), ModelSwitchToolObserver)


# ---------------------------------------------------------------------------
# AC 3 — ActiveModelState
# ---------------------------------------------------------------------------


def test_active_model_state_renders_the_key_in_full() -> None:
    rendered = ActiveModelState(key="openai:gpt-5.2").render_full()

    assert rendered.count("\n") == 0
    assert "openai:gpt-5.2" in rendered


def test_active_model_state_delta_is_none_when_the_key_is_unchanged() -> None:
    state = ActiveModelState(key="openai:gpt-5.2")

    assert state.render_delta(ActiveModelState(key="openai:gpt-5.2")) is None


def test_active_model_state_delta_names_both_ends_of_the_move() -> None:
    delta = ActiveModelState(key="openai:gpt-5.2").render_delta(
        ActiveModelState(key="anthropic:claude-opus-5")
    )

    assert delta is not None
    assert delta.count("\n") == 0
    assert "anthropic:claude-opus-5" in delta
    assert "openai:gpt-5.2" in delta


def test_active_model_state_carries_exactly_one_field() -> None:
    """The state is O(1) by construction — every other column is derivable."""
    assert list(ActiveModelState.model_fields) == ["key"]


def test_active_model_state_round_trips() -> None:
    state = ActiveModelState(key="openai:gpt-5.2")

    assert ActiveModelState.model_validate(state.model_dump()) == state


def test_active_model_state_round_trips_out_of_tool_state_as_its_concrete_type() -> None:
    """``SerializableBaseModel``'s ``__model__`` stamp is what makes this work."""
    tool_state = ToolState(context_baselines={"active_model_state": ActiveModelState(key="a:b")})

    restored = ToolState.model_validate(tool_state.model_dump())

    baseline = restored.context_baselines["active_model_state"]
    assert isinstance(baseline, ActiveModelState)
    assert baseline.key == "a:b"


# ---------------------------------------------------------------------------
# AC 1, AC 2 — the card's shape and its serializability
# ---------------------------------------------------------------------------


def test_the_card_declares_exactly_the_three_capability_fields() -> None:
    assert list(ModelTool.model_fields) == ["list_models", "switch_model", "active_model"]


@pytest.mark.parametrize(
    ("param", "expected"),
    [
        (ListModels, {TOOL_CALL, COMMAND}),
        (SwitchModel, {TOOL_CALL, COMMAND}),
        (ActiveModel, {LLM_CONTEXT}),
    ],
)
def test_each_param_declares_its_channels(param: type, expected: set[object]) -> None:
    assert param().expose == expected


def test_the_card_adds_no_private_attr_and_no_config_of_its_own() -> None:
    """Golden Rule 1b: no escape hatch — the card carries serializable fields only.

    Compared against the base rather than against empty: ``_observer_ref`` is the
    weak observer edge every card inherits, and ``arbitrary_types_allowed`` is
    ``SerializableBaseModel``'s, in ``akgentic-core``.
    """
    assert ModelTool.__private_attributes__ == ToolCard.__private_attributes__
    assert ModelTool.model_config == ToolCard.model_config


def test_the_card_does_not_override_observer_or_depends_on() -> None:
    """No actor, no proxy call, no dependency — the card falls through to the base."""
    assert "observer" not in vars(ModelTool)
    assert "depends_on" not in vars(ModelTool)
    assert ModelTool().depends_on == []


def test_default_card_round_trips_through_pydantic_and_json() -> None:
    card = ModelTool()

    assert ModelTool.model_validate(card.model_dump()) == card
    assert ModelTool.model_validate_json(card.model_dump_json()) == card


def test_a_wired_card_still_round_trips_and_the_observer_never_enters_the_dump() -> None:
    """The observer lives in the inherited ``_observer_ref`` ``PrivateAttr``."""
    card, observer = _wired_card()

    dumped = card.model_dump()
    assert "_observer_ref" not in dumped
    assert "observer" not in dumped
    assert ModelTool.model_validate(dumped) == ModelTool()


@pytest.mark.parametrize("field", ["list_models", "switch_model", "active_model"])
def test_card_with_disabled_capability_round_trips(field: str) -> None:
    card = ModelTool.model_validate({field: False})

    restored = ModelTool.model_validate(card.model_dump())
    assert restored == card
    assert getattr(restored, field) is False


def test_a_narrowed_exposure_survives_the_round_trip() -> None:
    card = ModelTool(list_models=ListModels(expose={COMMAND}))

    restored = ModelTool.model_validate(card.model_dump())

    assert restored == card
    assert isinstance(restored.list_models, ListModels)
    assert restored.list_models.expose == {COMMAND}


def test_no_param_normalizes_a_persisted_system_prompt_exposure() -> None:
    """These params are new — nothing persisted can carry the old channel.

    ``normalize_system_prompt_to_llm_context`` is for params whose card *moved*
    ``SYSTEM_PROMPT`` content; attaching it here is the failure that rule exists
    to prevent, so an explicit ``SYSTEM_PROMPT`` must survive untouched (and be
    dropped by the channel gate, like any other unserved channel).
    """
    from akgentic.tool.core import SYSTEM_PROMPT

    for param in (ListModels, SwitchModel, ActiveModel):
        assert param(expose={SYSTEM_PROMPT}).expose == {SYSTEM_PROMPT}


# ---------------------------------------------------------------------------
# AC 4 — the wiring table: assert the actual lists, never the declaration
# ---------------------------------------------------------------------------


def test_default_card_serves_two_tools_two_commands_and_one_provider() -> None:
    card, observer = _wired_card()

    assert [tool.__name__ for tool in card.get_tools()] == ["list_models", "switch_model"]
    assert set(card.get_commands()) == {ListModels, SwitchModel}
    assert len(card.get_context_states()) == 1


def test_disabling_list_models_removes_only_that_capability() -> None:
    card, observer = _wired_card(list_models=False)

    assert [tool.__name__ for tool in card.get_tools()] == ["switch_model"]
    assert set(card.get_commands()) == {SwitchModel}
    assert len(card.get_context_states()) == 1


def test_disabling_switch_model_removes_only_that_capability() -> None:
    card, observer = _wired_card(switch_model=False)

    assert [tool.__name__ for tool in card.get_tools()] == ["list_models"]
    assert set(card.get_commands()) == {ListModels}
    assert len(card.get_context_states()) == 1


def test_disabling_active_model_removes_only_the_provider() -> None:
    card, observer = _wired_card(active_model=False)

    assert [tool.__name__ for tool in card.get_tools()] == ["list_models", "switch_model"]
    assert set(card.get_commands()) == {ListModels, SwitchModel}
    assert card.get_context_states() == []


def test_narrowing_list_models_to_command_drops_it_from_the_tool_list() -> None:
    """The silent drop: a capability off its served channel vanishes with no error."""
    card, observer = _wired_card(list_models=ListModels(expose={COMMAND}))

    assert [tool.__name__ for tool in card.get_tools()] == ["switch_model"]
    assert set(card.get_commands()) == {ListModels, SwitchModel}
    assert len(card.get_context_states()) == 1


def test_active_model_exposed_on_command_serves_nothing() -> None:
    """The card serves ``active_model`` on ``LLM_CONTEXT`` only."""
    card, observer = _wired_card(active_model=ActiveModel(expose={COMMAND}))

    assert [tool.__name__ for tool in card.get_tools()] == ["list_models", "switch_model"]
    assert set(card.get_commands()) == {ListModels, SwitchModel}
    assert card.get_context_states() == []


# ---------------------------------------------------------------------------
# AC 5, AC 6 — the provider name and the command registry
# ---------------------------------------------------------------------------


def test_the_factory_yields_exactly_one_provider_named_active_model_state() -> None:
    """The ``__name__`` is the aggregation key and the persisted baseline key."""
    observer = _FakeObserver(_roster())
    factory = ToolFactory([ModelTool()], observer=observer)

    providers = factory.get_context_states()

    assert [provider.__name__ for provider in providers] == ["active_model_state"]


def test_the_registry_names_exactly_the_two_commands_in_order() -> None:
    observer = _FakeObserver(_roster())
    registry = ToolFactory([ModelTool()], observer=observer).get_command_registry()

    assert [d.name for d in registry.descriptors()] == ["list_models", "switch_model"]


def test_the_switch_model_descriptor_carries_one_required_string_argument() -> None:
    observer = _FakeObserver(_roster())
    registry = ToolFactory([ModelTool()], observer=observer).get_command_registry()

    descriptor = next(d for d in registry.descriptors() if d.name == "switch_model")

    assert [(arg.name, arg.type, arg.required) for arg in descriptor.args] == [
        ("model", "string", True)
    ]


def test_list_models_dispatches_to_the_rendered_listing() -> None:
    observer = _FakeObserver(_roster())
    registry = ToolFactory([ModelTool()], observer=observer).get_command_registry()

    dispatched = registry.dispatch("/list_models")

    assert dispatched == observer_free_render(observer)


def observer_free_render(observer: _FakeObserver) -> str:
    """The listing the card should produce for *observer*'s roster."""
    from akgentic.tool.model.tool import _render_model_rows

    return _render_model_rows(observer.rows)


def test_switch_model_dispatch_delivers_the_colon_bearing_key_verbatim() -> None:
    """``shlex`` and the keyword rule must leave ``openai:gpt-5.2`` positional."""
    observer = _FakeObserver(_roster())
    registry = ToolFactory([ModelTool()], observer=observer).get_command_registry()

    registry.dispatch("/switch_model openai:gpt-5.2")

    assert observer.switch_calls == ["openai:gpt-5.2"]


def test_switch_model_binds_by_the_declared_parameter_name() -> None:
    """The descriptor advertises ``model`` (ADR-018 §5), not the observer's ``key``."""
    observer = _FakeObserver(_roster())
    registry = ToolFactory([ModelTool()], observer=observer).get_command_registry()

    registry.dispatch("/switch_model model=openai:gpt-5.2")

    assert observer.switch_calls == ["openai:gpt-5.2"]


def test_an_unrecognised_keyword_token_binds_positionally_and_reaches_the_observer() -> None:
    """``key=…`` is NOT rejected as a keyword — the whole token binds to ``model``.

    ``_classify_tokens`` counts a token as a keyword only when the text before its
    first ``=`` is a known parameter name, so a value containing ``=`` is never
    silently swallowed. ``key`` is the *observer's* name, not the callable's, so
    the entire token is classified positional and arrives at the observer verbatim,
    to be refused downstream as an unknown roster key. The registry's own
    ``unknown keyword argument`` branch is unreachable from ``dispatch`` for it.
    """
    observer = _FakeObserver(_roster())
    registry = ToolFactory([ModelTool()], observer=observer).get_command_registry()

    dispatched = registry.dispatch("/switch_model key=openai:gpt-5.2")

    assert observer.switch_calls == ["key=openai:gpt-5.2"]
    assert "unknown keyword" not in dispatched


def test_the_switch_model_callable_declares_model_as_its_only_parameter() -> None:
    card, observer = _wired_card(_roster())
    switch = next(tool for tool in card.get_tools() if tool.__name__ == "switch_model")

    signature = inspect.signature(switch, eval_str=True)

    assert list(signature.parameters) == ["model"]
    assert signature.parameters["model"].annotation is str


# ---------------------------------------------------------------------------
# AC 7, AC 8, AC 9 — the switch writes the live slot, and only on success
# ---------------------------------------------------------------------------


def _switch_callable(card: ModelTool) -> Any:
    return next(tool for tool in card.get_tools() if tool.__name__ == "switch_model")


def test_a_successful_switch_writes_the_key_to_the_live_slot() -> None:
    card, observer = _wired_card(_roster())

    outcome = _switch_callable(card)("anthropic:claude-opus-5")

    # Read off the state object the observer exposes, never off a held reference.
    assert observer.state.tool_state.active_model == "anthropic:claude-opus-5"
    assert outcome == "Now using anthropic:claude-opus-5."


def test_the_slot_starts_empty_so_the_write_is_what_fills_it() -> None:
    card, observer = _wired_card(_roster())

    assert observer.state.tool_state.active_model is None


def test_the_switch_reads_the_live_state_object_not_the_one_bound_at_build_time() -> None:
    """AC 8 — the staleness guard, this story's central acceptance criterion.

    ``init_state()`` replaces the agent's state object wholesale on restore. A
    card that dereferenced ``observer.state.tool_state`` at bind time would keep
    writing into the abandoned carrier, silently, forever.
    """
    card, observer = _wired_card(_roster())
    switch = _switch_callable(card)

    switch("openai:gpt-5.2")
    stale_carrier = observer.state

    observer.state = _Carrier()  # what init_state() does on restore
    switch("anthropic:claude-opus-5")

    assert observer.state.tool_state.active_model == "anthropic:claude-opus-5"
    assert stale_carrier.tool_state.active_model == "openai:gpt-5.2"
    assert observer.state is not stale_carrier


def test_the_switch_dereferences_the_slot_after_the_call_not_before_it() -> None:
    """The second forbidden form, which the bind-time guard above cannot see.

    Task 6 forbids *a local computed before the observer call*, not only a
    bind-time capture. The two are different defects: the guard above replaces
    the carrier **between** calls, so a per-call hoist taken before
    ``observer.switch_model(...)`` still lands correctly and stays green.

    It is wrong all the same. ``observer.switch_model`` reaches the llm layer,
    and a restore that replaces the agent's state object while that call is in
    flight leaves the hoisted local pointing at the abandoned carrier — the write
    then vanishes with no error. Here the observer performs that replacement
    itself, which is the only way to put the swap inside the call.
    """

    class _SwappingObserver(_FakeObserver):
        """Replaces its own state carrier mid-switch, as a restore would."""

        def switch_model(self, key: str) -> str:
            outcome = super().switch_model(key)
            self.state = _Carrier()  # init_state() lands while the call is in flight
            return outcome

    observer = _SwappingObserver(_roster())
    card = ModelTool()
    card.observer(observer)
    stale_carrier = observer.state

    _switch_callable(card)("anthropic:claude-opus-5")

    assert observer.state is not stale_carrier
    assert observer.state.tool_state.active_model == "anthropic:claude-opus-5"
    assert stale_carrier.tool_state.active_model is None


def test_a_refused_switch_raises_retriable_and_names_the_key_and_the_reason() -> None:
    card, observer = _wired_card(_roster())
    observer.raise_on_switch = ValueError("unknown roster key")

    with pytest.raises(RetriableError) as excinfo:
        _switch_callable(card)("openai:nope")

    message = str(excinfo.value)
    assert "openai:nope" in message
    assert "unknown roster key" in message


def test_a_refused_switch_leaves_an_empty_slot_untouched() -> None:
    card, observer = _wired_card(_roster())
    observer.raise_on_switch = ValueError("unknown roster key")

    with pytest.raises(RetriableError):
        _switch_callable(card)("openai:nope")

    assert observer.state.tool_state.active_model is None


def test_a_refused_switch_does_not_overwrite_a_previously_written_key() -> None:
    card, observer = _wired_card(_roster())
    switch = _switch_callable(card)
    switch("openai:gpt-5.2")

    observer.raise_on_switch = ValueError("compaction bounds violated")
    with pytest.raises(RetriableError):
        switch("mistral:medium-3")

    assert observer.state.tool_state.active_model == "openai:gpt-5.2"


def test_a_retriable_error_from_the_observer_passes_through_as_the_same_object() -> None:
    """A retry from below is already model-readable; re-wrapping would bury it."""
    card, observer = _wired_card(_roster())
    raised = RetriableError("that entry cannot be built")
    observer.raise_on_switch = raised

    with pytest.raises(RetriableError) as excinfo:
        _switch_callable(card)("mistral:medium-3")

    assert excinfo.value is raised


def test_dispatching_a_refused_switch_returns_the_failure_string() -> None:
    """A different surface, and a different fact: ``_invoke`` catches everything.

    ``CommandRegistry._invoke`` turns any exception into a plain result string, so
    a ``pytest.raises`` around ``dispatch`` could never fire — the refusal is
    asserted on the callable above, and the string here.
    """
    observer = _FakeObserver(_roster())
    observer.raise_on_switch = ValueError("unknown roster key")
    registry = ToolFactory([ModelTool()], observer=observer).get_command_registry()

    result = registry.dispatch("/switch_model openai:nope")

    assert result is not None
    assert result.startswith("Command 'switch_model' failed:")
    assert "openai:nope" in result


# ---------------------------------------------------------------------------
# AC 10 — list_models renders deterministically, in roster order
# ---------------------------------------------------------------------------


def _list_callable(card: ModelTool) -> Any:
    return next(tool for tool in card.get_tools() if tool.__name__ == "list_models")


def test_list_models_preserves_the_roster_order_it_was_given() -> None:
    """A deliberately non-alphabetical roster: the card must not sort it."""
    card, observer = _wired_card(_roster())

    lines = _list_callable(card)().splitlines()

    assert [line.split()[0].lstrip("*- ") for line in lines] == [
        "openai:gpt-5.2",
        "anthropic:claude-opus-5",
        "mistral:medium-3",
    ]


def test_list_models_marks_the_active_row_and_only_that_row() -> None:
    card, observer = _wired_card(_roster())

    lines = _list_callable(card)().splitlines()

    marked = [line for line in lines if "active" in line.lower()]
    assert len(marked) == 1
    assert "openai:gpt-5.2" in marked[0]


def test_list_models_is_byte_identical_across_two_calls() -> None:
    """Nothing turn-varying — no timestamp, no counter."""
    card, observer = _wired_card(_roster())
    list_models = _list_callable(card)

    assert list_models() == list_models()


def test_list_models_returns_a_sentinel_for_an_empty_roster() -> None:
    card, observer = _wired_card([])

    rendered = _list_callable(card)()

    assert rendered.strip() != ""
    assert "openai" not in rendered


def test_list_models_renders_one_line_per_row() -> None:
    card, observer = _wired_card(_roster())

    assert len(_list_callable(card)().splitlines()) == 3


# ---------------------------------------------------------------------------
# AC 11 — the provider degrades to None and never raises
# ---------------------------------------------------------------------------


def _provider(card: ModelTool) -> Any:
    return card.get_context_states()[0]


def test_the_provider_returns_the_active_rows_key() -> None:
    card, observer = _wired_card(_roster())

    state = _provider(card)()

    assert isinstance(state, ActiveModelState)
    assert state.key == "openai:gpt-5.2"


def test_the_provider_reads_the_roster_flag_not_the_persisted_slot() -> None:
    """The slot is a persisted preference the agent re-applies; the flag is truth."""
    card, observer = _wired_card(_roster())
    observer.state.tool_state.active_model = "mistral:medium-3"

    state = _provider(card)()

    assert isinstance(state, ActiveModelState)
    assert state.key == "openai:gpt-5.2"


def test_the_provider_returns_none_for_an_empty_roster() -> None:
    card, observer = _wired_card([])

    assert _provider(card)() is None


def test_the_provider_returns_none_when_no_row_is_active() -> None:
    card, observer = _wired_card([_row("openai:gpt-5.2"), _row("anthropic:claude-opus-5")])

    assert _provider(card)() is None


def test_the_provider_returns_none_when_the_observer_raises() -> None:
    card, observer = _wired_card(_roster())
    observer.raise_on_list = RuntimeError("the roster is unreachable")

    assert _provider(card)() is None


def test_the_provider_returns_none_once_the_observer_is_collected() -> None:
    card, observer = _wired_card(_roster())
    provider = _provider(card)

    del observer
    gc.collect()

    assert provider() is None


# ---------------------------------------------------------------------------
# AC 12 — a collected observer fails the callables with ToolObserverGone
# ---------------------------------------------------------------------------


def test_the_narrowing_accessor_raises_once_the_agent_is_collected() -> None:
    """The raising accessor is the in-life form: gone means gone, not ``None``."""
    card, observer = _wired_card(_roster())

    assert card._model_observer() is observer

    del observer
    gc.collect()

    assert card._model_observer_or_none() is None
    with pytest.raises(ToolObserverGone):
        card._model_observer()


def test_both_callables_raise_tool_observer_gone_once_the_agent_is_collected() -> None:
    card, observer = _wired_card(_roster())
    list_models, switch = _list_callable(card), _switch_callable(card)

    del observer
    gc.collect()

    with pytest.raises(ToolObserverGone):
        list_models()
    with pytest.raises(ToolObserverGone):
        switch("openai:gpt-5.2")


def _wire_and_drop_the_factory(observer: _FakeObserver) -> tuple[ModelTool, Any, Any]:
    """Card, registry and providers, with the ``ToolFactory`` itself out of scope.

    ``ToolFactory`` stores its observer **strongly** (``self.observer``), so a
    factory kept alive would pin the agent no matter how the card behaves. Only
    the card, its closures and the registry are under test here — the same reason
    ``tests/test_command_registry_weak_observer.py`` builds its registry inside a
    helper and returns only the registry.
    """
    card = ModelTool()
    factory = ToolFactory([card], observer=observer)
    return card, factory.get_command_registry(), factory.get_context_states()


def test_the_card_its_closures_and_its_registry_do_not_pin_the_observer() -> None:
    import weakref

    observer = _FakeObserver(_roster())
    card, registry, providers = _wire_and_drop_the_factory(observer)

    reference = weakref.ref(observer)
    del observer
    gc.collect()

    assert reference() is None
    # Still held, yet the agent was reclaimed.
    assert card is not None and registry is not None and providers


# ---------------------------------------------------------------------------
# AC 13 — exports
# ---------------------------------------------------------------------------


def test_model_tool_is_the_same_object_through_both_export_paths() -> None:
    import akgentic.tool as tool_package
    import akgentic.tool.model as model_package
    from akgentic.tool import ModelTool as RootModelTool
    from akgentic.tool.model import ModelTool as PackageModelTool

    assert PackageModelTool is ModelTool
    assert RootModelTool is ModelTool
    assert "ModelTool" in model_package.__all__
    assert "ModelTool" in tool_package.__all__


@pytest.mark.parametrize("name", ["ActiveModelState", "ListModels", "SwitchModel", "ActiveModel"])
def test_the_domain_local_names_are_exported_from_the_model_package_only(name: str) -> None:
    """Params and states stay domain-local — the ``ReadMailbox``/``Stop`` precedent."""
    import akgentic.tool as tool_package
    import akgentic.tool.model as model_package

    assert name in model_package.__all__
    assert hasattr(model_package, name)
    assert name not in tool_package.__all__
    assert not hasattr(tool_package, name)
