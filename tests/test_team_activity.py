"""Unit tests for ``who_is_working``, the team-activity capability on ``TeamTool``.

Mock-based throughout: telemetry is hand-built and handed to a fake orchestrator
proxy, so no real team runs. Every test is synchronous — the package pyproject
carries no ``[tool.pytest.ini_options]``, so CI runs ``asyncio_mode=strict``
while local runs inherit ``auto`` from the workspace root, and staying
synchronous keeps that divergence benign.
"""

from __future__ import annotations

import ast
import inspect
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, get_args
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core import ActorAddressProxy
from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent_config import BaseConfig
from akgentic.core.messages.message import Message, ResultMessage, UserMessage
from akgentic.core.messages.orchestrator import (
    ProcessedMessage,
    ReceivedMessage,
    SentMessage,
    StartMessage,
)
from akgentic.core.orchestrator import Orchestrator

import akgentic.tool.team.activity as activity_module
from akgentic.tool.core import COMMAND, TOOL_CALL, BaseToolParam
from akgentic.tool.core.deferred import DeferredPayload, DeferredWorker, poll_deferred
from akgentic.tool.errors import RetriableError
from akgentic.tool.team import (
    ActivitySummarizer,
    AgentActivity,
    FireTeamMember,
    GetRoleProfiles,
    GetTeamActivity,
    GetTeamRoster,
    HireTeamMember,
    TeamActivityReport,
    TeamTool,
)
from akgentic.tool.team.activity import (
    TEAM_ACTIVITY_ACTOR_NAME,
    TEAM_ACTIVITY_ACTOR_ROLE,
    UNRESOLVED_TASK,
    SummarizePayload,
    SummarizerWorker,
    TeamActivityActor,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _address(
    name: str,
    role: str = "Developer",
    *,
    is_user_proxy: bool = False,
    agent_id: uuid.UUID | None = None,
) -> ActorAddressProxy:
    """A real ``ActorAddress`` implementation built from a serialized payload."""
    return ActorAddressProxy(
        {
            "__actor_address__": True,
            "__actor_type__": "test.Agent",
            "agent_id": str(agent_id or uuid.uuid4()),
            "name": name,
            "role": role,
            "team_id": str(uuid.uuid4()),
            "squad_id": str(uuid.uuid4()),
            "is_user_proxy": is_user_proxy,
        }
    )


class _RecordingList(list[Message]):
    """A list that records which indices were read.

    Turns "entries older than the last resolution are never inspected" into an
    observable property instead of an intended one.
    """

    def __init__(self, items: list[Message]) -> None:
        super().__init__(items)
        self.read_indices: list[int] = []

    def __getitem__(self, index: Any) -> Any:
        self.read_indices.append(index)
        return super().__getitem__(index)


class _FakeOrchestrator:
    """Serves canned telemetry and records how it was asked for."""

    def __init__(self, messages: list[Message]) -> None:
        self._messages = messages
        self.get_messages_calls: list[type | None] = []
        self.children_created: list[tuple[type, BaseConfig]] = []
        self.sent_view: _RecordingList | None = None
        self.team: list[ActorAddress] = []

    def get_messages(
        self, sender: ActorAddress | None = None, message_type: type | None = None
    ) -> list[Message]:
        self.get_messages_calls.append(message_type)
        if message_type is None:
            raise AssertionError(
                "unfiltered get_messages returns the orchestrator's live list by reference"
            )
        matched = [msg for msg in self._messages if isinstance(msg, message_type)]
        if message_type is SentMessage:
            self.sent_view = _RecordingList(matched)
            return self.sent_view
        return matched

    def get_team(self) -> list[ActorAddress]:
        return self.team

    def getChildrenOrCreate(  # noqa: N802
        self, actor_class: type, config: BaseConfig
    ) -> ActorAddress:
        self.children_created.append((actor_class, config))
        return _address(config.name, config.role)


class _FakeObserver:
    """Minimal ``TeamManagementToolObserver``: one address, two proxies, no actor system.

    ``TeamTool`` casts its observer to the team protocol and the cast is
    unchecked, so the hire/fire members are present as no-ops: a test that
    exercises the shared card through hire or fire must not explode on a missing
    attribute instead of failing on what it asserts.
    """

    def __init__(
        self,
        orchestrator_actor: _FakeOrchestrator,
        activity_actor: TeamActivityActor,
        my_address: ActorAddress,
    ) -> None:
        self.orchestrator: ActorAddress | None = _address("@Orchestrator", "Orchestrator")
        self.myAddress = my_address  # noqa: N815 — protocol member name
        self.team_id = uuid.uuid4()
        self.proxy_ask_calls: list[type | None] = []
        self._orchestrator_actor = orchestrator_actor
        self._activity_actor = activity_actor

    def notify_event(self, event: object) -> None:
        """Unused here; present so the observer satisfies the protocol."""

    def createActor(self, actor_class: type, *, config: object) -> ActorAddress:  # noqa: N802
        """Unused here; present so the team protocol is satisfied."""
        return _address("@Child")

    def on_hire(self, address: ActorAddress) -> None:
        """Unused here; present so the team protocol is satisfied."""

    def on_fire(self, address: ActorAddress) -> None:
        """Unused here; present so the team protocol is satisfied."""

    def proxy_ask(
        self, actor: ActorAddress, actor_type: type | None = None, timeout: int | None = None
    ) -> Any:
        self.proxy_ask_calls.append(actor_type)
        if actor_type is Orchestrator:
            return self._orchestrator_actor
        return self._activity_actor


class _SpyWorker(DeferredWorker):
    """Counts its own instantiation, so "no worker was spawned" is provable."""

    spawned: ClassVar[list[str]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _SpyWorker.spawned.append("spawned")

    def produce(self, payload: DeferredPayload) -> Any:
        return "spy-summary"


class _SpyActivityActor(TeamActivityActor):
    """Cache actor whose worker class is the spy."""

    def worker_class(self) -> type[DeferredWorker]:
        return _SpyWorker


def _spawning_create(actor_class: type, *, config: BaseConfig) -> Any:
    """Stand-in for ``createActor`` that really instantiates the worker class."""
    actor_class()
    return MagicMock()


def _make_activity_actor(spy: bool = False) -> TeamActivityActor:
    """An initialised cache actor without a running actor system."""
    actor: TeamActivityActor = _SpyActivityActor() if spy else TeamActivityActor()
    actor.config = BaseConfig(name=TEAM_ACTIVITY_ACTOR_NAME, role=TEAM_ACTIVITY_ACTOR_ROLE)
    actor.on_start()
    return actor


# ---------------------------------------------------------------------------
# Telemetry builders
# ---------------------------------------------------------------------------


def _received(
    sender: ActorAddress, message_id: uuid.UUID, *, age_seconds: float = 1.0
) -> ReceivedMessage:
    return ReceivedMessage(
        message_id=message_id,
        sender=sender,
        timestamp=datetime.now(UTC) - timedelta(seconds=age_seconds),
    )


def _processed(sender: ActorAddress, message_id: uuid.UUID) -> ProcessedMessage:
    return ProcessedMessage(message_id=message_id, sender=sender)


def _sent(sender: ActorAddress, message: Message) -> SentMessage:
    return SentMessage(message=message, recipient=sender, sender=sender)


def _task(content: str, message_id: uuid.UUID | None = None) -> UserMessage:
    return UserMessage(content=content, id=message_id or uuid.uuid4())


# ---------------------------------------------------------------------------
# Fixtures / harness
# ---------------------------------------------------------------------------


class _Harness:
    """A bound ``TeamTool`` plus the doubles behind it.

    ``get_team_activity`` defaults to a **summarizer-configured** capability, so
    the derivation tests exercise the path that owns the cache actor. Pass the
    field explicitly (``True``, ``False``, or a hand-built ``GetTeamActivity``) to
    test the gates themselves.
    """

    def __init__(
        self,
        messages: list[Message],
        *,
        caller: ActorAddress | None = None,
        spy_worker: bool = False,
        get_team_activity: GetTeamActivity | bool | None = None,
        max_task_chars: int = 200,
        stale_after_seconds: float = 300.0,
        model: str = "openai:gpt-5.2-mini",
        poll_attempts: int = 5,
        poll_delay_seconds: float = 0.4,
    ) -> None:
        if get_team_activity is None:
            get_team_activity = GetTeamActivity(
                summarizer=ActivitySummarizer(
                    model=model,
                    poll_attempts=poll_attempts,
                    poll_delay_seconds=poll_delay_seconds,
                ),
                max_task_chars=max_task_chars,
                stale_after_seconds=stale_after_seconds,
            )
        self.caller = caller or _address("@Manager", "Manager")
        self.orchestrator = _FakeOrchestrator(messages)
        self.activity = _make_activity_actor(spy=spy_worker)
        self.observer = _FakeObserver(self.orchestrator, self.activity, self.caller)
        self.tool = TeamTool(get_team_activity=get_team_activity)
        self.tool.observer(self.observer)

    @property
    def who_is_working(self) -> Callable[..., TeamActivityReport]:
        """The activity callable, selected by name — ``get_tools()[0]`` is hire."""
        return next(
            tool for tool in self.tool.get_tools() if tool.__name__ == "who_is_working"
        )

    def run(self, summarize_over: int | None = None) -> TeamActivityReport:
        with (
            patch.object(self.activity, "createActor", side_effect=_spawning_create),
            patch.object(self.activity, "proxy_tell", return_value=MagicMock()),
        ):
            callable_ = self.who_is_working
            if "summarize_over" in inspect.signature(callable_).parameters:
                return callable_(summarize_over=summarize_over)
            assert summarize_over is None, "the truncate-only variant takes no threshold"
            return callable_()


@pytest.fixture(autouse=True)
def _reset_spy() -> None:
    _SpyWorker.spawned.clear()


# ---------------------------------------------------------------------------
# AC1 — the two report models
# ---------------------------------------------------------------------------


class TestReportModels:
    """AC1: typed report shapes, no ``dict[str, Any]`` anywhere."""

    def test_agent_activity_fields(self) -> None:
        expected = {
            "name",
            "agent_id",
            "role",
            "message_id",
            "task",
            "summarized",
            "started_at",
            "busy_for_seconds",
            "suspect",
        }
        assert set(AgentActivity.model_fields) == expected

    def test_report_fields(self) -> None:
        assert set(TeamActivityReport.model_fields) == {
            "generated_at",
            "members",
            "pending_summaries",
        }

    def test_suspect_defaults_to_false(self) -> None:
        row = AgentActivity(
            name="@Dev1",
            agent_id=uuid.uuid4(),
            role="Developer",
            message_id=uuid.uuid4(),
            task="t",
            summarized=False,
            started_at=datetime.now(UTC),
            busy_for_seconds=1.0,
        )
        assert row.suspect is False

    def test_models_round_trip_through_serialization(self) -> None:
        report = TeamActivityReport(
            generated_at=datetime.now(UTC), members=[], pending_summaries=0
        )
        assert report.model_dump()["pending_summaries"] == 0


# ---------------------------------------------------------------------------
# AC2 — a capability ON TeamTool, configured through the param models
# ---------------------------------------------------------------------------


class TestCardShape:
    """AC2: configuration on the param models, runtime handles in PrivateAttr."""

    def test_the_capability_is_off_by_default(self) -> None:
        assert TeamTool().get_team_activity is False

    def test_configuration_defaults(self) -> None:
        params = GetTeamActivity()
        assert params.summarizer is None
        assert params.stale_after_seconds == 300.0
        assert params.max_task_chars == 200

        summarizer = ActivitySummarizer()
        assert summarizer.model == "openai:gpt-5.2-mini"
        assert summarizer.poll_attempts == 5
        assert summarizer.poll_delay_seconds == 0.4

    def test_a_bound_card_still_round_trips_through_serialization(self) -> None:
        """Golden Rule #1b, stated as the guarantee it exists for.

        Asserting on ``arbitrary_types_allowed`` would be vacuous — it is already
        True on ``SerializableBaseModel``, so every card in the package inherits
        it. What the rule actually protects is that a card carrying **live actor
        proxies** still serializes, which holds only while those proxies are
        private attributes rather than fields.
        """
        harness = _Harness([], max_task_chars=99)
        assert harness.tool._activity_proxy is not None  # genuinely bound

        restored = TeamTool.model_validate(harness.tool.model_dump())
        assert isinstance(restored.get_team_activity, GetTeamActivity)
        assert restored.get_team_activity.max_task_chars == 99
        assert restored.get_team_activity.summarizer is not None
        assert restored._activity_proxy is None, "runtime state must not survive a round trip"

    def test_every_field_is_a_capability_union(self) -> None:
        """No raw runtime type may reach a Pydantic field (Golden Rule #1b).

        Asserted on the union's *members*, not on ``str(annotation)``: a substring
        check for ``"bool"`` also passes for ``dict[str, bool]``, which is exactly
        the shape the rule forbids.
        """
        for name, field in TeamTool.model_fields.items():
            members = set(get_args(field.annotation))
            assert bool in members, f"{name} is not a `ParamModel | bool` capability field"
            others = members - {bool}
            assert len(others) == 1, f"{name} is not a two-member union: {members}"
            param = others.pop()
            assert isinstance(param, type) and issubclass(param, BaseToolParam), (
                f"{name} carries {param!r}, which is not a BaseToolParam"
            )

    def test_the_activity_proxy_is_a_private_attribute_not_a_field(self) -> None:
        assert "_activity_proxy" not in TeamTool.model_fields
        assert "_activity_proxy" in TeamTool.__private_attributes__

    def test_the_param_models_survive_a_json_round_trip(self) -> None:
        """Golden Rule #1b: configuration must reach a catalog entry and come back.

        Asserting on ``arbitrary_types_allowed`` would be vacuous — it is already
        True on ``SerializableBaseModel`` and every model in the package inherits
        it. JSON is the guarantee that actually matters: it fails the moment a
        non-serializable type reaches a field.
        """
        params = GetTeamActivity(
            summarizer=ActivitySummarizer(model="openai:gpt-5.2-mini", poll_attempts=2),
            max_task_chars=42,
        )
        restored = GetTeamActivity.model_validate_json(params.model_dump_json())
        assert restored.max_task_chars == 42
        assert restored.summarizer is not None
        assert restored.summarizer.poll_attempts == 2
        assert restored.expose == params.expose

    def test_card_serializes_without_runtime_state(self) -> None:
        harness = _Harness([])
        dumped = harness.tool.model_dump()
        assert dumped["get_team_activity"]["summarizer"]["model"] == "openai:gpt-5.2-mini"
        assert "_activity_proxy" not in dumped

    def test_observer_requires_an_orchestrator(self) -> None:
        observer = _FakeObserver(_FakeOrchestrator([]), _make_activity_actor(), _address("@Me"))
        observer.orchestrator = None
        with pytest.raises(ValueError, match="requires access to the orchestrator"):
            TeamTool().observer(observer)

    def test_the_summarizing_callable_requires_a_bound_observer(self) -> None:
        """Hire and fire are switched off so the activity gate is what is reached."""
        tool = TeamTool(
            hire_team_members=False,
            fire_team_members=False,
            get_team_activity=GetTeamActivity(summarizer=ActivitySummarizer()),
        )
        with pytest.raises(ValueError, match="observer\\(\\) must run"):
            tool.get_tools()

    def test_a_summary_budget_refuses_to_exist_without_a_summarizer(self) -> None:
        """The summarizing path must be unreachable without a configured summarizer."""
        with pytest.raises(ValueError, match="requires a configured summarizer"):
            activity_module.SummaryBudget.from_params(GetTeamActivity())


class TestNoTeamActivityToolAnywhere:
    """AC2/AC16: the separate card is gone from both namespaces."""

    def test_absent_from_the_team_package(self) -> None:
        import akgentic.tool.team as team_package

        assert "TeamActivityTool" not in team_package.__all__
        assert not hasattr(team_package, "TeamActivityTool")

    def test_absent_from_the_top_level_facade(self) -> None:
        import akgentic.tool as tool_package

        assert "TeamActivityTool" not in tool_package.__all__
        assert not hasattr(tool_package, "TeamActivityTool")

    def test_absent_from_the_activity_module(self) -> None:
        assert not hasattr(activity_module, "TeamActivityTool")


# ---------------------------------------------------------------------------
# AC2a — two independent gates; the summarizer is what creates the actor
# ---------------------------------------------------------------------------


class TestActorGate:
    """AC2a: the actor exists only to cache summaries, so no summarizer, no actor."""

    def test_no_actor_when_the_summarizer_is_absent(self) -> None:
        """RED when actor creation stops being gated on the summarizer.

        The assertion is on the ORCHESTRATOR'S CHILDREN, not on
        ``tool._activity_proxy is None``. The proxy-flavoured assertion passes
        against an implementation that calls ``getChildrenOrCreate`` and merely
        forgets to store the result — and that is the expensive half of the bug:
        the team has paid for an actor it will never use. Drop the
        ``summarizer is not None`` clause from ``_bind_activity_actor`` and this
        list holds one entry instead of none.
        """
        harness = _Harness([], get_team_activity=True)
        assert harness.orchestrator.children_created == []

    def test_no_actor_on_a_default_card(self) -> None:
        harness = _Harness([], get_team_activity=False)
        assert harness.orchestrator.children_created == []
        assert harness.tool._activity_proxy is None

    def test_the_summarizer_is_what_creates_the_actor(self) -> None:
        """Guard on the guard: the probe fires when the gate opens."""
        harness = _Harness([])
        assert len(harness.orchestrator.children_created) == 1
        assert harness.tool._activity_proxy is not None

    def test_no_proxy_ask_for_the_cache_actor_without_a_summarizer(self) -> None:
        harness = _Harness([], get_team_activity=True)
        assert harness.observer.proxy_ask_calls == [Orchestrator]


# ---------------------------------------------------------------------------
# AC2b — the tool signature is built from the configuration
# ---------------------------------------------------------------------------


class TestSignatureFollowsConfiguration:
    """AC2b: absent from the schema, not merely defaulted off."""

    def test_no_summarize_over_parameter_without_a_summarizer(self) -> None:
        """RED against a merely-defaulted parameter.

        A parameter given a default still appears in ``inspect.signature``, and it
        still appears in the JSON schema pydantic-ai derives for the model. Only
        its absence from ``parameters`` proves the model cannot ask for a summary
        that nothing is configured to produce — a behavioural assertion on the
        returned report cannot tell the two apart.
        """
        harness = _Harness([], get_team_activity=True)
        signature = inspect.signature(harness.who_is_working)
        assert "summarize_over" not in signature.parameters
        assert signature.parameters == {}

    def test_the_summarize_over_parameter_appears_with_a_summarizer(self) -> None:
        signature = inspect.signature(_Harness([]).who_is_working)
        assert signature.parameters["summarize_over"].default is None

    def test_both_variants_are_named_who_is_working(self) -> None:
        """The command registry derives the command name from ``__name__``."""
        assert _Harness([], get_team_activity=True).who_is_working.__name__ == "who_is_working"
        assert _Harness([]).who_is_working.__name__ == "who_is_working"

    def test_the_truncating_docstring_never_mentions_the_threshold(self) -> None:
        doc = _Harness([], get_team_activity=True).who_is_working.__doc__
        assert doc is not None
        assert "summarize_over" not in doc

    def test_the_summarizing_docstring_documents_the_threshold(self) -> None:
        doc = _Harness([]).who_is_working.__doc__
        assert doc is not None
        assert "summarize_over" in doc

    def test_instructions_are_appended_to_both_variants(self) -> None:
        truncating = _Harness(
            [], get_team_activity=GetTeamActivity(instructions="Ask sparingly.")
        ).who_is_working
        summarizing = _Harness(
            [],
            get_team_activity=GetTeamActivity(
                instructions="Ask sparingly.", summarizer=ActivitySummarizer()
            ),
        ).who_is_working
        assert "Ask sparingly." in (truncating.__doc__ or "")
        assert "Ask sparingly." in (summarizing.__doc__ or "")


# ---------------------------------------------------------------------------
# AC2c — backward compatibility, proven not assumed
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """AC2c: a payload that predates the field still validates and still behaves."""

    def _bound_default_tool(self) -> tuple[TeamTool, _FakeOrchestrator, _FakeObserver]:
        orchestrator = _FakeOrchestrator([])
        observer = _FakeObserver(orchestrator, _make_activity_actor(), _address("@Manager"))
        tool = TeamTool()
        tool.observer(observer)
        return tool, orchestrator, observer

    def test_a_payload_without_the_field_still_validates(self) -> None:
        payload = TeamTool().model_dump()
        del payload["get_team_activity"]
        restored = TeamTool.model_validate(payload)
        assert restored.get_team_activity is False

    def test_the_dispatch_surface_is_unchanged(self) -> None:
        tool, _, _ = self._bound_default_tool()

        assert [callable_.__name__ for callable_ in tool.get_tools()] == [
            "hire_members",
            "fire_members",
        ]
        assert set(tool.get_commands()) == {
            HireTeamMember,
            FireTeamMember,
            GetTeamRoster,
            GetRoleProfiles,
        }
        assert len(tool.get_system_prompts()) == 2
        assert GetTeamActivity not in tool.get_commands()
        assert all(
            callable_.__name__ != "who_is_working"
            for callable_ in [*tool.get_tools(), *tool.get_system_prompts()]
        )

    def test_a_default_card_creates_no_actor(self) -> None:
        _, orchestrator, _ = self._bound_default_tool()
        assert orchestrator.children_created == []

    def test_an_old_payload_behaves_exactly_like_a_fresh_default_card(self) -> None:
        payload = TeamTool().model_dump()
        del payload["get_team_activity"]
        restored = TeamTool.model_validate(payload)
        orchestrator = _FakeOrchestrator([])
        restored.observer(_FakeObserver(orchestrator, _make_activity_actor(), _address("@M")))

        assert [callable_.__name__ for callable_ in restored.get_tools()] == [
            "hire_members",
            "fire_members",
        ]
        assert orchestrator.children_created == []


# ---------------------------------------------------------------------------
# AC3 — channel registration and the callable name
# ---------------------------------------------------------------------------


class TestChannels:
    """AC3: both channels registered alongside hire/fire."""

    def test_default_expose_set(self) -> None:
        assert GetTeamActivity().expose == {TOOL_CALL, COMMAND}

    def test_tool_channel_registers_the_callable(self) -> None:
        tools = _Harness([]).tool.get_tools()
        assert [callable_.__name__ for callable_ in tools] == [
            "hire_members",
            "fire_members",
            "who_is_working",
        ]

    def test_command_channel_maps_the_param_class(self) -> None:
        commands = _Harness([]).tool.get_commands()
        assert GetTeamActivity in commands
        assert commands[GetTeamActivity].__name__ == "who_is_working"

    def test_disabled_capability_registers_nothing(self) -> None:
        harness = _Harness([], get_team_activity=False)
        assert all(
            callable_.__name__ != "who_is_working" for callable_ in harness.tool.get_tools()
        )
        assert GetTeamActivity not in harness.tool.get_commands()

    def test_tool_call_only_leaves_the_command_channel_empty(self) -> None:
        harness = _Harness([], get_team_activity=GetTeamActivity(expose={TOOL_CALL}))
        assert harness.who_is_working.__name__ == "who_is_working"
        assert GetTeamActivity not in harness.tool.get_commands()

    def test_command_only_leaves_the_tool_channel_empty(self) -> None:
        harness = _Harness([], get_team_activity=GetTeamActivity(expose={COMMAND}))
        assert all(
            callable_.__name__ != "who_is_working" for callable_ in harness.tool.get_tools()
        )
        assert GetTeamActivity in harness.tool.get_commands()

    def test_hire_and_fire_are_untouched_by_the_new_capability(self) -> None:
        harness = _Harness([])
        assert HireTeamMember in harness.tool.get_commands()
        assert FireTeamMember in harness.tool.get_commands()

    def test_gone_observer_raises_retriable(self) -> None:
        harness = _Harness([])
        callable_ = harness.who_is_working
        harness.tool._observer_ref = None
        with pytest.raises(RetriableError, match="shutting down"):
            callable_()


# ---------------------------------------------------------------------------
# AC4 — only the filtered get_messages form
# ---------------------------------------------------------------------------


class TestFilteredHistoryAccess:
    """AC4: the unfiltered call races the orchestrator's live list."""

    def test_every_call_is_filtered_to_a_telemetry_type(self) -> None:
        sender = _address("@Dev1")
        message = _task("build the thing")
        harness = _Harness([_sent(sender, message), _received(sender, message.id)])
        harness.run()

        calls = harness.orchestrator.get_messages_calls
        assert calls, "who_is_working never read the message history"
        assert all(call is not None for call in calls)
        assert set(calls) <= {ReceivedMessage, ProcessedMessage, SentMessage}

    def test_history_is_scanned_at_most_three_times(self) -> None:
        sender = _address("@Dev1")
        message = _task("build the thing")
        harness = _Harness([_sent(sender, message), _received(sender, message.id)])
        harness.run()
        assert len(harness.orchestrator.get_messages_calls) == 3

    def test_the_truncating_variant_is_filtered_too(self) -> None:
        sender = _address("@Dev1")
        message = _task("build the thing")
        harness = _Harness(
            [_sent(sender, message), _received(sender, message.id)], get_team_activity=True
        )
        harness.run()
        assert set(harness.orchestrator.get_messages_calls) <= {
            ReceivedMessage,
            ProcessedMessage,
            SentMessage,
        }


# ---------------------------------------------------------------------------
# AC5 — grouping by agent_id, never by name
# ---------------------------------------------------------------------------


class TestGroupingKey:
    """AC5: fire-then-re-hire reuses names, so a name key merges two actors."""

    def test_same_name_different_agent_ids_yields_two_rows(self) -> None:
        """RED against a name-keyed report.

        Both senders are called ``@Dev123`` — the shape a fire followed by a
        re-hire produces, since ``_hire_single_member`` only rejects a name that
        collides with a *currently live* member. Grouped by name the two open
        handlers collapse into a single row and this assertion fails; grouped by
        ``agent_id`` they stay two.
        """
        first = _address("@Dev123", agent_id=uuid.uuid4())
        second = _address("@Dev123", agent_id=uuid.uuid4())
        assert first.agent_id != second.agent_id

        task_a = _task("first incarnation task")
        task_b = _task("second incarnation task")
        harness = _Harness(
            [
                _sent(first, task_a),
                _sent(second, task_b),
                _received(first, task_a.id),
                _received(second, task_b.id),
            ]
        )
        report = harness.run()

        assert len(report.members) == 2
        assert {row.agent_id for row in report.members} == {first.agent_id, second.agent_id}
        assert {row.name for row in report.members} == {"@Dev123"}
        assert {row.task for row in report.members} == {
            "first incarnation task",
            "second incarnation task",
        }

    def test_one_actor_is_one_row_even_across_reported_names(self) -> None:
        """The converse: identity, not the label, decides how many rows there are."""
        agent_id = uuid.uuid4()
        old = _address("@Dev123", agent_id=agent_id)
        new = _address("@Renamed", agent_id=agent_id)
        task_a = _task("older open task")
        task_b = _task("newer open task")
        harness = _Harness(
            [
                _sent(old, task_a),
                _sent(new, task_b),
                _received(old, task_a.id, age_seconds=5),
                _received(new, task_b.id, age_seconds=1),
            ]
        )
        report = harness.run()

        assert len(report.members) == 1
        assert report.members[0].suspect is True


# ---------------------------------------------------------------------------
# AC6 — the busy predicate and the suspect flag
# ---------------------------------------------------------------------------


class TestBusyPredicate:
    """AC6: ``>= 1`` is busy; above 1 is an anomaly that is STILL reported."""

    def test_single_open_message_is_working_and_not_suspect(self) -> None:
        sender = _address("@Dev1")
        message = _task("a normal turn")
        harness = _Harness([_sent(sender, message), _received(sender, message.id)])
        report = harness.run()

        assert len(report.members) == 1
        assert report.members[0].suspect is False
        assert report.members[0].message_id == message.id

    def test_two_open_messages_are_suspect_and_still_reported(self) -> None:
        sender = _address("@Dev1")
        older = _task("older open task")
        newer = _task("newer open task")
        harness = _Harness(
            [
                _sent(sender, older),
                _sent(sender, newer),
                _received(sender, older.id, age_seconds=9),
                _received(sender, newer.id, age_seconds=2),
            ]
        )
        report = harness.run()

        assert len(report.members) == 1, "an anomalous member must never be dropped"
        assert report.members[0].suspect is True
        assert report.members[0].message_id == older.id, "report the oldest open entry"
        assert report.members[0].task == "older open task"

    def test_a_closed_handler_is_not_reported(self) -> None:
        sender = _address("@Dev1")
        message = _task("finished work")
        harness = _Harness(
            [
                _sent(sender, message),
                _received(sender, message.id),
                _processed(sender, message.id),
            ]
        )
        assert _Harness([]).run().members == []
        assert harness.run().members == []

    def test_busy_for_seconds_tracks_the_open_entry(self) -> None:
        sender = _address("@Dev1")
        message = _task("long-running turn")
        harness = _Harness([_sent(sender, message), _received(sender, message.id, age_seconds=42)])
        report = harness.run()
        assert report.members[0].busy_for_seconds == pytest.approx(42, abs=2)


# ---------------------------------------------------------------------------
# AC7 — the replay cut-off
# ---------------------------------------------------------------------------


class TestStaleCutOff:
    """AC7: a replayed, permanently unbalanced pair must not read as busy."""

    def test_stale_member_is_excluded_while_a_fresh_one_survives(self) -> None:
        """RED when the cut-off is deleted.

        The stale entry is the resume shape: telemetry is persisted and replayed,
        so a team stopped mid-turn carries a ``ReceivedMessage`` whose
        ``ProcessedMessage`` never existed and never will. Delete the cut-off and
        that member reads as busy for ever — this report would then hold two rows
        instead of one.

        The second, *fresh* member is what makes the assertion mean something: it
        distinguishes "the cut-off was applied" from "the report came back empty
        for some unrelated reason", which an absence-only assertion cannot.
        """
        stale_sender = _address("@Ghost")
        fresh_sender = _address("@Dev1")
        stale_task = _task("interrupted before the team was stopped")
        fresh_task = _task("actually running right now")

        harness = _Harness(
            [
                _sent(stale_sender, stale_task),
                _sent(fresh_sender, fresh_task),
                _received(stale_sender, stale_task.id, age_seconds=360.0),
                _received(fresh_sender, fresh_task.id, age_seconds=2.0),
            ],
            stale_after_seconds=300.0,
        )
        report = harness.run()

        names = {row.name for row in report.members}
        assert "@Ghost" not in names, "a replayed open handler must not read as busy"
        assert names == {"@Dev1"}

    def test_the_cut_off_is_configurable(self) -> None:
        sender = _address("@Dev1")
        message = _task("recent enough under a wider window")
        telemetry = [_sent(sender, message), _received(sender, message.id, age_seconds=100.0)]

        assert _Harness(telemetry, stale_after_seconds=50.0).run().members == []
        assert len(_Harness(telemetry, stale_after_seconds=300.0).run().members) == 1

    def test_a_timestampless_entry_is_excluded(self) -> None:
        sender = _address("@Dev1")
        message = _task("no timestamp at all")
        received = _received(sender, message.id)
        received.timestamp = None
        harness = _Harness([_sent(sender, message), received])
        assert harness.run().members == []


# ---------------------------------------------------------------------------
# AC8 — structural exclusions
# ---------------------------------------------------------------------------


class TestExclusions:
    """AC8: self, ``#``-prefixed actors and the user proxy never appear."""

    def test_the_caller_excludes_itself_by_agent_id(self) -> None:
        caller = _address("@Manager", "Manager")
        # A different actor that happens to carry the SAME name: only the
        # agent_id match may exclude, so this one must still be reported.
        namesake = _address("@Manager", "Manager")
        own_task = _task("what the caller itself is doing")
        other_task = _task("what the namesake is doing")

        harness = _Harness(
            [
                _sent(caller, own_task),
                _sent(namesake, other_task),
                _received(caller, own_task.id),
                _received(namesake, other_task.id),
            ],
            caller=caller,
        )
        report = harness.run()

        assert len(report.members) == 1
        assert report.members[0].agent_id == namesake.agent_id

    def test_hash_prefixed_tool_actors_are_excluded(self) -> None:
        tool_actor = _address("#TeamActivity", "ToolActor")
        message = _task("an internal tool-actor turn")
        harness = _Harness([_sent(tool_actor, message), _received(tool_actor, message.id)])
        assert harness.run().members == []

    def test_the_user_proxy_is_excluded_by_type(self) -> None:
        """A human blocked on input is not working."""
        proxy = _address("@User", "Human", is_user_proxy=True)
        message = _task("waiting for the human to answer")
        harness = _Harness([_sent(proxy, message), _received(proxy, message.id)])
        assert harness.run().members == []

    def test_a_senderless_entry_is_ignored(self) -> None:
        message = _task("telemetry with no sender")
        orphan = ReceivedMessage(message_id=message.id, timestamp=datetime.now(UTC))
        harness = _Harness([_sent(_address("@Dev1"), message), orphan])
        assert harness.run().members == []

    def test_the_report_is_not_intersected_with_the_roster(self) -> None:
        """A busy member absent from ``get_team()`` is still reported.

        Every exclusion is read off the address itself, and the hard-kill case a
        roster filter would catch is already covered by the staleness cut-off.
        """
        sender = _address("@Dev1")
        message = _task("working, but missing from the roster")
        harness = _Harness([_sent(sender, message), _received(sender, message.id)])
        harness.orchestrator.team = []

        report = harness.run()
        assert [row.name for row in report.members] == ["@Dev1"]


# ---------------------------------------------------------------------------
# AC9 — task-text resolution
# ---------------------------------------------------------------------------


class TestTaskTextResolution:
    """AC9: reverse walk over ``SentMessage``, stopping as soon as it can."""

    def test_resolves_the_matching_sent_message_content(self) -> None:
        sender = _address("@Dev1")
        message = _task("draft the migration plan")
        harness = _Harness([_sent(sender, message), _received(sender, message.id)])
        assert harness.run().members[0].task == "draft the migration plan"

    def test_walk_stops_once_every_open_id_is_resolved(self) -> None:
        """Entries older than the last resolution are never inspected."""
        sender = _address("@Dev1")
        wanted = _task("the only open task")
        noise = [_sent(sender, _task(f"older chatter {index}")) for index in range(6)]
        harness = _Harness([*noise, _sent(sender, wanted), _received(sender, wanted.id)])
        harness.run()

        view = harness.orchestrator.sent_view
        assert view is not None
        # Seven SentMessages; the match is the last one, so exactly one read.
        assert len(view) == 7
        assert view.read_indices == [6]

    def test_the_walk_reads_only_as_far_back_as_it_must(self) -> None:
        sender = _address("@Dev1")
        wanted = _task("the open task")
        harness = _Harness(
            [
                _sent(sender, _task("ancient history")),
                _sent(sender, wanted),
                _sent(sender, _task("later, unrelated")),
                _received(sender, wanted.id),
            ]
        )
        harness.run()
        view = harness.orchestrator.sent_view
        assert view is not None
        assert view.read_indices == [2, 1]
        assert 0 not in view.read_indices

    def test_an_unresolvable_id_degrades_to_a_placeholder(self) -> None:
        """No matching ``SentMessage``: the text degrades, the member stays."""
        sender = _address("@Dev1")
        harness = _Harness([_received(sender, uuid.uuid4())])
        report = harness.run()

        assert len(report.members) == 1
        assert report.members[0].task == UNRESOLVED_TASK

    def test_an_unresolvable_id_degrades_on_the_truncating_path_too(self) -> None:
        sender = _address("@Dev1")
        harness = _Harness([_received(sender, uuid.uuid4())], get_team_activity=True)
        assert harness.run().members[0].task == UNRESOLVED_TASK

    def test_a_contentless_message_degrades_to_its_class_name(self) -> None:
        sender = _address("@Dev1")
        contentless = StartMessage(config=BaseConfig(name="@Dev1", role="Developer"))
        harness = _Harness([_sent(sender, contentless), _received(sender, contentless.id)])
        assert harness.run().members[0].task == "<StartMessage>"

    def test_result_messages_resolve_through_their_content(self) -> None:
        sender = _address("@Dev1")
        message = ResultMessage(content="the produced answer")
        harness = _Harness([_sent(sender, message), _received(sender, message.id)])
        assert harness.run().members[0].task == "the produced answer"

    def test_no_open_member_skips_the_sent_scan_entirely(self) -> None:
        harness = _Harness([])
        harness.run()
        assert SentMessage not in harness.orchestrator.get_messages_calls


# ---------------------------------------------------------------------------
# AC10 — zero LLM calls unless activated
# ---------------------------------------------------------------------------


class TestActivationCostsNothing:
    """AC10: two cases — no summarizer at all, and a summarizer left dormant."""

    def test_no_summarizer_means_no_actor_and_no_worker(self) -> None:
        """AC10(a). RED against un-gated actor creation or an eager warm.

        Asserting on the returned string does NOT satisfy this AC, and that is not
        a hypothetical: the superseded attempt asserted exactly that, and a
        mutation that warmed the cache and decided afterwards still returned the
        truncation, still reported ``summarized=False`` and still passed green
        while a worker had been spawned and paid for.

        So the probes are on the **cost**: ``getChildrenOrCreate`` was never
        called, and the worker spy — which counts its own ``__init__`` — is at
        zero even though the history holds a task twenty-five times longer than
        the character budget.
        """
        sender = _address("@Dev1")
        message = _task("x" * 5_000)
        harness = _Harness(
            [_sent(sender, message), _received(sender, message.id)],
            get_team_activity=GetTeamActivity(max_task_chars=200),
            spy_worker=True,
        )

        report = harness.run()

        assert harness.orchestrator.children_created == [], "an unused cache actor was created"
        assert _SpyWorker.spawned == [], "a summarizer worker was spawned and paid for"
        assert harness.tool._activity_proxy is None
        assert report.members[0].summarized is False
        assert len(report.members[0].task) == 200
        assert report.pending_summaries == 0

    def test_a_configured_summarizer_left_dormant_touches_nothing(self) -> None:
        """AC10(b). The actor exists; the cache must still be untouched.

        ``summarize_over=None`` is the caller withholding consent, so neither
        ``get`` nor ``request`` may fire — a ``get`` alone would be harmless today
        but is exactly the eager-warm shape that the returned string cannot
        distinguish from the correct one.
        """
        sender = _address("@Dev1")
        message = _task("y" * 5_000)
        harness = _Harness(
            [_sent(sender, message), _received(sender, message.id)],
            spy_worker=True,
            max_task_chars=200,
        )
        assert harness.tool._activity_proxy is not None, "case (b) needs the actor to exist"

        with (
            patch.object(harness.activity, "get", wraps=harness.activity.get) as get_spy,
            patch.object(
                harness.activity, "request", wraps=harness.activity.request
            ) as request_spy,
        ):
            report = harness.run(summarize_over=None)

        get_spy.assert_not_called()
        request_spy.assert_not_called()
        assert _SpyWorker.spawned == [], "a summarizer worker was spawned and paid for"
        assert report.members[0].summarized is False
        assert len(report.members[0].task) == 200
        assert report.pending_summaries == 0

    def test_the_spy_would_catch_a_spawn(self) -> None:
        """Guard on the guard: the spy is wired to the path it is watching.

        Without this, a spy that could never fire would make both tests above
        vacuous — they would pass whether or not the short-circuits exist.
        """
        sender = _address("@Dev1")
        message = _task("z" * 5_000)
        harness = _Harness(
            [_sent(sender, message), _received(sender, message.id)], spy_worker=True
        )
        harness.run(summarize_over=100)
        assert _SpyWorker.spawned == ["spawned"]


# ---------------------------------------------------------------------------
# AC11 — opt-in summarization
# ---------------------------------------------------------------------------


class TestSummarization:
    """AC11: threshold, one poll for the whole set, cache reuse, degradation."""

    def _long_task_harness(self, count: int = 1, **kwargs: Any) -> _Harness:
        messages: list[Message] = []
        for index in range(count):
            sender = _address(f"@Dev{index}")
            task = _task(f"{index}-" + "z" * 400)
            messages.append(_sent(sender, task))
            messages.append(_received(sender, task.id))
        return _Harness(messages, spy_worker=True, **kwargs)

    def test_a_zero_character_budget_reports_no_text(self) -> None:
        sender = _address("@Dev1")
        message = _task("some task text")
        harness = _Harness(
            [_sent(sender, message), _received(sender, message.id)], max_task_chars=0
        )
        assert harness.run().members[0].task == ""

    def test_short_tasks_are_truncated_and_never_summarized(self) -> None:
        sender = _address("@Dev1")
        message = _task("short enough")
        harness = _Harness([_sent(sender, message), _received(sender, message.id)], spy_worker=True)

        with patch.object(
            harness.activity, "request", wraps=harness.activity.request
        ) as request_spy:
            report = harness.run(summarize_over=100)

        request_spy.assert_not_called()
        assert _SpyWorker.spawned == []
        assert report.members[0].task == "short enough"
        assert report.members[0].summarized is False

    def test_a_cached_summary_is_returned_and_spawns_nothing(self) -> None:
        sender = _address("@Dev1")
        message = _task("q" * 400)
        harness = _Harness(
            [_sent(sender, message), _received(sender, message.id)], spy_worker=True
        )
        harness.activity.deliver(message.id, "a previously produced summary")

        report = harness.run(summarize_over=100)

        assert _SpyWorker.spawned == []
        assert report.members[0].task == "a previously produced summary"
        assert report.members[0].summarized is True
        assert report.pending_summaries == 0

    def test_a_cached_summary_is_clipped_to_this_card_s_budget(self) -> None:
        """``#TeamActivity`` is one singleton shared by every card on the team.

        The worker clips to the budget of whoever requested the summary, so a card
        configured with a smaller ``max_task_chars`` can read back a summary
        produced under a larger one. The reported text must respect the budget of
        the card doing the reporting, not the one that paid for the summary.
        """
        sender = _address("@Dev1")
        message = _task("q" * 400)
        harness = _Harness(
            [_sent(sender, message), _received(sender, message.id)],
            spy_worker=True,
            max_task_chars=50,
        )
        harness.activity.deliver(message.id, "s" * 300)

        report = harness.run(summarize_over=100)

        assert _SpyWorker.spawned == []
        assert report.members[0].summarized is True
        assert len(report.members[0].task) == 50

    def test_a_harvested_summary_is_clipped_to_the_budget_too(self) -> None:
        """The post-poll sweep goes through the same budget as the cache hit."""
        harness = self._long_task_harness(
            count=1, poll_attempts=1, poll_delay_seconds=0.0, max_task_chars=50
        )
        message_id = next(
            msg.message_id
            for msg in harness.orchestrator._messages
            if isinstance(msg, ReceivedMessage)
        )
        landed: dict[uuid.UUID, str] = {}

        def fake_get(key: uuid.UUID) -> str | None:
            return landed.get(key)

        def fake_poll(fetch: Callable[[], Any], attempts: int = 5, delay: float = 0.4) -> Any:
            fetch()  # one honest attempt: nothing has landed yet
            landed[message_id] = "t" * 300
            return None

        with (
            patch.object(harness.activity, "get", side_effect=fake_get),
            patch.object(activity_module, "poll_deferred", side_effect=fake_poll),
        ):
            report = harness.run(summarize_over=100)

        assert report.members[0].summarized is True
        assert len(report.members[0].task) == 50
        assert report.pending_summaries == 0

    def test_the_summary_request_carries_the_message_id_as_its_key(self) -> None:
        sender = _address("@Dev1")
        message = _task("w" * 400)
        harness = _Harness(
            [_sent(sender, message), _received(sender, message.id)],
            spy_worker=True,
            model="openai:gpt-5.2-mini",
            max_task_chars=180,
        )

        with patch.object(
            harness.activity, "request", wraps=harness.activity.request
        ) as request_spy:
            harness.run(summarize_over=100)

        key, payload = request_spy.call_args.args
        assert key == message.id
        assert isinstance(payload, SummarizePayload)
        assert payload.deferred_key == message.id, "the key must never drift from the request key"
        assert payload.model == "openai:gpt-5.2-mini"
        assert payload.max_chars == 180
        assert payload.text.startswith("w")

    def test_one_poll_serves_the_whole_pending_set(self) -> None:
        harness = self._long_task_harness(count=3, poll_attempts=1, poll_delay_seconds=0.0)

        with patch.object(
            activity_module, "poll_deferred", wraps=poll_deferred
        ) as poll_spy:
            report = harness.run(summarize_over=100)

        assert len(report.members) == 3
        assert poll_spy.call_count == 1, "polling once per member would be N round-trip budgets"

    def test_unresolved_summaries_degrade_to_truncated_text(self) -> None:
        harness = self._long_task_harness(
            count=2, poll_attempts=1, poll_delay_seconds=0.0, max_task_chars=60
        )
        report = harness.run(summarize_over=100)

        assert report.pending_summaries == 2
        assert all(row.summarized is False for row in report.members)
        assert all(len(row.task) == 60 for row in report.members)

    def test_the_post_poll_sweep_harvests_a_late_arrival(self) -> None:
        """Without the sweep, ``pending_summaries`` over-counts.

        ``poll_deferred`` answers only when EVERY pending key has landed, so a
        summary that arrives during the last attempt is invisible to the poll's
        own return value. The sweep afterwards is what picks it up.
        """
        harness = self._long_task_harness(count=2, poll_attempts=1, poll_delay_seconds=0.0)
        landed: dict[uuid.UUID, str] = {}

        def fake_get(key: uuid.UUID) -> str | None:
            return landed.get(key)

        def fake_poll(
            fetch: Callable[[], Any], attempts: int = 5, delay: float = 0.4
        ) -> Any:
            fetch()  # one honest attempt: nothing has landed yet
            landed[first_id] = "arrived just after the budget ran out"
            return None

        first_id = next(
            msg.message_id
            for msg in harness.orchestrator._messages
            if isinstance(msg, ReceivedMessage)
        )

        with (
            patch.object(harness.activity, "get", side_effect=fake_get),
            patch.object(activity_module, "poll_deferred", side_effect=fake_poll),
        ):
            report = harness.run(summarize_over=100)

        summarized = [row for row in report.members if row.summarized]
        assert len(summarized) == 1
        assert summarized[0].message_id == first_id
        assert summarized[0].task == "arrived just after the budget ran out"
        assert report.pending_summaries == 1

    def test_a_summary_that_lands_during_the_poll_clears_the_pending_count(self) -> None:
        """The poll itself resolves the set, so the sweep is not needed.

        The first ``get`` is the pre-request cache probe and must miss, otherwise
        the row is served from cache and the poll is never reached at all.
        """
        harness = self._long_task_harness(count=1, poll_attempts=2, poll_delay_seconds=0.0)
        message_id = next(
            msg.message_id
            for msg in harness.orchestrator._messages
            if isinstance(msg, ReceivedMessage)
        )
        landed: dict[uuid.UUID, str] = {}

        def fake_get(key: uuid.UUID) -> str | None:
            if key not in landed:
                landed[key] = "the finished summary"  # produced right after the miss
                return None
            return landed[key]

        with patch.object(harness.activity, "get", side_effect=fake_get):
            report = harness.run(summarize_over=100)

        assert report.pending_summaries == 0
        assert report.members[0].summarized is True
        assert report.members[0].task == "the finished summary"
        assert report.members[0].message_id == message_id


# ---------------------------------------------------------------------------
# AC12 — actor wiring and the summarizer worker
# ---------------------------------------------------------------------------


class TestActorWiring:
    """AC12: the ``#TeamActivity`` singleton and the pydantic-ai call."""

    def test_singleton_is_created_with_the_tool_actor_name_and_role(self) -> None:
        harness = _Harness([])
        assert len(harness.orchestrator.children_created) == 1
        actor_class, config = harness.orchestrator.children_created[0]
        assert actor_class is TeamActivityActor
        assert config.name == TEAM_ACTIVITY_ACTOR_NAME == "#TeamActivity"
        assert config.role == TEAM_ACTIVITY_ACTOR_ROLE == "ToolActor"

    def test_the_actor_name_reads_as_a_tool_actor(self) -> None:
        """The ``#`` prefix keeps the singleton out of the non-tool roster."""
        assert TEAM_ACTIVITY_ACTOR_NAME.startswith("#")

    def test_both_proxies_are_built_through_proxy_ask(self) -> None:
        harness = _Harness([])
        assert harness.observer.proxy_ask_calls == [Orchestrator, TeamActivityActor]

    def test_the_cache_actor_spawns_the_summarizer_worker(self) -> None:
        assert _make_activity_actor().worker_class() is SummarizerWorker

    def test_summarizer_builds_the_agent_from_the_payload_model_spec(self) -> None:
        worker = SummarizerWorker()
        worker.config = BaseConfig(name="#defer-abc", role="ToolActor")
        worker.on_start()
        payload = SummarizePayload(
            deferred_key=uuid.uuid4(),
            text="a very long task description",
            model="openai:gpt-5.2-mini",
            max_chars=40,
        )
        agent = MagicMock()
        agent.run_sync.return_value = MagicMock(output="  a concise summary  ")

        with patch("akgentic.tool.team.activity.Agent", return_value=agent) as agent_cls:
            produced = worker.produce(payload)

        agent_cls.assert_called_once_with("openai:gpt-5.2-mini")
        assert produced == "a concise summary"
        prompt = agent.run_sync.call_args.args[0]
        assert "a very long task description" in prompt
        assert "40" in prompt

    def test_the_worker_hands_its_budget_to_the_model_call(self) -> None:
        """A timeout that does not reach the client is decoration."""
        worker = SummarizerWorker()
        worker.config = BaseConfig(name="#defer-abc", role="ToolActor")
        worker.on_start()
        payload = SummarizePayload(
            deferred_key=uuid.uuid4(), text="text", model="openai:gpt-5.2-mini", max_chars=50
        )
        agent = MagicMock()
        agent.run_sync.return_value = MagicMock(output="summary")

        with patch("akgentic.tool.team.activity.Agent", return_value=agent):
            worker.produce(payload)

        settings = agent.run_sync.call_args.kwargs["model_settings"]
        assert settings["timeout"] == worker.timeout_s

    def test_the_summary_is_clipped_to_the_payload_budget(self) -> None:
        worker = SummarizerWorker()
        worker.config = BaseConfig(name="#defer-abc", role="ToolActor")
        worker.on_start()
        payload = SummarizePayload(
            deferred_key=uuid.uuid4(), text="text", model="openai:gpt-5.2-mini", max_chars=10
        )
        agent = MagicMock()
        agent.run_sync.return_value = MagicMock(output="k" * 500)

        with patch("akgentic.tool.team.activity.Agent", return_value=agent):
            assert worker.produce(payload) == "k" * 10

    def test_a_foreign_payload_is_rejected(self) -> None:
        worker = SummarizerWorker()
        worker.config = BaseConfig(name="#defer-abc", role="ToolActor")
        worker.on_start()
        with pytest.raises(TypeError, match="SummarizePayload"):
            worker.produce(DeferredPayload(deferred_key="not-a-uuid"))

    def test_the_payload_narrows_its_key_to_a_uuid(self) -> None:
        with pytest.raises(ValueError, match="deferred_key"):
            SummarizePayload(
                deferred_key="not-a-uuid", text="t", model="openai:gpt-5.2-mini", max_chars=10
            )

    def test_the_budget_is_built_from_the_two_param_models(self) -> None:
        params = GetTeamActivity(
            summarizer=ActivitySummarizer(
                model="openai:gpt-5.2-mini", poll_attempts=3, poll_delay_seconds=0.1
            ),
            max_task_chars=77,
        )
        budget = activity_module.SummaryBudget.from_params(params)
        assert budget.model == "openai:gpt-5.2-mini"
        assert budget.max_chars == 77
        assert budget.poll_attempts == 3
        assert budget.poll_delay_seconds == 0.1


# ---------------------------------------------------------------------------
# AC12 (NFR1) / AC13 — the boundary and the public API
# ---------------------------------------------------------------------------


_STORY_MODULES = ("activity.py", "team.py")


def _imported_roots(module_path: Path) -> set[str]:
    """Every dotted module name imported by the file at *module_path*."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


class TestPackageBoundary:
    """NFR1: ``akgentic-tool`` never reaches for ``akgentic-llm``."""

    def test_no_akgentic_llm_import_in_the_story_modules(self) -> None:
        team_dir = Path(activity_module.__file__).parent
        violations: list[str] = []
        for module_name in _STORY_MODULES:
            for imported in sorted(_imported_roots(team_dir / module_name)):
                if imported == "akgentic.llm" or imported.startswith("akgentic.llm."):
                    violations.append(f"{module_name} imports {imported}")
        assert not violations, f"package boundary violated: {violations}"

    def test_no_akgentic_llm_import_in_this_story_s_tests(self) -> None:
        for imported in sorted(_imported_roots(Path(__file__))):
            assert not imported.startswith("akgentic.llm")

    def test_the_summarizer_depends_on_pydantic_ai_directly(self) -> None:
        team_dir = Path(activity_module.__file__).parent
        assert "pydantic_ai" in _imported_roots(team_dir / "activity.py")

    def test_the_folded_actor_module_is_gone(self) -> None:
        team_dir = Path(activity_module.__file__).parent
        assert not (team_dir / "activity_actor.py").exists()


class TestPublicApi:
    """AC13: the new names are exported; the dissolved card is not."""

    def test_team_package_exports(self) -> None:
        import akgentic.tool.team as team_package

        for name in (
            "GetTeamActivity",
            "ActivitySummarizer",
            "AgentActivity",
            "TeamActivityReport",
        ):
            assert name in team_package.__all__
            assert getattr(team_package, name) is not None

    def test_team_tool_and_its_param_classes_keep_their_exports(self) -> None:
        import akgentic.tool.team as team_package

        for name in (
            "TeamTool",
            "HireTeamMember",
            "FireTeamMember",
            "GetTeamRoster",
            "GetRoleProfiles",
        ):
            assert name in team_package.__all__
