"""Tests for ``MailboxTool``: the signal and the cancellation surface (ADR-010, ADR-040 §7)."""

from __future__ import annotations

import ast
import gc
import inspect
import json
import uuid
from pathlib import Path

import pytest
from akgentic.core.messages import Message

from akgentic.tool import ToolFactory
from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL
from akgentic.tool.errors import ToolObserverGone
from akgentic.tool.mailbox import (
    MailboxTool,
    MailboxToolObserver,
    ReadMailbox,
    Stop,
)
from tests.mailbox.conftest import _user_message


class _FakeObserver:
    """Structural MailboxToolObserver: real messages, no actor system.

    Every mailbox call is recorded in ``calls``. The signal must touch neither
    method, and a stub that silently did nothing would make that unfalsifiable —
    so ``consume_mailbox`` really removes from the backing list and returns what
    it removed.
    """

    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages = messages or []
        self.calls: list[str] = []

    def notify_event(self, event: object) -> None:
        pass

    def get_mailbox(self) -> list[Message]:
        self.calls.append("get_mailbox")
        return list(self._messages)

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]:
        self.calls.append("consume_mailbox")
        wanted = set(message_ids)
        removed = [message for message in self._messages if message.id in wanted]
        self._messages = [message for message in self._messages if message.id not in wanted]
        return removed

    def deliver(self, message: Message) -> None:
        """Queue one more message, as an arrival between two reads would."""
        self._messages.append(message)


class _Probe(Message):
    """A handler message class declared here, to be named by its built dotted path.

    It carries no ``content`` — deliberately, because that is the shape the
    whitelist exists to serve and the shape the deleted renderer emptied.
    """

    incident: str = ""


def _wired_card(
    messages: list[Message] | None = None, **fields: object
) -> tuple[MailboxTool, _FakeObserver]:
    """A card with an attached fake observer; the observer is returned to keep it alive."""
    observer = _FakeObserver(messages)
    assert isinstance(observer, MailboxToolObserver)  # structural conformance
    card = MailboxTool.model_validate(fields)
    card.observer(observer)
    return card, observer


# ── the silent-drop assertion (backlog row 28): assert the lists, not behaviour ──


def test_default_card_yields_one_tool_one_command_and_no_provider() -> None:
    card, observer = _wired_card()

    assert card.get_context_states() == []

    tools = card.get_tools()
    assert [tool.__name__ for tool in tools] == ["read_mailbox"]

    commands = card.get_commands()
    assert set(commands) == {Stop}
    assert commands[Stop].__name__ == "stop"


def test_factory_registers_exactly_one_command_under_canonical_name_stop() -> None:
    observer = _FakeObserver()
    factory = ToolFactory([MailboxTool()], observer=observer)

    registry = factory.get_command_registry()
    descriptors = registry.descriptors()
    assert [descriptor.name for descriptor in descriptors] == ["stop"]
    assert descriptors[0].args == []
    assert registry.has("stop")


# ── the LLM_CONTEXT capability is gone (ADR-019 §4b) ─────────────────────────


def test_card_serves_no_context_state_provider() -> None:
    card, observer = _wired_card([_user_message("@Alice", "hello")])

    assert card.get_context_states() == []


def test_factory_aggregates_no_mailbox_provider() -> None:
    observer = _FakeObserver([_user_message("@Alice", "hello")])
    factory = ToolFactory([MailboxTool()], observer=observer)

    assert factory.get_context_states() == []


def test_card_exposes_no_mailbox_status_field() -> None:
    # Removed, not deprecated. ToolCard keeps Pydantic's default extra="ignore",
    # so the value is dropped silently — assert the absent field, never a raise.
    card = MailboxTool(mailbox_status=True)  # type: ignore[call-arg]

    assert "mailbox_status" not in card.model_dump()
    assert not hasattr(card, "mailbox_status")
    assert "mailbox_status" not in MailboxTool.model_fields


def test_state_vocabulary_is_not_importable_from_this_package() -> None:
    # A stale re-export is exactly what this story removes; assert the absence.
    import akgentic.tool.mailbox as mailbox_module

    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import MailboxStatus  # noqa: F401
    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import MailboxState  # noqa: F401
    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import MailboxRow  # noqa: F401
    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import make_mailbox_state_provider  # noqa: F401

    for symbol in ("MailboxStatus", "MailboxState", "MailboxRow", "make_mailbox_state_provider"):
        assert symbol not in dir(mailbox_module)


# ── read_mailbox is a signal: it acknowledges an id and touches nothing (AC 1) ──


def test_read_mailbox_takes_a_message_id_and_returns_a_non_empty_acknowledgement() -> None:
    card, observer = _wired_card()
    read_mailbox = card.get_tools()[0]

    acknowledgement = read_mailbox(message_id="0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0")
    assert isinstance(acknowledgement, str)
    assert acknowledgement.strip() != ""


def test_the_parameter_is_named_message_id_and_is_addressable_by_that_name() -> None:
    # A cross-package contract, not a local choice: the agent capability reads
    # this name out of the completed tool call's arguments (agent Epic 23).
    # Rename it and the other half fails SILENTLY — the tool still acknowledges,
    # the capability finds no id, nothing is injected, nothing is logged, and
    # neither test suite can see it because the halves live in separate repos.
    card, observer = _wired_card()
    # eval_str: the module carries `from __future__ import annotations`, so the
    # raw annotation is the string "str" — and it is the resolved type that
    # pydantic-ai turns into the tool schema.
    signature = inspect.signature(card.get_tools()[0], eval_str=True)

    assert list(signature.parameters) == ["message_id"]
    assert signature.parameters["message_id"].annotation is str
    assert signature.return_annotation is str

    # The positive form, not `is not POSITIONAL_ONLY`: the kind must be one the
    # caller can address BY NAME. A positional-only parameter correctly *named*
    # message_id passes any name check and still breaks the lookup, and so would
    # a **message_id — the negative form admits both VAR_ kinds.
    assert signature.parameters["message_id"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def test_read_mailbox_reads_nothing_and_consumes_nothing() -> None:
    # The whole point of the story: the card no longer touches the mailbox, so
    # nothing is emptied on its way past a renderer written for one message class.
    waiting = _user_message("@Alice", "the incident report")
    card, observer = _wired_card([waiting])
    read_mailbox = card.get_tools()[0]

    read_mailbox(message_id=str(waiting.id))

    assert observer.calls == []
    assert observer.get_mailbox() == [waiting]


@pytest.mark.parametrize(
    "message_id", ["", "not-a-uuid", "  ", "an id that was never issued", "00000000"]
)
def test_read_mailbox_performs_no_lookup_parsing_or_validation_of_the_id(message_id: str) -> None:
    # No uuid parsing, no membership check: resolving the id is the agent
    # capability's job, and the card raising here would only teach the model to
    # retry a call the card cannot answer either way.
    card, observer = _wired_card([_user_message("@Alice", "hello")])

    assert card.get_tools()[0](message_id=message_id).strip() != ""
    assert observer.calls == []


def test_read_mailbox_docstring_carries_the_absorption_contract() -> None:
    # The docstring IS the model-facing tool description — functional surface,
    # not documentation: it is what tells the model the signal is one-shot.
    card, observer = _wired_card()
    doc = (card.get_tools()[0].__doc__ or "").lower()

    assert "absorbs" in doc
    assert "not be delivered to you again" in doc
    assert "as its own turn" in doc


def test_read_mailbox_docstring_documents_message_id_and_promises_no_content() -> None:
    # pydantic-ai derives the tool schema from the signature plus this docstring,
    # so the Args: block is functional surface. And the return must not be sold
    # as carrying the message — it carries an acknowledgement.
    card, observer = _wired_card()
    doc = card.get_tools()[0].__doc__ or ""

    assert "Args:" in doc
    assert "message_id" in doc
    assert "arrival" in doc.lower()  # where the ids come from


def test_read_mailbox_after_observer_collected_raises_tool_observer_gone() -> None:
    # Liveness is load-bearing, not defensive: an acknowledgement from a card
    # whose agent has stopped is a false one — the capability that would absorb
    # the named message went with the agent.
    card, observer = _wired_card([])
    read_mailbox = card.get_tools()[0]

    del observer
    gc.collect()
    with pytest.raises(ToolObserverGone):
        read_mailbox(message_id="any id at all")


# ── the deletions (AC 3) ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "_render_mailbox",
        "_render_message",
        "_content_of",
        "_protocol_line",
        "_REPLY_PROTOCOLS",
        "_is_cancel_envelope",
        "sender_name",
        "_EMPTY_MAILBOX",
        "_mailbox_observer_or_none",
        "CancelMessage",
    ],
)
def test_the_rendering_and_cancel_machinery_is_gone_from_the_module(name: str) -> None:
    # Rendering a message is the message's job and delivering one is the agent's;
    # a second path left here for compatibility is a path the wrong caller reaches.
    import akgentic.tool.mailbox.mailbox as mailbox_module

    assert name not in dir(mailbox_module)
    assert not hasattr(MailboxTool, name)


def test_message_stays_because_the_whitelist_resolver_needs_it() -> None:
    import akgentic.tool.mailbox.mailbox as mailbox_module

    assert mailbox_module.Message is Message


def test_no_module_under_src_imports_from_the_agent_package() -> None:
    # Module boundary rules: akgentic-tool may import from akgentic-core only.
    # The card names message classes as strings and resolves them at runtime.
    #
    # AST, not a text search, and the difference is load-bearing:
    # notification/models.py sets DEFAULT_MESSAGE_CLASS to the dotted string
    # "akgentic.agent.messages.AgentMessage", resolved at runtime. A grep would
    # call that an offender — it is the same stringly-typed pattern this story
    # adopts, and the whole point of the design.
    package_root = Path(inspect.getfile(MailboxTool)).parent.parent
    sources = sorted(package_root.rglob("*.py"))
    offenders: list[str] = []
    for source_file in sources:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if (node.module or "").startswith("akgentic.agent"):
                    offenders.append(f"{source_file.name}:{node.lineno}")
            if isinstance(node, ast.Import):
                offenders += [
                    f"{source_file.name}:{node.lineno}"
                    for alias in node.names
                    if alias.name.startswith("akgentic.agent")
                ]

    assert offenders == []
    assert len(sources) > 40, "the sweep found almost no modules — the root is wrong"


# ── the whitelist field (AC 4, AC 6) ─────────────────────────────────────────


def test_the_whitelist_defaults_to_none_meaning_every_handler_shows_the_preview() -> None:
    assert MailboxTool().mailbox_preview_handlers is None


@pytest.mark.parametrize(
    "handlers",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty"),
        pytest.param(["akgentic.core.messages.UserMessage"], id="populated"),
    ],
)
def test_the_whitelist_round_trips_unchanged(handlers: list[str] | None) -> None:
    card = MailboxTool(mailbox_preview_handlers=handlers)
    restored = MailboxTool.model_validate(card.model_dump())

    assert restored == card
    assert restored.mailbox_preview_handlers == handlers


def test_empty_list_is_not_coerced_to_none_on_round_trip() -> None:
    # None and [] are different values: None means every handler shows the
    # preview, [] means none does. Collapsing them would silently re-enable it.
    restored = MailboxTool.model_validate(MailboxTool(mailbox_preview_handlers=[]).model_dump())

    assert restored.mailbox_preview_handlers == []
    assert restored.mailbox_preview_handlers is not None


def test_the_whitelist_survives_a_json_round_trip_as_plain_strings() -> None:
    # Golden Rule #1b: the whitelist names classes as strings precisely so the
    # card stays JSON-serializable — a resolved `type[Message]` on the model
    # would need an arbitrary-types escape hatch and would not survive a catalog.
    card = MailboxTool(mailbox_preview_handlers=["akgentic.core.messages.UserMessage"])

    encoded = json.dumps(card.model_dump(mode="json"))
    restored = MailboxTool.model_validate(json.loads(encoded))

    assert restored.mailbox_preview_handlers == ["akgentic.core.messages.UserMessage"]


# ── the whitelist is validated at wiring, not at construction (AC 5, AC 6) ───


@pytest.mark.parametrize(
    "dotted_path",
    [
        pytest.param("NotDotted", id="no-module-part"),
        pytest.param("no_such_module.Thing", id="module-not-importable"),
        pytest.param("akgentic.tool.mailbox.NoSuchClass", id="no-such-attribute"),
        pytest.param("akgentic.tool.mailbox.MailboxTool", id="not-a-message-subclass"),
    ],
)
def test_wiring_an_unresolvable_handler_raises_value_error_naming_the_path(
    dotted_path: str,
) -> None:
    # Dotted paths are stringly typed: a rename breaks one silently, and the
    # capability it gates then never fires again for the agent's whole life.
    card = MailboxTool(mailbox_preview_handlers=[dotted_path])
    observer = _FakeObserver()

    with pytest.raises(ValueError) as excinfo:
        card.observer(observer)

    assert dotted_path in str(excinfo.value)


def test_an_unresolvable_entry_is_caught_even_when_the_read_capability_is_off() -> None:
    # get_tools() returns early when read_mailbox is False, so validating there
    # would let a typo through on exactly the cards nobody reads back.
    card = MailboxTool(read_mailbox=False, mailbox_preview_handlers=["NotDotted"])

    with pytest.raises(ValueError):
        card.observer(_FakeObserver())


def test_the_offending_entry_is_named_even_among_valid_ones() -> None:
    card = MailboxTool(
        mailbox_preview_handlers=["akgentic.core.messages.UserMessage", "no_such_module.Thing"]
    )

    with pytest.raises(ValueError) as excinfo:
        card.observer(_FakeObserver())

    assert "no_such_module.Thing" in str(excinfo.value)


@pytest.mark.parametrize(
    "dotted_path",
    ["NotDotted", "no_such_module.Thing", "akgentic.tool.mailbox.MailboxTool"],
)
def test_constructing_or_deserializing_a_bad_entry_never_raises(dotted_path: str) -> None:
    # Validation is a wiring-time event. A field validator would make a card
    # carrying a perfectly valid entry undeserializable wherever that class
    # happens not to be importable — a catalog reader, a serialization round trip.
    assert MailboxTool(mailbox_preview_handlers=[dotted_path]).mailbox_preview_handlers == [
        dotted_path
    ]
    assert MailboxTool.model_validate(
        {"mailbox_preview_handlers": [dotted_path]}
    ).mailbox_preview_handlers == [dotted_path]


@pytest.mark.parametrize(
    "handlers",
    [
        pytest.param(None, id="none-means-all"),
        pytest.param([], id="empty-means-none"),
        pytest.param(["akgentic.core.messages.UserMessage"], id="a-core-message"),
        # Built, never hardcoded: akgentic-agent is not installed in this
        # package's CI, so no test may name a class under it in either polarity.
        pytest.param([f"{_Probe.__module__}.{_Probe.__name__}"], id="a-locally-declared-message"),
        pytest.param(
            ["akgentic.core.messages.UserMessage", f"{_Probe.__module__}.{_Probe.__name__}"],
            id="both",
        ),
    ],
)
def test_a_resolvable_whitelist_wires_cleanly(handlers: list[str] | None) -> None:
    card, observer = _wired_card(mailbox_preview_handlers=handlers)

    assert card.mailbox_preview_handlers == handlers
    assert [tool.__name__ for tool in card.get_tools()] == ["read_mailbox"]


def test_a_handler_class_without_a_content_field_is_accepted() -> None:
    # The resolver deliberately does not require content/type, unlike the
    # notification one. Requiring them would reject exactly the classes this
    # whitelist exists to name — the ones carrying their own fields.
    assert "content" not in _Probe.model_fields

    card, observer = _wired_card(
        mailbox_preview_handlers=[f"{_Probe.__module__}.{_Probe.__name__}"]
    )

    assert card.mailbox_preview_handlers is not None


# ── stop (FR5, AC 7) ─────────────────────────────────────────────────────────


def test_stop_invoked_directly_reports_there_is_nothing_to_cancel() -> None:
    card, observer = _wired_card()
    stop = card.get_commands()[Stop]

    assert stop() == "There is no run to cancel."


def test_stop_dispatch_through_registry_returns_the_idle_string() -> None:
    # Dispatch reaches a command only while the agent is idle, and a mid-run
    # cancel never survives to be dequeued — so this is the only case there is.
    observer = _FakeObserver()
    factory = ToolFactory([MailboxTool()], observer=observer)

    assert factory.get_command_registry().dispatch("/stop") == "There is no run to cancel."


def test_stop_still_announces_a_description_to_the_command_palette() -> None:
    # The docstring is the wire surface: descriptors() feeds
    # CommandDescriptor.description from it and CommandsAnnouncedEvent carries it
    # to every frontend. Rewriting the docstring must not empty the palette entry.
    observer = _FakeObserver()
    factory = ToolFactory([MailboxTool()], observer=observer)

    descriptor = factory.get_command_registry().descriptors()[0]
    assert "cancel" in descriptor.description.lower()


# ── ownership: the cancel vocabulary lives with the enforcement, not the card ──


def test_cancel_vocabulary_is_not_importable_from_this_package() -> None:
    # The card's exclusion filter is gone entirely — the cancel filter moved to
    # the agent's offer rule — and must not reappear as a public vocabulary here.
    import akgentic.tool.mailbox as mailbox_module

    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import is_cancel  # noqa: F401
    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import render_arrival_notice  # noqa: F401

    assert "is_cancel" not in dir(mailbox_module)
    assert "render_arrival_notice" not in dir(mailbox_module)


def test_mailbox_tool_observer_still_imports_from_the_package_root() -> None:
    # The agent's capability module imports this symbol from exactly this path,
    # and still calls both mailbox methods even though the card no longer does.
    from akgentic.tool.mailbox import MailboxToolObserver as ObserverProtocol

    assert isinstance(_FakeObserver(), ObserverProtocol)


# ── channel gating (FR3): disabling removes exactly one capability ───────────


def test_disabling_read_mailbox_removes_only_the_tool() -> None:
    card, observer = _wired_card(read_mailbox=False)

    assert card.get_tools() == []
    assert set(card.get_commands()) == {Stop}


def test_disabling_stop_removes_only_the_command() -> None:
    card, observer = _wired_card(stop=False)

    assert [tool.__name__ for tool in card.get_tools()] == ["read_mailbox"]
    assert card.get_commands() == {}


def test_capability_exposed_off_its_served_channel_is_dropped() -> None:
    # The gate is param resolved AND the served channel in its expose set.
    card, observer = _wired_card(
        read_mailbox=ReadMailbox(expose={COMMAND}),
        stop=Stop(expose={LLM_CONTEXT}),
    )

    assert card.get_tools() == []
    assert card.get_commands() == {}
    assert card.get_context_states() == []


# ── serializability (FR7 / Golden Rule #1b) ──────────────────────────────────


def test_default_card_round_trips_through_pydantic() -> None:
    card = MailboxTool()
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card


@pytest.mark.parametrize("field", ["read_mailbox", "stop"])
def test_card_with_disabled_param_round_trips(field: str) -> None:
    card = MailboxTool.model_validate({field: False})
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card
    assert getattr(restored, field) is False


def test_card_with_explicit_params_round_trips() -> None:
    card = MailboxTool(
        read_mailbox=ReadMailbox(instructions="prefer wrapping up", expose={TOOL_CALL}),
        stop=Stop(),
        mailbox_preview_handlers=["akgentic.core.messages.UserMessage"],
    )
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card
    assert isinstance(restored.read_mailbox, ReadMailbox)
    assert restored.read_mailbox.instructions == "prefer wrapping up"
    assert isinstance(restored.stop, Stop)
    assert restored.mailbox_preview_handlers == ["akgentic.core.messages.UserMessage"]
