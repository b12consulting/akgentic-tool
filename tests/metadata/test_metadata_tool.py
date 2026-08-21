"""``MetadataTool``: the card, its template grammar, and its one-shot render.

Story 32-1 (ADR-038). The metadata model is declared **here**, locally, with two
neutral fields: this package must never learn a deployment's business shape, and a
shared fixture model would be a standing invitation for production code to import
one. A second, structurally unrelated model renders through the same card, which is
what proves the card assumes no shape at all.

The observer is a detached-``Mock`` pair — a ``Mock(spec=ActorToolObserver)`` whose
orchestrator proxy is built *independently* rather than as a child mock. A child's
parent chain would pin the observer through the card's proxy attribute and quietly
defeat the reclamation assertions at the bottom of this file; the idiom is the one
``test_team_tool_weak_observer.py`` already relies on.
"""

from __future__ import annotations

import ast
import gc
import logging
import uuid
import weakref
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from akgentic.core import ActorAddressProxy
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils import SerializableBaseModel
from pydantic import ValidationError

import akgentic.tool
import akgentic.tool.metadata
from akgentic.tool.core import COMMAND, SYSTEM_PROMPT, ToolCard
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.metadata import MetadataTool, RenderMetadata

TEMPLATE = "Fiscal year: {fiscal_year}. Engagement: {engagement}."
RENDERED = "Fiscal year: FY26. Engagement: Advisory."


class _TeamContext(SerializableBaseModel):
    """The deployment's business context, as this test suite invents it."""

    fiscal_year: str = "FY26"
    engagement: str = "Advisory"


class _OtherContext(SerializableBaseModel):
    """A second metadata model sharing no field name with the first."""

    region: str = "EMEA"
    quarter: int = 3


def _address(name: str, role: str = "Agent") -> ActorAddressProxy:
    """Build a serializable stand-in address."""
    return ActorAddressProxy(
        {
            "__actor_address__": True,
            "__actor_type__": "test.Agent",
            "agent_id": str(uuid.uuid4()),
            "name": name,
            "role": role,
            "team_id": str(uuid.uuid4()),
            "squad_id": str(uuid.uuid4()),
            "is_user_proxy": False,
        }
    )


def _make_observer(metadata: SerializableBaseModel | None) -> Mock:
    """Weak-referenceable observer whose orchestrator proxy answers *metadata*."""
    observer = Mock(spec=ActorToolObserver)
    observer.orchestrator = _address("@Orchestrator", "Orchestrator")
    observer.myAddress = _address("@Manager", "Manager")

    # Detached orchestrator mock: NOT a child of ``observer``, so the card's
    # ``_orchestrator_proxy`` cannot pin the agent through a parent chain.
    orchestrator = Mock(spec=Orchestrator)
    orchestrator.get_metadata.return_value = metadata
    observer.proxy_ask = Mock(return_value=orchestrator)
    return observer


def _wire(metadata: SerializableBaseModel | None, **params: Any) -> tuple[MetadataTool, Mock]:
    """Wire a card configured with *params* against an observer holding *metadata*."""
    params.setdefault("template", TEMPLATE)
    observer = _make_observer(metadata)
    card = MetadataTool(render_metadata=RenderMetadata(**params))
    card.observer(observer)
    return card, observer


def _proxy(observer: Mock) -> Mock:
    """The orchestrator proxy behind *observer*."""
    proxy: Mock = observer.proxy_ask.return_value
    return proxy


def _prompt(card: MetadataTool) -> Any:
    """The card's single system-prompt callable."""
    return card.get_system_prompts()[0]


def _render_through_any_channel(card: MetadataTool) -> str:
    """Render through whichever channel *card* exposes, prompt first."""
    prompts = card.get_system_prompts()
    if prompts:
        rendered: str = prompts[0]()
        return rendered
    command: str = next(iter(card.get_commands().values()))()
    return command


# ── AC1, AC12: card and capability shape ─────────────────────────────────────


class TestCardSurface:
    def test_the_card_declares_exactly_one_field(self) -> None:
        assert set(MetadataTool.model_fields) == {"render_metadata"}

    def test_the_capability_declares_template_header_and_the_inherited_pair(self) -> None:
        assert set(RenderMetadata.model_fields) == {
            "instructions",
            "expose",
            "template",
            "header",
        }

    def test_it_ships_on_the_prompt_and_command_channels_by_default(self) -> None:
        assert RenderMetadata(template=TEMPLATE).expose == {SYSTEM_PROMPT, COMMAND}

    def test_the_header_is_optional_and_defaults_to_none(self) -> None:
        assert RenderMetadata(template=TEMPLATE).header is None

    def test_a_template_is_required(self) -> None:
        with pytest.raises(ValidationError):
            RenderMetadata()

    def test_an_empty_template_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            RenderMetadata(template="")

    def test_runtime_state_is_private_never_a_field(self) -> None:
        for name in ("_orchestrator_proxy", "_rendered", "_names"):
            assert name not in MetadataTool.model_fields
            assert name in MetadataTool.__private_attributes__

    def test_it_is_a_tool_card_exported_from_both_roots(self) -> None:
        assert issubclass(MetadataTool, ToolCard)
        assert akgentic.tool.metadata.MetadataTool is MetadataTool
        assert akgentic.tool.metadata.RenderMetadata is RenderMetadata
        assert set(akgentic.tool.metadata.__all__) == {"MetadataTool", "RenderMetadata"}
        assert akgentic.tool.MetadataTool is MetadataTool
        assert "MetadataTool" in akgentic.tool.__all__

    def test_the_model_is_never_given_a_tool_to_fetch_metadata(self) -> None:
        """§Alternatives B: a round trip for content that never changes."""
        card, _ = _wire(_TeamContext())
        assert card.get_tools() == []
        assert MetadataTool(render_metadata=False).get_tools() == []
        assert (
            MetadataTool(
                render_metadata=RenderMetadata(template=TEMPLATE, expose={COMMAND})
            ).get_tools()
            == []
        )


class TestSerialization:
    def test_a_bound_card_still_round_trips(self) -> None:
        """Golden Rule #1b as the guarantee it exists for: a *bound* card serializes."""
        card, _ = _wire(_TeamContext(), header="Team context", expose={COMMAND})
        _render_through_any_channel(card)
        assert card._orchestrator_proxy is not None  # genuinely bound
        assert card._rendered is not None  # and genuinely holding a snapshot

        dumped = card.model_dump()
        assert "_orchestrator_proxy" not in dumped
        assert "_rendered" not in dumped

        restored = MetadataTool.model_validate(dumped)
        assert isinstance(restored.render_metadata, RenderMetadata)
        assert restored.render_metadata.template == TEMPLATE
        assert restored.render_metadata.header == "Team context"
        assert restored.render_metadata.expose == {COMMAND}
        assert restored._orchestrator_proxy is None, "runtime state must not survive"
        assert restored._rendered is None, "the snapshot must not survive"

    def test_it_survives_a_json_round_trip(self) -> None:
        card = MetadataTool(
            render_metadata=RenderMetadata(
                template=TEMPLATE, header="Team context", expose={SYSTEM_PROMPT}
            )
        )
        restored = MetadataTool.model_validate_json(card.model_dump_json())

        assert isinstance(restored.render_metadata, RenderMetadata)
        assert restored.render_metadata.template == TEMPLATE
        assert restored.render_metadata.header == "Team context"
        assert restored.render_metadata.expose == {SYSTEM_PROMPT}

    def test_a_disabled_capability_round_trips_as_false(self) -> None:
        card = MetadataTool(render_metadata=False)
        assert MetadataTool.model_validate_json(card.model_dump_json()).render_metadata is False


# ── AC2: placeholder grammar, rejected at observer() ──────────────────────────


class TestTemplateGrammar:
    @pytest.mark.parametrize(
        ("template", "offender"),
        [
            ("Account: {account.name}", "account.name"),
            ("First: {items[0]}", "items[0]"),
            ("Year: {fiscal_year!r}", "fiscal_year"),
            ("Year: {fiscal_year:>10}", "fiscal_year"),
            ("Auto: {}", "{}"),
            ("Positional: {0}", "0"),
            ("Hyphen: {fiscal-year}", "fiscal-year"),
            ("Nested spec: {fiscal_year:{engagement}}", "fiscal_year"),
        ],
    )
    @pytest.mark.parametrize("metadata_is_set", [False, True], ids=["unset", "set"])
    def test_anything_but_a_bare_identifier_raises_at_wiring(
        self, template: str, offender: str, metadata_is_set: bool
    ) -> None:
        """The ``unset`` half is the honest guard, and it is why both halves run.

        With metadata present a relaxed grammar still raises — from the *name*
        check, over a dotted name that is not a field either — so that half alone
        stays green under exactly the mistake this table exists to catch. With no
        metadata to check names against, only the grammar rule can raise.
        """
        observer = _make_observer(_TeamContext() if metadata_is_set else None)
        card = MetadataTool(render_metadata=RenderMetadata(template=template))

        with pytest.raises(ValueError) as excinfo:
            card.observer(observer)

        assert offender in str(excinfo.value)

    def test_escaped_braces_are_literal_text_not_placeholders(self) -> None:
        card, _ = _wire(_TeamContext(), template="{{not a placeholder}} {fiscal_year}")
        assert _prompt(card)() == "{not a placeholder} FY26"

    def test_a_template_with_no_placeholders_at_all_is_valid(self) -> None:
        card, _ = _wire(_TeamContext(), template="This team works on nothing in particular.")
        assert _prompt(card)() == "This team works on nothing in particular."


# ── AC3: unknown field name, rejected at observer() ───────────────────────────


class TestNameValidation:
    def test_an_unknown_name_raises_at_wiring_naming_the_real_fields(self) -> None:
        observer = _make_observer(_TeamContext())
        card = MetadataTool(render_metadata=RenderMetadata(template="Account: {account}"))

        with pytest.raises(ValueError) as excinfo:
            card.observer(observer)

        message = str(excinfo.value)
        assert "account" in message
        assert "fiscal_year" in message
        assert "engagement" in message

    def test_a_template_whose_names_all_resolve_wires_without_error(self) -> None:
        card, observer = _wire(_TeamContext())
        assert card._orchestrator_proxy is _proxy(observer)

    def test_wiring_fails_when_the_observer_has_no_orchestrator(self) -> None:
        observer = _make_observer(_TeamContext())
        observer.orchestrator = None
        card = MetadataTool(render_metadata=RenderMetadata(template=TEMPLATE))

        with pytest.raises(ValueError, match="orchestrator"):
            card.observer(observer)

    def test_enabling_the_capability_without_a_template_fails_at_wiring(self) -> None:
        """``render_metadata=True`` means "defaults", and there is no default template."""
        with pytest.raises(ValueError):
            MetadataTool().observer(_make_observer(_TeamContext()))


# ── AC4: deferred name validation when metadata is unset at wiring ────────────


class TestDeferredValidation:
    def test_grammar_is_still_enforced_when_metadata_is_unset(self) -> None:
        observer = _make_observer(None)
        card = MetadataTool(render_metadata=RenderMetadata(template="Account: {account.name}"))

        with pytest.raises(ValueError, match="account.name"):
            card.observer(observer)

    def test_names_that_cannot_be_checked_yet_wire_without_error(self) -> None:
        card, _ = _wire(None, template="Account: {account}")
        assert card.get_system_prompts()  # wired, and contributing

    def test_a_bad_name_then_surfaces_at_render_as_a_log_not_an_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        card, observer = _wire(None, template="Account: {account}")
        _proxy(observer).get_metadata.return_value = _TeamContext()

        with caplog.at_level(logging.DEBUG):
            rendered = _prompt(card)()

        assert rendered == ""
        errors = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert errors, "a deferred name failure must be visible"
        assert "account" in errors[0].getMessage()
        assert "_TeamContext" in errors[0].getMessage()


# ── AC5: rendering and header ─────────────────────────────────────────────────


class TestRendering:
    def test_every_placeholder_is_replaced_by_the_field_value(self) -> None:
        card, _ = _wire(_TeamContext())
        assert _prompt(card)() == RENDERED

    def test_a_header_is_bolded_on_its_own_line(self) -> None:
        card, _ = _wire(_TeamContext(), header="Team context")
        assert _prompt(card)() == f"**Team context**\n{RENDERED}"

    def test_without_a_header_the_block_is_the_rendered_text_alone(self) -> None:
        card, _ = _wire(_TeamContext())
        rendered = _prompt(card)()
        assert not rendered.startswith("\n")
        assert "**" not in rendered

    def test_a_name_used_twice_is_collected_once_and_rendered_twice(self) -> None:
        card, _ = _wire(_TeamContext(), template="{fiscal_year}, again {fiscal_year}")
        assert card._names == ["fiscal_year"]
        assert _prompt(card)() == "FY26, again FY26"

    def test_a_non_string_field_is_rendered_with_str(self) -> None:
        card, _ = _wire(_OtherContext(), template="Q{quarter} in {region}")
        assert _prompt(card)() == "Q3 in EMEA"


# ── AC6: degradation at render — the callable never raises ────────────────────


class TestDegradation:
    def test_absent_metadata_contributes_nothing_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        card, _ = _wire(None)

        with caplog.at_level(logging.DEBUG):
            rendered = _prompt(card)()

        assert rendered == ""
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert warnings
        assert "MetadataTool" in warnings[0].getMessage()

    def test_a_missing_name_contributes_nothing_and_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        card, observer = _wire(None, template="Account: {account}")
        _proxy(observer).get_metadata.return_value = _OtherContext()

        with caplog.at_level(logging.DEBUG):
            rendered = _prompt(card)()

        assert rendered == ""
        errors = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert errors
        assert "account" in errors[0].getMessage()
        assert "_OtherContext" in errors[0].getMessage()

    def test_an_unexpected_proxy_failure_degrades_the_same_way(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        card, observer = _wire(_TeamContext())
        _proxy(observer).get_metadata.side_effect = RuntimeError("orchestrator is unreachable")

        with caplog.at_level(logging.DEBUG):
            rendered = _prompt(card)()

        assert rendered == ""
        assert [record for record in caplog.records if record.levelno == logging.ERROR]

    def test_an_unwired_card_degrades_rather_than_raising(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        card = MetadataTool(render_metadata=RenderMetadata(template=TEMPLATE))

        with caplog.at_level(logging.DEBUG):
            rendered = _prompt(card)()

        assert rendered == ""
        assert [record for record in caplog.records if record.levelno == logging.ERROR]


# ── AC7: the block is a snapshot, taken at the first successful render ────────


class TestSnapshot:
    def test_two_renders_are_byte_identical_and_read_the_orchestrator_once(self) -> None:
        card, observer = _wire(_TeamContext())
        proxy = _proxy(observer)
        prompt = _prompt(card)

        before = proxy.get_metadata.call_count
        first = prompt()
        second = prompt()

        assert first == second == RENDERED
        # Byte-identity alone passes under a live re-read of unchanged metadata;
        # the call count is what actually pins the snapshot.
        assert proxy.get_metadata.call_count - before == 1

    def test_a_later_set_metadata_is_not_reflected(self) -> None:
        card, observer = _wire(_TeamContext())
        prompt = _prompt(card)
        first = prompt()

        _proxy(observer).get_metadata.return_value = _TeamContext(
            fiscal_year="FY27", engagement="Audit"
        )

        assert prompt() == first == RENDERED

    def test_a_degraded_render_caches_nothing_and_is_retried(self) -> None:
        """FR3b: a ``set_metadata`` landing after start-up still gets its block."""
        card, observer = _wire(None)
        prompt = _prompt(card)
        assert prompt() == ""

        _proxy(observer).get_metadata.return_value = _TeamContext()

        assert prompt() == RENDERED

    def test_the_snapshot_is_shared_across_channels(self) -> None:
        card, observer = _wire(_TeamContext())
        proxy = _proxy(observer)
        prompt = _prompt(card)
        command = card.get_commands()[RenderMetadata]

        before = proxy.get_metadata.call_count
        assert prompt() == command() == RENDERED
        assert proxy.get_metadata.call_count - before == 1


# ── AC8, AC9: channels and capability gating ──────────────────────────────────


class TestChannels:
    def test_the_command_is_named_team_metadata_and_takes_no_arguments(self) -> None:
        card, _ = _wire(_TeamContext())
        command = card.get_commands()[RenderMetadata]
        assert command.__name__ == "team_metadata"
        assert command() == RENDERED

    def test_the_command_renders_the_same_string_as_the_prompt(self) -> None:
        card, _ = _wire(_TeamContext(), header="Team context")
        assert card.get_commands()[RenderMetadata]() == _prompt(card)()

    def test_instructions_reach_the_command_docstring(self) -> None:
        card, _ = _wire(_TeamContext(), instructions="Quote it verbatim.")
        assert "Quote it verbatim." in (card.get_commands()[RenderMetadata].__doc__ or "")

    def test_command_only_exposure_contributes_no_prompt(self) -> None:
        card, _ = _wire(_TeamContext(), expose={COMMAND})
        assert card.get_system_prompts() == []
        assert list(card.get_commands()) == [RenderMetadata]

    def test_prompt_only_exposure_contributes_no_command(self) -> None:
        card, _ = _wire(_TeamContext(), expose={SYSTEM_PROMPT})
        assert card.get_commands() == {}
        assert len(card.get_system_prompts()) == 1

    def test_a_disabled_capability_contributes_nothing_anywhere(self) -> None:
        card = MetadataTool(render_metadata=False)
        card.observer(_make_observer(_TeamContext()))

        assert card.get_system_prompts() == []
        assert card.get_commands() == {}
        assert card.get_tools() == []

    def test_a_disabled_capability_validates_no_template(self) -> None:
        """There is no template to validate, so a broken one cannot be reached."""
        card = MetadataTool(render_metadata=False)
        card.observer(_make_observer(_TeamContext()))  # would raise if it validated
        assert card._orchestrator_proxy is None


# ── AC10: the metadata stays opaque ───────────────────────────────────────────


class TestMetadataStaysOpaque:
    def test_the_card_renders_two_unrelated_metadata_shapes(self) -> None:
        first, _ = _wire(_TeamContext(), template="{engagement}")
        second, _ = _wire(_OtherContext(), template="{region}")

        assert first.get_system_prompts()[0]() == "Advisory"
        assert second.get_system_prompts()[0]() == "EMEA"

    def test_the_package_declares_no_shape_of_its_own(self) -> None:
        """No ``TypedDict``, no ``Protocol``, no example model under ``metadata/``."""
        package = Path(str(akgentic.tool.metadata.__file__)).parent
        forbidden = {"TypedDict", "Protocol"}

        for module in sorted(package.rglob("*.py")):
            tree = ast.parse(module.read_text())
            names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
                node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            }
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            }
            assert not (forbidden & (names | imported)), module


# ── AC11: no retention (ADR-030) ──────────────────────────────────────────────


class TestNoRetention:
    def test_the_closures_do_not_pin_a_stopped_agent(self) -> None:
        observer = _make_observer(_TeamContext())
        card = MetadataTool(render_metadata=RenderMetadata(template=TEMPLATE))
        card.observer(observer)

        ref = weakref.ref(observer)
        held = [*card.get_system_prompts(), *card.get_commands().values()]

        del observer
        gc.collect()

        assert card._observer_or_none() is None
        assert ref() is None
        assert held  # the closures are still referenced, yet the agent was reclaimed

    def test_a_closure_still_renders_after_its_agent_is_gone(self) -> None:
        """The card holds the orchestrator proxy — a different actor, a legal strong edge."""
        observer = _make_observer(_TeamContext())
        card = MetadataTool(render_metadata=RenderMetadata(template=TEMPLATE))
        card.observer(observer)
        prompt = _prompt(card)

        del observer
        gc.collect()

        assert prompt() == RENDERED
