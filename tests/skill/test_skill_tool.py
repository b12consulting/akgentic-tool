"""``SkillTool``: the menu in the prefix, the bodies on demand, and nothing kept.

Story 33-1 (ADR-039). Three properties carry the whole design, and each has a class
below whose failure means the card no longer does what it was built for:

* ``use_skill`` returns the **body**, in the tool result — not an acknowledgement that
  the body is on its way (``TestUseSkill``);
* the card records **nothing** — two calls leave it byte-identical to a card that was
  never called (``TestTheCardKeepsNothing``);
* the menu is O(skills), never O(content), which is what lets it live in the frozen
  prefix at all (``TestMenuCostIsIndependentOfBodySize``).

The observer here is a bare ``Mock(spec=ActorToolObserver)``: this card needs no
observer, so nothing is wired onto it. It exists only so ``TestNoRetention`` has
something weak-referenceable to collect.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import akgentic.tool
import akgentic.tool.skill
from akgentic.tool.core import COMMAND, SYSTEM_PROMPT, TOOL_CALL, ToolCard
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.errors import RetriableError
from akgentic.tool.skill import SkillEntry, Skills, SkillTool
from akgentic.tool.skill.tool import MENU_HEADER as HEADER

REFUND = SkillEntry(
    name="refund-policy",
    description="How refunds are approved, and the thresholds that need a second signature.",
    content="Refunds under 100 EUR are approved by the agent handling the case.",
)
ESCALATION = SkillEntry(
    name="escalation",
    description="When to escalate to a human, and what the handover must contain.",
    content="Escalate whenever the customer asks for a person, or after two failed fixes.",
)

MENU = (
    f"{HEADER}\n"
    "refund-policy — How refunds are approved, and the thresholds that need a second "
    "signature.\n"
    "escalation — When to escalate to a human, and what the handover must contain."
)


def _card(**params: Any) -> SkillTool:
    """A card carrying the two example skills, plus any *params* overrides."""
    params.setdefault("skills", [REFUND, ESCALATION])
    return SkillTool(param=Skills(**params))


def _menu(card: SkillTool) -> str:
    """The card's single system-prompt callable, rendered."""
    rendered: str = card.get_system_prompts()[0]()
    return rendered


def _use_skill(card: SkillTool) -> Any:
    """The card's single ``use_skill`` callable."""
    return card.get_tools()[0]


def _observer() -> Mock:
    """A weak-referenceable observer. This card asks it for nothing."""
    return Mock(spec=ActorToolObserver)


# ── AC1, AC2, AC12: the models and the exports ────────────────────────────────


class TestCardSurface:
    def test_the_card_declares_exactly_one_field(self) -> None:
        assert set(SkillTool.model_fields) == {"param"}

    def test_the_capability_ships_on(self) -> None:
        assert SkillTool.model_fields["param"].default is True
        assert SkillTool().param is True

    def test_the_entry_declares_exactly_name_description_and_content(self) -> None:
        assert set(SkillEntry.model_fields) == {"name", "description", "content"}

    def test_the_capability_declares_skills_and_the_inherited_pair(self) -> None:
        assert set(Skills.model_fields) == {"instructions", "expose", "skills"}

    def test_it_ships_on_all_three_channels_by_default(self) -> None:
        assert Skills().expose == {SYSTEM_PROMPT, TOOL_CALL, COMMAND}

    def test_a_card_with_no_skills_configured_carries_an_empty_library(self) -> None:
        assert Skills().skills == []

    @pytest.mark.parametrize("field", ["name", "description", "content"])
    def test_an_empty_string_is_rejected_at_construction(self, field: str) -> None:
        values = {"name": "refund-policy", "description": "One line.", "content": "A body."}
        values[field] = ""
        with pytest.raises(ValidationError):
            SkillEntry(**values)

    def test_it_is_a_tool_card_exported_from_both_roots(self) -> None:
        assert issubclass(SkillTool, ToolCard)
        assert akgentic.tool.skill.SkillTool is SkillTool
        assert akgentic.tool.skill.Skills is Skills
        assert akgentic.tool.skill.SkillEntry is SkillEntry
        assert set(akgentic.tool.skill.__all__) == {"SkillEntry", "SkillTool", "Skills"}
        assert akgentic.tool.SkillTool is SkillTool
        assert "SkillTool" in akgentic.tool.__all__

    def test_only_the_card_joins_the_package_root(self) -> None:
        """``Skills`` and ``SkillEntry`` reach an operator through the subpackage."""
        assert "Skills" not in akgentic.tool.__all__
        assert "SkillEntry" not in akgentic.tool.__all__


# ── AC3: unique names, at validation rather than at call time ─────────────────


class TestUniqueNames:
    def test_a_duplicate_name_raises_at_model_validation(self) -> None:
        twin = SkillEntry(
            name="refund-policy",
            description="A second entry claiming the same handle.",
            content="Whatever this says, the model could never ask for it.",
        )
        with pytest.raises(ValidationError) as excinfo:
            Skills(skills=[REFUND, twin])

        assert "refund-policy" in str(excinfo.value)

    def test_distinct_names_validate(self) -> None:
        assert len(Skills(skills=[REFUND, ESCALATION]).skills) == 2


# ── AC4: the menu ─────────────────────────────────────────────────────────────


class TestMenu:
    def test_one_line_per_skill_under_the_header(self) -> None:
        assert _menu(_card()) == MENU

    def test_the_header_is_rendered_verbatim(self) -> None:
        assert _menu(_card()).splitlines()[0] == HEADER

    def test_entries_render_in_declaration_order_never_sorted(self) -> None:
        """``escalation`` sorts before ``refund-policy``; the operator's order wins."""
        lines = _menu(_card()).splitlines()[1:]
        assert [line.split(" — ")[0] for line in lines] == ["refund-policy", "escalation"]

    def test_an_empty_library_contributes_the_empty_string(self) -> None:
        card = SkillTool(param=Skills(skills=[]))
        assert card.get_system_prompts(), "the capability is still enabled"
        assert _menu(card) == ""

    def test_the_menu_never_carries_a_body(self) -> None:
        rendered = _menu(_card())
        assert REFUND.content not in rendered
        assert ESCALATION.content not in rendered

    def test_the_default_card_renders_an_empty_menu(self) -> None:
        """``param=True`` means "enabled with defaults", and the default library is empty."""
        assert _menu(SkillTool()) == ""


# ── AC5 / NFR1: the prefix cost is O(skills), never O(content) ────────────────


class TestMenuCostIsIndependentOfBodySize:
    def test_fifty_kilobyte_bodies_render_the_same_menu_as_twenty_character_ones(self) -> None:
        def library(content: str) -> SkillTool:
            return SkillTool(
                param=Skills(
                    skills=[
                        SkillEntry(name="refund-policy", description="Refunds.", content=content),
                        SkillEntry(name="escalation", description="Escalation.", content=content),
                    ]
                )
            )

        large = _menu(library("x" * 50_000))
        small = _menu(library("y" * 20))

        assert large == small
        # Bounded against the header, not against a bare number: the header is prose and
        # gets reworded, while what must never change is that the menu does not grow with
        # the bodies. Inlining one 50 kB body blows past this by two orders of magnitude.
        assert len(large) < len(HEADER) + 200


# ── AC6: use_skill returns the body ───────────────────────────────────────────


class TestUseSkill:
    def test_the_tool_is_named_use_skill(self) -> None:
        assert _use_skill(_card()).__name__ == "use_skill"

    def test_it_returns_the_name_then_a_blank_line_then_the_body(self) -> None:
        assert _use_skill(_card())("refund-policy") == f"refund-policy\n\n{REFUND.content}"

    def test_the_body_is_the_tool_result_not_an_acknowledgement(self) -> None:
        """Invariant 1: the model asked because it needs the body for *this* turn."""
        assert ESCALATION.content in _use_skill(_card())("escalation")

    def test_an_unknown_name_raises_retriable_listing_the_available_names(self) -> None:
        with pytest.raises(RetriableError) as excinfo:
            _use_skill(_card())("refunds")

        message = str(excinfo.value)
        assert "refunds" in message
        assert "refund-policy" in message
        assert "escalation" in message

    def test_an_unknown_name_against_an_empty_library_still_raises_retriable(self) -> None:
        card = SkillTool(param=Skills(skills=[]))
        with pytest.raises(RetriableError):
            card.get_tools()[0]("refund-policy")

    def test_calling_it_twice_on_one_name_returns_the_body_both_times(self) -> None:
        """No "already loaded" case exists — after a fold, the repeat call is the point."""
        use_skill = _use_skill(_card())
        assert use_skill("refund-policy") == use_skill("refund-policy")
        assert REFUND.content in use_skill("refund-policy")

    def test_instructions_reach_the_tool_docstring(self) -> None:
        card = _card(instructions="Load a skill before answering a policy question.")
        doc = _use_skill(card).__doc__ or ""
        assert "Load a skill before answering a policy question." in doc

    def test_the_docstring_is_dedented_before_the_instructions_are_appended(self) -> None:
        """griffe dedents only when every line past the first shares one margin.

        The appended instructions are flush left, so a docstring still carrying its
        source margin would leave the two mismatched — ``Args:`` would go unparsed and
        the ``name`` parameter would reach the model undescribed. A flush-left
        ``Args:`` header is exactly what says the dedent ran first.
        """
        card = _card(instructions="Anything at all.")
        doc = _use_skill(card).__doc__ or ""
        assert "\nArgs:\n" in doc, doc
        assert "\nAdditional Instructions:\n" in doc, doc


# ── AC7: the skills() command ─────────────────────────────────────────────────


class TestCommand:
    def test_the_command_is_registered_under_the_param_class(self) -> None:
        assert list(_card().get_commands()) == [Skills]

    def test_it_is_named_skills_and_takes_no_arguments(self) -> None:
        command = _card().get_commands()[Skills]
        assert command.__name__ == "skills"
        assert command() == MENU

    def test_it_renders_exactly_what_the_prompt_renders(self) -> None:
        card = _card()
        assert card.get_commands()[Skills]() == _menu(card)

    def test_instructions_reach_the_command_docstring(self) -> None:
        card = _card(instructions="Quote the menu verbatim.")
        assert "Quote the menu verbatim." in (card.get_commands()[Skills].__doc__ or "")


# ── AC8: the three channels are independent ───────────────────────────────────


class TestChannels:
    def test_prompt_only_exposure_contributes_the_menu_alone(self) -> None:
        card = _card(expose={SYSTEM_PROMPT})
        assert len(card.get_system_prompts()) == 1
        assert card.get_tools() == []
        assert card.get_commands() == {}

    def test_tool_only_exposure_contributes_use_skill_alone(self) -> None:
        card = _card(expose={TOOL_CALL})
        assert card.get_system_prompts() == []
        assert len(card.get_tools()) == 1
        assert card.get_commands() == {}

    def test_command_only_exposure_contributes_the_command_alone(self) -> None:
        card = _card(expose={COMMAND})
        assert card.get_system_prompts() == []
        assert card.get_tools() == []
        assert list(card.get_commands()) == [Skills]

    def test_a_disabled_card_contributes_nothing_and_raises_nothing(self) -> None:
        card = SkillTool(param=False)
        card.observer(_observer())

        assert card.get_system_prompts() == []
        assert card.get_tools() == []
        assert card.get_commands() == {}

    def test_the_default_card_contributes_on_all_three_channels(self) -> None:
        card = SkillTool()
        assert len(card.get_system_prompts()) == 1
        assert len(card.get_tools()) == 1
        assert list(card.get_commands()) == [Skills]


# ── AC9 / NFR2: the card keeps no per-agent runtime state ─────────────────────


class TestTheCardKeepsNothing:
    def test_it_declares_no_private_attribute_of_its_own(self) -> None:
        """Invariant 2: the inherited weak observer is the only one there is."""
        assert set(SkillTool.__private_attributes__) == set(ToolCard.__private_attributes__)

    def test_two_calls_leave_the_card_indistinguishable_from_a_fresh_one(self) -> None:
        card = _card()
        before = card.model_dump()

        use_skill = _use_skill(card)
        use_skill("refund-policy")
        use_skill("escalation")

        assert card.model_dump() == before == _card().model_dump()

    def test_the_menu_is_unchanged_by_a_use_skill_call(self) -> None:
        card = _card()
        _use_skill(card)("refund-policy")
        assert _menu(card) == MENU


# ── AC10: serialization round-trip ────────────────────────────────────────────


class TestSerialization:
    def test_it_round_trips_through_a_dict(self) -> None:
        card = _card(expose={SYSTEM_PROMPT, TOOL_CALL})
        restored = SkillTool.model_validate(card.model_dump())

        assert isinstance(restored.param, Skills)
        assert restored.param.skills == [REFUND, ESCALATION]
        assert restored.param.expose == {SYSTEM_PROMPT, TOOL_CALL}

    def test_it_round_trips_through_json_preserving_every_entry_in_order(self) -> None:
        card = _card()
        restored = SkillTool.model_validate_json(card.model_dump_json())

        assert isinstance(restored.param, Skills)
        assert [
            (entry.name, entry.description, entry.content) for entry in restored.param.skills
        ] == [
            (REFUND.name, REFUND.description, REFUND.content),
            (ESCALATION.name, ESCALATION.description, ESCALATION.content),
        ]
        assert restored.param.expose == {SYSTEM_PROMPT, TOOL_CALL, COMMAND}

    def test_a_disabled_card_round_trips_as_false(self) -> None:
        card = SkillTool(param=False)
        assert SkillTool.model_validate_json(card.model_dump_json()).param is False

    def test_a_restored_card_renders_the_same_menu(self) -> None:
        restored = SkillTool.model_validate_json(_card().model_dump_json())
        assert _menu(restored) == MENU


# ── AC11 / NFR3: no retention (ADR-030) ───────────────────────────────────────


class TestNoRetention:
    def test_the_closures_do_not_pin_a_stopped_agent(self) -> None:
        observer = _observer()
        card = _card()
        card.observer(observer)

        ref = weakref.ref(observer)
        held = [
            *card.get_system_prompts(),
            *card.get_tools(),
            *card.get_commands().values(),
        ]

        del observer
        gc.collect()

        assert card._observer_or_none() is None
        assert ref() is None
        assert held  # the closures are still referenced, yet the agent was reclaimed

    def test_every_closure_still_works_after_its_agent_is_gone(self) -> None:
        """This card reads only its own configuration, so nothing degrades."""
        observer = _observer()
        card = _card()
        card.observer(observer)
        menu = card.get_system_prompts()[0]
        use_skill = card.get_tools()[0]
        command = card.get_commands()[Skills]

        del observer
        gc.collect()

        assert menu() == command() == MENU
        assert use_skill("escalation") == f"escalation\n\n{ESCALATION.content}"
