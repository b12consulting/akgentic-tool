"""Tests for the model domain's vocabulary and observer contract (Story 36-1).

Covers the ``ModelRow`` projection round-trip, the runtime conformance of
``ModelSwitchToolObserver`` in both directions, and the export identity of both
symbols through ``akgentic.tool.model`` and the package root.

The conforming fake supplies the six members ``ActorToolObserver`` declares on
this branch base — ``notify_event``, ``myAddress``, ``orchestrator``,
``team_id``, ``state``, ``proxy_ask`` — and nothing more, so the two negative
cases isolate the two new members rather than an incidental omission.

The module boundary is swept rather than eyeballed. It is the epic's blocking
invariant, and the package already learned once that a hand-checked structural
rule is only ever "someone remembered" — see the same reasoning in
``test_core_domain_purity.test_every_subpackage_is_listed_as_a_domain``.
"""

from __future__ import annotations

import ast
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType

import akgentic.tool as tool_package
import akgentic.tool.model as model_package
from akgentic.tool import ModelRow as RootModelRow
from akgentic.tool import ModelSwitchToolObserver as RootModelSwitchToolObserver
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.core.state import ToolState
from akgentic.tool.model import ModelRow as PackageModelRow
from akgentic.tool.model import ModelSwitchToolObserver as PackageModelSwitchToolObserver
from akgentic.tool.model.observer import ModelSwitchToolObserver
from akgentic.tool.model.state import ModelRow

# ---------------------------------------------------------------------------
# AC 2 — ModelRow is a serializable projection
# ---------------------------------------------------------------------------


def test_model_row_round_trip_preserves_every_field() -> None:
    """A populated row survives dump/validate with every field equal."""
    row = ModelRow(
        key="openai:gpt-5.2",
        provider="openai",
        model="gpt-5.2",
        active=True,
        context_length=400_000,
    )

    restored = ModelRow.model_validate(row.model_dump())

    assert restored == row
    assert restored.key == "openai:gpt-5.2"
    assert restored.provider == "openai"
    assert restored.model == "gpt-5.2"
    assert restored.active is True
    assert restored.context_length == 400_000


def test_model_row_round_trips_an_undeclared_context_length() -> None:
    """``context_length`` is optional in value: ``None`` survives the round-trip."""
    row = ModelRow(
        key="anthropic:claude-opus-5",
        provider="anthropic",
        model="claude-opus-5",
        active=False,
        context_length=None,
    )

    restored = ModelRow.model_validate(row.model_dump())

    assert restored == row
    assert restored.context_length is None
    assert restored.active is False


def test_model_row_declares_its_fields_in_the_decided_order() -> None:
    """Field order is part of the contract, and it is what the dump emits."""
    assert list(ModelRow.model_fields) == [
        "key",
        "provider",
        "model",
        "active",
        "context_length",
    ]


# ---------------------------------------------------------------------------
# AC 4 — runtime conformance of ModelSwitchToolObserver, both ways
# ---------------------------------------------------------------------------


class _Carrier:
    """Minimal ``ToolStateCarrier``: an object exposing ``tool_state``."""

    def __init__(self) -> None:
        self.tool_state = ToolState()


class _ObserverBase:
    """Every member ``ActorToolObserver`` declares on this base, and nothing more."""

    def __init__(self) -> None:
        self.myAddress = SimpleNamespace()  # noqa: N815 — protocol member name
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

    def proxy_tell(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
    ) -> Any:
        return None


class _ModelSwitchObserver(_ObserverBase):
    """The base plus both new members — the conforming case."""

    def list_model_rows(self) -> list[ModelRow]:
        return []

    def switch_model(self, key: str) -> str:
        return f"switched to {key}"


class _WithoutListModelRows(_ObserverBase):
    """Otherwise identical, missing ``list_model_rows``."""

    def switch_model(self, key: str) -> str:
        return f"switched to {key}"


class _WithoutSwitchModel(_ObserverBase):
    """Otherwise identical, missing ``switch_model``."""

    def list_model_rows(self) -> list[ModelRow]:
        return []


def test_conforming_observer_satisfies_the_protocol() -> None:
    """Every base member plus both new methods passes the runtime check."""
    assert isinstance(_ModelSwitchObserver(), ModelSwitchToolObserver)


def test_observer_missing_list_model_rows_fails() -> None:
    """``list_model_rows`` is required — its absence alone fails the check."""
    assert not isinstance(_WithoutListModelRows(), ModelSwitchToolObserver)


def test_observer_missing_switch_model_fails() -> None:
    """``switch_model`` is required — its absence alone fails the check."""
    assert not isinstance(_WithoutSwitchModel(), ModelSwitchToolObserver)


def test_the_negative_cases_isolate_the_new_members() -> None:
    """Guard the guards: both negatives still satisfy the base protocol.

    Without this, a fake that failed for a missing ``team_id`` would look like
    proof that ``switch_model`` is required.
    """
    assert isinstance(_WithoutListModelRows(), ActorToolObserver)
    assert isinstance(_WithoutSwitchModel(), ActorToolObserver)


def test_the_base_protocol_is_not_widened() -> None:
    """A plain ``ActorToolObserver`` stand-in is unaffected by the sibling protocol."""
    base = _ObserverBase()
    assert isinstance(base, ActorToolObserver)
    assert not isinstance(base, ModelSwitchToolObserver)


# ---------------------------------------------------------------------------
# AC 8 — exports
# ---------------------------------------------------------------------------


def test_both_symbols_are_exported_by_identity() -> None:
    """One class each, reachable from the domain package and from the root."""
    assert PackageModelRow is ModelRow
    assert RootModelRow is ModelRow
    assert PackageModelSwitchToolObserver is ModelSwitchToolObserver
    assert RootModelSwitchToolObserver is ModelSwitchToolObserver


def test_both_symbols_are_named_in_both_all_lists() -> None:
    """The export surface declares them, not merely the import machinery."""
    assert "ModelRow" in model_package.__all__
    assert "ModelSwitchToolObserver" in model_package.__all__
    assert "ModelRow" in tool_package.__all__
    assert "ModelSwitchToolObserver" in tool_package.__all__


# ---------------------------------------------------------------------------
# AC 1 — the module boundary, swept over the package rather than checked by eye
# ---------------------------------------------------------------------------

MODEL_DIR = Path(model_package.__file__).parent

# Dotted parts of the package ``MODEL_DIR`` holds, for resolving relative imports.
# ``from ...llm import ModelConfig`` reaches ``akgentic.llm`` from inside
# ``akgentic.tool.model`` without ever writing the string, so the sweep resolves
# levels instead of matching on absolute module text.
_MODEL_PACKAGE_PARTS = ("akgentic", "tool", "model")

_FORBIDDEN_ROOT = "akgentic.llm"

# The one name this package must never bind. ``ModelRow`` exists precisely so that
# ``ModelConfig`` — which lives in ``akgentic-llm`` — has no reason to appear here.
_FORBIDDEN_NAME = "ModelConfig"


def _model_package_modules() -> list[Path]:
    """Every module in the model domain package, including any 36-2 adds."""
    return sorted(MODEL_DIR.rglob("*.py"))


def _imported_modules(tree: ast.Module, package_parts: tuple[str, ...]) -> set[str]:
    """Every dotted module path *tree* imports, with relative imports resolved.

    Each ``from`` clause contributes the module it names **and** that module plus
    each imported name, so ``from akgentic import llm`` is reported as
    ``akgentic.llm`` rather than as the bare ``akgentic`` the clause carries.
    """
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                module = node.module or ""
            else:
                base = package_parts[: len(package_parts) - node.level + 1]
                module = ".".join([*base, node.module] if node.module else list(base))
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def _binds_forbidden_name(tree: ast.Module) -> bool:
    """Whether *tree* binds or reads the identifier ``ModelConfig``.

    Deliberately an identifier check and not a text search. The package names
    ``ModelConfig`` in prose in two places already — ``team/activity.py`` and
    ``team/README.md`` — precisely in order to forbid it, and a text ban would
    outlaw the warning while catching nothing a reader could not already see.
    What must not exist is a *use*.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == _FORBIDDEN_NAME:
            return True
        if isinstance(node, ast.Attribute) and node.attr == _FORBIDDEN_NAME:
            return True
        if isinstance(node, ast.ImportFrom | ast.Import) and any(
            alias.name == _FORBIDDEN_NAME or alias.asname == _FORBIDDEN_NAME
            for alias in node.names
        ):
            return True
    return False


def test_the_model_package_imports_no_akgentic_llm() -> None:
    """Epic 36 invariant 1: this package may reach for ``akgentic-core`` only."""
    violations: list[str] = []
    for module_path in _model_package_modules():
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        parts = (*_MODEL_PACKAGE_PARTS, *module_path.relative_to(MODEL_DIR).parts[:-1])
        for imported in sorted(_imported_modules(tree, parts)):
            if imported == _FORBIDDEN_ROOT or imported.startswith(f"{_FORBIDDEN_ROOT}."):
                violations.append(f"{module_path.relative_to(MODEL_DIR)} imports {imported}")
    assert not violations, f"module boundary violated: {violations}"


def test_the_model_package_never_names_model_config() -> None:
    """``ModelRow`` is the projection that keeps ``ModelConfig`` out of this package."""
    violations = [
        module_path.relative_to(MODEL_DIR).as_posix()
        for module_path in _model_package_modules()
        if _binds_forbidden_name(ast.parse(module_path.read_text(encoding="utf-8")))
    ]
    assert not violations, f"{_FORBIDDEN_NAME} is used in: {violations}"


def test_the_boundary_sweep_is_not_vacuous() -> None:
    """Guard the guard: a mistyped glob would make both sweeps trivially green."""
    swept = {path.relative_to(MODEL_DIR).as_posix() for path in _model_package_modules()}
    assert {"__init__.py", "observer.py", "state.py", "tool.py"} <= swept, swept


@pytest.mark.parametrize(
    "source",
    [
        "from akgentic.llm import ModelConfig",
        "from akgentic.llm.models import ModelConfig",
        "import akgentic.llm",
        "import akgentic.llm as llm_package",
        "from akgentic import llm",
        "from ...llm import ModelConfig",
        "def f():\n    from akgentic.llm import ModelConfig\n",
    ],
)
def test_every_import_form_of_the_forbidden_package_is_reported(source: str) -> None:
    """Guard the guard: the sweep matches module paths, so the parser must produce them.

    Asking whether the sweep *would* go red is a question about the parser, not
    about today's tree — a boundary breach will not be written in whichever form
    happens to be absent now. The relative form reaches ``akgentic.llm`` without
    the string appearing in the source at all.
    """
    reported = _imported_modules(ast.parse(source), _MODEL_PACKAGE_PARTS)
    assert any(
        name == _FORBIDDEN_ROOT or name.startswith(f"{_FORBIDDEN_ROOT}.") for name in reported
    ), reported


@pytest.mark.parametrize(
    "source",
    [
        "from akgentic.llm import ModelConfig",
        "from akgentic.llm import ModelConfig as Cfg",
        "from akgentic.llm import models as ModelConfig",
        "row = ModelConfig(provider='openai')",
        "def f(cfg: ModelConfig) -> None: ...",
        "llm.ModelConfig",
    ],
)
def test_every_use_form_of_the_forbidden_name_is_reported(source: str) -> None:
    """Guard the guard: the name check must see bindings, aliases and reads alike."""
    assert _binds_forbidden_name(ast.parse(source))


def test_a_docstring_mentioning_the_forbidden_name_is_not_a_violation() -> None:
    """The rule bans a use, not the prose that explains the rule.

    ``team/activity.py`` and ``team/README.md`` both name ``ModelConfig`` in order
    to forbid it. A text-matching guard would turn that documentation red.
    """
    assert not _binds_forbidden_name(
        ast.parse('"""Never a ModelConfig here — see ModelRow."""\n# not a ModelConfig\n')
    )
