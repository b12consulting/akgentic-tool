"""Structural guard for the ``akgentic.tool.core`` intra-package import chain."""

import ast
from pathlib import Path

import akgentic.tool.core as core_package

# Strict layering: a module may only import siblings that appear EARLIER here.
#
# ``channels``, ``context_state`` and ``event`` are chain heads, not outside-chain
# modules: ``card`` and ``factory`` import ``ToolObserver``, and ``commands`` imports
# the command-discovery models, so the chain depends on them. ``context_state`` sits
# ahead of ``observer`` because ``observer`` imports ``state``, which imports
# ``context_state``; ``context_state`` itself imports no core sibling (its outward
# edge is ``akgentic.core``), which is what makes it legal at the front.
CHAIN: list[str] = [
    "channels",
    "context_state",
    "state",
    "observer",
    "event",
    "params",
    "card",
    "dependencies",
    "commands",
    "context_update",
    "factory",
]

# Modules that sit OUTSIDE the chain: they depend on ``akgentic.core`` only, and
# nothing in the chain may reach for them either.
OUTSIDE_CHAIN: list[str] = ["deferred"]

_TRACKED: set[str] = set(CHAIN) | set(OUTSIDE_CHAIN)

CORE_DIR = Path(core_package.__file__).parent

_PREFIX = "akgentic.tool.core."


def _sibling_imports(module_path: Path) -> set[str]:
    """Return the ``core`` siblings imported by the module at *module_path*.

    Covers relative (``from .card import ...``), sibling-as-name (``from . import
    card``), absolute (``from akgentic.tool.core.card import ...``) and plain-``import``
    forms, at module level and inside function bodies alike. The sibling-as-name form is
    tracked because it names the module on the alias rather than on ``node.module``,
    which is exactly where a guard matching only on ``node.module`` stops looking.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    siblings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module is None:
                siblings.update(alias.name for alias in node.names)
            elif node.level == 1 and node.module:
                siblings.add(node.module.split(".")[0])
            elif node.level == 0 and node.module and node.module.startswith(_PREFIX):
                siblings.add(node.module.removeprefix(_PREFIX).split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_PREFIX):
                    siblings.add(alias.name.removeprefix(_PREFIX).split(".")[0])
    return siblings & _TRACKED


def test_core_modules_import_only_earlier_chain_members() -> None:
    """No ``core/`` module imports a sibling that appears later in the chain."""
    violations: list[str] = []
    for position, module_name in enumerate(CHAIN):
        module_path = CORE_DIR / f"{module_name}.py"
        assert module_path.is_file(), f"missing core module: {module_path}"
        for imported in sorted(_sibling_imports(module_path) & set(CHAIN)):
            if CHAIN.index(imported) >= position:
                violations.append(f"{module_name} imports {imported}")
    assert not violations, f"core import chain violated: {violations}"


def test_outside_chain_modules_import_no_core_sibling() -> None:
    """A module outside the chain depends on ``akgentic.core`` only."""
    violations: list[str] = []
    for module_name in OUTSIDE_CHAIN:
        module_path = CORE_DIR / f"{module_name}.py"
        assert module_path.is_file(), f"missing core module: {module_path}"
        for imported in sorted(_sibling_imports(module_path)):
            violations.append(f"{module_name} imports {imported}")
    assert not violations, f"outside-chain module reached into core/: {violations}"


def test_chain_modules_do_not_import_outside_chain_modules() -> None:
    """The chain must not grow a dependency on a module that sits outside it."""
    violations: list[str] = []
    for module_name in CHAIN:
        module_path = CORE_DIR / f"{module_name}.py"
        for imported in sorted(_sibling_imports(module_path) & set(OUTSIDE_CHAIN)):
            violations.append(f"{module_name} imports {imported}")
    assert not violations, f"chain module imports an outside-chain module: {violations}"
