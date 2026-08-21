"""Structural guards that ``core/`` names no domain, in both directions.

``02-core.md`` has stated this in prose for a while. These tests make it mechanical:
prose does not go red when someone adds an import.

Four invariants:

* no module under ``core/`` imports a domain package, a deprecated façade, or the package
  root — the root is listed because its lazy ``__getattr__`` hands out knowledge-graph
  types, so an import through it restores the edge without ever naming a domain;
* no module anywhere under ``src/`` imports either deprecated façade — a package must not
  consume its own deprecated path, and that reasoning was never specific to ``core/``;
* no module under ``knowledge_graph/`` reaches back for the event modules, and
  ``models.py`` calls no ``model_rebuild`` — the bottom-of-file rebuild was one of
  three coupled mechanisms holding the old cycle open, and it is the cheapest one to
  reintroduce by reflex when a resolution error shows up;
* building a ``ToolStateEvent`` does not drag the knowledge-graph package into
  ``sys.modules`` — the behavioural proof that the edge is gone rather than moved.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import akgentic.tool as tool_package
import akgentic.tool.core as core_package
import akgentic.tool.knowledge_graph as kg_package

TOOL_DIR = Path(tool_package.__file__).parent
CORE_DIR = Path(core_package.__file__).parent
KG_DIR = Path(kg_package.__file__).parent

# Every domain package under ``akgentic.tool``. ``core/`` may name none of them.
DOMAIN_PACKAGES: frozenset[str] = frozenset(
    {
        "knowledge_graph",
        "team",
        "workspace",
        "planning",
        "sandbox",
        "search",
        "mcp",
        "vector_store",
        "vector",
        "notification",
        "metadata",
        "skill",
    }
)

# The deprecated façades. A ``core/`` module reaching through ``event`` would re-create the
# domain edge indirectly, since that façade resolves knowledge-graph symbols. More
# generally, no module in this package may consume a path the package itself deprecated —
# which is why the sweep below covers all of ``src/`` and not only ``core/``.
FACADE_MODULES: frozenset[str] = frozenset({"akgentic.tool.event", "akgentic.tool.vector"})

# The façade files themselves, which are the one legitimate place these names appear.
FACADE_FILENAMES: frozenset[str] = frozenset({"event.py", "vector.py"})

_TOOL_PREFIX = "akgentic.tool."

# The package root, which is a violation in its own right and not merely an ancestor of
# one. ``akgentic/tool/__init__.py`` serves ``KnowledgeGraphStateEvent`` from a lazy
# module ``__getattr__``, so ``from akgentic.tool import KnowledgeGraphStateEvent``
# inside ``core/`` restores the runtime edge while naming no domain package at all —
# the cheapest way past a guard that matches on domain names.
ROOT_PACKAGE = "akgentic.tool"


def _is_tool_module(module: str) -> bool:
    """Whether *module* names the tool package root or something inside it."""
    return module == ROOT_PACKAGE or module.startswith(_TOOL_PREFIX)


def _package_parts(package_dir: Path) -> tuple[str, ...]:
    """Dotted parts of the package *package_dir* holds, e.g. ``('akgentic', 'tool', 'core')``.

    Derived from the path rather than from ``package_dir.name`` so the tool package root
    itself is a legal argument — ``TOOL_DIR.name`` is ``"tool"``, which a name-based form
    would render as ``akgentic.tool.tool`` and silently mis-resolve every relative import
    in the whole-``src/`` sweep.
    """
    return ("akgentic", "tool", *package_dir.relative_to(TOOL_DIR).parts)


def _imports_in(tree: ast.Module, package_parts: tuple[str, ...]) -> set[str]:
    """Return the ``akgentic.tool*`` module paths *tree* imports.

    Covers ``from x import y`` (relative and absolute) and plain ``import x``, at
    module level and inside function bodies alike. Relative imports are resolved
    against *package_parts*, the package the module actually lives in, so
    ``from ..team.observer import X`` is reported with its absolute path, exactly as an
    absolute import would be.

    Each ``from`` clause contributes the module it names **and** that module plus each
    imported name. ``from akgentic.tool import vector`` and ``from . import event`` carry
    the module on the alias rather than on the clause, so a set built from the clause
    alone reports only ``akgentic.tool`` and the façade slips through a guard matching on
    module paths.
    """
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if not (node.module and _is_tool_module(node.module)):
                    continue
                module = node.module
            else:
                base = package_parts[: len(package_parts) - node.level + 1]
                module = ".".join([*base, node.module] if node.module else list(base))
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_tool_module(alias.name):
                    imported.add(alias.name)
    return imported


def _imported_modules(module_path: Path, package_dir: Path) -> set[str]:
    """Return the ``akgentic.tool*`` module paths imported by *module_path*."""
    return _imports_in(
        ast.parse(module_path.read_text(encoding="utf-8")),
        (
            *_package_parts(package_dir),
            *module_path.relative_to(package_dir).parts[:-1],
        ),
    )


def _domain_of(module: str) -> str | None:
    """Return the ``akgentic.tool`` domain a module path belongs to, if any."""
    if not module.startswith(_TOOL_PREFIX):
        return None
    head = module.removeprefix(_TOOL_PREFIX).split(".")[0]
    return head if head in DOMAIN_PACKAGES else None


def test_core_modules_import_no_domain_package() -> None:
    """No module under ``core/`` names a domain package, the façade, or the root."""
    violations: list[str] = []
    for module_path in sorted(CORE_DIR.rglob("*.py")):
        name = module_path.relative_to(CORE_DIR).as_posix()
        for imported in sorted(_imported_modules(module_path, CORE_DIR)):
            if _domain_of(imported) is not None:
                violations.append(f"{name} imports {imported}")
            elif imported in FACADE_MODULES:
                violations.append(f"{name} imports the deprecated {imported}")
            elif imported == ROOT_PACKAGE:
                violations.append(f"{name} imports {imported}, which serves domain types lazily")
    assert not violations, f"core/ reached into a domain: {violations}"


def test_core_directory_is_not_empty() -> None:
    """Guard the guard: an empty glob would make the purity test vacuously green."""
    modules = {path.name for path in CORE_DIR.glob("*.py")}
    assert {"event.py", "observer.py", "card.py", "factory.py"} <= modules


def test_every_subpackage_is_listed_as_a_domain() -> None:
    """Guard the guard: a new card's package must join the denylist with it.

    ``DOMAIN_PACKAGES`` is hand-written and claims to name every domain, so a package
    absent from it is invisible to the purity sweep — ``core/`` could import it and
    nothing would go red. Nothing made the omission surface, and two packages had
    already drifted out of the list. Deriving the expectation from the tree is what
    turns "someone remembered" into "the suite noticed".
    """
    packages = {path.name for path in TOOL_DIR.iterdir() if (path / "__init__.py").is_file()}
    missing = packages - {"core"} - DOMAIN_PACKAGES
    assert not missing, f"subpackages absent from DOMAIN_PACKAGES: {sorted(missing)}"


def test_knowledge_graph_does_not_import_the_deprecated_facade() -> None:
    """A KG module going through the façade would re-open the edge indirectly."""
    violations: list[str] = []
    for module_path in sorted(KG_DIR.rglob("*.py")):
        if _imported_modules(module_path, KG_DIR) & FACADE_MODULES:
            violations.append(module_path.relative_to(KG_DIR).as_posix())
    assert not violations, f"knowledge_graph/ imports a deprecated façade: {violations}"


def _swept_source_modules() -> list[Path]:
    """Every module under ``src/akgentic/tool/`` except the façades themselves."""
    return [
        path
        for path in sorted(TOOL_DIR.rglob("*.py"))
        if not (path.parent == TOOL_DIR and path.name in FACADE_FILENAMES)
    ]


def test_no_source_module_imports_a_deprecated_facade() -> None:
    """The package does not consume its own deprecated paths — anywhere under ``src/``.

    27.6 forbade this for ``core/`` and ``knowledge_graph/`` against one façade. There are
    two façades now, and the eleven internal import sites the vector move re-pointed were
    re-pointed by hand: nothing would have caught a twelfth. The sweep is the thing that
    would, and it covers function-body imports because ``_imported_modules`` walks the
    whole AST rather than only the module-level statements.
    """
    violations: list[str] = []
    for module_path in _swept_source_modules():
        name = module_path.relative_to(TOOL_DIR).as_posix()
        for imported in sorted(_imported_modules(module_path, TOOL_DIR) & FACADE_MODULES):
            violations.append(f"{name} imports the deprecated {imported}")
    assert not violations, f"src/ consumes its own deprecated path: {violations}"


def test_the_source_sweep_is_not_vacuous() -> None:
    """Guard the guard: a mistyped glob would make the sweep above trivially green."""
    swept = {path.relative_to(TOOL_DIR).as_posix() for path in _swept_source_modules()}
    assert "vector_store/vector.py" in swept, "the moved module is outside the sweep"
    assert {"core/event.py", "core/observer.py", "team/observer.py"} <= swept
    assert "event.py" not in swept and "vector.py" not in swept, "façades must be excluded"


@pytest.mark.parametrize(
    "source",
    [
        "from akgentic.tool.vector import VectorEntry",
        "import akgentic.tool.vector",
        "import akgentic.tool.vector as vector_mod",
        "from akgentic.tool import vector",
        "from . import vector",
        "def f():\n    from akgentic.tool import event\n",
    ],
)
def test_every_import_form_of_a_facade_is_reported(source: str) -> None:
    """Guard the guard: the sweep matches module paths, so the parser must produce them.

    Asking whether the sweep *would* go red is a question about this function, not about
    today's tree — a reintroduced import will not be written in the form that was just
    removed. The last three forms name the façade on the alias instead of on the ``from``
    clause, and each one was invisible to the sweep until the parser reported it.
    """
    reported = _imports_in(ast.parse(source), ("akgentic", "tool"))
    assert reported & FACADE_MODULES, reported


def test_knowledge_graph_has_no_bottom_of_file_event_import() -> None:
    """No KG module defers an event import to the bottom of the file.

    A domain module importing the envelope at the top is ordinary and allowed —
    ``kg_actor.py`` emits ``ToolStateEvent``. What is forbidden is the shape the old
    cycle used: an import pushed below the class definitions so it runs late enough
    to paper over a circular dependency. That shape is the tell, not the import.
    """
    event_modules = {"akgentic.tool.event", "akgentic.tool.core.event"}
    violations: list[str] = []
    for module_path in sorted(KG_DIR.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        seen_definition = False
        for node in tree.body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                seen_definition = True
            elif seen_definition and isinstance(node, ast.ImportFrom | ast.Import):
                names = (
                    {node.module} if isinstance(node, ast.ImportFrom) and node.module else set()
                ) | {alias.name for alias in node.names}
                if names & event_modules:
                    rel = module_path.relative_to(KG_DIR).as_posix()
                    violations.append(f"{rel}:{node.lineno}")
    assert not violations, f"bottom-of-file event import is back: {violations}"


def test_models_calls_no_model_rebuild() -> None:
    """``models.py`` no longer patches the envelope's annotations at import time."""
    tree = ast.parse((KG_DIR / "models.py").read_text(encoding="utf-8"))
    rebuilds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "model_rebuild"
    ]
    assert not rebuilds, "models.py calls model_rebuild — the cycle workaround is back"


def test_building_a_tool_state_event_does_not_import_knowledge_graph() -> None:
    """The behavioural proof: the envelope no longer needs the KG package at all.

    Under any of the three deleted workarounds this fails — the envelope either
    could not be built without the rebuild, or dragged the domain package in with it.
    """
    from akgentic.core.utils.serializer import SerializableBaseModel

    kg_modules = [key for key in sys.modules if "knowledge_graph" in key]
    saved = {key: sys.modules.pop(key) for key in kg_modules}
    try:
        from akgentic.tool.core.event import ToolStateEvent

        class _TrivialPayload(SerializableBaseModel):
            note: str

        event = ToolStateEvent(tool_id="#Trivial", seq=1, payload=_TrivialPayload(note="hi"))

        assert event.payload.note == "hi"  # type: ignore[attr-defined]
        assert "akgentic.tool.knowledge_graph" not in sys.modules
    finally:
        sys.modules.update(saved)
