"""A capability module depends on the spine, and on no other capability.

The property ADR-053 Decision 1 exists to obtain, and the one that makes "a
capability's code is deletable" true: ``workspace/read/`` names the package's
shared machinery and nothing else, so removing ``card/`` — or any sibling
capability — leaves it standing.

**Why this is static, and how far it goes.** ``workspace/__init__.py`` eagerly
re-exports 149 names from every module in the package, so after *any* import of
anything under ``akgentic.tool.workspace`` — in a fresh interpreter as surely as
in a shared test session — ``sys.modules`` holds the journal, the exec machinery
and the RAG pipeline. A runtime absence probe would therefore assert something
the package facade contradicts on its third line, and a subprocess buys nothing a
same-process probe does not. **No subprocess harness is built here.** What is
checked instead is what the code *declares*, which is the thing a capability
module can actually control and the thing that makes it deletable.

Making the facade lazy is the only way the runtime claim could ever be literally
true. It is a public-API change with a cross-package blast radius and it would
cost mypy the type of every re-exported name; it is recorded, not taken.

Three rules the graph follows, each of which a weaker version of this guard has
to get wrong to pass:

- **A ``from a.b.c import X`` is a dependency on the package ``a.b``, not only on
  ``a.b.c``**, because importing a submodule executes its parents' ``__init__``
  first. A graph that treated ``card.params`` as a leaf would answer "clean" for a
  capability that drags in the whole card package.
- **The package root is not attributed**, and that is a stated scope rather than a
  convenience: *every* module of the package sits under
  ``akgentic.tool.workspace``, so attributing it would put the eager facade — and
  therefore everything — in every closure, and the guard would assert nothing at
  all. Its eagerness is the limit recorded above.
- **The closure is compared against an allow-list, never a deny-list.** A
  deny-list passes silently for the module nobody thought to forbid.

**The runtime graph excludes ``if TYPE_CHECKING:`` imports, deliberately**, and
:class:`TestNoCapabilityNamesAnother` closes the gap that opens. Such an import
never executes, closes no cycle and costs nothing at bind — which is exactly why
``documents/splitter.py`` and ``actor/documents.py`` use one to name
``WorkspaceRagIndex`` without taking on ``card/``. Counting it would condemn the
package's own recommended pattern. So the runtime closure ignores it, and a
second, narrower rule then forbids a capability module from naming another
capability in **any** import, annotation-only ones included.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import akgentic.tool.workspace as ws

PACKAGE = "akgentic.tool.workspace"
"""The package whose internal structure is under analysis."""

PACKAGE_ROOT = Path(ws.__file__).parent
"""Read from the installed package itself, never assembled from this file's path."""

SPINE = frozenset(
    {
        # ``errors.py`` is not here: ``RetriableError`` lives at
        # ``akgentic.tool.errors``, outside this package, so it is never a node in
        # this graph at all. Only the package's own modules are.
        f"{PACKAGE}.event",
        f"{PACKAGE}.models",
        f"{PACKAGE}.readers",
        f"{PACKAGE}.workspace",
        # Reached transitively: ``models`` imports ``documents.models`` for the
        # extraction constants, which attributes the ``documents`` package, whose
        # ``__init__`` pulls in its context and splitter modules. It deliberately
        # does **not** import ``documents/worker.py`` — the one module there that
        # imports ``card.params`` — and says so in its own docstring.
        f"{PACKAGE}.documents",
        f"{PACKAGE}.documents.context",
        f"{PACKAGE}.documents.models",
        f"{PACKAGE}.documents.splitter",
    }
)
"""The shared machinery every capability is allowed to stand on.

``readers.py`` is in here rather than inside ``read/`` even though the read
closures are its heaviest consumer: ``card/rag.py``, ``actor/documents.py``,
``documents/worker.py``, ``card/params.py`` and ``models.py`` all use it too, and
``akgentic-agent`` imports ``MediaContent`` from its deep path. It is shared
machinery and it stays in the spine.
"""

CAPABILITY_CLOSURES: dict[str, frozenset[str]] = {
    "read": SPINE | {f"{PACKAGE}.read", f"{PACKAGE}.read.params"},
}
"""One row per capability module. The four stories after 55-2 add four more rows.

The value is the complete allow-list for that capability's transitive closure:
its own modules, plus the spine modules it genuinely needs.
"""

MINIMUM_MODULES_PARSED = 20
"""Below this the sweep is looking at the wrong directory, not at a clean package.

Calibrated for **this package**, which holds 27 modules. Story 55-1's sibling
sweep uses 40 because it walks the whole of ``src/akgentic/tool/``; carrying that
number over to a 27-module root would fail a correct sweep, which is the opposite
of what a sentinel is for.
"""


def _module_name(path: Path) -> str:
    """The dotted name of the package module at *path*."""
    parts = path.relative_to(PACKAGE_ROOT).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts))


def _package_modules() -> dict[str, Path]:
    """Every module of the package, by dotted name."""
    return {_module_name(path): path for path in sorted(PACKAGE_ROOT.rglob("*.py"))}


def _containing_package(module: str, path: Path) -> str:
    """The package a relative import inside *module* resolves against."""
    return module if path.name == "__init__.py" else module.rsplit(".", 1)[0]


def _targets(node: ast.Import | ast.ImportFrom, home: str) -> list[str]:
    """Every dotted name *node* imports, relative levels resolved against *home*.

    *home* is the importing module's own package. This package uses absolute
    imports throughout, so the relative branch is unexercised today; it is
    resolved properly anyway, because a graph that silently mis-resolved one
    would under-report edges and the guard would pass for the wrong reason.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        base = home
        for _ in range(node.level - 1):
            base = base.rsplit(".", 1)[0]
        prefix = f"{base}.{node.module}" if node.module else base
    else:
        prefix = node.module or ""
    # ``from a.b import c`` where ``c`` is itself a module is an import of ``a.b.c``.
    return [prefix, *(f"{prefix}.{alias.name}" for alias in node.names)]


def _runtime_imports(source: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Every import that actually executes — ``if TYPE_CHECKING:`` blocks excluded."""
    skipped: set[int] = set()
    for node in ast.walk(source):
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for guarded in node.body:
                for inner in ast.walk(guarded):
                    skipped.add(id(inner))
    return [
        node
        for node in ast.walk(source)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in skipped
    ]


def _all_imports(source: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Every import the module names, annotation-only ones included."""
    return [node for node in ast.walk(source) if isinstance(node, (ast.Import, ast.ImportFrom))]


def _is_type_checking(test: ast.expr) -> bool:
    """Whether an ``if`` guards a type-checking-only block."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _edges(module: str, path: Path, modules: set[str], annotations_too: bool) -> set[str]:
    """The package modules *module* depends on, parents of a submodule included."""
    source = ast.parse(path.read_text(encoding="utf-8"))
    nodes = _all_imports(source) if annotations_too else _runtime_imports(source)
    home = _containing_package(module, path)
    found: set[str] = set()
    for node in nodes:
        for target in _targets(node, home):
            if not target.startswith(f"{PACKAGE}."):
                continue
            # Every prefix below the package root, because importing a submodule
            # executes each parent ``__init__`` on the way down.
            parts = target[len(PACKAGE) + 1 :].split(".")
            for depth in range(1, len(parts) + 1):
                candidate = ".".join((PACKAGE, *parts[:depth]))
                if candidate in modules and candidate != module:
                    found.add(candidate)
    return found


def _closure(roots: set[str], modules: dict[str, Path], annotations_too: bool = False) -> set[str]:
    """Every package module reachable from *roots*, transitively."""
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(_edges(current, modules[current], set(modules), annotations_too))
    return seen


def _modules_under(capability: str, modules: dict[str, Path]) -> set[str]:
    """The capability's own modules — ``read`` and everything below it."""
    prefix = f"{PACKAGE}.{capability}"
    return {name for name in modules if name == prefix or name.startswith(f"{prefix}.")}


@pytest.fixture(scope="module")
def modules() -> dict[str, Path]:
    """Every module of the package, by dotted name, from the installed source."""
    return _package_modules()


class TestTheSweepLooksAtTheRightThing:
    """Non-vacuity. A guard that parsed nothing would pass every rule below."""

    def test_it_parses_the_whole_package(self, modules: dict[str, Path]) -> None:
        """Too few modules means the root is wrong, not that the package is small."""
        assert len(modules) > MINIMUM_MODULES_PARSED, (
            f"the sweep found {len(modules)} modules under {PACKAGE_ROOT} — the root is wrong"
        )

    def test_every_capability_it_claims_to_cover_exists(self, modules: dict[str, Path]) -> None:
        """A row naming a module that is gone would be checked against nothing."""
        for capability in CAPABILITY_CLOSURES:
            assert _modules_under(capability, modules), f"{capability}/ has no modules"

    def test_the_spine_it_allows_exists(self, modules: dict[str, Path]) -> None:
        """An allow-list entry naming no module allows nothing and hides a typo."""
        assert SPINE <= set(modules)

    def test_the_closure_attributes_a_package_nobody_imports_by_name(
        self, modules: dict[str, Path]
    ) -> None:
        """The parent-attribution rule, asserted rather than left to a mutation.

        No module anywhere imports ``akgentic.tool.workspace.documents`` by name —
        every reference is to ``documents.models``, ``documents.context`` or
        ``documents.splitter``. The package is in ``read/``'s closure **only**
        because importing one of those executes its ``__init__`` first, which is
        the whole of the rule. A graph that recorded leaves alone would leave it
        out, and would then answer "clean" for a capability that dragged in a
        whole package through one of its submodules.

        **Asserted once, on ``read``, and not per capability**, because it is a
        property of :func:`_edges` — shared by every row — and not of any one
        capability. Parametrised, it would assert of *each* later capability that
        it reaches ``documents``, which is a fact about ``read/`` importing
        ``models``. The first capability that legitimately never reaches it would
        redden a guard that is working correctly, for a reason unconnected to
        what the guard tests.
        """
        closure = _closure(_modules_under("read", modules), modules)

        assert f"{PACKAGE}.documents" in closure, (
            "the graph is recording leaves without their parent packages — "
            "importing a submodule executes every parent __init__ on the way down"
        )


@pytest.mark.parametrize("capability", sorted(CAPABILITY_CLOSURES))
class TestACapabilityDependsOnTheSpineAndNothingElse:
    """One row per capability module; the four later stories add four more."""

    def test_the_closure_is_inside_the_allow_list(
        self, capability: str, modules: dict[str, Path]
    ) -> None:
        """An allow-list, so a module nobody thought to forbid is still caught."""
        closure = _closure(_modules_under(capability, modules), modules)
        outside = closure - CAPABILITY_CLOSURES[capability]

        assert outside == set(), (
            f"{capability}/ reaches {sorted(outside)} — a capability module may import "
            f"the spine and its own modules, and nothing else"
        )

    def test_the_closure_is_not_empty_and_reaches_the_spine(
        self, capability: str, modules: dict[str, Path]
    ) -> None:
        """The containment above is satisfied by an empty set; this is why it is not.

        Spelled as the specific spine modules the capability actually imports,
        not as "non-empty": a closure holding only the capability's own modules
        would satisfy a mere emptiness check while proving the graph followed no
        edge at all.
        """
        closure = _closure(_modules_under(capability, modules), modules)

        assert closure, f"{capability}/ has an empty closure — no edge was followed"
        assert {f"{PACKAGE}.workspace", f"{PACKAGE}.models", f"{PACKAGE}.readers"} <= closure

    def test_it_names_no_other_capability_even_in_an_annotation(
        self, capability: str, modules: dict[str, Path]
    ) -> None:
        """The narrow rule that closes the ``TYPE_CHECKING`` gap.

        The runtime closure above ignores annotation-only imports, because such an
        import executes nothing and the package recommends exactly that pattern
        elsewhere. Here the question is different and the answer is the same
        either way: a capability module must not *name* another capability at all,
        because a reader following the reference lands in ``card/`` and because
        deleting ``card/`` would break this module's type check.

        **Direct edges only, and deliberately so** — this is not the transitive
        closure with a wider filter. Following annotation edges transitively runs
        straight into ``documents/splitter.py``'s ``TYPE_CHECKING`` import of
        ``card.params``, which is the package's own recommended cycle-breaker and
        is nothing to do with the capability under test. The transitive question
        is the runtime closure's, and it is answered above.
        """
        own = _modules_under(capability, modules)
        named = {
            target
            for name in own
            for target in _edges(name, modules[name], set(modules), annotations_too=True)
        }
        outside = named - CAPABILITY_CLOSURES[capability]

        assert outside == set(), (
            f"{capability}/ names {sorted(outside)} — including under TYPE_CHECKING, "
            f"a capability module may name only the spine and its own modules"
        )
