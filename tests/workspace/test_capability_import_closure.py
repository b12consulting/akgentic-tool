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
:meth:`TestACapabilityDependsOnTheSpineAndNothingElse.test_it_names_no_other_capability_even_in_an_annotation`
closes the gap that opens. Such an import never executes, closes no cycle and
costs nothing at bind — which is exactly why ``rag/splitter.py`` and
``actor/documents.py`` use one to name each other's types without taking the
other on. Counting it would condemn the package's own recommended pattern. So the
runtime closure ignores it, and a second, narrower rule then forbids a capability
module from naming another capability in **any** import, annotation-only ones
included.

**A function-level import is a runtime import here, and that is not a gap.** It
executes when the method runs, so :func:`_runtime_imports` counts it, and it
should: ``IndexWorker._report``'s import of ``WorkspaceActor`` put seven modules
into ``rag/``'s closure — three of them capabilities — for an argument
``proxy_tell`` ignores. The guard is what found it.
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
        # extraction constants, which attributes the ``documents`` package. Those
        # two, and nothing else under ``documents/``: the extraction cache is
        # genuinely shared by the read and retrieval capabilities, so it stays
        # shared (ADR-053 Decision 1), while ``documents/store.py`` is reached by
        # neither capability's closure.
        f"{PACKAGE}.documents",
        f"{PACKAGE}.documents.models",
    }
)
"""The shared machinery every capability is allowed to stand on.

``readers.py`` is in here rather than inside ``read/`` even though the read
closures are its heaviest consumer: ``rag/__init__.py``, ``actor/documents.py``,
``rag/worker.py`` and ``models.py`` all use it too, and ``akgentic-agent`` imports
``MediaContent`` from its deep path. It is shared machinery and it stays in the
spine.

**It shrank by two when the retrieval capability became a module**, and that is
the structural observable of story 55-6 rather than a tidy-up.
``documents.context`` and ``documents.splitter`` were retrieval machinery sitting
in the allow-list *every* capability stands on — so ``journal/``, which has
nothing whatever to do with retrieval, was permitted to reach both. They were
there because ``workspace/models.py`` takes two cap constants from
``documents/models.py``, and importing that executed ``documents/__init__.py``,
which re-exported the splitter and the context state. Moving those two modules
into ``rag/`` and dropping the re-exports broke the chain. The claim "a capability
the card does not enable costs nothing" was, at the static level, partly false
until then.
"""

CAPABILITY_CLOSURES: dict[str, frozenset[str]] = {
    "read": SPINE | {f"{PACKAGE}.read", f"{PACKAGE}.read.params"},
    "write": SPINE
    | {
        f"{PACKAGE}.write",
        f"{PACKAGE}.write.gate",
        f"{PACKAGE}.write.params",
        # ``EditItem`` in the closures' signatures, and twelve more names in the
        # gate. **On this row rather than in ``SPINE``, and the distinction is
        # the one that decides where a later capability's shared module goes.**
        # ``readers.py`` earns ``SPINE`` on the evidence: seven consumers in
        # ``src/`` spread over four capabilities *and the spine itself*
        # (``models.py`` imports it). ``edit.py`` has exactly two — this package
        # and the facade's re-export — so it is not shared machinery at all; it
        # is the write capability's own machinery, left in place only because
        # moving it was out of this story's scope. Promoting it would hand five
        # capabilities a permission none of them needs, and would make ``SPINE``
        # — "the shared machinery every capability is allowed to stand on" —
        # say something untrue. A later capability that genuinely needs it
        # states it on its own row; promote it only once a second capability
        # outside ``write/`` does.
        f"{PACKAGE}.edit",
        # ``Identity`` is constructed in ``_gated`` — the commit's author.
        # Runtime, so it is in both the closure and the direct-edge rule.
        # Becomes 55-4's ``journal/``.
        f"{PACKAGE}.journal",
        # ``mutation_busy`` composes the busy refusal. Runtime, so both rules.
        # Becomes 55-7's ``exec/``.
        f"{PACKAGE}.execution",
        # ``LockBackend`` annotates ``_lock_backend`` and nothing else. Becomes
        # 55-7's. It is on this row because the direct-edge rule below counts a
        # ``TYPE_CHECKING`` import too — moving it under one would take it out of
        # the runtime closure and buy no row.
        f"{PACKAGE}.lock",
        # ``WorkspaceActor`` annotates ``_workspace_tell`` and nothing else, and
        # is already imported under ``TYPE_CHECKING`` — so it is *not* in the
        # runtime closure, and is here only for the direct-edge rule. Becomes
        # 55-7's.
        f"{PACKAGE}.actor",
    },
    # **The first row with no non-spine entry at all**, and that is the finding
    # rather than an omission. ``journal/`` imports exactly four names from
    # ``models`` and nothing else in the package: no ``akgentic.tool.core``
    # module, no sibling capability, under ``TYPE_CHECKING`` or otherwise. Where
    # ``write/`` needs five, this needs none.
    "journal": SPINE | {f"{PACKAGE}.journal"},
    "rag": SPINE
    | {
        f"{PACKAGE}.rag",
        f"{PACKAGE}.rag.context",
        f"{PACKAGE}.rag.params",
        f"{PACKAGE}.rag.splitter",
        f"{PACKAGE}.rag.worker",
        # ``RagFactories._rag_reader`` resolves the card's extraction
        # configuration, which is nested inside ``WorkspaceRead.document_reader``
        # — so the capability that extracts depends on the capability that reads.
        #
        # **This is a capability naming another capability, admitted deliberately
        # and different in kind from ``write/``'s ``journal`` entry.** ``read/`` is
        # the *always-available* capability, present in every bind, so depending
        # on it is depending on machinery that is there anyway; ``write/``'s
        # dependency was on a **default-off** capability, which is why ``Identity``
        # is moving to the spine and why nothing analogous is owed here. Moving
        # ``_rag_reader`` into ``card/__init__.py`` to shorten this row would be
        # reorganising to satisfy the map, and would grow the longest module in
        # the package.
        f"{PACKAGE}.read",
        f"{PACKAGE}.read.params",
        # ``WorkspaceActor`` annotates ``RagFactories``'s two proxy slots and is
        # the type ``IndexWorker._report`` casts its reply proxy to. Both are
        # ``TYPE_CHECKING`` imports, so this is **not** in the runtime closure and
        # is on the row only for the direct-edge rule — exactly ``write/``'s
        # ``actor`` entry.
        #
        # It was briefly a *runtime* entry, and the measurement is worth
        # recording: ``IndexWorker._report`` imported ``WorkspaceActor`` inside
        # the method purely to pass it to ``proxy_tell``, whose second argument
        # core ignores. That one executed import put ``actor``, ``actor.documents``,
        # ``actor.execution``, ``documents.store``, ``execution``, ``journal`` and
        # ``lock`` into this closure — seven modules, none of them a real
        # dependency of retrieval, and three of them capabilities. Naming the
        # actor under ``TYPE_CHECKING`` instead removed all seven.
        f"{PACKAGE}.actor",
    },
}
"""One row per capability module. Story 55-7 adds the last one.

The value is the complete allow-list for that capability's transitive closure:
its own modules, plus the spine modules it genuinely needs.

**The two assertions below read a row differently**, and ``write/``'s four
non-spine entries show why. The runtime closure excludes annotation-only imports,
so ``lock`` and ``actor`` are not in it; the direct-edge rule includes them, so
they must still be on the row. A ``TYPE_CHECKING`` import therefore keeps a module
out of the *runtime* closure and never off the row.
"""

CAPABILITY_SPINE_REACH: dict[str, frozenset[str]] = {
    "read": frozenset({f"{PACKAGE}.workspace", f"{PACKAGE}.models", f"{PACKAGE}.readers"}),
    "write": frozenset({f"{PACKAGE}.workspace", f"{PACKAGE}.models", f"{PACKAGE}.readers"}),
    # **Not the same triple, and that is the whole reason this dict exists.**
    # ``journal/`` never reaches ``workspace.py``: it imports only ``models``,
    # which imports ``documents.models`` and — inside a function — ``readers``,
    # and none of those imports ``workspace``. ``readers`` is the interesting
    # entry here rather than the obvious one: it is reached *only* through that
    # in-function import, so a graph that followed no transitive edge would miss
    # it while still satisfying a mere "non-empty" check.
    "journal": frozenset({f"{PACKAGE}.models", f"{PACKAGE}.readers"}),
    # The widest of the four, and stated in full rather than trimmed to match the
    # others: ``rag/`` reaches ``workspace`` (``meta_dir_for``, ``get_workspace``),
    # ``readers`` (the extraction configuration), ``documents.models`` (the
    # collection name, the chunk record and the id rule) and, through ``read/``,
    # ``models``. Listing all five is what makes this row report a graph that
    # silently narrowed.
    "rag": frozenset(
        {
            f"{PACKAGE}.documents",
            f"{PACKAGE}.documents.models",
            f"{PACKAGE}.models",
            f"{PACKAGE}.readers",
            f"{PACKAGE}.workspace",
        }
    ),
}
"""The spine modules each capability actually reaches — per row, never one literal.

**Split out of the non-vacuity assertion below, which carried the triple inline
and would have reddened the first correct row that differed.** The value had been
read off ``read/``'s closure and then asserted of every capability, so it encoded
one capability's dependencies as a property of all of them; ``journal/`` is the
row that finds it, by not reaching ``workspace``.

This is the second instance of that shape in this module — the first was
:meth:`TestTheSweepLooksAtTheRightThing.test_the_closure_attributes_a_package_nobody_imports_by_name`,
which was **de-parametrised** instead. The two fixes differ deliberately, and the
distinction is which kind of claim was being made. That one asserts a property of
:func:`_edges` that every row shares, so asserting it once is the honest form.
*Which spine modules a capability reaches* genuinely differs per capability, so
here the claim is real for every row and only the expected value is per-row. It
becomes data.

**Its keys must match :data:`CAPABILITY_CLOSURES`'s exactly**, and a sentinel
below asserts it. Two dicts one story can update by half is the new way this
guard can rot, and a row present in the first and missing from the second would
assert nothing at all about that capability.
"""

MINIMUM_MODULES_PARSED = 20
"""Below this the sweep is looking at the wrong directory, not at a clean package.

Calibrated for **this package**, which holds 29 modules. Story 55-1's sibling
sweep uses 40 because it walks the whole of ``src/akgentic/tool/``; carrying that
number over to a root of this size would fail a correct sweep, which is the
opposite of what a sentinel is for.

**The floor does not move when the package grows.** It is deliberately below the
smallest correct sweep, not a number chosen to pass: story 55-6 took the count
from 27 to 29 and this stayed at 20, which is the whole point of a floor.
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

    def test_no_spine_entry_lies_under_a_capability_directory(
        self, modules: dict[str, Path]
    ) -> None:
        """The guard for the rot story 55-6 removed, rather than a restatement of it.

        ``documents.context`` and ``documents.splitter`` sat in :data:`SPINE` while
        they were retrieval machinery, so every capability — ``journal/`` included
        — was allowed to reach the splitter and the retrieval context state. It was
        not a typo in the row; it was the shape of the package, and it made "a
        capability the card does not enable costs nothing" partly false at the
        static level.

        This is green today. What it is for is the *next* time a row will not go
        green: promoting a capability module into the spine to satisfy one entry is
        the cheap fix, it is exactly what the epic's traps forbid, and it would
        otherwise pass every assertion in this module while meaning the opposite of
        what they claim.
        """
        offenders = {
            entry
            for entry in SPINE
            for capability in CAPABILITY_CLOSURES
            if entry == f"{PACKAGE}.{capability}" or entry.startswith(f"{PACKAGE}.{capability}.")
        }

        assert offenders == set(), (
            f"{sorted(offenders)} are capability modules in SPINE — the allow-list "
            f"every capability stands on must not carry any one capability's code"
        )

    def test_the_two_capability_tables_describe_the_same_capabilities(self) -> None:
        """Key parity, because two tables are two things one story can update by half.

        A capability present in :data:`CAPABILITY_CLOSURES` and missing from
        :data:`CAPABILITY_SPINE_REACH` would have its containment checked and its
        non-vacuity not checked at all — the assertion that stops an empty
        closure from passing would be the one that went missing. The subscript in
        that spec is deliberately direct rather than a ``.get(…, frozenset())``:
        an empty ``frozenset`` is a subset of everything, so the defensive
        spelling would pass silently and this sentinel is what reports the rot
        instead.
        """
        assert set(CAPABILITY_CLOSURES) == set(CAPABILITY_SPINE_REACH)

    def test_the_closure_attributes_a_package_nobody_imports_by_name(
        self, modules: dict[str, Path]
    ) -> None:
        """The parent-attribution rule, asserted rather than left to a mutation.

        No module anywhere imports ``akgentic.tool.workspace.documents`` by name —
        every reference is to ``documents.models`` or ``documents.store``. The
        package is in ``read/``'s closure **only** because importing one of those
        executes its ``__init__`` first, which is the whole of the rule. A graph
        that recorded leaves alone would leave it out, and would then answer
        "clean" for a capability that dragged in a whole package through one of
        its submodules.

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
    """One row per capability module; story 55-7 adds the last one."""

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

        **The expected value is per-row data**
        (:data:`CAPABILITY_SPINE_REACH`), not one literal. It was one literal,
        read off ``read/``'s closure, and ``journal/`` — which reaches ``models``
        and ``readers`` but never ``workspace`` — is a correct row that would
        have reddened it.
        """
        closure = _closure(_modules_under(capability, modules), modules)

        assert closure, f"{capability}/ has an empty closure — no edge was followed"
        assert CAPABILITY_SPINE_REACH[capability] <= closure

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
        straight into ``actor/documents.py``'s ``TYPE_CHECKING`` import of
        ``rag.params``, which is the package's own recommended cycle-breaker and is
        nothing to do with the capability under test. The transitive question is
        the runtime closure's, and it is answered above.
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
