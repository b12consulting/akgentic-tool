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
costs nothing at bind — which is exactly why ``rag/__init__.py`` and
``execution/card.py`` both use one to name ``WorkspaceActor`` without taking the
actor on. Counting it would condemn the package's own recommended pattern. So the
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
        # The one ``flock`` idiom, shared by four capabilities' four lock families
        # since story 57-2 and importing nothing but the standard library. It earns
        # ``SPINE`` on the same evidence ``readers.py`` does — consumers spread
        # across several capabilities rather than one — and it is the entry whose
        # *absence* is measurable: removing it from this set reddens both
        # assertions below for ``write``, ``journal`` and ``rag``, and neither one
        # for ``read`` or ``execution``. Adding it reddens nothing, because an
        # extra permitted entry never reddens a subset test.
        f"{PACKAGE}.locks",
        f"{PACKAGE}.models",
        f"{PACKAGE}.readers",
        f"{PACKAGE}.workspace",
    }
)
"""The shared machinery every capability is allowed to stand on.

``readers.py`` is in here rather than inside ``read/`` even though the read
closures are its heaviest consumer: ``rag/__init__.py``, ``rag/actor.py``,
``rag/worker.py`` and ``models.py`` all use it too, and ``akgentic-agent`` imports
``MediaContent`` from its deep path. It is shared machinery and it stays in the
spine.

**It shrank by two more in story 55-8, and those two are the same shape as
55-6's.** ``documents`` and ``documents.models`` were here because
``workspace/models.py`` imported two cap constants from the second to default two
``WorkspaceConfig`` fields — so *every* capability, ``journal/`` included, was
allowed to reach the documents package through one spine module that named it for
an unrelated reason. Those fields went card-side with the extraction cache, the
import went with them, and the two entries are now stated on the rows that
genuinely reach them: ``rag/``, which imports the records, and ``write/``, which
names the cache the stale-mark calls.

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
        # ``GitJournal`` types ``_journal`` under ``TYPE_CHECKING`` and nothing
        # else — **annotation-only since story 55-7 moved ``Identity`` into the
        # spine**, which is what took the journal out of this capability's
        # *runtime* closure. It stays on the row because the direct-edge rule
        # below counts an annotation too, so removing it reddens exactly one of
        # the two assertions.
        #
        # There is no ``execution`` entry any more, and its absence is the
        # structural observable of story 55-7 rather than a tidy-up: the busy
        # refusal ``_busy_refusal`` composes is ``lock.mutation_busy`` now, so
        # nothing under ``write/`` names the exec capability at all. An allow-list
        # cannot report that — an extra *permitted* entry never reddens a subset
        # test — so :class:`TestTheDeletabilityClaimIsObservable` asserts it
        # directly.
        f"{PACKAGE}.journal",
        # ``LockBackend`` annotates ``_lock_backend`` and ``mutation_busy`` is
        # called on every refused mutation, so this is a **runtime** entry now
        # rather than the annotation-only one it was. The hold is the tree's
        # state and is read whether or not exec is enabled, which is the whole
        # argument for ``lock.py`` staying in the spine while ``execution/``
        # became a capability.
        f"{PACKAGE}.lock",
        # ``DocumentCache`` annotates ``_document_cache`` under ``TYPE_CHECKING``
        # and nothing else — the stale-mark an accepted mutation applies is a
        # plain method call, which is no import at all — so neither entry is in
        # the runtime closure and both are here for the direct-edge rule. The
        # package entry rides on the submodule: importing
        # ``documents.cache`` executes ``documents/__init__`` on the way down.
        #
        # **There is no ``actor`` entry any more, and that is story 55-8's
        # structural observable.** ``_mark_stale`` was the last thing under
        # ``write/`` that named the actor — it told ``mark_paths_stale`` over the
        # tell proxy — and the actor is only created by a card that dispatches,
        # which a write card does not. Removing the entry reddens nothing, which
        # is the point: an extra *permitted* entry never reddens a subset test, so
        # :class:`TestTheDeletabilityClaimIsObservable` asserts the absence
        # directly.
        f"{PACKAGE}.documents",
        f"{PACKAGE}.documents.cache",
    },
    # **The first row with no non-spine entry at all**, and that is the finding
    # rather than an omission. ``journal/`` imports exactly four names from
    # ``models`` and nothing else in the package: no ``akgentic.tool.core``
    # module, no sibling capability, under ``TYPE_CHECKING`` or otherwise. Where
    # ``write/`` needs five, this needs none.
    "journal": SPINE | {f"{PACKAGE}.journal"},
    "rag": SPINE
    | {
        # The records themselves, and the package they live in. **On this row
        # since story 55-8 rather than in ``SPINE``**: the retrieval capability is
        # the one that genuinely reads a ``RagFile`` and mints a chunk id, and the
        # spine stopped naming the documents package at all when the two document
        # caps left ``WorkspaceConfig``.
        f"{PACKAGE}.documents",
        # Measured, story 55-9: ``rag/actor.py`` imports ``DocumentCache`` and
        # ``DocumentEntry`` at module scope, so both are edge targets from a module
        # under this capability. Removing either **reddens both**
        # ``test_the_closure_is_inside_the_allow_list`` and
        # ``test_it_names_no_other_capability_even_in_an_annotation``, run one at a
        # time. ``documents.cache`` was not predicted by the story at all — it
        # arrived card-side in 55-8 and the pipeline reaches the cache through it.
        f"{PACKAGE}.documents.cache",
        f"{PACKAGE}.documents.models",
        f"{PACKAGE}.documents.store",
        f"{PACKAGE}.rag",
        # Measured, story 55-9: the mixin's own module. It is a **closure root**,
        # so removing it reddens ``test_the_closure_is_inside_the_allow_list``;
        # it reddens the direct-edge spec **not at all**, because no sibling under
        # ``rag/`` imports it — ``actor/__init__.py`` does, and the actor is the
        # assembly point rather than a capability. That asymmetry is the
        # allow-list rule read off this row rather than inherited from another.
        f"{PACKAGE}.rag.actor",
        f"{PACKAGE}.rag.context",
        f"{PACKAGE}.rag.params",
        # Measured, story 57-1, one mutation at a time: the search itself — both
        # legs, the fusion and the render — as its own module. Removing it reddens
        # **both** ``test_the_closure_is_inside_the_allow_list`` (it is one of this
        # capability's own modules, so it is a closure root) **and**
        # ``test_it_names_no_other_capability_even_in_an_annotation``
        # (``rag/__init__.py`` imports it at module scope). That is the asymmetry
        # with ``rag.actor`` one row up, which reddens only the first because no
        # sibling under ``rag/`` imports it — the assembly point does.
        f"{PACKAGE}.rag.search",
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
        # core ignores. That one executed import put ``actor``, the mixin module
        # that was then ``actor.documents``, ``actor.execution``,
        # ``documents.store``, ``execution``, ``journal`` and
        # ``lock`` into this closure — seven modules, none of them a real
        # dependency of retrieval, and three of them capabilities. Naming the
        # actor under ``TYPE_CHECKING`` instead removed all seven.
        f"{PACKAGE}.actor",
    },
    "execution": SPINE
    | {
        f"{PACKAGE}.execution",
        f"{PACKAGE}.execution.actor",
        f"{PACKAGE}.execution.card",
        f"{PACKAGE}.execution.params",
        # ``LEASE_GRACE_S``, ``LockTicket`` and ``LockBackend`` in the mixin, and
        # ``_BUSY_PREFIX`` in the package root's ``lock_unavailable``. **Runtime,
        # and not a defect to remove**: ``lock.py`` is the spine's vocabulary of
        # the hold, and it is where this capability's own lease grace, run-id mint
        # and two busy refusals now live — precisely so that the spine never has
        # to import back into ``execution/``.
        #
        # **Not predicted by the story that added this row; measured.** The story
        # expected the six entries around it and no seventh. Transcribing its list
        # would have produced a row that was wrong in the safe direction — too
        # narrow — and reddened a correct capability.
        f"{PACKAGE}.lock",
        # ``WorkspaceActor`` annotates ``_bound``'s signature and the
        # ``_workspace_proxy`` declaration in ``execution/card.py``, both under
        # ``TYPE_CHECKING``, so this is **not** in the runtime closure and is on
        # the row only for the direct-edge rule. Exactly ``write/``'s ``actor``
        # entry, and exactly the edge story 55-3 removed from ``write/``.
        f"{PACKAGE}.actor",
        # ``GitJournal`` types ``_journal`` and ``configure_journal``'s parameter
        # in the mixin, both under ``TYPE_CHECKING``. **Stated as a known residual
        # rather than removed**: an exec run's discovered commit is a real
        # dependency of this capability on the journal capability, and the epic's
        # deletability guard is scoped to ``write/``.
        #
        # Story 57-3 added the tell and predicted this entry would stay; the
        # prediction was checked by deleting the entry and watching the
        # direct-edge assertion redden with
        # ``execution/ names ['akgentic.tool.workspace.journal']``. The actor's
        # own **runtime** import of the journal went in the same story and moved
        # nothing here, because the actor is an assembly point rather than a
        # capability and reaches these rows only under ``TYPE_CHECKING``.
        # Closing it means a journal Protocol at the spine, which is one ADR
        # decision and not a story's to take.
        f"{PACKAGE}.journal",
    },
}
"""One row per capability module — all five of them since story 55-7.

The value is the complete allow-list for that capability's transitive closure:
its own modules, plus the spine modules it genuinely needs.

**The two assertions below read a row differently**, and ``write/``'s four
non-spine entries show why. The runtime closure excludes annotation-only imports,
so ``lock`` and ``actor`` are not in it; the direct-edge rule includes them, so
they must still be on the row. A ``TYPE_CHECKING`` import therefore keeps a module
out of the *runtime* closure and never off the row.
"""

CAPABILITY_SPINE_REACH: dict[str, frozenset[str]] = {
    # **No ``locks`` entry, and that is measured rather than assumed.** ``read/``
    # takes no ``flock`` at all, so adding it here goes red immediately — this is
    # an ``EXPECTED <= closure`` assertion, which reports a row claiming a reach
    # the capability does not have.
    "read": frozenset({f"{PACKAGE}.workspace", f"{PACKAGE}.models", f"{PACKAGE}.readers"}),
    # ``locks`` joins the three here: ``CardGate._hold`` calls it on every gated
    # mutation. Removing this entry reddens nothing (a subset assertion), which is
    # why the closure row above carries the observable instead.
    "write": frozenset(
        {
            f"{PACKAGE}.locks",
            f"{PACKAGE}.models",
            f"{PACKAGE}.readers",
            f"{PACKAGE}.workspace",
        }
    ),
    # **Not the same triple, and that is the whole reason this dict exists.**
    # ``journal/`` never reaches ``workspace.py``: it imports only ``models``,
    # which imports ``documents.models`` and — inside a function — ``readers``,
    # and none of those imports ``workspace``. ``readers`` is the interesting
    # entry here rather than the obvious one: it is reached *only* through that
    # in-function import, so a graph that followed no transitive edge would miss
    # it while still satisfying a mere "non-empty" check.
    "journal": frozenset({f"{PACKAGE}.locks", f"{PACKAGE}.models", f"{PACKAGE}.readers"}),
    # The widest of the four, and stated in full rather than trimmed to match the
    # others: ``rag/`` reaches ``workspace`` (``meta_dir_for``, ``get_workspace``),
    # ``readers`` (the extraction configuration), ``documents.models`` (the
    # collection name, the chunk record and the id rule) and, through ``read/``,
    # ``models``.
    #
    # **It grew by two in story 55-9, and the row is widened rather than left
    # generous.** ``rag/actor.py`` brought ``documents.store`` and
    # ``documents.cache`` with it, and this table exists to report a graph that
    # silently narrows: listing every module the row actually reaches is what
    # makes a later narrowing visible, where a shorter row would absorb it.
    "rag": frozenset(
        {
            f"{PACKAGE}.documents",
            f"{PACKAGE}.documents.cache",
            f"{PACKAGE}.documents.models",
            f"{PACKAGE}.documents.store",
            f"{PACKAGE}.locks",
            f"{PACKAGE}.models",
            f"{PACKAGE}.readers",
            f"{PACKAGE}.workspace",
        }
    ),
    # Three, not the five it was: ``execution/`` gets to ``workspace`` through
    # ``lock.py``'s ``meta_dir_for``, to ``models`` through both
    # the mixin's ``WorkspaceConfig`` and its ``Identity``, and to ``readers``
    # only through ``models.gitignore_seed``'s in-function import — the entry a
    # graph that followed no transitive edge would miss while still satisfying a
    # mere "non-empty" check.
    #
    # **It lost ``documents`` and ``documents.models`` in story 55-8**, and the
    # row is narrowed rather than left generous: it reached both only because
    # ``models.py`` imported two cap constants for ``WorkspaceConfig`` fields that
    # no longer exist. A row that kept claiming them would be a reach this
    # capability does not have, and this table exists to report exactly that.
    "execution": frozenset(
        {
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

Calibrated for **this package**, which holds 33 modules — counted from the sweep
itself rather than by adding one to the number that was here. The previous figure
said 31 while 32 were on disk, so incrementing it would have published an
inherited off-by-one as a fact; a count nobody measures is how a number stops
meaning anything. Story 55-1's sibling
sweep uses 40 because it walks the whole of ``src/akgentic/tool/``; carrying that
number over to a root of this size would fail a correct sweep, which is the
opposite of what a sentinel is for.

**The floor does not move when the package grows.** It is deliberately below the
smallest correct sweep, not a number chosen to pass: story 55-6 took the count
from 27 to 29, story 55-8 to 31 and story 57-2 to 33, and this stayed at 20
throughout, which is the whole point of a floor. Story 55-9 moved a module
without changing the count, which is what a move does.
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
        every reference is to ``documents.models``, ``documents.store`` or
        ``documents.cache``. The package is in ``rag/``'s closure **only** because
        importing one of those executes its ``__init__`` first, which is the whole
        of the rule. A graph that recorded leaves alone would leave it out, and
        would then answer "clean" for a capability that dragged in a whole package
        through one of its submodules.

        **Asserted once, and not per capability**, because it is a property of
        :func:`_edges` — shared by every row — and not of any one capability.
        Parametrised, it would assert of *each* capability that it reaches
        ``documents``, and three of them do not.

        **Read off ``rag`` rather than ``read``, since story 55-8.** It was
        ``read``, which reached the package transitively through
        ``workspace/models.py``'s import of the two document caps; those caps went
        card-side with the extraction cache and the import went with them, so
        ``read/`` legitimately never reaches ``documents`` any more — exactly the
        case this docstring warned would redden a working guard. ``rag/`` imports
        ``documents.models`` outright, which is the same rule read off a different
        row.
        """
        closure = _closure(_modules_under("rag", modules), modules)

        assert f"{PACKAGE}.documents" in closure, (
            "the graph is recording leaves without their parent packages — "
            "importing a submodule executes every parent __init__ on the way down"
        )


@pytest.mark.parametrize("capability", sorted(CAPABILITY_CLOSURES))
class TestACapabilityDependsOnTheSpineAndNothingElse:
    """One row per capability module — all five of them since story 55-7."""

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
        straight into ``rag/__init__.py``'s ``TYPE_CHECKING`` import of
        ``WorkspaceActor``, which is the package's own recommended cycle-breaker and
        is nothing to do with the capability under test. The transitive question is
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


class TestTheDeletabilityClaimIsObservable:
    """ADR-053's Consequences, asserted instead of asserted-about.

    *"A capability's code is deletable, which is the test of whether this layout
    is real."* For ``journal/`` and ``execution/`` that was **false** until story
    55-7: ``write/gate.py`` constructed an ``Identity`` out of the journal and
    composed its busy refusal out of the exec module, both at runtime, on every
    accepted mutation. Delete either directory and the write capability — the
    default-on one — stopped importing.

    **Why this is not the deny-list the module docstring forbids.** That rule is
    about how a capability's closure is *bounded*: an allow-list catches the
    module nobody thought to forbid, and a deny-list passes silently for it. This
    spec bounds nothing. It is a named claim about two named capabilities, and it
    exists because the allow-list **cannot** make it: containment asserts
    ``closure - allowed == set()``, so an entry that is permitted and no longer
    needed is invisible — ``journal`` could silently return to ``write/``'s
    runtime closure tomorrow and every assertion above would stay green. The
    allow-list still runs on every row, unchanged; this runs beside it, and its
    subject is named by the claim rather than by a list somebody has to remember
    to extend.
    """

    def test_the_write_capability_reaches_neither_at_runtime(
        self, modules: dict[str, Path]
    ) -> None:
        """The two default-off capabilities ``write/`` used to drag in with it.

        ``GitJournal`` still **types** ``_journal`` under ``TYPE_CHECKING`` —
        which is why ``journal`` is still on ``write/``'s allow-list row and why
        this is the *runtime* closure rather than the direct-edge set. An
        annotation executes nothing, closes no cycle and costs nothing at bind; it
        is not a reason a directory cannot be deleted.
        """
        closure = _closure(_modules_under("write", modules), modules)

        assert f"{PACKAGE}.journal" not in closure, (
            "write/ imports the journal capability at runtime again — deleting "
            "journal/ would break the default-on mutation path, which is the "
            "deletability claim ADR-053 rests on"
        )
        assert f"{PACKAGE}.execution" not in closure, (
            "write/ imports the exec capability at runtime again — the mutation "
            "busy refusal belongs to lock.py, in the spine, precisely so that it "
            "does not"
        )

    def test_the_write_capability_no_longer_names_the_actor_at_all(
        self, modules: dict[str, Path]
    ) -> None:
        """Story 55-8, and it is a **direct-edge** claim rather than a runtime one.

        ``write/gate.py`` annotated ``_workspace_tell`` as a ``WorkspaceActor``
        and sent ``mark_paths_stale`` over it on every accepted mutation. The
        stale-mark is a direct call on the card's own ``DocumentCache`` now,
        because a read/write card creates no actor — so nothing under ``write/``
        names the actor, under ``TYPE_CHECKING`` or otherwise.

        The allow-list cannot report this: removing a permitted-and-unneeded entry
        reddens no subset test, which is exactly why the entry was removed *and*
        this assertion added in the same change.
        """
        own = _modules_under("write", modules)
        named = {
            target
            for name in own
            for target in _edges(name, modules[name], set(modules), annotations_too=True)
        }

        assert f"{PACKAGE}.actor" not in named, (
            "write/ names the workspace actor again — the mutation gate is "
            "card-side and the actor is dispatch-only, so a write card has none"
        )

    def test_the_names_it_denies_are_real_modules(self, modules: dict[str, Path]) -> None:
        """Non-vacuity, and it is the whole risk of an assertion shaped like this.

        Two ``not in`` assertions over a misspelled dotted name pass for ever and
        prove nothing — the shape ``test_exec.py::TestTheCapability`` already
        refuses for absence assertions over a registered-name list. So the two
        subjects are checked to exist, here, where a rename that moved either
        capability would otherwise leave the guard green and empty.
        """
        assert f"{PACKAGE}.journal" in modules
        assert f"{PACKAGE}.execution" in modules
        assert f"{PACKAGE}.actor" in modules
