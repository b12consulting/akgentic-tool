"""What a read-only ``WorkspaceTool`` costs, and what its parameters are stored as.

Two properties, both written against the **un-moved** tree and confirmed green
there — the discipline ``test_card_public_api.py`` states about itself — so what
they assert is what the move preserved rather than what the move produced.

**The cost table is a characterisation guard, not a bug fix.** Most of it is
already true, and saying so is the point: it is a fence around a property while
five stories move code past it. Every row is an *equality* over a list or a set,
never ``name not in names`` — the vocabulary ``test_exec.py::TestTheCapability``
already uses, and for its stated reason: an absence assertion passes over an
empty list, so a bind that registered nothing at all would satisfy it.

**Two rows this module deliberately does not claim**, because they are not this
story's to change:

- a read-only bind still creates the ``#Workspace`` actor. Removing it belongs to
  story 55-7, which turns the actor into dispatch; until then the honest
  assertion is that **exactly one** actor is created and it is that one.
- a read-only bind still constructs a :class:`GitJournal` object. It initialises
  nothing and creates no repository, so the observable — no ``<leaf>.git`` on
  disk — is already what it will be after story 55-4 removes the construction.

Neither carries an ``xfail`` marker. Story 55-1 removed this package's last
strict-xfail tripwire, and a second one would redden the suite the day 55-7 lands
rather than the day someone wants it to.

**Two resolutions that run unconditionally are not counted as cost**, because
they are deliberate: ``resolve_lock_backend()`` and ``resolve_document_store()``
are admin-facing fail-fast checks over stateless objects that touch no disk until
a first ``acquire`` or ``put``. "Resolves no backend" below is scoped to the
*vector* backend, which is the one that opens a client and creates an actor.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from akgentic.core.utils import deserialize_object, import_class, serialize

from akgentic.tool.workspace.actor import workspace_actor_name
from akgentic.tool.workspace.card import WorkspaceTool
from akgentic.tool.workspace.journal import git_dir_for

# From where they are **defined**, never through ``card/params.py``'s re-export:
# this module is what holds that re-export in place, so a spec that imported
# through it would be reddened by its own import line rather than by the
# assertions that state the property.
from akgentic.tool.workspace.read.params import (
    ExpandMediaRefs,
    WorkspaceGlob,
    WorkspaceRead,
)
from akgentic.tool.workspace.workspace import meta_dir_for
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
)

READ_CALLABLES = [
    "workspace_read",
    "workspace_list",
    "workspace_glob",
    "workspace_grep",
    "workspace_view",
]
"""The five read callables, **in registration order**.

The order is part of the contract, not incidental: ``test_read_tool.py`` indexes
``get_tools()`` positionally, so a reordering of ``_read_tools`` would redden it
somewhere far from the reorder. Pinned here as a list so it reddens here too, and
says why.
"""


class _Tripwire:
    """A stand-in that records every call and then raises.

    The raise states the intent — nothing here may be reached during a read-only
    bind — but it cannot be the assertion: ``_bind_vector_store`` catches
    ``Exception`` and degrades to a WARNING, so a bind that *did* reach one of
    these would swallow the raise and stay green. :attr:`calls` is what actually
    carries the property, and it is asserted empty.
    """

    def __init__(self, what: str) -> None:
        self.what = what
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(args)
        raise AssertionError(f"a read-only bind reached {self.what}")


@pytest.fixture
def vector_store_tripwires(monkeypatch: pytest.MonkeyPatch) -> dict[str, _Tripwire]:
    """Record-and-raise stand-ins over both ways a card can obtain a vector store.

    Patched on ``card``'s own module namespace, which is where the two names are
    bound and therefore where the card looks them up.
    """
    tripwires = {
        "get_backend_spec": _Tripwire("the vector-store registry"),
        "ensure_store_actor": _Tripwire("ensure_store_actor"),
    }
    for name, tripwire in tripwires.items():
        monkeypatch.setattr(f"akgentic.tool.workspace.card.{name}", tripwire)
    return tripwires


@pytest.fixture
def read_only_card(
    orchestrator_proxy: FakeOrchestratorProxy,
    workspace_tree: Path,
    vector_store_tripwires: dict[str, _Tripwire],
) -> WorkspaceTool:
    """``WorkspaceTool(read_only=True)`` with every optional field at its default.

    Bound through the real harness — the fake observer and the fake orchestrator
    the rest of this package's specs use — because a spec that hand-builds a
    ``Filesystem`` proves the helper and not the capability.
    """
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, read_only=True)
    card.observer(FakeActorToolObserver(orchestrator_proxy))
    return card


class TestEnablingNothingCostsNothing:
    """The whole cost of a read-only bind, stated as equalities."""

    def test_it_registers_the_five_read_callables_and_no_others(
        self, read_only_card: WorkspaceTool
    ) -> None:
        """An equality over the registered list, so a sixth callable reddens it."""
        assert [tool.__name__ for tool in read_only_card.get_tools()] == READ_CALLABLES

    def test_the_command_channel_carries_only_the_media_ref_expansion(
        self, read_only_card: WorkspaceTool
    ) -> None:
        """The two retrieval capabilities are off, so neither reaches COMMAND."""
        assert set(read_only_card.get_commands()) == {ExpandMediaRefs}

    def test_it_contributes_no_context_state(self, read_only_card: WorkspaceTool) -> None:
        """``workspace_rag_list`` is what puts a provider here, and it is off."""
        assert read_only_card.get_context_states() == []

    def test_it_creates_exactly_one_actor(
        self, read_only_card: WorkspaceTool, orchestrator_proxy: FakeOrchestratorProxy
    ) -> None:
        """The tree's own ``#Workspace`` and nothing else — no ``#VectorStore``.

        The row story 55-7 tightens: today the actor is created, and the claim
        this guard makes is that it is the *only* one.
        """
        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert created == [workspace_actor_name(WORKSPACE_PATH)]

    def test_it_resolves_no_vector_backend(
        self,
        read_only_card: WorkspaceTool,
        vector_store_tripwires: dict[str, _Tripwire],
        orchestrator_proxy: FakeOrchestratorProxy,
    ) -> None:
        """Neither the registry nor the store actor is reached, and none is looked up."""
        assert vector_store_tripwires["get_backend_spec"].calls == []
        assert vector_store_tripwires["ensure_store_actor"].calls == []
        assert orchestrator_proxy.member_lookups == []

    def test_it_leaves_no_journal_repository_on_disk(
        self, read_only_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        """``git_journal`` is off by default, so ``initialise`` never runs.

        The row story 55-4 tightens: today a ``GitJournal`` object is still
        constructed. Constructing one creates nothing, which is why the
        observable — the absent sibling — is already its final value.
        """
        assert not git_dir_for(workspace_tree).exists()

    def test_the_tree_and_its_metadata_sibling_hold_exactly_this(
        self, read_only_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        """Both listings by equality, with every entry explained.

        - the tree itself is **empty**: nothing is seeded into it, because
          ``.gitignore`` is the journal's and the journal never initialises;
        - the metadata sibling does not exist at all: the lock backend and the
          document store are stateless until a first ``acquire`` or ``put``, and
          nothing else at bind has reason to create it.
        """
        meta_dir = meta_dir_for(WORKSPACE_PATH)
        # The sibling relation, asserted before the absence: a ``meta_dir`` that
        # resolved somewhere else entirely would be absent for the wrong reason,
        # and the row would pass while looking at nothing.
        assert meta_dir.parent == workspace_tree.parent

        assert sorted(entry.name for entry in workspace_tree.iterdir()) == []
        assert not meta_dir.exists()


##
## AC 5 — a card a deployment already persisted still loads
##

STORED_MARKER_MODULE = "akgentic.tool.workspace.card.params"
"""The module path a deployment's stored read parameters carry in ``__model__``.

**Captured by serializing a configured card on the working tree, not transcribed
from a design document** — the same discipline ``test_card_public_api.py``'s
frozen sets follow.

``serialize_base_model`` stamps ``f"{cls.__module__}.{cls.__name__}"`` on every
:class:`~akgentic.core.utils.SerializableBaseModel`, and ``BaseToolParam`` is
one, so every card written since the card decomposition with a read parameter
set explicitly carries this literal string. Reading one back is ``import_module``
plus ``getattr`` on exactly it
(:func:`akgentic.core.utils.deserializer.import_class`); a path that has gone
raises ``UnresolvableClassError``, which turns a stored team's tool card into a
bad record rather than a card.

So the six read parameters keep resolving *here* whatever module actually defines
them — and the specs below are written so that the same source asserts the same
thing before and after any such move.
"""

READ_PARAM_NAMES = [
    "WorkspaceRead",
    "WorkspaceList",
    "WorkspaceGlob",
    "WorkspaceGrep",
    "WorkspaceView",
    "ExpandMediaRefs",
]
"""The six parameters of the read capability, by the name a marker can carry."""


@pytest.fixture
def explicitly_configured_card() -> WorkspaceTool:
    """A card carrying two read parameters the author set by hand.

    Two rather than one, and two *different* capabilities: a marker is stamped
    per nested model, so a single one could be preserved by an accident that a
    second would expose.
    """
    return WorkspaceTool(
        workspace_id=WORKSPACE_NAME,
        workspace_read=WorkspaceRead(document_reader=False),
        workspace_glob=WorkspaceGlob(max_results=3),
    )


class TestStoredReadParamsStillResolve:
    """The persisted ``__model__`` markers, end to end through core's own path."""

    def test_every_read_param_resolves_through_the_stored_module_path(self) -> None:
        """``import_module`` + ``getattr``, exactly as ``import_class`` does it.

        ``hasattr`` on the package would not catch this: the mechanism that keeps
        these six resolving after they are defined elsewhere is a **re-export**,
        which only an import of this precise module path exercises.
        """
        module = importlib.import_module(STORED_MARKER_MODULE)
        for name in READ_PARAM_NAMES:
            assert getattr(module, name, None) is not None, (
                f"{STORED_MARKER_MODULE}.{name} no longer resolves — every card "
                f"persisted with that parameter set explicitly carries that literal string"
            )

    def test_the_stored_path_serves_the_same_classes_the_card_uses(self) -> None:
        """A second definition would deserialise into a class nothing else uses.

        A re-export satisfies this; a copy of the class body, which is the
        tempting way to "keep the path working", does not.
        """
        stored = importlib.import_module(STORED_MARKER_MODULE)
        facade = importlib.import_module("akgentic.tool.workspace")
        for name in READ_PARAM_NAMES:
            assert getattr(stored, name) is getattr(facade, name)

    def test_a_freshly_dumped_card_carries_markers_that_resolve(
        self, explicitly_configured_card: WorkspaceTool
    ) -> None:
        """Whatever module a parameter lives in, the path a dump stamps must import back.

        This is the row that legitimately *changes* with a move — a dump names
        wherever the class is defined — so it is written as the invariant rather
        than as a literal: the stamped path resolves, and to the very class the
        card is holding.
        """
        dumped = serialize(explicitly_configured_card)
        assert isinstance(dumped, dict)

        for field in ("workspace_read", "workspace_glob"):
            marker = dumped[field]["__model__"]
            assert import_class(marker) is type(getattr(explicitly_configured_card, field))

    def test_a_record_written_before_the_move_still_validates_into_an_equal_card(
        self, explicitly_configured_card: WorkspaceTool
    ) -> None:
        """The end-to-end property, on the literal a deployment's database holds.

        The markers are rewritten to :data:`STORED_MARKER_MODULE` rather than
        left as the dump produced them, so this spec asserts the **same thing
        before and after** the parameters move: on the un-moved tree the rewrite
        is a no-op and the record is exactly what a deployment stored; afterwards
        it is the pre-move record, which is the one that has to keep loading.

        Its non-vacuity is the two specs above: they prove the path is real.
        """
        stored = serialize(explicitly_configured_card)
        assert isinstance(stored, dict)
        stored["workspace_read"]["__model__"] = f"{STORED_MARKER_MODULE}.WorkspaceRead"
        stored["workspace_glob"]["__model__"] = f"{STORED_MARKER_MODULE}.WorkspaceGlob"

        restored = deserialize_object(stored)

        assert isinstance(restored, WorkspaceTool)
        assert restored.model_dump() == explicitly_configured_card.model_dump()
