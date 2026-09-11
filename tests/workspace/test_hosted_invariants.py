"""What the workspace actor must never do — and the two rules that stopped being rules.

**Two spec classes were deleted here by story 52-5, by decision rather than by
failure**, and the distinction is the whole reason this paragraph exists:

- ``TestTheActorNeverReadsItsOrchestrator`` (three specs, AST canary included)
  held that no module under ``workspace/actor/`` may read ``self.orchestrator``,
  because a **hosted** actor is started with ``orchestrator=None`` and every such
  read would degrade or raise. The actor is a team child again (ADR-051,
  *Ruling A*), so it *has* an orchestrator and reading one is ordinary.
- ``TestTheWorkspaceIsNeverBoundAsAChild`` (two specs) held that no module under
  ``workspace/`` may call ``getChildrenOrCreate``. The card calls exactly that,
  once, on purpose.

Neither was removed for being inconvenient. Their premise was reversed by a
decision recorded in ADR-051 and in story 52-5's *Ruling A*: hosting existed to
keep one actor per tree because the actor held the shared state, and every piece
of that state — the exec hold, the document cache, the retrieval index, the write
gate — has since moved onto the tree itself, where the filesystem serialises it
across processes as well as teams. A reader who finds these classes missing
should find that sentence here rather than have to reconstruct it.

What the deletion does **not** license is a silent loss of coverage. The
invariant that replaced them is asserted from the other side, in
``test_observation_recording.py``: the card binds as a child, once, and forwards
to **no host at all** — ``resource_calls`` stays empty on every card shape, which
is ``akgentic-infra`` story 69-1's acceptance guard read from this side of the
seam.

The live class below is untouched by any of that: a workspace still owns no
vector store of its own.
"""

from __future__ import annotations

from pathlib import Path

import pykka
import pytest

from akgentic.tool.workspace.actor import workspace_actor_name
from akgentic.tool.workspace.tool import WorkspaceTool
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    tool_named,
)


class TestAWorkspaceOwnsNoStoreOfItsOwn:
    """**Premise reversed.** The workspace held a store child; it holds none.

    Story 51-1 made the in-memory store this actor's own child so that a *hosted*
    actor never had to ask an orchestrator for anything (core ADR-022 Decision 3).
    Epic 52 retires hosting, and ADR-022 Decision 2 records that the prohibition
    was always on the actor rather than on the card — "a card keeps talking to its
    orchestrator" — so the card binds the team's ``#VectorStore`` again and the
    actor creates nothing.

    The invariant the two classes above hold is untouched by that and is what
    makes this safe: the actor still reads no orchestrator, and still is not
    bound as anybody's child. These two specs are the *other* side of it, and
    they are inverted rather than deleted.
    """

    def test_a_bound_retrieval_card_leaves_the_workspace_with_no_store_child(
        self, threaded_orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The store exists — once, in the team — and the workspace did not make it.

        ``FakeOrchestratorProxy`` in live mode starts the workspace with no
        orchestrator at all, which is what ``ResourceHost`` does. So the one store
        that exists cannot have come from it: a workspace that still spawned a
        child would show two.
        """
        pytest.importorskip("numpy", reason="the [vector_search] extra is not installed")
        from akgentic.tool.vector_store.actor import (
            VS_ACTOR_NAME,
            VS_ACTOR_ROLE,
            VectorStoreActor,
        )

        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []
        observer = FakeActorToolObserver(threaded_orchestrator_proxy, name="alice")
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_index=True)
        card.observer(observer)

        answer = tool_named(card, "workspace_rag_index")("")

        assert answer == "0 file(s) queued, 0 already current, 0 unsupported"
        [store_ref] = pykka.ActorRegistry.get_by_class(VectorStoreActor)
        config = store_ref.proxy().config.get()
        # The team's singleton name, not the per-tree one a child carried.
        assert config.name == VS_ACTOR_NAME
        assert config.role == VS_ACTOR_ROLE
        assert f"{VS_ACTOR_NAME}-{WORKSPACE_PATH}" != config.name
        assert workspace_actor_name(WORKSPACE_PATH) in threaded_orchestrator_proxy.children

        threaded_orchestrator_proxy.stop_all()

        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []

    def test_the_store_outlives_the_workspace_it_serves(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """**Inverted.** A child died with its parent; a team member does not.

        That is the point of going back to the team's store: two trees in a team
        share it, so one workspace stopping must not take the other's index down.
        The workspace actor is stopped on its own here — it is a sibling of the
        store rather than its parent since 52-4 — and the store outlives it.
        """
        pytest.importorskip("numpy", reason="the [vector_search] extra is not installed")
        from akgentic.tool.vector_store.actor import VS_ACTOR_NAME, VectorStoreActor

        observer = FakeActorToolObserver(orchestrator_proxy, name="alice")
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_index=True)
        card.observer(observer)
        address, store = orchestrator_proxy.children[VS_ACTOR_NAME]
        assert isinstance(store, VectorStoreActor)

        name = workspace_actor_name(WORKSPACE_PATH)
        _workspace_address, workspace = orchestrator_proxy.children.pop(name)
        workspace.stop_children()
        workspace.on_stop()

        assert name not in orchestrator_proxy.children
        assert orchestrator_proxy.children[VS_ACTOR_NAME][1] is store
        assert address.is_alive()
