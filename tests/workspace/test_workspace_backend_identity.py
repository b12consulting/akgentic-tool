"""A workspace chunk keeps one id however many teams index the tree.

**Premise re-pointed, invariant kept.** Story 51-4 reached this by handing the
workspace's cluster backend ``team_id=None``: a hosted actor's team id was
auto-generated per lifetime, so a team inside the row identity moved every time
the actor restarted and a re-added chunk minted a second point. Story 52-4 gives
the backend the team's **real** id — a card builds it now, and a sweep and a
team-scoped filter both need that id — and moves the rule to where it belongs:
:func:`~akgentic.tool.vector_store.protocol.row_object_id` drops the team from
the identity of a **shared** collection, whoever writes it.

That is the stronger statement, and this file asserts the stronger case. It is no
longer two lifetimes of one team-less actor; it is **two different teams** binding
one tree, each with a real and different id, writing one chunk each — and landing
on one point. The team is still *stamped* on the row, because filtering and
sweeping read it; only the identity drops it.

**Every backend here is built through a real ``WorkspaceTool.observer()`` and the
real registered factory.** Only the network client is a double, handed out where
the factory fetches its shared client. Building the backends by hand with a
chosen ``team_id`` would be green whatever the card passed — the check narrowed
until it agreed.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from akgentic.tool.vector_store.protocol import VectorStoreParam, VectorStoreService
from akgentic.tool.vector_store.vector import VectorEntry
from akgentic.tool.workspace.actor import workspace_actor_name
from akgentic.tool.workspace.documents.models import RAG_COLLECTION
from akgentic.tool.workspace.tool import WorkspaceTool
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    FakeWorkspaceHost,
    factory_for,
)

_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")
"""The literal namespace every existing Qdrant point id was derived under."""


def _expected_id(team_id: str, tenant: str, ref_id: str) -> str:
    """The id the derivation must produce, computed here from the literal namespace."""
    identity = json.dumps([team_id, tenant, ref_id], ensure_ascii=True, separators=(",", ":"))
    return str(uuid.uuid5(_ID_NAMESPACE, identity))


def _chunk(ref_id: str = "chunk-1") -> VectorEntry:
    """One workspace chunk, as an ``#embed-`` worker hands it back."""
    return VectorEntry(
        ref_type="workspace_chunk",
        ref_id=ref_id,
        text="the chunk",
        vector=[0.1, 0.2],
        scope=WORKSPACE_PATH,
        path="a.md",
        ordinal=0,
    )


def _teams(
    count: int, param: VectorStoreParam, use: Callable[[VectorStoreService], None]
) -> list[uuid.UUID]:
    """Bind *count* teams' cards onto one tree and hand each one's backend to *use*.

    One shared :class:`FakeWorkspaceHost`, so the teams genuinely land on the same
    workspace actor — which is what makes "two teams, one tree" the case under
    test rather than two unrelated trees. Each card builds its backend through its
    own ``observer()`` and the real registered factory.

    Returns:
        Each binding team's id, in order.
    """
    host = FakeWorkspaceHost()
    team_ids: list[uuid.UUID] = []
    try:
        for index in range(count):
            orchestrator_proxy = FakeOrchestratorProxy(host=host)
            observer = FakeActorToolObserver(orchestrator_proxy, name=f"agent-{index}")
            team_ids.append(observer.team_id)
            card = WorkspaceTool(
                workspace_id=WORKSPACE_NAME, workspace_rag_index=True, vector_store=param
            )
            card.observer(observer)
            backend = card._vector_store
            assert backend is not None, "the card degraded instead of building a backend"
            use(backend)
    finally:
        host.stop_all()
    return team_ids


@pytest.fixture(autouse=True)
def _cluster_is_provisioned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both cluster URLs, for every spec in this file.

    ``require_backend_configured`` runs at ``observer()`` time now, so a card
    naming a cluster fails the bind unless the environment claims one — which is
    its own spec in ``test_rag_card.py`` and not what any spec here is about.
    """
    monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "http://weaviate.test:8080")
    monkeypatch.setenv("AKGENTIC_QDRANT_URL", "http://qdrant.test:6333")


@pytest.fixture
def qdrant_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """One client double, handed to every Qdrant backend the real factory builds."""
    import akgentic.tool.vector_store.backends.qdrant as qdrant_module

    client = MagicMock()
    client.collection_exists.return_value = False
    monkeypatch.setenv(qdrant_module.QDRANT_URL_ENV, "http://qdrant.test:6333")
    monkeypatch.setattr(qdrant_module, "get_client", lambda _key, _connect: client)
    return client


@pytest.fixture
def weaviate_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """One client double, handed to every Weaviate backend the real factory builds."""
    import akgentic.tool.vector_store.backends.weaviate as weaviate_module

    client = MagicMock()
    monkeypatch.setenv(weaviate_module.WEAVIATE_URL_ENV, "http://weaviate.test:8080")
    monkeypatch.setattr(weaviate_module, "_weaviate_client", lambda _url, _api_key=None: client)
    return client


def _weaviate_targets(predicate: Any) -> list[str]:
    """The property every leg of a real Weaviate filter names, in order.

    A conjunction carries its legs in ``filters``; a leg carries its property in
    ``target``. A team leg would show up here as ``team_id``.
    """
    legs = getattr(predicate, "filters", None)
    if legs is not None:
        return [target for leg in legs for target in _weaviate_targets(leg)]
    return [str(predicate.target)]


def _write_one_chunk(backend: VectorStoreService) -> None:
    backend.create_collection(RAG_COLLECTION, VectorStoreParam())
    backend.add(RAG_COLLECTION, [_chunk()])


def _added_objects(client: MagicMock) -> list[Any]:
    """Every ``add_object`` call on the one batch the shared collection handle opens."""
    batch = client.collections.get.return_value.batch.dynamic.return_value.__enter__.return_value
    return list(batch.add_object.call_args_list)


##
## The same chunk, the same id — now across two *teams*, not two lifetimes
##
class TestTheSameChunkKeepsItsIdAcrossTwoTeams:
    def test_two_teams_upsert_one_qdrant_point(
        self, workspace_tree: Path, qdrant_client: MagicMock
    ) -> None:
        """qdrant-client overwrites a point whose id exists, so one id is one point.

        The double cannot show a point count; it shows the id each upsert carried,
        and that is what this spec asserts. The two teams being genuinely
        different is asserted first — without it the spec could not see a team
        leaking into the identity.
        """
        team_ids = _teams(2, VectorStoreParam(backend="qdrant"), _write_one_chunk)

        assert team_ids[0] != team_ids[1], "one team id — the spec could not see a team leak"
        points = [call.kwargs["points"][0] for call in qdrant_client.upsert.call_args_list]
        assert len(points) == 2
        assert points[0].id == points[1].id == _expected_id("", "", "chunk-1")
        # Stamped, though: the identity drops the team, the row keeps it.
        assert [point.payload["team_id"] for point in points] == [
            str(team_ids[0]),
            str(team_ids[1]),
        ]

    def test_two_teams_add_one_weaviate_object(
        self, workspace_tree: Path, weaviate_client: MagicMock
    ) -> None:
        """weaviate-client replaces an object whose uuid exists and mints a v4 when none is given.

        The object's identity is the ``uuid`` keyword, so that — not the
        ``team_id`` property — is what makes it one object.
        """
        team_ids = _teams(2, VectorStoreParam(backend="weaviate"), _write_one_chunk)

        assert team_ids[0] != team_ids[1], "one team id — the spec could not see a team leak"
        added = _added_objects(weaviate_client)
        assert len(added) == 2
        assert [call.kwargs["uuid"] for call in added] == [_expected_id("", "", "chunk-1")] * 2
        assert [call.kwargs["properties"]["team_id"] for call in added] == [
            str(team_ids[0]),
            str(team_ids[1]),
        ]


##
## **Premise reversed.** The backend carried no team; it carries the real one
##
class TestTheWorkspaceBackendCarriesItsRealTeam:
    """Story 51-4 passed ``None`` here. The identity rule replaced that (see the module
    docstring), so the team is free to be what it is — and it has to be, because a
    row a sweep cannot attribute is a row a deleted team leaves behind."""

    @pytest.mark.parametrize("backend", ["weaviate", "qdrant"])
    def test_the_factory_receives_the_binding_teams_id(
        self, workspace_tree: Path, backend: str
    ) -> None:
        with factory_for(backend, lambda _context: MagicMock()) as contexts:
            [team_id] = _teams(1, VectorStoreParam(backend=backend), lambda _built: None)

        [context] = contexts
        assert context.team_id == str(team_id)
        assert context.team_id is not None

    def test_a_workspace_backend_still_queries_the_shared_collection_by_scope_alone(
        self, workspace_tree: Path, weaviate_client: MagicMock
    ) -> None:
        """The team leg is dropped **per collection**, never per backend.

        That is the load-bearing half: the filter must lose its team leg because
        ``workspace_chunks`` is shared, not because the writer happened to have no
        team. A backend with a real team is what proves it.
        """

        def use(backend: VectorStoreService) -> None:
            _write_one_chunk(backend)
            backend.search(RAG_COLLECTION, [0.1, 0.2], top_k=5, scope=WORKSPACE_PATH)
            backend.remove(RAG_COLLECTION, ["chunk-1"], scope=WORKSPACE_PATH)

        [team_id] = _teams(1, VectorStoreParam(backend="weaviate"), use)

        [added] = _added_objects(weaviate_client)
        assert added.kwargs["properties"]["team_id"] == str(team_id)
        collection = weaviate_client.collections.get.return_value
        search_filter = collection.query.near_vector.call_args.kwargs["filters"]
        remove_filter = collection.data.delete_many.call_args.kwargs["where"]
        assert _weaviate_targets(search_filter) == ["scope"]
        assert _weaviate_targets(remove_filter) == ["ref_id", "scope"]

    def test_a_workspace_backend_may_now_query_a_team_scoped_collection(
        self, workspace_tree: Path, weaviate_client: MagicMock
    ) -> None:
        """**Inverted.** A team-less backend was refused one; this one is not.

        The refusal survives — it is what a genuinely administrative backend still
        meets — but it is no longer what a workspace's backend meets, and a spec
        that still asserted the raise would be pinning the ``team_id=None`` wiring
        rather than the rule.
        """
        weaviate_client.collections.get.return_value.query.near_vector.return_value = MagicMock(
            objects=[]
        )

        def use(backend: VectorStoreService) -> None:
            backend.create_collection("planning", VectorStoreParam())
            backend.search("planning", [0.1, 0.2], top_k=5)

        [team_id] = _teams(1, VectorStoreParam(backend="weaviate"), use)

        search_filter = (
            weaviate_client.collections.get.return_value.query.near_vector.call_args.kwargs[
                "filters"
            ]
        )
        assert "team_id" in _weaviate_targets(search_filter)
        assert str(team_id)

    def test_a_workspace_qdrant_backend_queries_the_shared_collection_by_scope_alone(
        self, workspace_tree: Path, qdrant_client: MagicMock
    ) -> None:
        qdrant_client.query_points.return_value = MagicMock(points=[])

        def use(backend: VectorStoreService) -> None:
            _write_one_chunk(backend)
            backend.search(RAG_COLLECTION, [0.1, 0.2], top_k=5, scope=WORKSPACE_PATH)
            backend.remove(RAG_COLLECTION, ["chunk-1"], scope=WORKSPACE_PATH)

        [team_id] = _teams(1, VectorStoreParam(backend="qdrant"), use)

        [upsert] = qdrant_client.upsert.call_args_list
        assert upsert.kwargs["points"][0].payload["team_id"] == str(team_id)
        search_filter = qdrant_client.query_points.call_args.kwargs["query_filter"]
        remove_filter = qdrant_client.delete.call_args.kwargs["points_selector"]
        for predicate in (search_filter, remove_filter):
            assert "scope" in repr(predicate)
            assert "team_id" not in repr(predicate)


##
## The workspace actor's name is still what the backend is built under
##
class TestTheBackendNamesTheStore:
    def test_the_context_config_names_the_store_rather_than_the_tree(
        self, workspace_tree: Path
    ) -> None:
        """The card builds it, so the config it passes is the store's, not the tree's.

        Recorded because it moved: the workspace **actor's** name used to be in
        that slot, and no built-in factory reads ``config`` at all — so a reader
        who finds the old name there would be reading a stale claim rather than a
        wiring fact.
        """
        from akgentic.tool.vector_store.actor import VS_ACTOR_NAME

        with factory_for("weaviate", lambda _context: MagicMock()) as contexts:
            _teams(1, VectorStoreParam(backend="weaviate"), lambda _built: None)

        [context] = contexts
        assert context.config.name == VS_ACTOR_NAME
        assert context.config.name != workspace_actor_name(WORKSPACE_PATH)
