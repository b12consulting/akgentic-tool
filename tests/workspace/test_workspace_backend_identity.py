"""Story 51-4: a workspace row names no team, and the same chunk keeps the same id.

A hosted workspace's ``team_id`` is auto-generated per actor lifetime and names no
team, so its cluster backend is built with none: Weaviate and Qdrant stamp ``""``.
With no team in the identity, a chunk re-added by the next lifetime — after a reap,
a restart — lands on the same point (Qdrant) and the same object (Weaviate), which
both clients overwrite in place.

**Every backend here is built through two real workspace actors' ``_resolve_store``
and the real registered factory.** Only the network client is a double, handed
out where the factory fetches its shared client. Building the backends by hand
with ``team_id=None`` would be green whatever ``_resolve_store`` passed — the
check narrowed until it agreed.
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
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.documents.models import RAG_COLLECTION
from akgentic.tool.workspace.models import WorkspaceConfig
from tests.workspace.conftest import WORKSPACE_PATH, factory_for

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


def _lifetimes(
    count: int, param: VectorStoreParam, use: Callable[[VectorStoreService], None]
) -> list[uuid.UUID]:
    """Start, use and stop *count* workspace actors on one path, one after the other.

    Each builds its backend through its own ``_resolve_store``, exactly as
    ``enable_rag`` does, and hands it to *use*. Returns each actor's team id.
    """
    team_ids: list[uuid.UUID] = []
    for _ in range(count):
        actor = WorkspaceActor(
            config=WorkspaceConfig(
                name=workspace_actor_name(WORKSPACE_PATH),
                role=WORKSPACE_ACTOR_ROLE,
                workspace_path=WORKSPACE_PATH,
            )
        )
        actor.on_start()
        try:
            team_ids.append(actor.team_id)
            backend = actor._resolve_store(param)
            assert backend is not None, "the workspace degraded instead of building a backend"
            use(backend)
        finally:
            actor.stop_children()
            actor.on_stop()
    return team_ids


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
## AC 10 (a) and (b) — the same chunk, the same id, across two lifetimes
##
class TestTheSameChunkKeepsItsIdAcrossTwoLifetimes:
    def test_two_lifetimes_upsert_one_qdrant_point(
        self, workspace_tree: Path, qdrant_client: MagicMock
    ) -> None:
        """qdrant-client overwrites a point whose id exists, so one id is one point.

        The double cannot show a point count; it shows the id each upsert carried,
        and that is what this spec asserts.
        """
        team_ids = _lifetimes(2, VectorStoreParam(backend="qdrant"), _write_one_chunk)

        assert team_ids[0] != team_ids[1], "one team id — the spec could not see a team leak"
        points = [call.kwargs["points"][0] for call in qdrant_client.upsert.call_args_list]
        assert len(points) == 2
        assert points[0].id == points[1].id == _expected_id("", "", "chunk-1")
        assert [point.payload["team_id"] for point in points] == ["", ""]

    def test_two_lifetimes_add_one_weaviate_object(
        self, workspace_tree: Path, weaviate_client: MagicMock
    ) -> None:
        """weaviate-client replaces an object whose uuid exists and mints a v4 when none is given.

        The object's identity is the ``uuid`` keyword, so that — not the
        ``team_id`` property — is what makes it one object.
        """
        team_ids = _lifetimes(2, VectorStoreParam(backend="weaviate"), _write_one_chunk)

        assert team_ids[0] != team_ids[1], "one team id — the spec could not see a team leak"
        added = _added_objects(weaviate_client)
        assert len(added) == 2
        assert [call.kwargs["uuid"] for call in added] == [_expected_id("", "", "chunk-1")] * 2
        assert [call.kwargs["properties"]["team_id"] for call in added] == ["", ""]


##
## AC 9 (a) and (b) — the workspace hands its backend no team
##
class TestTheWorkspaceBackendCarriesNoTeam:
    @pytest.mark.parametrize("backend", ["weaviate", "qdrant"])
    def test_the_factory_receives_no_team_and_the_actor_name(
        self, workspace_tree: Path, backend: str
    ) -> None:
        with factory_for(backend, lambda _context: MagicMock()) as contexts:
            _lifetimes(1, VectorStoreParam(backend=backend), lambda _built: None)

        [context] = contexts
        assert context.team_id is None
        assert context.config.name == workspace_actor_name(WORKSPACE_PATH)

    def test_a_teamless_weaviate_backend_serves_the_shared_collection_and_no_other(
        self, workspace_tree: Path, weaviate_client: MagicMock
    ) -> None:
        def use(backend: VectorStoreService) -> None:
            _write_one_chunk(backend)
            backend.search(RAG_COLLECTION, [0.1, 0.2], top_k=5, scope=WORKSPACE_PATH)
            backend.remove(RAG_COLLECTION, ["chunk-1"], scope=WORKSPACE_PATH)
            backend.create_collection("planning", VectorStoreParam())
            with pytest.raises(ValueError, match="without a team_id"):
                backend.search("planning", [0.1, 0.2], top_k=5)

        _lifetimes(1, VectorStoreParam(backend="weaviate"), use)

        [added] = _added_objects(weaviate_client)
        assert added.kwargs["properties"]["team_id"] == ""
        collection = weaviate_client.collections.get.return_value
        search_filter = collection.query.near_vector.call_args.kwargs["filters"]
        remove_filter = collection.data.delete_many.call_args.kwargs["where"]
        assert _weaviate_targets(search_filter) == ["scope"]
        assert _weaviate_targets(remove_filter) == ["ref_id", "scope"]

    def test_a_teamless_qdrant_backend_serves_the_shared_collection_and_no_other(
        self, workspace_tree: Path, qdrant_client: MagicMock
    ) -> None:
        qdrant_client.query_points.return_value = MagicMock(points=[])

        def use(backend: VectorStoreService) -> None:
            _write_one_chunk(backend)
            backend.search(RAG_COLLECTION, [0.1, 0.2], top_k=5, scope=WORKSPACE_PATH)
            backend.remove(RAG_COLLECTION, ["chunk-1"], scope=WORKSPACE_PATH)
            backend.create_collection("planning", VectorStoreParam())
            with pytest.raises(ValueError, match="without a team_id"):
                backend.search("planning", [0.1, 0.2], top_k=5)

        _lifetimes(1, VectorStoreParam(backend="qdrant"), use)

        [upsert] = qdrant_client.upsert.call_args_list
        assert upsert.kwargs["points"][0].payload["team_id"] == ""
        search_filter = qdrant_client.query_points.call_args.kwargs["query_filter"]
        remove_filter = qdrant_client.delete.call_args.kwargs["points_selector"]
        for predicate in (search_filter, remove_filter):
            assert "scope" in repr(predicate)
            assert "team_id" not in repr(predicate)
