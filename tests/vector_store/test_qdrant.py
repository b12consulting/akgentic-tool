"""QdrantBackend — protocol conformance and team-scoped query construction.

The Qdrant client is mocked: these tests assert the backend builds the right
requests (points, filters, scope) and maps responses, not that a live cluster
behaves.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import akgentic.tool.vector_store.qdrant as qdrant_module
from akgentic.tool.vector_store.protocol import VectorQuery, VectorStoreParam
from akgentic.tool.vector_store.qdrant import (
    QDRANT_URL_ENV,
    QdrantBackend,
    require_qdrant_configured,
)
from akgentic.tool.vector_store.registry import BackendContext, get_backend_spec
from akgentic.tool.vector_store.vector import VectorEntry


def _make_backend(team_id: str | None = "team-42") -> tuple[QdrantBackend, MagicMock]:
    """Build a QdrantBackend over an injected client double.

    The backend takes a connected client and never builds one, so every spec
    here drives it through a double rather than patching the vendor module.
    """
    client = MagicMock()
    client.collection_exists.return_value = False
    backend = QdrantBackend(client=client, team_id=team_id)
    return backend, client


def _entry(ref_id: str = "e1", vector: list[float] | None = None) -> VectorEntry:
    return VectorEntry(
        ref_type="entity", ref_id=ref_id, text="hello", vector=vector or [0.1, 0.2, 0.3]
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TestEnvironment:
    def test_require_raises_when_unset(self) -> None:
        with pytest.raises(ValueError, match=QDRANT_URL_ENV):
            require_qdrant_configured("KnowledgeGraphTool")

    def test_require_passes_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(QDRANT_URL_ENV, "http://localhost:6333")
        require_qdrant_configured("KnowledgeGraphTool")  # does not raise

    def test_require_raises_when_client_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(QDRANT_URL_ENV, "http://localhost:6333")
        with (
            patch(
                "akgentic.tool.vector_store.qdrant.qdrant_dependencies_available",
                return_value=False,
            ),
            pytest.raises(ValueError, match=r"pip install akgentic-tool\[qdrant\]"),
        ):
            require_qdrant_configured("KnowledgeGraphTool")


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------


class TestCreateCollection:
    def test_creates_when_absent(self) -> None:
        backend, client = _make_backend()
        backend.create_collection("kg", VectorStoreParam(dimension=384))
        client.create_collection.assert_called_once()
        kwargs = client.create_collection.call_args[1]
        assert kwargs["collection_name"] == "kg"
        assert kwargs["vectors_config"].size == 384

    def test_no_op_when_present(self) -> None:
        from qdrant_client import models

        backend, client = _make_backend()
        client.collection_exists.return_value = True
        client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=models.VectorParams(size=1536, distance=models.Distance.COSINE)
                )
            )
        )
        backend.create_collection("kg", VectorStoreParam())
        client.create_collection.assert_not_called()

    def test_rejects_non_cosine_distance(self) -> None:
        from qdrant_client import models

        backend, _client = _make_backend()
        with pytest.raises(ValueError, match="only cosine distance"):
            backend.create_collection(
                "kg",
                VectorStoreParam(params={"distance": models.Distance.EUCLID}),
            )

    def test_rejects_existing_non_cosine_collection(self) -> None:
        from qdrant_client import models

        backend, client = _make_backend()
        client.collection_exists.return_value = True
        client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=models.VectorParams(size=1536, distance=models.Distance.EUCLID)
                )
            )
        )
        with pytest.raises(ValueError, match="does not use cosine distance"):
            backend.create_collection("kg", VectorStoreParam())


# ---------------------------------------------------------------------------
# Add / remove
# ---------------------------------------------------------------------------


class TestAdd:
    def test_stamps_team_id_on_every_point(self) -> None:
        backend, client = _make_backend(team_id="team-42")
        backend.create_collection("kg", VectorStoreParam())
        backend.add("kg", [_entry("e1"), _entry("e2")])
        points = client.upsert.call_args[1]["points"]
        assert len(points) == 2
        assert all(p.payload["team_id"] == "team-42" for p in points)
        assert all(p.payload["ref_id"] in {"e1", "e2"} for p in points)

    def test_point_ids_are_scoped_by_team(self) -> None:
        first, first_client = _make_backend(team_id="team-a")
        second, second_client = _make_backend(team_id="team-b")
        for backend in (first, second):
            backend.create_collection("planning", VectorStoreParam())

        first.add("planning", [_entry("3")])
        second.add("planning", [_entry("3")])

        first_id = first_client.upsert.call_args[1]["points"][0].id
        second_id = second_client.upsert.call_args[1]["points"][0].id
        assert first_id != second_id

    def test_point_ids_are_scoped_by_collection_tenant(self) -> None:
        first, first_client = _make_backend(team_id="team-a")
        second, second_client = _make_backend(team_id="team-a")
        first.create_collection("planning", VectorStoreParam(tenant="tenant-a"))
        second.create_collection("planning", VectorStoreParam(tenant="tenant-b"))

        first.add("planning", [_entry("3")])
        second.add("planning", [_entry("3")])

        first_point = first_client.upsert.call_args[1]["points"][0]
        second_point = second_client.upsert.call_args[1]["points"][0]
        assert first_point.id != second_point.id
        assert first_point.payload["tenant"] == "tenant-a"
        assert second_point.payload["tenant"] == "tenant-b"

    def test_add_unknown_collection_raises(self) -> None:
        backend, _client = _make_backend()
        with pytest.raises(ValueError, match="does not exist"):
            backend.add("missing", [_entry()])


class TestRemove:
    def test_remove_is_team_scoped(self) -> None:
        backend, client = _make_backend(team_id="team-42")
        backend.create_collection("kg", VectorStoreParam())
        backend.remove("kg", ["e1", "e2"])
        selector = client.delete.call_args[1]["points_selector"]
        # FilterSelector.filter.must carries both the team predicate and ref_id match.
        keys = {c.key for c in selector.filter.must}
        assert "team_id" in keys
        assert "ref_id" in keys


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def _response(self) -> SimpleNamespace:
        point = SimpleNamespace(
            score=0.9,
            payload={"ref_type": "entity", "ref_id": "e1", "text": "hello"},
        )
        return SimpleNamespace(points=[point])

    def test_maps_hits_and_scopes_to_team(self) -> None:
        backend, client = _make_backend(team_id="team-42")
        backend.create_collection("kg", VectorStoreParam())
        client.query_points.return_value = self._response()

        result = backend.search("kg", [0.1, 0.2, 0.3], top_k=5)

        assert len(result.hits) == 1
        assert result.hits[0].ref_id == "e1"
        assert result.hits[0].score == pytest.approx(0.9)
        query_filter = client.query_points.call_args[1]["query_filter"]
        assert any(c.key == "team_id" for c in query_filter.must)

    def test_query_filters_are_anded_onto_team_scope(self) -> None:
        backend, client = _make_backend(team_id="team-42")
        backend.create_collection("kg", VectorStoreParam())
        client.query_points.return_value = self._response()

        backend.search(
            "kg",
            [0.1, 0.2, 0.3],
            top_k=5,
            query=VectorQuery(filters={"ref_type": "entity"}, score_threshold=0.25),
        )

        call = client.query_points.call_args[1]
        keys = {c.key for c in call["query_filter"].must}
        assert {"team_id", "ref_type"} <= keys
        assert call["score_threshold"] == 0.25

    def test_native_params_passed_through(self) -> None:
        backend, client = _make_backend(team_id="team-42")
        backend.create_collection("kg", VectorStoreParam())
        client.query_points.return_value = self._response()

        backend.search(
            "kg", [0.1, 0.2, 0.3], top_k=5, query=VectorQuery(params={"timeout": 3})
        )
        assert client.query_points.call_args[1]["timeout"] == 3


class TestTeamlessBackendCannotQuery:
    """A backend with no team refuses to search/remove rather than guessing one."""

    def test_search_refuses(self) -> None:
        backend, client = _make_backend(team_id=None)
        backend.create_collection("kg", VectorStoreParam())
        with pytest.raises(ValueError, match="without a team_id"):
            backend.search("kg", [0.1, 0.2, 0.3], top_k=5)

    def test_remove_refuses(self) -> None:
        backend, _client = _make_backend(team_id=None)
        backend.create_collection("kg", VectorStoreParam())
        with pytest.raises(ValueError, match="without a team_id"):
            backend.remove("kg", ["e1"])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_factory_builds_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(QDRANT_URL_ENV, "http://localhost:6333")
        spec = get_backend_spec("qdrant")
        config = MagicMock()
        with patch("qdrant_client.QdrantClient", return_value=MagicMock()):
            backend = spec.factory(BackendContext(config=config, team_id="team-42"))
        assert isinstance(backend, QdrantBackend)

    def test_the_backend_takes_a_client_and_closes_none(self) -> None:
        """It holds a shared client, so it has no ``close`` at all."""
        assert not hasattr(QdrantBackend, "close")

    def test_the_factory_shares_one_client_per_cluster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two backends for one cluster hold one client, obtained through the cache."""
        # The cache ``qdrant.py`` actually writes into. Reached through the bound
        # function's globals rather than a fresh import: another spec file evicts
        # ``client`` from ``sys.modules``, so importing it again would hand back a
        # *different* module object with an empty cache.
        cache = qdrant_module.get_client.__globals__

        monkeypatch.setenv(QDRANT_URL_ENV, "http://localhost:6333")
        spec = get_backend_spec("qdrant")
        cache["close_all"]()
        made: list[MagicMock] = []

        def _new_client(**_kwargs: object) -> MagicMock:
            client = MagicMock()
            made.append(client)
            return client

        try:
            with patch("qdrant_client.QdrantClient", side_effect=_new_client):
                first = spec.factory(BackendContext(config=MagicMock(), team_id="t1"))
                second = spec.factory(BackendContext(config=MagicMock(), team_id="t2"))
            assert len(made) == 1
            assert first._client is second._client is made[0]
            assert [key.backend for key in cache["_clients"]] == ["qdrant"]
        finally:
            cache["close_all"]()

    def test_the_factory_keys_a_portless_url_on_6333(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``http://h`` and ``http://h:6333`` are one cluster, not two clients."""
        cache = qdrant_module.get_client.__globals__

        spec = get_backend_spec("qdrant")
        cache["close_all"]()
        made: list[MagicMock] = []

        def _new_client(**_kwargs: object) -> MagicMock:
            client = MagicMock()
            made.append(client)
            return client

        try:
            with patch("qdrant_client.QdrantClient", side_effect=_new_client):
                monkeypatch.setenv(QDRANT_URL_ENV, "http://qhost")
                first = spec.factory(BackendContext(config=MagicMock(), team_id="t1"))
                monkeypatch.setenv(QDRANT_URL_ENV, "http://qhost:6333")
                second = spec.factory(BackendContext(config=MagicMock(), team_id="t2"))
            assert len(made) == 1
            assert first._client is second._client
        finally:
            cache["close_all"]()

    def test_the_connect_builds_a_remote_client_only(self) -> None:
        """``url=`` is the remote path; the embedded modes are never reachable."""
        from akgentic.tool.vector_store.client import ClusterKey
        from akgentic.tool.vector_store.qdrant import _connect_qdrant

        with patch("qdrant_client.QdrantClient") as client_cls:
            _connect_qdrant(ClusterKey.from_url("qdrant", "https://q:6333", "k"))

        client_cls.assert_called_once_with(url="https://q:6333", api_key="k")
        kwargs = client_cls.call_args.kwargs
        assert "path" not in kwargs
        assert kwargs["url"] != ":memory:"

    def test_factory_raises_without_url(self) -> None:
        spec = get_backend_spec("qdrant")
        with pytest.raises(ValueError, match="qdrant_url is not configured"):
            spec.factory(BackendContext(config=MagicMock(), team_id="team-42"))


# ---------------------------------------------------------------------------
# The team predicate is a per-collection choice
# ---------------------------------------------------------------------------


def _keys(qfilter: object) -> list[str]:
    """Return the payload keys of the conditions that were sent to the cluster."""
    return [c.key for c in qfilter.must]  # type: ignore[attr-defined]


class TestTheTeamLegFollowsTheCollection:
    """Read what was *sent*: a double returns what it was told to, filter or no filter."""

    def _searching_backend(
        self, team_id: str | None = "team-42"
    ) -> tuple[QdrantBackend, MagicMock]:
        backend, client = _make_backend(team_id=team_id)
        client.query_points.return_value = SimpleNamespace(points=[])
        return backend, client

    def test_a_team_scoped_search_names_the_team(self) -> None:
        backend, client = self._searching_backend()
        backend.create_collection("planning", VectorStoreParam())

        backend.search("planning", [0.1, 0.2, 0.3], top_k=5)

        assert _keys(client.query_points.call_args[1]["query_filter"]) == ["team_id"]

    def test_a_shared_search_names_the_scope_and_not_the_team(self) -> None:
        backend, client = self._searching_backend()
        backend.create_collection("workspace_chunks", VectorStoreParam())

        backend.search("workspace_chunks", [0.1, 0.2, 0.3], top_k=5, scope="u/t")

        keys = _keys(client.query_points.call_args[1]["query_filter"])
        assert keys == ["scope"]
        assert "team_id" not in keys

    def test_a_team_scoped_removal_names_the_team(self) -> None:
        backend, client = _make_backend()
        backend.create_collection("planning", VectorStoreParam())

        backend.remove("planning", ["e1"])

        keys = _keys(client.delete.call_args[1]["points_selector"].filter)
        assert set(keys) == {"team_id", "ref_id"}

    def test_a_shared_removal_names_ref_id_and_scope_and_not_the_team(self) -> None:
        backend, client = _make_backend()
        backend.create_collection("workspace_chunks", VectorStoreParam())

        backend.remove("workspace_chunks", ["e1", "e2"], scope="u/t")

        keys = _keys(client.delete.call_args[1]["points_selector"].filter)
        assert set(keys) == {"scope", "ref_id"}
        assert "team_id" not in keys

    def test_a_shared_query_keeps_the_tenant_leg(self) -> None:
        """Tenancy is a deployment partition, orthogonal to which team wrote a row."""
        backend, client = self._searching_backend()
        backend.create_collection("workspace_chunks", VectorStoreParam(tenant="acme"))

        backend.search("workspace_chunks", [0.1, 0.2, 0.3], top_k=5, scope="u/t")

        keys = _keys(client.query_points.call_args[1]["query_filter"])
        assert set(keys) == {"tenant", "scope"}
        assert "team_id" not in keys


class TestATeamlessBackendAndTheSharedCollection:
    """No team leg means no identity to invent, so a team-less backend may query it."""

    def test_it_can_search_the_shared_collection(self) -> None:
        backend, client = _make_backend(team_id=None)
        client.query_points.return_value = SimpleNamespace(points=[])
        backend.create_collection("workspace_chunks", VectorStoreParam())

        backend.search("workspace_chunks", [0.1, 0.2, 0.3], top_k=5, scope="u/t")

        assert _keys(client.query_points.call_args[1]["query_filter"]) == ["scope"]

    def test_it_can_remove_from_the_shared_collection(self) -> None:
        backend, client = _make_backend(team_id=None)
        backend.create_collection("workspace_chunks", VectorStoreParam())

        backend.remove("workspace_chunks", ["e1"], scope="u/t")

        assert set(_keys(client.delete.call_args[1]["points_selector"].filter)) == {
            "scope",
            "ref_id",
        }

    def test_it_still_cannot_search_a_team_scoped_collection(self) -> None:
        backend, client = _make_backend(team_id=None)
        backend.create_collection("planning", VectorStoreParam())

        with pytest.raises(ValueError, match="without a team_id"):
            backend.search("planning", [0.1, 0.2, 0.3], top_k=5)
        client.query_points.assert_not_called()

    def test_it_still_cannot_remove_from_a_team_scoped_collection(self) -> None:
        backend, client = _make_backend(team_id=None)
        backend.create_collection("planning", VectorStoreParam())

        with pytest.raises(ValueError, match="without a team_id"):
            backend.remove("planning", ["e1"])
        client.delete.assert_not_called()


class TestASharedCollectionRefusesAnUnscopedQuery:
    """Scope is the shared collection's only boundary once the team leg is gone."""

    def test_search_refuses_before_reaching_the_cluster(self) -> None:
        backend, client = _make_backend()
        backend.create_collection("workspace_chunks", VectorStoreParam())

        with pytest.raises(ValueError, match="workspace_chunks") as excinfo:
            backend.search("workspace_chunks", [0.1, 0.2, 0.3], top_k=3)

        assert "scope" in str(excinfo.value)
        client.query_points.assert_not_called()

    def test_remove_refuses_before_reaching_the_cluster(self) -> None:
        backend, client = _make_backend()
        backend.create_collection("workspace_chunks", VectorStoreParam())

        with pytest.raises(ValueError, match="workspace_chunks"):
            backend.remove("workspace_chunks", ["a"])

        client.delete.assert_not_called()

    def test_a_team_scoped_collection_is_unaffected(self) -> None:
        backend, client = _make_backend()
        client.query_points.return_value = SimpleNamespace(points=[])
        backend.create_collection("planning", VectorStoreParam())

        backend.search("planning", [0.1, 0.2, 0.3], top_k=3)

        client.query_points.assert_called_once()


class TestDeleteByTeamRefusesASharedCollection:
    """A sweeper pointed at the workspace collection would reap a live team's points."""

    def test_it_refuses_before_any_cluster_call(self) -> None:
        backend, client = _make_backend()
        client.collection_exists.return_value = True

        with pytest.raises(ValueError, match="workspace_chunks"):
            backend.delete_by_team("workspace_chunks", "team-42")

        client.collection_exists.assert_not_called()
        client.delete.assert_not_called()

    def test_the_refusal_holds_on_an_administrative_backend(self) -> None:
        """No team of its own and no collection created here — the only kind a sweeper has."""
        backend, client = _make_backend(team_id=None)
        client.collection_exists.return_value = True

        with pytest.raises(ValueError, match="workspace_chunks"):
            backend.delete_by_team("workspace_chunks", "team-42")

        client.delete.assert_not_called()

    def test_a_team_scoped_collection_is_still_reaped(self) -> None:
        backend, client = _make_backend(team_id=None)
        client.collection_exists.return_value = True

        backend.delete_by_team("planning", "team-gone")

        assert _keys(client.delete.call_args[1]["points_selector"].filter) == ["team_id"]
