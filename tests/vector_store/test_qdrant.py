"""QdrantBackend — protocol conformance and team-scoped query construction.

The Qdrant client is mocked: these tests assert the backend builds the right
requests (points, filters, scope) and maps responses, not that a live cluster
behaves.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.vector_store.protocol import CollectionConfig, VectorQuery
from akgentic.tool.vector_store.qdrant import (
    QDRANT_URL_ENV,
    QdrantBackend,
    require_qdrant_configured,
)
from akgentic.tool.vector_store.registry import BackendContext, get_backend_spec
from akgentic.tool.vector_store.vector import VectorEntry


def _make_backend(team_id: str | None = "team-42") -> tuple[QdrantBackend, MagicMock]:
    """Build a QdrantBackend whose client is a MagicMock."""
    client = MagicMock()
    client.collection_exists.return_value = False
    with patch("qdrant_client.QdrantClient", return_value=client):
        backend = QdrantBackend(url="http://localhost:6333", team_id=team_id)
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
        backend.create_collection("kg", CollectionConfig(dimension=384))
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
        backend.create_collection("kg", CollectionConfig())
        client.create_collection.assert_not_called()

    def test_rejects_non_cosine_distance(self) -> None:
        from qdrant_client import models

        backend, _client = _make_backend()
        with pytest.raises(ValueError, match="only cosine distance"):
            backend.create_collection(
                "kg",
                CollectionConfig(params={"distance": models.Distance.EUCLID}),
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
            backend.create_collection("kg", CollectionConfig())


# ---------------------------------------------------------------------------
# Add / remove
# ---------------------------------------------------------------------------


class TestAdd:
    def test_stamps_team_id_on_every_point(self) -> None:
        backend, client = _make_backend(team_id="team-42")
        backend.create_collection("kg", CollectionConfig())
        backend.add("kg", [_entry("e1"), _entry("e2")])
        points = client.upsert.call_args[1]["points"]
        assert len(points) == 2
        assert all(p.payload["team_id"] == "team-42" for p in points)
        assert all(p.payload["ref_id"] in {"e1", "e2"} for p in points)

    def test_point_ids_are_scoped_by_team(self) -> None:
        first, first_client = _make_backend(team_id="team-a")
        second, second_client = _make_backend(team_id="team-b")
        for backend in (first, second):
            backend.create_collection("planning", CollectionConfig())

        first.add("planning", [_entry("3")])
        second.add("planning", [_entry("3")])

        first_id = first_client.upsert.call_args[1]["points"][0].id
        second_id = second_client.upsert.call_args[1]["points"][0].id
        assert first_id != second_id

    def test_point_ids_are_scoped_by_collection_tenant(self) -> None:
        first, first_client = _make_backend(team_id="team-a")
        second, second_client = _make_backend(team_id="team-a")
        first.create_collection("planning", CollectionConfig(tenant="tenant-a"))
        second.create_collection("planning", CollectionConfig(tenant="tenant-b"))

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
        backend.create_collection("kg", CollectionConfig())
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
        backend.create_collection("kg", CollectionConfig())
        client.query_points.return_value = self._response()

        result = backend.search("kg", [0.1, 0.2, 0.3], top_k=5)

        assert len(result.hits) == 1
        assert result.hits[0].ref_id == "e1"
        assert result.hits[0].score == pytest.approx(0.9)
        query_filter = client.query_points.call_args[1]["query_filter"]
        assert any(c.key == "team_id" for c in query_filter.must)

    def test_query_filters_are_anded_onto_team_scope(self) -> None:
        backend, client = _make_backend(team_id="team-42")
        backend.create_collection("kg", CollectionConfig())
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
        backend.create_collection("kg", CollectionConfig())
        client.query_points.return_value = self._response()

        backend.search(
            "kg", [0.1, 0.2, 0.3], top_k=5, query=VectorQuery(params={"timeout": 3})
        )
        assert client.query_points.call_args[1]["timeout"] == 3


class TestTeamlessBackendCannotQuery:
    """A backend with no team refuses to search/remove rather than guessing one."""

    def test_search_refuses(self) -> None:
        backend, client = _make_backend(team_id=None)
        backend.create_collection("kg", CollectionConfig())
        with pytest.raises(ValueError, match="without a team_id"):
            backend.search("kg", [0.1, 0.2, 0.3], top_k=5)

    def test_remove_refuses(self) -> None:
        backend, _client = _make_backend(team_id=None)
        backend.create_collection("kg", CollectionConfig())
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

    def test_factory_raises_without_url(self) -> None:
        spec = get_backend_spec("qdrant")
        with pytest.raises(ValueError, match="qdrant_url is not configured"):
            spec.factory(BackendContext(config=MagicMock(), team_id="team-42"))
