"""Unit tests for WeaviateBackend with mocked Weaviate client.

Covers: protocol compliance, create_collection (idempotent), add, remove,
search, multi-tenancy, team_id metadata and team-scoped cleanup, import guard,
and close().
"""

from __future__ import annotations

import inspect
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Recording Filter double
# ---------------------------------------------------------------------------


class _RecordedFilter:
    """A predicate the backend built, kept as the legs it is made of.

    The real ``Filter`` returns an opaque object, and a double that collapses
    every predicate to one sentinel string cannot tell ``ref_id`` from
    ``team_id``, cannot represent a conjunction, and passes just as happily
    against a query carrying no team predicate at all.
    """

    def __init__(self, legs: list[tuple[str, str, Any]]) -> None:
        self.legs = list(legs)

    def __and__(self, other: _RecordedFilter) -> _RecordedFilter:
        """Conjoin two predicates, keeping every leg in the order written."""
        return _RecordedFilter([*self.legs, *other.legs])

    def __repr__(self) -> str:
        return f"_RecordedFilter({self.legs!r})"


class _RecordedProperty:
    """Builder bound to one property name, as ``Filter.by_property`` returns."""

    def __init__(self, name: str) -> None:
        self._name = name

    def equal(self, value: Any) -> _RecordedFilter:
        """Record an equality leg on this property."""
        return _RecordedFilter([(self._name, "equal", value)])

    def contains_any(self, values: list[str]) -> _RecordedFilter:
        """Record a membership leg on this property."""
        return _RecordedFilter([(self._name, "contains_any", list(values))])

    def like(self, pattern: str) -> _RecordedFilter:
        """Record a wildcard leg on this property, as a path-prefix filter uses."""
        return _RecordedFilter([(self._name, "like", pattern)])


def _legs(predicate: Any) -> list[tuple[str, str, Any]]:
    """Flatten what was sent to the cluster into its recorded legs."""
    assert isinstance(predicate, _RecordedFilter), f"not a recorded predicate: {predicate!r}"
    return predicate.legs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_weaviate_module() -> MagicMock:
    """Build a mock ``weaviate`` module tree that satisfies WeaviateBackend imports."""
    mock_weaviate = MagicMock()

    # weaviate.connect_to_custom returns a mock client
    mock_client = MagicMock()
    mock_weaviate.connect_to_custom.return_value = mock_client

    # weaviate.auth.AuthApiKey
    mock_weaviate.auth = MagicMock()
    mock_weaviate.auth.AuthApiKey.return_value = MagicMock()

    # weaviate.classes.config
    mock_config = MagicMock()
    mock_config.Configure.Vectorizer.none.return_value = "none_vectorizer"
    mock_config.Configure.multi_tenancy.return_value = "multi_tenancy_config"
    mock_config.DataType.TEXT = "TEXT"
    mock_config.DataType.INT = "INT"
    mock_config.Property = MagicMock(side_effect=lambda **kw: kw)
    mock_weaviate.classes = MagicMock()
    mock_weaviate.classes.config = mock_config

    # weaviate.classes.tenants
    mock_tenants_mod = MagicMock()
    mock_tenants_mod.Tenant = MagicMock(side_effect=lambda name: f"Tenant({name})")
    mock_weaviate.classes.tenants = mock_tenants_mod

    # weaviate.classes.query
    mock_query_mod = MagicMock()
    mock_query_mod.MetadataQuery.return_value = "metadata_query"
    mock_filter = MagicMock()
    mock_filter.by_property = MagicMock(side_effect=_RecordedProperty)
    mock_query_mod.Filter = mock_filter
    mock_weaviate.classes.query = mock_query_mod

    return mock_weaviate


def _install_mock_weaviate() -> tuple[MagicMock, MagicMock]:
    """Patch sys.modules so ``import weaviate`` resolves to our mock.

    Returns (mock_weaviate_module, mock_client).
    """
    mock_weaviate = _make_mock_weaviate_module()
    mock_client = mock_weaviate.connect_to_custom.return_value

    modules = {
        "weaviate": mock_weaviate,
        "weaviate.auth": mock_weaviate.auth,
        "weaviate.classes": mock_weaviate.classes,
        "weaviate.classes.config": mock_weaviate.classes.config,
        "weaviate.classes.tenants": mock_weaviate.classes.tenants,
        "weaviate.classes.query": mock_weaviate.classes.query,
    }
    for name, mod in modules.items():
        sys.modules[name] = mod

    return mock_weaviate, mock_client


def _cleanup_weaviate_modules() -> None:
    """Remove all weaviate-related modules from sys.modules."""
    to_remove = [k for k in sys.modules if k.startswith("weaviate")]
    for k in to_remove:
        del sys.modules[k]
    # Also force-reload our module so it picks up the mock state
    backend_key = "akgentic.tool.vector_store.weaviate"
    if backend_key in sys.modules:
        del sys.modules[backend_key]


def _make_entry(
    ref_id: str = "e1",
    ref_type: str = "entity",
    text: str = "hello",
    vector: list[float] | None = None,
) -> MagicMock:
    """Return a mock VectorEntry."""
    entry = MagicMock()
    entry.ref_id = ref_id
    entry.ref_type = ref_type
    entry.text = text
    entry.vector = vector if vector is not None else [0.1, 0.2, 0.3]
    return entry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_modules() -> Any:
    """Ensure weaviate mock modules are cleaned up before and after each test."""
    _cleanup_weaviate_modules()
    yield
    _cleanup_weaviate_modules()


# ---------------------------------------------------------------------------
# Test: Import Guard (AC10)
# ---------------------------------------------------------------------------


class TestImportGuard:
    """AC10: ImportError with install instructions when weaviate-client missing."""

    def test_import_error_when_weaviate_missing(self) -> None:
        """Instantiating WeaviateBackend without weaviate-client raises ImportError."""
        # Ensure weaviate is NOT in sys.modules
        _cleanup_weaviate_modules()

        # Patch so that import weaviate fails
        with patch.dict(sys.modules, {"weaviate": None}):
            # Force reload to pick up the missing module
            if "akgentic.tool.vector_store.weaviate" in sys.modules:
                del sys.modules["akgentic.tool.vector_store.weaviate"]

            from akgentic.tool.vector_store.weaviate import WeaviateBackend

            with pytest.raises(ImportError, match="weaviate-client"):
                WeaviateBackend(url="http://localhost:8080")


# ---------------------------------------------------------------------------
# Test: create_collection (AC1, AC3)
# ---------------------------------------------------------------------------


class TestCreateCollection:
    """AC1, AC3: VectorStoreService compliance and idempotent creation."""

    def test_creates_collection_with_correct_config(self) -> None:
        """Collection created with cosine distance and no vectorizer."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        config = CollectionConfig(dimension=384)
        backend.create_collection("test_col", config)

        mock_client.collections.create.assert_called_once()
        call_kwargs = mock_client.collections.create.call_args
        assert call_kwargs[1]["name"] == "test_col" or call_kwargs[0][0] == "test_col"

    def test_idempotent_second_call(self) -> None:
        """Second call with same name is a no-op."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        config = CollectionConfig(dimension=384)
        backend.create_collection("test_col", config)

        # Now simulate exists=True
        mock_client.collections.exists.return_value = True
        mock_client.collections.create.reset_mock()
        backend.create_collection("test_col", config)

        mock_client.collections.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test: add (AC4)
# ---------------------------------------------------------------------------


class TestAdd:
    """AC4: add stores VectorEntry records with pre-populated vectors."""

    def test_add_entries_with_batch(self) -> None:
        """Entries are added via batch.dynamic() context manager."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_batch = MagicMock()
        mock_collection.batch.dynamic.return_value.__enter__ = MagicMock(return_value=mock_batch)
        mock_collection.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        backend.create_collection("col1", CollectionConfig())

        entry = _make_entry(ref_id="r1", text="test text", vector=[0.1, 0.2])
        backend.add("col1", [entry])

        mock_batch.add_object.assert_called_once()
        call_kwargs = mock_batch.add_object.call_args[1]
        assert call_kwargs["properties"]["ref_id"] == "r1"
        assert call_kwargs["vector"] == [0.1, 0.2]
        # A backend built without a team_id still writes the property, empty.
        assert call_kwargs["properties"]["team_id"] == ""

    def test_add_raises_on_unknown_collection(self) -> None:
        """add raises ValueError for non-existent collection."""
        _install_mock_weaviate()

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        entry = _make_entry()

        with pytest.raises(ValueError, match="does not exist"):
            backend.add("nonexistent", [entry])


# ---------------------------------------------------------------------------
# Test: remove (AC5)
# ---------------------------------------------------------------------------


class TestRemove:
    """AC5: remove deletes entries by ref_id filter."""

    def test_remove_by_ref_ids(self) -> None:
        """delete_many is called once, with a membership leg on the given ref_ids."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="team-42")
        backend.create_collection("col1", CollectionConfig())
        backend.remove("col1", ["id1", "id2"])

        mock_collection.data.delete_many.assert_called_once()
        where = mock_collection.data.delete_many.call_args[1]["where"]
        assert ("ref_id", "contains_any", ["id1", "id2"]) in _legs(where)

    def test_remove_is_scoped_to_the_backends_team(self) -> None:
        """delete_many carries the conjunction: the ref_ids AND the owning team."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="team-42")
        backend.create_collection("col1", CollectionConfig())
        backend.remove("col1", ["id1", "id2"])

        where = mock_collection.data.delete_many.call_args[1]["where"]
        assert _legs(where) == [
            ("ref_id", "contains_any", ["id1", "id2"]),
            ("team_id", "equal", "team-42"),
        ]

    def test_remove_raises_on_unknown_collection(self) -> None:
        """remove raises ValueError for non-existent collection."""
        _install_mock_weaviate()

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        with pytest.raises(ValueError, match="does not exist"):
            backend.remove("nonexistent", ["id1"])


# ---------------------------------------------------------------------------
# Test: search (AC6)
# ---------------------------------------------------------------------------


class TestSearch:
    """AC6: search performs cosine similarity search."""

    def test_search_returns_search_result(self) -> None:
        """near_vector query returns correctly-mapped SearchResult."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        # Mock search result objects
        mock_obj = MagicMock()
        mock_obj.properties = {"ref_type": "entity", "ref_id": "r1", "text": "hello"}
        mock_obj.metadata.distance = 0.2
        mock_result = MagicMock()
        mock_result.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_result

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="team-42")
        backend.create_collection("col1", CollectionConfig())
        result = backend.search("col1", [0.1, 0.2, 0.3], top_k=5)

        assert len(result.hits) == 1
        assert result.hits[0].ref_id == "r1"
        assert result.hits[0].score == pytest.approx(0.8)
        mock_collection.query.near_vector.assert_called_once()

    def test_search_clamps_negative_scores(self) -> None:
        """Scores are clamped to [0, 1] when distance > 1.0."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_obj = MagicMock()
        mock_obj.properties = {"ref_type": "entity", "ref_id": "r1", "text": "hello"}
        mock_obj.metadata.distance = 1.5  # distance > 1 => would produce negative score
        mock_result = MagicMock()
        mock_result.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_result

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="team-42")
        backend.create_collection("col1", CollectionConfig())
        result = backend.search("col1", [0.1, 0.2], top_k=5)

        assert result.hits[0].score == 0.0

    def test_search_is_scoped_to_the_backends_team(self) -> None:
        """near_vector carries the team predicate, so the cluster applies it before limit."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="team-42")
        backend.create_collection("col1", CollectionConfig())
        backend.search("col1", [0.1, 0.2], top_k=5)

        filters = mock_collection.query.near_vector.call_args[1]["filters"]
        assert _legs(filters) == [("team_id", "equal", "team-42")]

    def test_search_raises_on_unknown_collection(self) -> None:
        """search raises ValueError for non-existent collection."""
        _install_mock_weaviate()

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        with pytest.raises(ValueError, match="does not exist"):
            backend.search("nonexistent", [0.1], top_k=5)


# ---------------------------------------------------------------------------
# Test: Multi-tenancy (AC7)
# ---------------------------------------------------------------------------


class TestMultiTenancy:
    """AC7: tenant is passed on all operations when configured."""

    def test_collection_created_with_multi_tenancy(self) -> None:
        """Multi-tenancy config passed when tenant is set."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_col = MagicMock()
        mock_client.collections.get.return_value = mock_col

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", tenant="team-42")
        config = CollectionConfig(dimension=384)
        backend.create_collection("col1", config)

        create_kwargs = mock_client.collections.create.call_args[1]
        assert "multi_tenancy_config" in create_kwargs
        mock_col.tenants.create.assert_called_once()

    def test_operations_scoped_to_tenant(self) -> None:
        """get().with_tenant() is called for tenant-scoped backends."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_col = MagicMock()
        mock_tenant_col = MagicMock()
        mock_col.with_tenant.return_value = mock_tenant_col
        mock_client.collections.get.return_value = mock_col

        # Set up batch mock on tenant collection
        mock_batch = MagicMock()
        mock_tenant_col.batch.dynamic.return_value.__enter__ = MagicMock(return_value=mock_batch)
        mock_tenant_col.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", tenant="team-42")
        backend.create_collection("col1", CollectionConfig())

        entry = _make_entry()
        backend.add("col1", [entry])

        mock_col.with_tenant.assert_called_with("team-42")
        mock_batch.add_object.assert_called_once()

    def test_tenant_from_collection_config(self) -> None:
        """Tenant from CollectionConfig.tenant is used when backend tenant is None."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_col = MagicMock()
        mock_client.collections.get.return_value = mock_col

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")  # no tenant
        config = CollectionConfig(dimension=384, tenant="workspace-99")
        backend.create_collection("col1", config)

        create_kwargs = mock_client.collections.create.call_args[1]
        assert "multi_tenancy_config" in create_kwargs
        mock_col.tenants.create.assert_called_once()

    def test_config_tenant_scoped_on_operations(self) -> None:
        """Operations use per-collection tenant from CollectionConfig, not backend."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_col = MagicMock()
        mock_tenant_col = MagicMock()
        mock_col.with_tenant.return_value = mock_tenant_col
        mock_client.collections.get.return_value = mock_col

        # Set up batch mock on tenant collection
        mock_batch = MagicMock()
        mock_tenant_col.batch.dynamic.return_value.__enter__ = MagicMock(return_value=mock_batch)
        mock_tenant_col.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")  # no backend tenant
        config = CollectionConfig(tenant="workspace-99")
        backend.create_collection("col1", config)

        entry = _make_entry()
        backend.add("col1", [entry])

        mock_col.with_tenant.assert_called_with("workspace-99")
        mock_batch.add_object.assert_called_once()


# ---------------------------------------------------------------------------
# Test: team_id metadata and team-scoped cleanup
# ---------------------------------------------------------------------------


def _batch_for(mock_client: MagicMock) -> MagicMock:
    """Wire a batch context manager onto the mock collection and return the batch."""
    mock_collection = MagicMock()
    mock_client.collections.get.return_value = mock_collection
    mock_batch = MagicMock()
    mock_collection.batch.dynamic.return_value.__enter__ = MagicMock(return_value=mock_batch)
    mock_collection.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)
    return mock_batch


class TestTeamIdMetadata:
    """Every stored object carries the owning team's id, so a sweep can find it."""

    def test_schema_declares_team_id_property(self) -> None:
        """create_collection declares team_id alongside ref_type/ref_id/text."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="team-42")
        backend.create_collection("col1", CollectionConfig())

        properties = mock_client.collections.create.call_args[1]["properties"]
        assert {p["name"] for p in properties} == {
            "ref_type",
            "ref_id",
            "text",
            "team_id",
            "scope",
            "path",
            "ordinal",
        }

    def test_add_stamps_team_id_on_every_object(self) -> None:
        """Each batched object carries the backend's team_id."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_batch = _batch_for(mock_client)

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="team-42")
        backend.create_collection("col1", CollectionConfig())
        backend.add("col1", [_make_entry(ref_id="r1"), _make_entry(ref_id="r2")])

        assert mock_batch.add_object.call_count == 2
        for call in mock_batch.add_object.call_args_list:
            assert call[1]["properties"]["team_id"] == "team-42"

    def test_team_id_is_independent_of_tenant(self) -> None:
        """A tenant-scoped backend still stamps its own team_id, not the tenant."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_col = MagicMock()
        mock_tenant_col = MagicMock()
        mock_col.with_tenant.return_value = mock_tenant_col
        mock_client.collections.get.return_value = mock_col
        mock_batch = MagicMock()
        mock_tenant_col.batch.dynamic.return_value.__enter__ = MagicMock(return_value=mock_batch)
        mock_tenant_col.batch.dynamic.return_value.__exit__ = MagicMock(return_value=False)

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(
            url="http://localhost:8080", tenant="workspace-99", team_id="team-42"
        )
        backend.create_collection("col1", CollectionConfig())
        backend.add("col1", [_make_entry()])

        assert mock_batch.add_object.call_args[1]["properties"]["team_id"] == "team-42"


class TestProtocolCarriesNoTeam:
    """The isolation boundary lives in the backend, so no caller can omit it."""

    def test_protocol_search_and_remove_take_no_team_argument(self) -> None:
        """VectorStoreService takes no team; the boundary stays inside the backend.

        The scope and path predicates added for workspace retrieval are ordinary
        query arguments and narrow *within* a team — they are deliberately not the
        team leg, which no caller may pass and none may omit.
        """
        from akgentic.tool.vector_store.protocol import VectorStoreService

        search = inspect.signature(VectorStoreService.search)
        remove = inspect.signature(VectorStoreService.remove)

        assert list(search.parameters) == [
            "self",
            "collection",
            "query_vector",
            "top_k",
            "scope",
            "path_prefix",
        ]
        assert list(remove.parameters) == [
            "self",
            "collection",
            "ref_ids",
            "scope",
            "path_prefix",
        ]
        assert not any("team" in name for name in search.parameters)
        assert not any("team" in name for name in remove.parameters)


class TestTeamlessBackendCannotQuery:
    """A backend that does not know its team refuses to query, rather than guessing one.

    Filtering on ``""`` would not be a safe default: ``""`` is a real value in the
    data — ``add`` stamps it for a writer with no team — so a team-less query would
    silently answer *as* the unattributed team, an identity the caller never claimed.
    """

    def test_search_and_remove_refuse_without_a_team(self) -> None:
        """Both query paths raise rather than filtering on the empty team."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")  # no team_id
        backend.create_collection("col1", CollectionConfig())

        with pytest.raises(ValueError, match="without a team_id"):
            backend.search("col1", [0.1, 0.2], top_k=5)
        with pytest.raises(ValueError, match="without a team_id"):
            backend.remove("col1", ["id1"])

        mock_collection.query.near_vector.assert_not_called()
        mock_collection.data.delete_many.assert_not_called()

    def test_an_empty_string_team_is_refused_too(self) -> None:
        """`team_id=""` is not an identity; it must not slip past the guard."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_client.collections.get.return_value = MagicMock()

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="")
        backend.create_collection("col1", CollectionConfig())

        with pytest.raises(ValueError, match="without a team_id"):
            backend.search("col1", [0.1], top_k=5)

    def test_cluster_administration_still_works_without_a_team(self) -> None:
        """list_collections and delete_by_team need no team, and must stay usable.

        This is what the guard must not break: a sweeper reaping a deleted team is
        built with no team of its own.
        """
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = True
        mock_client.collections.list_all.return_value = {"planning": object()}

        mock_collection = MagicMock()
        mock_collection.data.delete_many.return_value = MagicMock(successful=4)
        mock_client.collections.get.return_value = mock_collection

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")  # no team_id

        assert backend.list_collections() == ["planning"]
        assert backend.delete_by_team("planning", "team-gone") == 4


class TestDeleteByTeam:
    """delete_by_team removes exactly the objects of one team."""

    def test_deletes_with_team_id_equality_filter(self) -> None:
        """delete_many is called with an equality filter on team_id."""
        mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_collection.data.delete_many.return_value = MagicMock(successful=7)
        mock_client.collections.get.return_value = mock_collection

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        deleted = backend.delete_by_team("col1", "team-42")

        mock_weaviate.classes.query.Filter.by_property.assert_called_with("team_id")
        where = mock_collection.data.delete_many.call_args[1]["where"]
        assert _legs(where) == [("team_id", "equal", "team-42")]
        assert deleted == 7

    def test_reaps_the_named_team_not_the_backends_own(self) -> None:
        """A sweeper deletes the team in its argument — one leg, and it is not its own."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = True

        mock_collection = MagicMock()
        mock_collection.data.delete_many.return_value = MagicMock(successful=4)
        mock_client.collections.get.return_value = mock_collection

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", team_id="team-sweeper")
        assert backend.delete_by_team("col1", "team-gone") == 4

        where = mock_collection.data.delete_many.call_args[1]["where"]
        assert _legs(where) == [("team_id", "equal", "team-gone")]

    def test_works_without_having_created_the_collection(self) -> None:
        """A sweeper never created the collection — cluster existence is what counts."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value.data.delete_many.return_value = MagicMock(
            successful=0
        )

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        assert backend.delete_by_team("never_created", "team-42") == 0

    def test_raises_when_collection_absent_from_cluster(self) -> None:
        """delete_by_team raises ValueError when the cluster has no such collection."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        with pytest.raises(ValueError, match="does not exist"):
            backend.delete_by_team("nonexistent", "team-42")

    def test_returns_zero_when_cluster_reports_nothing(self) -> None:
        """A delete result without a usable count degrades to 0, never raises."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value.data.delete_many.return_value = None

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        assert backend.delete_by_team("col1", "team-42") == 0

    def test_scoped_to_tenant_when_configured(self) -> None:
        """A tenant-scoped backend deletes inside its tenant."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = True

        mock_col = MagicMock()
        mock_tenant_col = MagicMock()
        mock_tenant_col.data.delete_many.return_value = MagicMock(successful=3)
        mock_col.with_tenant.return_value = mock_tenant_col
        mock_client.collections.get.return_value = mock_col

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080", tenant="team-42")
        assert backend.delete_by_team("col1", "team-42") == 3
        mock_col.with_tenant.assert_called_with("team-42")


class TestListCollections:
    """list_collections enumerates the cluster, not the backend's bookkeeping."""

    def test_lists_cluster_collections(self) -> None:
        """Names come from client.collections.list_all()."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.list_all.return_value = {
            "planning": object(),
            "knowledge_graph": object(),
        }

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        assert sorted(backend.list_collections()) == ["knowledge_graph", "planning"]

    def test_lists_collections_never_created_here(self) -> None:
        """A freshly-built backend reports collections it did not create."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.list_all.return_value = {"planning": object()}

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        assert backend.list_collections() == ["planning"]


# ---------------------------------------------------------------------------
# Test: close (AC2)
# ---------------------------------------------------------------------------


class TestClose:
    """close() disconnects the Weaviate client."""

    def test_close_calls_client_close(self) -> None:
        """close() delegates to client.close()."""
        _mock_weaviate, mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        backend.close()

        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Connection config (AC8)
# ---------------------------------------------------------------------------


class TestConnectionConfig:
    """AC8: Connection parameters from url and api_key."""

    def test_connects_with_api_key(self) -> None:
        """AuthApiKey is used when api_key is provided."""
        mock_weaviate, mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        WeaviateBackend(url="http://localhost:8080", api_key="test-key")

        mock_weaviate.connect_to_custom.assert_called_once()
        call_kwargs = mock_weaviate.connect_to_custom.call_args[1]
        assert call_kwargs["auth_credentials"] is not None

    def test_connects_without_api_key(self) -> None:
        """No auth when api_key is None."""
        mock_weaviate, mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        WeaviateBackend(url="http://localhost:8080")

        call_kwargs = mock_weaviate.connect_to_custom.call_args[1]
        assert call_kwargs["auth_credentials"] is None

    def test_https_url_parsed_correctly(self) -> None:
        """HTTPS URL sets http_secure and grpc_secure to True."""
        mock_weaviate, _mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        WeaviateBackend(url="https://my-cluster.weaviate.cloud:443")

        call_kwargs = mock_weaviate.connect_to_custom.call_args[1]
        assert call_kwargs["http_secure"] is True
        assert call_kwargs["grpc_secure"] is True
        assert call_kwargs["http_port"] == 443


# ---------------------------------------------------------------------------
# scope / path / ordinal — the second predicate dimension
# ---------------------------------------------------------------------------


def _entry_with(
    ref_id: str = "e1",
    scope: str | None = None,
    path: str | None = None,
    ordinal: int | None = None,
) -> MagicMock:
    """Return a mock VectorEntry carrying the workspace metadata dimension."""
    entry = _make_entry(ref_id=ref_id)
    entry.scope = scope
    entry.path = path
    entry.ordinal = ordinal
    return entry


def _scoped_backend(mock_client: MagicMock) -> Any:
    """Build a team-scoped backend with ``col1`` already created."""
    from akgentic.tool.vector_store.protocol import CollectionConfig
    from akgentic.tool.vector_store.weaviate import WeaviateBackend

    backend = WeaviateBackend(url="http://localhost:8080", team_id="team-42")
    backend.create_collection("col1", CollectionConfig())
    return backend


class TestScopeSchemaAndStamping:
    """The schema carries the three properties; add stamps only what is set."""

    def test_schema_declares_scope_path_and_ordinal(self) -> None:
        """create_collection declares the three new properties with their data types."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        _scoped_backend(mock_client)

        properties = mock_client.collections.create.call_args[1]["properties"]
        by_name = {p["name"]: p["data_type"] for p in properties}
        assert by_name["scope"] == "TEXT"
        assert by_name["path"] == "TEXT"
        assert by_name["ordinal"] == "INT"

    def test_add_stamps_the_three_when_set(self) -> None:
        """A workspace chunk writes scope, path and ordinal."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_batch = _batch_for(mock_client)

        backend = _scoped_backend(mock_client)
        backend.add(
            "col1", [_entry_with(scope="ws-1", path="docs/report.md", ordinal=3)]
        )

        props = mock_batch.add_object.call_args[1]["properties"]
        assert props["scope"] == "ws-1"
        assert props["path"] == "docs/report.md"
        assert props["ordinal"] == 3

    def test_add_omits_the_three_when_unset(self) -> None:
        """A planning or knowledge-graph entry writes exactly what it always wrote.

        A Weaviate class created before a property exists never gains it, so stamping
        a default here would ask the cluster to auto-extend the live ``Planning`` and
        ``Knowledge_graph`` schemas.
        """
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_batch = _batch_for(mock_client)

        backend = _scoped_backend(mock_client)
        backend.add("col1", [_entry_with()])

        props = mock_batch.add_object.call_args[1]["properties"]
        assert set(props) == {"ref_type", "ref_id", "text", "team_id"}

    def test_add_stamps_each_field_independently(self) -> None:
        """Setting only ``scope`` writes only ``scope``."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_batch = _batch_for(mock_client)

        backend = _scoped_backend(mock_client)
        backend.add("col1", [_entry_with(scope="ws-1")])

        props = mock_batch.add_object.call_args[1]["properties"]
        assert props["scope"] == "ws-1"
        assert "path" not in props
        assert "ordinal" not in props


class TestScopeAndPathPredicatesReachTheCluster:
    """Both predicates go to the cluster, conjoined with the team leg."""

    def _empty_search_backend(self) -> tuple[Any, MagicMock]:
        """Return (backend, mock_collection) with an empty near_vector result."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])
        return _scoped_backend(mock_client), mock_collection

    def test_search_conjoins_scope_with_the_team_leg(self) -> None:
        """The scope leg is added to the team predicate, never instead of it."""
        backend, mock_collection = self._empty_search_backend()

        backend.search("col1", [0.1, 0.2], top_k=5, scope="ws-1")

        sent = mock_collection.query.near_vector.call_args[1]["filters"]
        assert _legs(sent) == [
            ("team_id", "equal", "team-42"),
            ("scope", "equal", "ws-1"),
        ]

    def test_search_path_prefix_is_a_wildcard_leg(self) -> None:
        """path_prefix becomes a like('prefix*') leg on the path property."""
        backend, mock_collection = self._empty_search_backend()

        backend.search("col1", [0.1, 0.2], top_k=5, path_prefix="docs/")

        sent = mock_collection.query.near_vector.call_args[1]["filters"]
        assert _legs(sent) == [
            ("team_id", "equal", "team-42"),
            ("path", "like", "docs/*"),
        ]

    def test_search_conjoins_both_predicates(self) -> None:
        """scope and path_prefix are both applied, together with the team leg."""
        backend, mock_collection = self._empty_search_backend()

        backend.search("col1", [0.1, 0.2], top_k=5, scope="ws-1", path_prefix="docs/")

        sent = mock_collection.query.near_vector.call_args[1]["filters"]
        assert _legs(sent) == [
            ("team_id", "equal", "team-42"),
            ("scope", "equal", "ws-1"),
            ("path", "like", "docs/*"),
        ]

    def test_predicates_go_as_filters_so_top_k_is_honoured_after_filtering(self) -> None:
        """They travel as ``filters=`` beside ``limit``, so the cluster applies them first."""
        backend, mock_collection = self._empty_search_backend()

        backend.search("col1", [0.1, 0.2], top_k=7, scope="ws-1")

        kwargs = mock_collection.query.near_vector.call_args[1]
        assert kwargs["limit"] == 7
        assert "filters" in kwargs

    def test_search_without_predicates_sends_only_the_team_leg(self) -> None:
        """The default is exactly the predicate this backend has always applied."""
        backend, mock_collection = self._empty_search_backend()

        backend.search("col1", [0.1, 0.2], top_k=5)

        sent = mock_collection.query.near_vector.call_args[1]["filters"]
        assert _legs(sent) == [("team_id", "equal", "team-42")]

    def test_remove_by_scope_is_one_conjoined_delete_many(self) -> None:
        """ref-ids, the team predicate and the scope predicate travel together."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        backend = _scoped_backend(mock_client)
        backend.remove("col1", ["r1", "r2"], scope="ws-1")

        mock_collection.data.delete_many.assert_called_once()
        sent = mock_collection.data.delete_many.call_args[1]["where"]
        assert _legs(sent) == [
            ("ref_id", "contains_any", ["r1", "r2"]),
            ("team_id", "equal", "team-42"),
            ("scope", "equal", "ws-1"),
        ]

    def test_remove_without_predicates_is_unchanged(self) -> None:
        """The existing two-leg removal is exactly what it was."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        backend = _scoped_backend(mock_client)
        backend.remove("col1", ["r1"])

        sent = mock_collection.data.delete_many.call_args[1]["where"]
        assert _legs(sent) == [
            ("ref_id", "contains_any", ["r1"]),
            ("team_id", "equal", "team-42"),
        ]

    def test_a_teamless_backend_still_cannot_query_with_a_scope(self) -> None:
        """A scope argument never substitutes for the team leg."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.protocol import CollectionConfig
        from akgentic.tool.vector_store.weaviate import WeaviateBackend

        backend = WeaviateBackend(url="http://localhost:8080")
        backend.create_collection("col1", CollectionConfig())

        with pytest.raises(ValueError, match="without a team_id"):
            backend.search("col1", [0.1], top_k=5, scope="ws-1")


class TestScopeReadBackOntoHits:
    """search reads the three properties back onto the SearchHit when present."""

    def _search_returning(self, properties: dict[str, Any]) -> Any:
        """Run a search against a single object carrying *properties*."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_obj = MagicMock()
        mock_obj.properties = properties
        mock_obj.metadata.distance = 0.2
        mock_collection.query.near_vector.return_value = MagicMock(objects=[mock_obj])

        backend = _scoped_backend(mock_client)
        return backend.search("col1", [0.1, 0.2], top_k=5)

    def test_hit_carries_the_three_fields(self) -> None:
        """A workspace chunk comes back with its scope, path and ordinal."""
        result = self._search_returning(
            {
                "ref_type": "chunk",
                "ref_id": "r1",
                "text": "hello",
                "scope": "ws-1",
                "path": "docs/report.md",
                "ordinal": 3,
            }
        )
        hit = result.hits[0]
        assert hit.scope == "ws-1"
        assert hit.path == "docs/report.md"
        assert hit.ordinal == 3

    def test_hit_carries_none_when_the_class_has_no_such_property(self) -> None:
        """A pre-existing Planning object reads back as three Nones, not 'None'."""
        result = self._search_returning(
            {"ref_type": "entity", "ref_id": "r1", "text": "hello"}
        )
        hit = result.hits[0]
        assert hit.scope is None
        assert hit.path is None
        assert hit.ordinal is None
