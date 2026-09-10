"""Unit tests for WeaviateBackend with mocked Weaviate client.

Covers: protocol compliance, create_collection (idempotent), add, remove,
search, multi-tenancy, team_id metadata and team-scoped cleanup, and the
client-taking constructor (the shared client, separate bookkeeping, and the
per-call batch context). The client cache itself is specified in
``test_client.py``.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.vector_store.backends.weaviate import WEAVIATE_API_KEY_ENV, WEAVIATE_URL_ENV

if TYPE_CHECKING:
    # Only under TYPE_CHECKING: at runtime the mock ``weaviate`` module must be
    # installed into ``sys.modules`` before this module is imported.
    from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
    from akgentic.tool.vector_store.protocol import VectorStoreConfig

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
    """Close the client cache, then evict it, the backend and every ``weaviate*`` module.

    The cache is process-global: without ``close_all()`` first, a client cached
    under ``localhost:8080`` by one test is handed to the next, whose mock then
    records no connect. Evicting ``client`` too makes the next import build a
    fresh dict against the fresh mock.
    """
    client_key = "akgentic.tool.vector_store.client"
    loaded = sys.modules.get(client_key)
    if loaded is not None:
        loaded.close_all()
    to_remove = [k for k in sys.modules if k.startswith("weaviate")]
    for k in to_remove:
        del sys.modules[k]
    # Also force-reload our modules so they pick up the mock state
    for key in (client_key, "akgentic.tool.vector_store.backends.weaviate"):
        sys.modules.pop(key, None)


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


_EVICTED_PACKAGE_MODULES = (
    "akgentic.tool.vector_store.client",
    "akgentic.tool.vector_store.backends.weaviate",
)


@pytest.fixture(autouse=True)
def _clean_modules() -> Any:
    """Evict the weaviate modules around each test, then put the originals back.

    Each test needs a fresh import of the backend against its own mock client, so the
    modules are evicted before and after it. Putting the evicted originals back — the
    ``sys.modules`` entries, their parent-package attributes, and the registry's
    ``weaviate`` spec, which the fresh import replaced — keeps the eviction from leaking:
    without it, every later test would see the package root's ``WeaviateBackend`` differ
    from the class the registry builds, and an identity check would depend on file order.
    """
    from akgentic.tool.vector_store.registry import get_backend_spec, register_backend

    saved = {
        key: module
        for key, module in sys.modules.items()
        if key.startswith("weaviate") or key in _EVICTED_PACKAGE_MODULES
    }
    saved_spec = get_backend_spec("weaviate")
    _cleanup_weaviate_modules()
    yield
    _cleanup_weaviate_modules()
    sys.modules.update(saved)
    for key in _EVICTED_PACKAGE_MODULES:
        parent, _, child = key.rpartition(".")
        if key in saved and parent in sys.modules:
            setattr(sys.modules[parent], child, saved[key])
    register_backend(saved_spec, replace=True)


# ---------------------------------------------------------------------------
# Test: create_collection (AC1, AC3)
# ---------------------------------------------------------------------------


class TestCreateCollection:
    """AC1, AC3: VectorStoreService compliance and idempotent creation."""

    def test_creates_collection_with_correct_config(self) -> None:
        """Collection created with cosine distance and no vectorizer."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client)
        config = VectorStoreParam(dimension=384)
        backend.create_collection("test_col", config)

        mock_client.collections.create.assert_called_once()
        call_kwargs = mock_client.collections.create.call_args
        assert call_kwargs[1]["name"] == "test_col" or call_kwargs[0][0] == "test_col"

    def test_idempotent_second_call(self) -> None:
        """Second call with same name is a no-op."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client)
        config = VectorStoreParam(dimension=384)
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client)
        backend.create_collection("col1", VectorStoreParam())

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
        _mock_weaviate, mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, team_id="team-42")
        backend.create_collection("col1", VectorStoreParam())
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, team_id="team-42")
        backend.create_collection("col1", VectorStoreParam())
        backend.remove("col1", ["id1", "id2"])

        where = mock_collection.data.delete_many.call_args[1]["where"]
        assert _legs(where) == [
            ("ref_id", "contains_any", ["id1", "id2"]),
            ("team_id", "equal", "team-42"),
        ]

    def test_remove_raises_on_unknown_collection(self) -> None:
        """remove raises ValueError for non-existent collection."""
        _mock_weaviate, mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, team_id="team-42")
        backend.create_collection("col1", VectorStoreParam())
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, team_id="team-42")
        backend.create_collection("col1", VectorStoreParam())
        result = backend.search("col1", [0.1, 0.2], top_k=5)

        assert result.hits[0].score == 0.0

    def test_search_is_scoped_to_the_backends_team(self) -> None:
        """near_vector carries the team predicate, so the cluster applies it before limit."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, team_id="team-42")
        backend.create_collection("col1", VectorStoreParam())
        backend.search("col1", [0.1, 0.2], top_k=5)

        filters = mock_collection.query.near_vector.call_args[1]["filters"]
        assert _legs(filters) == [("team_id", "equal", "team-42")]

    def test_search_raises_on_unknown_collection(self) -> None:
        """search raises ValueError for non-existent collection."""
        _mock_weaviate, mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, tenant="team-42")
        config = VectorStoreParam(dimension=384)
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, tenant="team-42")
        backend.create_collection("col1", VectorStoreParam())

        entry = _make_entry()
        backend.add("col1", [entry])

        mock_col.with_tenant.assert_called_with("team-42")
        mock_batch.add_object.assert_called_once()

    def test_tenant_from_collection_config(self) -> None:
        """Tenant from VectorStoreParam.tenant is used when backend tenant is None."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_col = MagicMock()
        mock_client.collections.get.return_value = mock_col

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client)  # no tenant
        config = VectorStoreParam(dimension=384, tenant="workspace-99")
        backend.create_collection("col1", config)

        create_kwargs = mock_client.collections.create.call_args[1]
        assert "multi_tenancy_config" in create_kwargs
        mock_col.tenants.create.assert_called_once()

    def test_config_tenant_scoped_on_operations(self) -> None:
        """Operations use per-collection tenant from VectorStoreParam, not backend."""
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client)  # no backend tenant
        config = VectorStoreParam(tenant="workspace-99")
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, team_id="team-42")
        backend.create_collection("col1", VectorStoreParam())

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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, team_id="team-42")
        backend.create_collection("col1", VectorStoreParam())
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, tenant="workspace-99", team_id="team-42")
        backend.create_collection("col1", VectorStoreParam())
        backend.add("col1", [_make_entry()])

        assert mock_batch.add_object.call_args[1]["properties"]["team_id"] == "team-42"


class TestProtocolCarriesNoTeam:
    """The isolation boundary lives in the backend, so no caller can omit it."""

    def test_protocol_search_and_remove_take_no_team_argument(self) -> None:
        """VectorStoreService takes no team; the boundary stays inside the backend.

        The scope and path predicates added for workspace retrieval, and the
        optional ``query`` refinement, are ordinary query arguments and narrow
        *within* a team — they are deliberately not the team leg, which no caller
        may pass and none may omit.
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
            "query",
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client)  # no team_id
        backend.create_collection("col1", VectorStoreParam())

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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client, team_id="")
        backend.create_collection("col1", VectorStoreParam())

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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)  # no team_id

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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client, team_id="team-sweeper")
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
        assert backend.delete_by_team("never_created", "team-42") == 0

    def test_raises_when_collection_absent_from_cluster(self) -> None:
        """delete_by_team raises ValueError when the cluster has no such collection."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
        with pytest.raises(ValueError, match="does not exist"):
            backend.delete_by_team("nonexistent", "team-42")

    def test_returns_zero_when_cluster_reports_nothing(self) -> None:
        """A delete result without a usable count degrades to 0, never raises."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value.data.delete_many.return_value = None

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client, tenant="team-42")
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
        assert sorted(backend.list_collections()) == ["knowledge_graph", "planning"]

    def test_lists_collections_never_created_here(self) -> None:
        """A freshly-built backend reports collections it did not create."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.list_all.return_value = {"planning": object()}

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)
        assert backend.list_collections() == ["planning"]


# ---------------------------------------------------------------------------
# The backend takes a client and never connects (AC 9, AC 13)
# ---------------------------------------------------------------------------


class TestBackendTakesAClient:
    """The constructor takes a ``WeaviateClient``; the connection is not its business."""

    def test_construction_does_not_connect(self) -> None:
        """A backend built with a client opens nothing of its own."""
        mock_weaviate, mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client, team_id="team-42")

        mock_weaviate.connect_to_custom.assert_not_called()
        assert backend._client is mock_client

    def test_url_and_api_key_left_the_signature(self) -> None:
        """The constructor is (client, tenant, team_id) and nothing else."""
        _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        params = inspect.signature(WeaviateBackend.__init__).parameters
        assert list(params) == ["self", "client", "tenant", "team_id"]

    def test_the_backend_has_no_close(self) -> None:
        """It does not own the connection, so it cannot close it under the other consumers."""
        _mock_weaviate, mock_client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=mock_client)

        assert not hasattr(backend, "close")
        mock_client.close.assert_not_called()


# ---------------------------------------------------------------------------
# Two backends, one client, separate bookkeeping (AC 10)
# ---------------------------------------------------------------------------


def _config() -> VectorStoreConfig:
    """Return a ``VectorStoreConfig`` for the factory."""
    from akgentic.tool.vector_store.actor import VS_ACTOR_NAME, VS_ACTOR_ROLE
    from akgentic.tool.vector_store.protocol import VectorStoreConfig

    return VectorStoreConfig(name=VS_ACTOR_NAME, role=VS_ACTOR_ROLE)


def _two_backends_for_one_cluster() -> tuple[Any, Any]:
    """Build backends for teams ``a`` and ``b`` through the registry factory."""
    from akgentic.tool.vector_store.backends.weaviate import _make_weaviate_backend
    from akgentic.tool.vector_store.registry import BackendContext

    config = _config()
    first = _make_weaviate_backend(BackendContext(config=config, team_id="team-a"))
    second = _make_weaviate_backend(BackendContext(config=config, team_id="team-b"))
    return first, second


class TestTwoBackendsShareOneClient:
    """One cluster, one client, and each backend keeps its own created-collections cache."""

    def test_the_client_is_the_same_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both factory calls resolve the exported cluster to one connect."""
        monkeypatch.setenv(WEAVIATE_URL_ENV, "http://localhost:8080")
        mock_weaviate, mock_client = _install_mock_weaviate()

        first, second = _two_backends_for_one_cluster()

        assert first._client is mock_client
        assert second._client is mock_client
        mock_weaviate.connect_to_custom.assert_called_once()
        assert (first._team_id, second._team_id) == ("team-a", "team-b")

    def test_bookkeeping_is_per_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Creating through one backend does not mark the collection created on the other."""
        monkeypatch.setenv(WEAVIATE_URL_ENV, "http://localhost:8080")
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.protocol import VectorStoreParam

        first, second = _two_backends_for_one_cluster()
        first.create_collection("col1", VectorStoreParam())

        assert first._collections_created is not second._collections_created
        assert first._collection_tenants is not second._collection_tenants
        assert "col1" in first._collections_created
        assert "col1" not in second._collections_created
        with pytest.raises(ValueError, match="does not exist"):
            second.add("col1", [_make_entry()])

    def test_the_second_backend_records_an_existing_collection_locally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cost of separate bookkeeping is one existence check, not a second create."""
        monkeypatch.setenv(WEAVIATE_URL_ENV, "http://localhost:8080")
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False

        from akgentic.tool.vector_store.protocol import VectorStoreParam

        first, second = _two_backends_for_one_cluster()
        first.create_collection("col1", VectorStoreParam())
        mock_client.collections.create.assert_called_once()

        mock_client.collections.exists.return_value = True
        mock_client.collections.create.reset_mock()
        second.create_collection("col1", VectorStoreParam())

        mock_client.collections.create.assert_not_called()
        assert "col1" in second._collections_created


# ---------------------------------------------------------------------------
# The factory reads the environment only, then hands the pair to get_client (AC 11)
# ---------------------------------------------------------------------------


def _config_carrying_the_removed_connection_keys() -> VectorStoreConfig:
    """A ``VectorStoreConfig`` validated from a record that still names a cluster.

    ``weaviate_url`` and ``weaviate_api_key`` are not fields any more, so validation drops
    them. The record is what a hand-built or stored config might still carry.
    """
    from akgentic.tool.vector_store.protocol import VectorStoreConfig

    return VectorStoreConfig.model_validate(
        {
            "name": "#VectorStore",
            "role": "ToolActor",
            "weaviate_url": "http://from-config:8080",
            "weaviate_api_key": "config-key",
        }
    )


class TestFactoryResolution:
    """The environment is the only source; ValueError when it names no cluster."""

    def test_a_config_carrying_connection_keys_does_not_outrank_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exported pair goes to get_client, whatever the config record carried."""
        monkeypatch.setenv(WEAVIATE_URL_ENV, "http://from-env:8080")
        monkeypatch.setenv(WEAVIATE_API_KEY_ENV, "env-key")
        _install_mock_weaviate()

        import akgentic.tool.vector_store.backends.weaviate as weaviate_module
        from akgentic.tool.vector_store.registry import BackendContext

        config = _config_carrying_the_removed_connection_keys()
        with patch.object(weaviate_module, "_weaviate_client") as get_client:
            backend = weaviate_module._make_weaviate_backend(
                BackendContext(config=config, team_id="team-42")
            )

        get_client.assert_called_once_with("http://from-env:8080", "env-key")
        assert backend._client is get_client.return_value
        assert backend._team_id == "team-42"

    def test_a_config_carrying_connection_keys_is_no_cluster_without_the_environment(
        self,
    ) -> None:
        """With nothing exported, a URL on the config record does not build a backend."""
        _install_mock_weaviate()

        import akgentic.tool.vector_store.backends.weaviate as weaviate_module
        from akgentic.tool.vector_store.registry import BackendContext

        config = _config_carrying_the_removed_connection_keys()
        with (
            patch.object(weaviate_module, "_weaviate_client") as get_client,
            pytest.raises(ValueError, match=WEAVIATE_URL_ENV),
        ):
            weaviate_module._make_weaviate_backend(BackendContext(config=config))

        get_client.assert_not_called()

    def test_the_environment_is_the_only_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A config carrying no connection settings resolves the exported cluster."""
        monkeypatch.setenv(WEAVIATE_URL_ENV, "http://from-env:8080")
        monkeypatch.setenv(WEAVIATE_API_KEY_ENV, "env-key")
        _install_mock_weaviate()

        import akgentic.tool.vector_store.backends.weaviate as weaviate_module
        from akgentic.tool.vector_store.registry import BackendContext

        with patch.object(weaviate_module, "_weaviate_client") as get_client:
            weaviate_module._make_weaviate_backend(BackendContext(config=_config()))

        get_client.assert_called_once_with("http://from-env:8080", "env-key")

    def test_no_url_anywhere_is_a_value_error(self) -> None:
        """The environment names no cluster: raise before touching the cache."""
        _install_mock_weaviate()

        import akgentic.tool.vector_store.backends.weaviate as weaviate_module
        from akgentic.tool.vector_store.registry import BackendContext

        with (
            patch.object(weaviate_module, "_weaviate_client") as get_client,
            pytest.raises(ValueError, match=WEAVIATE_URL_ENV),
        ):
            weaviate_module._make_weaviate_backend(BackendContext(config=_config()))

        get_client.assert_not_called()


# ---------------------------------------------------------------------------
# add() opens a fresh batch context per call and stores none (AC 12)
# ---------------------------------------------------------------------------


def _recording_batch_contexts(
    mock_client: MagicMock, enter_delay: float = 0.0
) -> tuple[MagicMock, list[MagicMock]]:
    """Make ``batch.dynamic()`` mint a new context per call and record each one."""
    contexts: list[MagicMock] = []

    def dynamic() -> MagicMock:
        context = MagicMock(name="batch-context")
        batch = MagicMock(name="batch")

        def enter() -> MagicMock:
            time.sleep(enter_delay)
            return batch

        context.__enter__ = MagicMock(side_effect=enter)
        context.__exit__ = MagicMock(return_value=False)
        contexts.append(context)
        return context

    mock_collection = MagicMock()
    mock_collection.batch.dynamic.side_effect = dynamic
    mock_client.collections.get.return_value = mock_collection
    return mock_collection, contexts


class TestBatchContextPerCall:
    """Invariant 1 of the thread-safety answer: a batch context never escapes ``add()``."""

    def test_two_adds_open_two_contexts_and_store_none(self) -> None:
        """Each add opens its own context and leaves it before returning.

        The **handle** is reused across the two adds (see the handle-cache specs);
        the **batch context** is not, and that is the invariant this pins.
        """
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_collection, contexts = _recording_batch_contexts(mock_client)

        backend = _scoped_backend(mock_client)
        backend.add("col1", [_make_entry(ref_id="r1")])
        backend.add("col1", [_make_entry(ref_id="r2")])

        assert mock_collection.batch.dynamic.call_count == 2
        assert len(contexts) == 2
        assert contexts[0] is not contexts[1]
        for context in contexts:
            context.__enter__.assert_called_once()
            context.__exit__.assert_called_once()

        # No batch object survives the call: what the backend holds is the
        # client and its cached handles, never a batch.
        assert not any(context is value for value in vars(backend).values() for context in contexts)

    def test_concurrent_adds_each_enter_their_own_context(self) -> None:
        """Two threads inside add() at once hold two distinct contexts, never one."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        _mock_collection, contexts = _recording_batch_contexts(mock_client, enter_delay=0.05)

        backend = _scoped_backend(mock_client)
        barrier = threading.Barrier(2)

        def worker(ref_id: str) -> None:
            barrier.wait()
            backend.add("col1", [_make_entry(ref_id=ref_id)])

        threads = [threading.Thread(target=worker, args=(ref_id,)) for ref_id in ("r1", "r2")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert len(contexts) == 2
        assert contexts[0] is not contexts[1]
        for context in contexts:
            context.__enter__.assert_called_once()
            context.__exit__.assert_called_once()


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
    from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
    from akgentic.tool.vector_store.protocol import VectorStoreParam

    backend = WeaviateBackend(client=mock_client, team_id="team-42")
    backend.create_collection("col1", VectorStoreParam())
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

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = WeaviateBackend(client=mock_client)
        backend.create_collection("col1", VectorStoreParam())

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


class TestPathPrefixWildcardsAreRefused:
    """A ``path_prefix`` carrying ``*`` or ``?`` never reaches the cluster.

    ``_query_filter`` builds ``like(f"{path_prefix}*")``, and ``*`` and ``?`` are
    both wildcards to ``Like`` with no escape in the v4 filter API — while the
    in-memory backend reads them literally through ``str.startswith``. The same
    query would mean two different things depending on where the collection lives,
    so both backends refuse them with the same sentence (ADR-045 §5). On
    ``remove()`` that is the sharp case: unguarded, a ``*`` widens the deletion
    here and narrows it to nothing there.
    """

    def _backend_and_collection(self) -> tuple[Any, MagicMock]:
        """Return (team-scoped backend, mock collection) with ``col1`` created."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])
        return _scoped_backend(mock_client), mock_collection

    @pytest.mark.parametrize("prefix", ["docs/*", "docs/?ne.md", "*", "?"])
    def test_search_refuses_a_wildcard_prefix(self, prefix: str) -> None:
        """The refusal happens before ``near_vector`` is called at all."""
        backend, mock_collection = self._backend_and_collection()

        with pytest.raises(ValueError, match="path_prefix cannot contain"):
            backend.search("col1", [0.1, 0.2], top_k=5, path_prefix=prefix)

        mock_collection.query.near_vector.assert_not_called()

    @pytest.mark.parametrize("prefix", ["docs/*", "docs/?ne.md", "*", "?"])
    def test_remove_refuses_a_wildcard_prefix(self, prefix: str) -> None:
        """No ``delete_many`` is issued, so nothing is widened on the cluster."""
        backend, mock_collection = self._backend_and_collection()

        with pytest.raises(ValueError, match="path_prefix cannot contain"):
            backend.remove("col1", ["r1"], path_prefix=prefix)

        mock_collection.data.delete_many.assert_not_called()

    def test_the_message_is_the_one_both_backends_share(self) -> None:
        """Identical wording to the in-memory backend and to the workspace answer."""
        from akgentic.tool.vector_store.protocol import PATH_PREFIX_REJECTED

        backend, _mock_collection = self._backend_and_collection()

        with pytest.raises(ValueError) as excinfo:
            backend.remove("col1", ["r1"], path_prefix="docs/*")
        assert str(excinfo.value) == PATH_PREFIX_REJECTED

    def test_a_clean_prefix_still_builds_its_like_leg(self) -> None:
        """The guard refuses two characters and changes nothing else."""
        backend, mock_collection = self._backend_and_collection()

        backend.search("col1", [0.1, 0.2], top_k=5, path_prefix="docs/")

        sent = mock_collection.query.near_vector.call_args[1]["filters"]
        assert _legs(sent) == [
            ("team_id", "equal", "team-42"),
            ("path", "like", "docs/*"),
        ]


# ---------------------------------------------------------------------------
# The connect callable and the dependency guard came back from client.py
# ---------------------------------------------------------------------------


class TestTheConnectCallable:
    """``client.py`` names no vendor; this module owns the keyword set."""

    def test_connects_with_the_pinned_keyword_set(self) -> None:
        """The whole keyword set, including the fixed gRPC port."""
        mock_weaviate, _client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import _connect_weaviate
        from akgentic.tool.vector_store.client import ClusterKey

        _connect_weaviate(ClusterKey.from_url("weaviate", "http://localhost:8080"))

        assert mock_weaviate.connect_to_custom.call_args[1] == {
            "http_host": "localhost",
            "http_port": 8080,
            "http_secure": False,
            "grpc_host": "localhost",
            "grpc_port": 50051,
            "grpc_secure": False,
            "auth_credentials": None,
        }

    def test_https_is_secure_on_both_transports(self) -> None:
        mock_weaviate, _client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import _connect_weaviate
        from akgentic.tool.vector_store.client import ClusterKey

        _connect_weaviate(ClusterKey.from_url("weaviate", "https://my-cluster.weaviate.cloud"))

        kwargs = mock_weaviate.connect_to_custom.call_args[1]
        assert kwargs["http_secure"] is True
        assert kwargs["grpc_secure"] is True
        assert kwargs["http_port"] == 443

    def test_an_api_key_becomes_auth_api_key(self) -> None:
        mock_weaviate, _client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import _connect_weaviate
        from akgentic.tool.vector_store.client import ClusterKey

        _connect_weaviate(ClusterKey.from_url("weaviate", "http://localhost:8080", "test-key"))

        mock_weaviate.auth.AuthApiKey.assert_called_once_with("test-key")
        kwargs = mock_weaviate.connect_to_custom.call_args[1]
        assert kwargs["auth_credentials"] is mock_weaviate.auth.AuthApiKey.return_value

    def test_no_api_key_means_no_auth(self) -> None:
        """An empty key is unauthenticated, not authenticated with ``''``."""
        mock_weaviate, _client = _install_mock_weaviate()

        from akgentic.tool.vector_store.backends.weaviate import _connect_weaviate
        from akgentic.tool.vector_store.client import ClusterKey

        _connect_weaviate(ClusterKey.from_url("weaviate", "http://localhost:8080", ""))

        assert mock_weaviate.connect_to_custom.call_args[1]["auth_credentials"] is None
        mock_weaviate.auth.AuthApiKey.assert_not_called()

    def test_the_shared_client_is_keyed_on_the_weaviate_backend_name(self) -> None:
        """``_weaviate_client`` builds a ``weaviate``-keyed ClusterKey and caches on it."""
        mock_weaviate, _client = _install_mock_weaviate()

        import akgentic.tool.vector_store.client as client_module
        from akgentic.tool.vector_store.backends.weaviate import _weaviate_client

        first = _weaviate_client("http://localhost:8080", "k")
        second = _weaviate_client("http://LOCALHOST:8080/", "k")

        assert second is first
        assert mock_weaviate.connect_to_custom.call_count == 1
        assert [key.backend for key in client_module._clients] == ["weaviate"]


class TestTheDependencyGuard:
    """``_check_weaviate_dependencies`` and its message live here, beside qdrant's."""

    def test_both_names_resolve_from_this_module(self) -> None:
        from akgentic.tool.vector_store.backends import weaviate as weaviate_module

        assert callable(weaviate_module._check_weaviate_dependencies)
        assert "akgentic-tool[weaviate]" in weaviate_module.WEAVIATE_MISSING_MESSAGE

    def test_import_error_when_weaviate_missing(self) -> None:
        """The connect raises ImportError naming the extra to install."""
        from akgentic.tool.vector_store.backends.weaviate import _connect_weaviate
        from akgentic.tool.vector_store.client import ClusterKey

        with (
            patch.dict(sys.modules, {"weaviate": None}),
            pytest.raises(ImportError, match=r"akgentic-tool\[weaviate\]"),
        ):
            _connect_weaviate(ClusterKey.from_url("weaviate", "http://localhost:8080"))

    def test_client_py_imports_no_vendor_module(self) -> None:
        """The cache is backend-agnostic: it names neither vendor at module scope."""
        import akgentic.tool.vector_store.client as client_module

        assert not hasattr(client_module, "_check_weaviate_dependencies")
        assert not hasattr(client_module, "WEAVIATE_MISSING_MESSAGE")
        assert not hasattr(client_module, "GRPC_PORT")


# ---------------------------------------------------------------------------
# The collection-handle cache
# ---------------------------------------------------------------------------


class TestCollectionHandleCache:
    """One ``Collection`` per (name, tenant): the vendor mints a new one per call."""

    def test_two_operations_on_one_collection_fetch_one_handle(self) -> None:
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        _recording_batch_contexts(mock_client)

        backend = _scoped_backend(mock_client)
        mock_client.collections.get.reset_mock()

        backend.add("col1", [_make_entry(ref_id="r1")])
        backend.add("col1", [_make_entry(ref_id="r2")])

        assert mock_client.collections.get.call_count == 1

    def test_a_second_collection_fetches_its_own_handle(self) -> None:
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        _recording_batch_contexts(mock_client)

        from akgentic.tool.vector_store.protocol import VectorStoreParam

        backend = _scoped_backend(mock_client)
        backend.create_collection("col2", VectorStoreParam())
        mock_client.collections.get.reset_mock()

        backend.add("col1", [_make_entry(ref_id="r1")])
        backend.add("col2", [_make_entry(ref_id="r2")])

        assert mock_client.collections.get.call_count == 2

    def test_the_same_collection_under_a_different_tenant_fetches_again(self) -> None:
        """``with_tenant()`` returns a different Collection, so the tenant is in the key."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        _recording_batch_contexts(mock_client)

        backend = _scoped_backend(mock_client)
        mock_client.collections.get.reset_mock()

        backend.add("col1", [_make_entry(ref_id="r1")])
        backend._collection_tenants["col1"] = "tenant-b"
        backend.add("col1", [_make_entry(ref_id="r2")])

        assert mock_client.collections.get.call_count == 2
        assert set(backend._collection_handles) == {("col1", None), ("col1", "tenant-b")}

    def test_the_cache_belongs_to_the_instance_not_the_class(self) -> None:
        """One backend instance per actor is the invariant the cache rests on."""
        _mock_weaviate, mock_client = _install_mock_weaviate()
        mock_client.collections.exists.return_value = False
        _recording_batch_contexts(mock_client)

        first = _scoped_backend(mock_client)
        second = _scoped_backend(mock_client)
        mock_client.collections.get.reset_mock()

        first.add("col1", [_make_entry(ref_id="r1")])
        second.add("col1", [_make_entry(ref_id="r2")])

        assert mock_client.collections.get.call_count == 2
        assert first._collection_handles is not second._collection_handles


# ---------------------------------------------------------------------------
# The team predicate is a per-collection choice
# ---------------------------------------------------------------------------


def _backend_with(
    mock_client: MagicMock, *collections: str, team_id: str | None = "team-42"
) -> WeaviateBackend:
    """Build a backend that has created each named collection.

    ``team_id=None`` builds the team-less kind — a hand-written script, or the
    administrative backend a sweeper uses.
    """
    from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
    from akgentic.tool.vector_store.protocol import VectorStoreParam

    backend = WeaviateBackend(client=mock_client, team_id=team_id)
    for name in collections:
        backend.create_collection(name, VectorStoreParam())
    return backend


def _empty_query_client() -> tuple[MagicMock, MagicMock]:
    """Return (client, collection) whose ``near_vector`` yields no objects."""
    _mock_weaviate, mock_client = _install_mock_weaviate()
    mock_client.collections.exists.return_value = False
    mock_collection = MagicMock()
    mock_client.collections.get.return_value = mock_collection
    mock_collection.query.near_vector.return_value = MagicMock(objects=[])
    return mock_client, mock_collection


class TestTheTeamLegFollowsTheCollection:
    """What was *sent* to the cluster, never what a double chose to return.

    A double returns whatever it was told to return, filter or no filter, so a
    search that yields no foreign hits proves nothing. Every spec here reads the
    recorded conjunction off the call.
    """

    def test_a_team_scoped_search_names_the_team(self) -> None:
        client, collection = _empty_query_client()
        backend = _backend_with(client, "planning")

        backend.search("planning", [0.1, 0.2], top_k=5)

        sent = collection.query.near_vector.call_args[1]["filters"]
        assert _legs(sent) == [("team_id", "equal", "team-42")]

    def test_a_shared_search_names_the_scope_and_not_the_team(self) -> None:
        """Two teams over one tree must read the same rows, so no team leg is built."""
        client, collection = _empty_query_client()
        backend = _backend_with(client, "workspace_chunks")

        backend.search("workspace_chunks", [0.1, 0.2], top_k=5, scope="u/t")

        sent = collection.query.near_vector.call_args[1]["filters"]
        assert _legs(sent) == [("scope", "equal", "u/t")]
        assert not any(leg[0] == "team_id" for leg in _legs(sent))

    def test_a_shared_search_still_conjoins_a_path_prefix(self) -> None:
        client, collection = _empty_query_client()
        backend = _backend_with(client, "workspace_chunks")

        backend.search("workspace_chunks", [0.1], top_k=5, scope="u/t", path_prefix="docs/")

        assert _legs(collection.query.near_vector.call_args[1]["filters"]) == [
            ("scope", "equal", "u/t"),
            ("path", "like", "docs/*"),
        ]

    def test_a_team_scoped_removal_names_the_team(self) -> None:
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.exists.return_value = False
        collection = MagicMock()
        client.collections.get.return_value = collection
        backend = _backend_with(client, "planning")

        backend.remove("planning", ["r1"])

        assert _legs(collection.data.delete_many.call_args[1]["where"]) == [
            ("ref_id", "contains_any", ["r1"]),
            ("team_id", "equal", "team-42"),
        ]

    def test_a_shared_removal_names_ref_id_and_scope_and_not_the_team(self) -> None:
        """The ref-id leg is still required: ids collide across scopes."""
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.exists.return_value = False
        collection = MagicMock()
        client.collections.get.return_value = collection
        backend = _backend_with(client, "workspace_chunks")

        backend.remove("workspace_chunks", ["r1", "r2"], scope="u/t")

        legs = _legs(collection.data.delete_many.call_args[1]["where"])
        assert legs == [
            ("ref_id", "contains_any", ["r1", "r2"]),
            ("scope", "equal", "u/t"),
        ]
        assert not any(leg[0] == "team_id" for leg in legs)


class TestATeamlessBackendAndTheSharedCollection:
    """A backend that does not know its team can query the shared collection only.

    On a shared collection there is no identity to invent, because the boundary is
    the scope — so ``_team_filter``'s refusal is simply never reached. On a
    team-scoped one it still bites, unchanged.
    """

    def test_it_can_search_the_shared_collection(self) -> None:
        client, collection = _empty_query_client()
        backend = _backend_with(client, "workspace_chunks", team_id=None)

        backend.search("workspace_chunks", [0.1], top_k=5, scope="u/t")

        assert _legs(collection.query.near_vector.call_args[1]["filters"]) == [
            ("scope", "equal", "u/t")
        ]

    def test_it_can_remove_from_the_shared_collection(self) -> None:
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.exists.return_value = False
        collection = MagicMock()
        client.collections.get.return_value = collection
        backend = _backend_with(client, "workspace_chunks", team_id=None)

        backend.remove("workspace_chunks", ["r1"], scope="u/t")

        assert _legs(collection.data.delete_many.call_args[1]["where"]) == [
            ("ref_id", "contains_any", ["r1"]),
            ("scope", "equal", "u/t"),
        ]

    def test_it_still_cannot_search_a_team_scoped_collection(self) -> None:
        client, collection = _empty_query_client()
        backend = _backend_with(client, "planning", team_id=None)

        with pytest.raises(ValueError, match="without a team_id"):
            backend.search("planning", [0.1], top_k=5)
        collection.query.near_vector.assert_not_called()

    def test_it_still_cannot_remove_from_a_team_scoped_collection(self) -> None:
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.exists.return_value = False
        collection = MagicMock()
        client.collections.get.return_value = collection
        backend = _backend_with(client, "planning", team_id=None)

        with pytest.raises(ValueError, match="without a team_id"):
            backend.remove("planning", ["r1"])
        collection.data.delete_many.assert_not_called()


class TestASharedCollectionRefusesAnUnscopedQuery:
    """Scope is the shared collection's only boundary, so it is mandatory.

    A caller that forgets it would otherwise read or delete every workspace's
    chunks on the cluster — a strictly worse version of the bug the team predicate
    was added to fix.
    """

    def test_search_refuses_before_reaching_the_cluster(self) -> None:
        client, collection = _empty_query_client()
        backend = _backend_with(client, "workspace_chunks")

        with pytest.raises(ValueError, match="workspace_chunks") as excinfo:
            backend.search("workspace_chunks", [0.1], top_k=3)

        assert "scope" in str(excinfo.value)
        collection.query.near_vector.assert_not_called()

    def test_remove_refuses_before_reaching_the_cluster(self) -> None:
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.exists.return_value = False
        collection = MagicMock()
        client.collections.get.return_value = collection
        backend = _backend_with(client, "workspace_chunks")

        with pytest.raises(ValueError, match="workspace_chunks"):
            backend.remove("workspace_chunks", ["a"])

        collection.data.delete_many.assert_not_called()

    def test_a_team_scoped_collection_is_unaffected(self) -> None:
        """It keeps its team predicate, so an unscoped query is still a bounded one."""
        client, collection = _empty_query_client()
        backend = _backend_with(client, "planning")

        backend.search("planning", [0.1], top_k=3)

        collection.query.near_vector.assert_called_once()


class TestDeleteByTeamRefusesASharedCollection:
    """A sweeper pointed at the workspace collection would reap a live team's rows."""

    def test_it_refuses_before_any_cluster_call(self) -> None:
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.exists.return_value = True
        collection = MagicMock()
        client.collections.get.return_value = collection

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=client, team_id="team-42")

        with pytest.raises(ValueError, match="workspace_chunks"):
            backend.delete_by_team("workspace_chunks", "team-42")

        client.collections.exists.assert_not_called()
        collection.data.delete_many.assert_not_called()

    def test_the_refusal_holds_on_an_administrative_backend(self) -> None:
        """The only kind a sweeper has: no team of its own, no collection created here.

        The guard consults a module-level fact rather than per-instance state, which
        is precisely why it is not vacuous on this backend.
        """
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.exists.return_value = True
        collection = MagicMock()
        client.collections.get.return_value = collection

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=client)  # no team_id, nothing created

        with pytest.raises(ValueError, match="workspace_chunks"):
            backend.delete_by_team("workspace_chunks", "team-42")

        collection.data.delete_many.assert_not_called()

    def test_a_team_scoped_collection_is_still_reaped(self) -> None:
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.exists.return_value = True
        collection = MagicMock()
        collection.data.delete_many.return_value = MagicMock(successful=4)
        client.collections.get.return_value = collection

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=client)

        assert backend.delete_by_team("planning", "team-42") == 4
        assert _legs(collection.data.delete_many.call_args[1]["where"]) == [
            ("team_id", "equal", "team-42")
        ]

    def test_list_collections_is_untouched_and_lists_the_shared_one(self) -> None:
        """A collection name identifies no team, so enumeration needs no scoping."""
        _mock_weaviate, client = _install_mock_weaviate()
        client.collections.list_all.return_value = {
            "planning": object(),
            "workspace_chunks": object(),
        }

        from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend

        backend = WeaviateBackend(client=client)  # no team_id

        assert sorted(backend.list_collections()) == ["planning", "workspace_chunks"]
