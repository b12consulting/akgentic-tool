"""The backend registry — the pluggability seam for custom vector-store backends.

A backend is registered by name and resolved through the registry, so the actor
never switches on a backend string and adding one needs no core edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from akgentic.tool.vector_store.protocol import (
    CollectionConfig,
    SearchResult,
    VectorQuery,
)
from akgentic.tool.vector_store.qdrant import QDRANT_URL_ENV
from akgentic.tool.vector_store.registry import (
    BackendContext,
    BackendSpec,
    available_backends,
    get_backend_spec,
    is_registered,
    register_backend,
    resolve_default_backend,
    unregister_backend,
)

if TYPE_CHECKING:
    from akgentic.tool.vector_store.vector import VectorEntry


class _StubBackend:
    """Minimal VectorStoreService-shaped stub for registry tests."""

    def create_collection(self, name: str, config: CollectionConfig) -> None:
        pass

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        pass

    def remove(self, collection: str, ref_ids: list[str]) -> None:
        pass

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        query: VectorQuery | None = None,
    ) -> SearchResult | None:
        return None


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------


class TestBuiltins:
    """The three shipped backends self-register on first registry use."""

    def test_builtins_are_registered(self) -> None:
        assert set(available_backends()) >= {"inmemory", "weaviate", "qdrant"}

    def test_inmemory_persists_in_actor_state(self) -> None:
        assert get_backend_spec("inmemory").persists_in_actor_state is True

    def test_external_backends_do_not_persist_in_actor_state(self) -> None:
        assert get_backend_spec("weaviate").persists_in_actor_state is False
        assert get_backend_spec("qdrant").persists_in_actor_state is False

    def test_inmemory_is_not_selectable_as_default(self) -> None:
        """It is the fallback, never auto-selected over a provisioned store."""
        assert get_backend_spec("inmemory").selectable_as_default is False

    def test_unknown_backend_raises_with_guidance(self) -> None:
        with pytest.raises(ValueError, match="Unknown vector-store backend 'nope'"):
            get_backend_spec("nope")


# ---------------------------------------------------------------------------
# Default resolution
# ---------------------------------------------------------------------------


class TestDefaultResolution:
    """`resolve_default_backend` follows the environment, falling back to inmemory."""

    def test_defaults_to_inmemory_when_nothing_configured(self) -> None:
        assert resolve_default_backend() == "inmemory"

    def test_qdrant_claims_default_when_only_qdrant_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(QDRANT_URL_ENV, "http://localhost:6333")
        assert resolve_default_backend() == "qdrant"
        assert CollectionConfig().backend == "qdrant"


# ---------------------------------------------------------------------------
# Registering a custom backend
# ---------------------------------------------------------------------------


class TestCustomRegistration:
    """A framework user registers a backend and the registry resolves it."""

    def test_register_and_resolve(self) -> None:
        built: list[BackendContext] = []

        def factory(ctx: BackendContext) -> _StubBackend:
            built.append(ctx)
            return _StubBackend()

        register_backend(BackendSpec(name="stub", factory=factory))
        try:
            assert is_registered("stub")
            spec = get_backend_spec("stub")
            assert spec.name == "stub"
        finally:
            unregister_backend("stub")
        assert not is_registered("stub")

    def test_duplicate_registration_raises_without_replace(self) -> None:
        register_backend(BackendSpec(name="dup", factory=lambda _ctx: _StubBackend()))
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_backend(BackendSpec(name="dup", factory=lambda _ctx: _StubBackend()))
            # replace=True overrides deliberately
            register_backend(
                BackendSpec(name="dup", factory=lambda _ctx: _StubBackend()), replace=True
            )
        finally:
            unregister_backend("dup")

    def test_custom_backend_can_claim_default(self) -> None:
        register_backend(
            BackendSpec(
                name="always_on",
                factory=lambda _ctx: _StubBackend(),
                is_configured=lambda: True,
            )
        )
        try:
            # inmemory is not selectable, so a configured external stub wins.
            assert resolve_default_backend() in {"always_on", "weaviate", "qdrant"}
        finally:
            unregister_backend("always_on")


# ---------------------------------------------------------------------------
# VectorQuery
# ---------------------------------------------------------------------------


class TestVectorQuery:
    """The per-call query refinement object round-trips and defaults cleanly."""

    def test_defaults_are_permissive(self) -> None:
        q = VectorQuery()
        assert q.filters is None
        assert q.score_threshold is None
        assert q.params == {}

    def test_round_trip(self) -> None:
        q = VectorQuery(
            filters={"ref_type": "entity"}, score_threshold=0.5, params={"hnsw_ef": 128}
        )
        assert VectorQuery.model_validate(q.model_dump()) == q
