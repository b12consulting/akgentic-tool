"""How a consumer resolves its storage engine, and who gets a store actor.

Covers the predicate (``needs_store_actor``), the card-side helper
(``ensure_store_actor``), and the two branches every consumer actor now takes:
an actor-state backend through the store actor's proxy, a cluster backend
through the registered factory.

Backends are swapped by **re-registering** the spec — ``BackendSpec`` is a frozen
dataclass — which is the seam ``test_registry.py`` already uses, and never by
monkeypatching a vendor module: neither ``weaviate-client`` nor ``qdrant-client``
is a dev dependency of this package.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import pytest

from akgentic.tool.vector_store import registry
from akgentic.tool.vector_store.actor import (
    VS_ACTOR_NAME,
    VS_ACTOR_ROLE,
    VectorStoreActor,
    ensure_store_actor,
)
from akgentic.tool.vector_store.protocol import (
    VectorStoreConfig,
    VectorStoreParam,
    needs_store_actor,
)
from akgentic.tool.vector_store.registry import BackendSpec, register_backend, unregister_backend
from tests.conftest import MockActorAddress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingStore:
    """A ``VectorStoreService`` double that records the four protocol calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        self.calls.append(("create_collection", (name, config)))

    def add(self, collection: str, entries: list[Any]) -> None:
        self.calls.append(("add", (collection, entries)))

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        self.calls.append(("remove", (collection, ref_ids, scope, path_prefix)))

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
        query: Any = None,
    ) -> Any:
        self.calls.append(("search", (collection, query_vector, top_k)))
        return None

    def of(self, name: str) -> list[tuple[Any, ...]]:
        """Return the arguments of every call to *name*."""
        return [args for called, args in self.calls if called == name]


@contextmanager
def _factory_for(backend: str, factory: Any) -> Iterator[list[Any]]:
    """Swap one registered backend's factory, yielding the contexts it receives."""
    contexts: list[Any] = []
    original = registry.get_backend_spec(backend)

    def _recording(context: Any) -> Any:
        contexts.append(context)
        return factory(context)

    register_backend(replace(original, factory=_recording), replace=True)
    try:
        yield contexts
    finally:
        register_backend(original, replace=True)


@contextmanager
def _registered(name: str, *, persists: bool) -> Iterator[None]:
    """Register a throwaway backend under *name* for the duration of a spec."""
    register_backend(
        BackendSpec(
            name=name,
            factory=lambda _context: _RecordingStore(),
            persists_in_actor_state=persists,
            selectable_as_default=False,
        )
    )
    try:
        yield
    finally:
        unregister_backend(name)


# ---------------------------------------------------------------------------
# needs_store_actor
# ---------------------------------------------------------------------------


class TestNeedsStoreActor:
    """The flag that already existed answers the larger question."""

    def test_the_in_memory_index_needs_an_actor(self) -> None:
        assert needs_store_actor(VectorStoreParam(backend="inmemory")) is True

    @pytest.mark.parametrize("backend", ["weaviate", "qdrant"])
    def test_a_cluster_backend_does_not(self, backend: str) -> None:
        assert needs_store_actor(VectorStoreParam(backend=backend)) is False

    def test_the_answer_comes_from_the_spec_not_from_a_name_list(self) -> None:
        """A registered backend nobody hardcoded gets the right answer for free."""
        with _registered("stateful_stub", persists=True):
            assert needs_store_actor(VectorStoreParam(backend="stateful_stub")) is True
        with _registered("cluster_stub", persists=False):
            assert needs_store_actor(VectorStoreParam(backend="cluster_stub")) is False

    def test_an_unregistered_backend_raises(self) -> None:
        """A card naming a backend nobody registered fails the build, loudly."""
        with pytest.raises(ValueError, match="Unknown vector-store backend"):
            needs_store_actor(VectorStoreParam(backend="nobody-registered-this"))


# ---------------------------------------------------------------------------
# ensure_store_actor
# ---------------------------------------------------------------------------


class TestEnsureStoreActor:
    """The card-side half: create the store only when the actor *is* the database."""

    def test_an_actor_state_backend_creates_exactly_one_actor(self) -> None:
        proxy = MagicMock()

        ensure_store_actor(VectorStoreParam(backend="inmemory"), proxy)

        proxy.getChildrenOrCreate.assert_called_once()
        actor_cls = proxy.getChildrenOrCreate.call_args.args[0]
        config = proxy.getChildrenOrCreate.call_args.kwargs["config"]
        assert actor_cls is VectorStoreActor
        assert isinstance(config, VectorStoreConfig)
        assert config.name == VS_ACTOR_NAME
        assert config.role == VS_ACTOR_ROLE

    def test_the_config_sets_neither_connection_field(self) -> None:
        """The actor it creates is the in-memory one, which needs neither."""
        proxy = MagicMock()

        ensure_store_actor(VectorStoreParam(backend="inmemory"), proxy)

        config = proxy.getChildrenOrCreate.call_args.kwargs["config"]
        assert config.weaviate_url is None
        assert config.weaviate_api_key is None

    @pytest.mark.parametrize("backend", ["weaviate", "qdrant"])
    def test_a_cluster_backend_creates_nothing(self, backend: str) -> None:
        proxy = MagicMock()

        ensure_store_actor(VectorStoreParam(backend=backend), proxy)

        assert proxy.getChildrenOrCreate.call_count == 0

    def test_three_calls_issue_three_creates(self) -> None:
        """The idempotence is the orchestrator's, per ADR-025 — not this helper's."""
        proxy = MagicMock()
        param = VectorStoreParam(backend="inmemory")

        for _ in range(3):
            ensure_store_actor(param, proxy)

        assert proxy.getChildrenOrCreate.call_count == 3

    def test_an_unregistered_backend_raises(self) -> None:
        proxy = MagicMock()

        with pytest.raises(ValueError, match="Unknown vector-store backend"):
            ensure_store_actor(VectorStoreParam(backend="nobody-registered-this"), proxy)

        assert proxy.getChildrenOrCreate.call_count == 0


# ---------------------------------------------------------------------------
# The consumer actors' two branches
# ---------------------------------------------------------------------------


def _plan_actor(param: VectorStoreParam, *, store_found: bool = True) -> tuple[Any, MagicMock]:
    """Build a bare PlanActor wired to a recording orchestrator proxy."""
    from akgentic.tool.planning.planning_actor import PlanActor, PlanConfig, PlanManagerState

    actor = PlanActor()
    actor.config = PlanConfig(name="#PlanningTool", role="ToolActor", vector_store=param)
    actor.state = PlanManagerState()
    actor.state.observer(actor)
    actor._vs_proxy = None
    actor._embedder = None

    orch_addr = MockActorAddress("orchestrator", "Orchestrator")
    actor._orchestrator = orch_addr  # type: ignore[assignment]
    orch_proxy = MagicMock()
    orch_proxy.get_team_member.return_value = (
        MockActorAddress(VS_ACTOR_NAME, "ToolActor") if store_found else None
    )
    store_proxy = _RecordingStore()

    def _proxy_ask(
        target: object, actor_type: type | None = None, timeout: int | None = None
    ) -> object:
        return orch_proxy if target is orch_addr else store_proxy

    actor.proxy_ask = _proxy_ask  # type: ignore[method-assign,assignment]
    actor._store_double = store_proxy  # type: ignore[attr-defined]
    return actor, orch_proxy


def _kg_actor(param: VectorStoreParam, *, store_found: bool = True) -> tuple[Any, MagicMock]:
    """Build a bare KnowledgeGraphActor wired to a recording orchestrator proxy."""
    from akgentic.tool.knowledge_graph.kg_actor import (
        KG_ACTOR_NAME,
        KG_ACTOR_ROLE,
        KnowledgeGraphActor,
        KnowledgeGraphConfig,
    )
    from akgentic.tool.knowledge_graph.models import KnowledgeGraphState

    actor = KnowledgeGraphActor()
    actor.config = KnowledgeGraphConfig(
        name=KG_ACTOR_NAME, role=KG_ACTOR_ROLE, vector_store=param
    )
    actor.state = KnowledgeGraphState()
    actor.state.observer(actor)
    actor._vs_proxy = None
    actor._embedder = None
    actor._state_event_seq = 0

    orch_addr = MockActorAddress("orchestrator", "Orchestrator")
    actor._orchestrator = orch_addr  # type: ignore[assignment]
    orch_proxy = MagicMock()
    orch_proxy.get_team_member.return_value = (
        MockActorAddress(VS_ACTOR_NAME, "ToolActor") if store_found else None
    )
    store_proxy = _RecordingStore()

    def _proxy_ask(
        target: object, actor_type: type | None = None, timeout: int | None = None
    ) -> object:
        return orch_proxy if target is orch_addr else store_proxy

    actor.proxy_ask = _proxy_ask  # type: ignore[method-assign,assignment]
    actor._store_double = store_proxy  # type: ignore[attr-defined]
    return actor, orch_proxy


class TestPlanActorResolution:
    """``PlanActor`` binds a proxy or a backend, and calls ``create_collection`` either way."""

    def test_an_in_memory_param_resolves_the_store_actor(self) -> None:
        from akgentic.tool.planning.planning_actor import PLAN_COLLECTION

        param = VectorStoreParam(backend="inmemory")
        actor, orch_proxy = _plan_actor(param)

        actor._acquire_vs_proxy()

        orch_proxy.get_team_member.assert_called_once_with(VS_ACTOR_NAME)
        assert actor._vs_proxy is actor._store_double
        assert actor._store_double.of("create_collection") == [(PLAN_COLLECTION, param)]

    def test_a_cluster_param_resolves_a_backend_and_never_looks_up(self) -> None:
        from akgentic.tool.planning.planning_actor import PLAN_COLLECTION

        param = VectorStoreParam(backend="weaviate")
        actor, orch_proxy = _plan_actor(param)
        double = _RecordingStore()

        with _factory_for("weaviate", lambda _context: double) as contexts:
            actor._acquire_vs_proxy()

        assert orch_proxy.get_team_member.call_count == 0
        assert actor._vs_proxy is double
        assert double.of("create_collection") == [(PLAN_COLLECTION, param)]
        assert len(contexts) == 1
        assert contexts[0].team_id == str(actor.team_id)
        assert contexts[0].config.name == "#PlanningTool"
        assert contexts[0].config.role == VS_ACTOR_ROLE

    def test_a_missing_store_actor_is_still_a_runtime_error(self) -> None:
        actor, _orch_proxy = _plan_actor(VectorStoreParam(backend="inmemory"), store_found=False)

        with pytest.raises(RuntimeError) as excinfo:
            actor._acquire_vs_proxy()

        message = str(excinfo.value)
        assert "#PlanningTool" in message
        assert VS_ACTOR_NAME in message
        assert "VectorStoreTool" not in message
        assert actor._vs_proxy is None

    def test_a_cluster_factory_that_raises_degrades_with_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, _orch_proxy = _plan_actor(VectorStoreParam(backend="weaviate"))

        def _boom(_context: Any) -> Any:
            raise ValueError("unreachable")

        logger_name = "akgentic.tool.planning.planning_actor"
        with (
            _factory_for("weaviate", _boom),
            caplog.at_level(logging.WARNING, logger=logger_name),
        ):
            actor._acquire_vs_proxy()  # must not raise

        assert actor._vs_proxy is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "unreachable" in warnings[0].getMessage()


class TestKnowledgeGraphActorResolution:
    """The same two branches, for the knowledge graph's own collection."""

    def test_an_in_memory_param_resolves_the_store_actor(self) -> None:
        from akgentic.tool.knowledge_graph.kg_actor import KG_COLLECTION

        param = VectorStoreParam(backend="inmemory")
        actor, orch_proxy = _kg_actor(param)

        actor._acquire_vs_proxy()

        orch_proxy.get_team_member.assert_called_once_with(VS_ACTOR_NAME)
        assert actor._vs_proxy is actor._store_double
        assert actor._store_double.of("create_collection") == [(KG_COLLECTION, param)]

    def test_a_cluster_param_resolves_a_backend_and_never_looks_up(self) -> None:
        from akgentic.tool.knowledge_graph.kg_actor import KG_ACTOR_NAME, KG_COLLECTION

        param = VectorStoreParam(backend="qdrant")
        actor, orch_proxy = _kg_actor(param)
        double = _RecordingStore()

        with _factory_for("qdrant", lambda _context: double) as contexts:
            actor._acquire_vs_proxy()

        assert orch_proxy.get_team_member.call_count == 0
        assert actor._vs_proxy is double
        assert double.of("create_collection") == [(KG_COLLECTION, param)]
        assert len(contexts) == 1
        assert contexts[0].team_id == str(actor.team_id)
        assert contexts[0].config.name == KG_ACTOR_NAME

    def test_a_missing_store_actor_is_still_a_runtime_error(self) -> None:
        from akgentic.tool.knowledge_graph.kg_actor import KG_ACTOR_NAME

        actor, _orch_proxy = _kg_actor(VectorStoreParam(backend="inmemory"), store_found=False)

        with pytest.raises(RuntimeError) as excinfo:
            actor._acquire_vs_proxy()

        message = str(excinfo.value)
        assert KG_ACTOR_NAME in message
        assert VS_ACTOR_NAME in message
        assert "VectorStoreTool" not in message
        assert actor._vs_proxy is None

    def test_a_cluster_factory_that_raises_degrades_with_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, _orch_proxy = _kg_actor(VectorStoreParam(backend="qdrant"))

        def _boom(_context: Any) -> Any:
            raise ValueError("unreachable")

        logger_name = "akgentic.tool.knowledge_graph.kg_actor"
        with (
            _factory_for("qdrant", _boom),
            caplog.at_level(logging.WARNING, logger=logger_name),
        ):
            actor._acquire_vs_proxy()  # must not raise

        assert actor._vs_proxy is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "unreachable" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# The card surface, across all three consumers
# ---------------------------------------------------------------------------


class TestNoCardOverridesDependsOn:
    """The dependency edge that ordered two cards is gone from the whole package."""

    def test_every_consumer_card_uses_the_base_property(self) -> None:
        from akgentic.tool.core import ToolCard
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool
        from akgentic.tool.workspace.card import WorkspaceTool

        for card_cls in (PlanningTool, KnowledgeGraphTool, WorkspaceTool):
            assert card_cls.depends_on is ToolCard.depends_on

    def test_the_deleted_card_is_gone_from_the_package(self) -> None:
        import akgentic.tool.vector_store as vector_store

        with pytest.raises(ModuleNotFoundError):
            import akgentic.tool.vector_store.tool  # noqa: F401

        assert "VectorStoreTool" not in vector_store.__all__
        assert not hasattr(vector_store, "VectorStoreTool")


class TestTheThreeCardsCarryOneParam:
    """One ``vector_store: VectorStoreParam`` field, and no lookup field anywhere."""

    def test_the_field_is_a_param_on_all_three(self) -> None:
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool
        from akgentic.tool.workspace.card import WorkspaceTool

        for card_cls in (PlanningTool, KnowledgeGraphTool, WorkspaceTool):
            assert "vector_store" in card_cls.model_fields
            assert card_cls.model_fields["vector_store"].annotation is VectorStoreParam

    def test_the_old_names_are_gone(self) -> None:
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool
        from akgentic.tool.workspace.card import WorkspaceTool

        assert "collection" not in PlanningTool.model_fields
        assert "collection" not in KnowledgeGraphTool.model_fields
        assert "rag_collection" not in WorkspaceTool.model_fields

    def test_the_old_catalog_shapes_fail_loudly(self) -> None:
        """``vector_store: true`` is a validation error, never an ignored key."""
        from pydantic import ValidationError

        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool

        with pytest.raises(ValidationError):
            PlanningTool(vector_store=True)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            KnowledgeGraphTool(vector_store="#VectorStore-RAG")  # type: ignore[arg-type]

    def test_a_persisted_card_naming_the_old_key_lands_on_the_default(self) -> None:
        """A known silent default: ``extra='ignore'`` drops the renamed key.

        A card that set a non-default dimension, tenant or embedding model under
        the old name loses it without a word. Recorded rather than mitigated —
        an alias would be the two-meanings state the rename exists to remove.
        """
        from akgentic.tool.planning.planning import PlanningTool

        card = PlanningTool.model_validate(
            {"collection": {"backend": "inmemory", "tenant": "lost", "dimension": 3072}}
        )
        assert card.vector_store == VectorStoreParam()
        assert card.vector_store.tenant is None
