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
    resolve_store_param,
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


def _plan_actor(
    param: VectorStoreParam | None, *, store_found: bool = True
) -> tuple[Any, MagicMock]:
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


def _kg_actor(param: VectorStoreParam | None, *, store_found: bool = True) -> tuple[Any, MagicMock]:
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
    """One ``vector_store`` field on each card, and no lookup field anywhere.

    Two of the three widened it to ``VectorStoreParam | bool`` so a card can
    decline a store; the workspace card did not, because it already carries that
    opt-out under a larger name.
    """

    def test_the_field_is_a_param_or_a_bool_on_the_two_consumers(self) -> None:
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool

        for card_cls in (PlanningTool, KnowledgeGraphTool):
            assert "vector_store" in card_cls.model_fields
            assert card_cls.model_fields["vector_store"].annotation == VectorStoreParam | bool

    def test_the_workspace_field_stays_narrow(self) -> None:
        """``WorkspaceTool`` is deliberately excluded, and this pins it.

        It already has this opt-out as ``_rag_enabled()``, which gates all four
        of its store sites. A second switch would be contradictable —
        ``workspace_rag_index=True, vector_store=False`` would leave retrieval
        announced and ``index_paths`` accepting work whose writes go nowhere —
        and ``derived_document_caps`` reads ``self.vector_store.backend``
        *outside* that guard, where a bool is an ``AttributeError``.
        """
        from akgentic.tool.workspace.card import WorkspaceTool

        assert "vector_store" in WorkspaceTool.model_fields
        assert WorkspaceTool.model_fields["vector_store"].annotation is VectorStoreParam

    def test_the_old_names_are_gone(self) -> None:
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool
        from akgentic.tool.workspace.card import WorkspaceTool

        assert "collection" not in PlanningTool.model_fields
        assert "collection" not in KnowledgeGraphTool.model_fields
        assert "rag_collection" not in WorkspaceTool.model_fields

    def test_the_boolean_catalog_shape_loads(self) -> None:
        """Every stored card writing ``vector_store: false`` keeps loading.

        This is the half of the old ``bool | str`` field that comes back. The
        two shapes are kept verbatim, not expanded into a param, which is what
        makes a stored record independent of the environment that wrote it.
        """
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool

        for card_cls in (PlanningTool, KnowledgeGraphTool):
            assert card_cls(vector_store=True).vector_store is True
            assert card_cls(vector_store=False).vector_store is False
            assert card_cls.model_validate({"vector_store": False}).vector_store is False

    def test_the_old_lookup_shape_still_fails_loudly(self) -> None:
        """A ``"#VectorStore-RAG"`` string is still a validation error.

        The string half of the old field named a ``VectorStoreActor`` to look up,
        and that lookup no longer exists. This is the only guard keeping that
        shape dead, so it does not move when the boolean half returns.
        """
        from pydantic import ValidationError

        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool

        with pytest.raises(ValidationError):
            PlanningTool(vector_store="#VectorStore")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            KnowledgeGraphTool(vector_store="#VectorStore-RAG")  # type: ignore[arg-type]

    def test_a_persisted_card_naming_the_old_key_lands_on_the_default(self) -> None:
        """A known silent default: ``extra='ignore'`` drops the renamed key.

        A card that set a non-default dimension, tenant or embedding model under
        the old name loses it without a word. Recorded rather than mitigated —
        an alias would be the two-meanings state the rename exists to remove.

        The default it lands on is now the bool ``True``, which resolves to a
        default param; the loss this spec records is unchanged.
        """
        from akgentic.tool.planning.planning import PlanningTool

        card = PlanningTool.model_validate(
            {"collection": {"backend": "inmemory", "tenant": "lost", "dimension": 3072}}
        )
        assert card.vector_store is True
        resolved = resolve_store_param(card.vector_store)
        assert resolved == VectorStoreParam()
        assert resolved is not None
        assert resolved.tenant is None


# ---------------------------------------------------------------------------
# A consumer whose collection cannot be created does not bind (AC 27)
# ---------------------------------------------------------------------------


class _RefusingStore(_RecordingStore):
    """A store whose ``create_collection`` refuses, as the actor now does."""

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        from akgentic.tool.errors import RetriableError

        super().create_collection(name, config)
        raise RetriableError(f"no backend could be built; collection '{name}' was not created.")


class TestAConsumerDoesNotBindWhenTheCollectionCannotBeCreated:
    """The store refuses; each consumer degrades with the answer it already had.

    None of the three raises: the ``RuntimeError`` belongs to a *missing actor*,
    not to a *failed collection*.
    """

    def test_plan_actor_leaves_the_slot_empty_and_logs_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, _orch_proxy = _plan_actor(VectorStoreParam(backend="inmemory"))
        actor._store_double = _RefusingStore()
        refusing = actor._store_double

        def _proxy_ask(
            target: object, actor_type: type | None = None, timeout: int | None = None
        ) -> object:
            return _orch_proxy if target is actor.orchestrator else refusing

        actor.proxy_ask = _proxy_ask  # type: ignore[method-assign,assignment]

        logger_name = "akgentic.tool.planning.planning_actor"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            actor._acquire_vs_proxy()  # must not raise

        assert actor._vs_proxy is None
        assert actor._embedder is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        # The collection *was* attempted — this is a refusal, not a skipped call.
        assert refusing.of("create_collection")

    def test_kg_actor_leaves_the_slot_empty_and_logs_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        actor, _orch_proxy = _kg_actor(VectorStoreParam(backend="inmemory"))
        refusing = _RefusingStore()

        def _proxy_ask(
            target: object, actor_type: type | None = None, timeout: int | None = None
        ) -> object:
            return _orch_proxy if target is actor.orchestrator else refusing

        actor.proxy_ask = _proxy_ask  # type: ignore[method-assign,assignment]

        logger_name = "akgentic.tool.knowledge_graph.kg_actor"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            actor._acquire_vs_proxy()  # must not raise

        assert actor._vs_proxy is None
        assert actor._embedder is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert refusing.of("create_collection")

    def test_a_cluster_consumer_whose_collection_is_refused_also_degrades(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same answer on the factory branch: refused is refused, whoever refused."""
        actor, _orch_proxy = _plan_actor(VectorStoreParam(backend="weaviate"))
        refusing = _RefusingStore()

        logger_name = "akgentic.tool.planning.planning_actor"
        with (
            _factory_for("weaviate", lambda _context: refusing),
            caplog.at_level(logging.WARNING, logger=logger_name),
        ):
            actor._acquire_vs_proxy()  # must not raise

        assert actor._vs_proxy is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# The opt-out: a card can decline a store without declining the tool
#
# Every negative below is paired with a positive in the same fixture. "A
# disabled card creates no store actor" passes whether or not the card is
# disabled, if nothing in the fixture would have created one anyway — so each
# spec also shows the enabled card doing the thing, and the disabled card still
# creating its own consumer actor, which proves it bound at all.
# ---------------------------------------------------------------------------


class TestResolveStoreParam:
    """The one place the card's three shapes collapse into the wiring's two."""

    def test_true_becomes_a_default_param(self) -> None:
        assert resolve_store_param(True) == VectorStoreParam()

    def test_false_becomes_none(self) -> None:
        assert resolve_store_param(False) is None

    def test_a_param_passes_through_by_identity(self) -> None:
        """Identity, not equality: a copy would break a param shared by ``__ref__``.

        A catalog can point two cards at one param with ``__ref__``. Returning a
        copy would silently give them two, so a later edit to one would stop
        reaching the other.
        """
        param = VectorStoreParam(tenant="team-7")
        assert resolve_store_param(param) is param

    def test_true_hands_each_caller_its_own_param(self) -> None:
        """Two enabled cards must not share one mutable object."""
        assert resolve_store_param(True) is not resolve_store_param(True)


class TestTheCardKeepsTheAuthorsDeclarationVerbatim:
    """No coercing validator: what the author wrote is what the catalog stores."""

    @pytest.mark.parametrize("declared", [True, False])
    def test_a_bool_round_trips_as_a_bool(self, declared: bool) -> None:
        """The bool is never expanded into a param on dump.

        A validator coercing ``True`` into ``VectorStoreParam()`` would resolve
        ``default_backend`` — which reads the environment *per instantiation* —
        and freeze the build environment's backend into a record whose author
        wrote ``true``. Catalogs are promoted between tiers, so that record would
        then name a backend the next tier has not provisioned.
        """
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
        from akgentic.tool.planning.planning import PlanningTool

        for card_cls in (PlanningTool, KnowledgeGraphTool):
            card = card_cls(vector_store=declared)
            dumped = card.model_dump()

            assert dumped["vector_store"] is declared
            assert card_cls.model_validate(dumped).vector_store is declared

    def test_an_explicit_param_still_round_trips_as_a_param(self) -> None:
        """The positive half: the param shape is untouched by the widening."""
        from akgentic.tool.planning.planning import PlanningTool

        card = PlanningTool(vector_store=VectorStoreParam(tenant="team-7"))
        reloaded = PlanningTool.model_validate(card.model_dump())

        assert isinstance(reloaded.vector_store, VectorStoreParam)
        assert reloaded.vector_store.tenant == "team-7"


def _run_card_observer(card: Any) -> list[tuple[type, Any]]:
    """Run ``card.observer()`` against one recording orchestrator double.

    Returns the ``(actor_class, config)`` pair of every ``getChildrenOrCreate``,
    in order. The **class** is what matters and a count is not enough: a count of
    one for a disabled card is ambiguous between "the store was skipped and the
    consumer created" and "the store was created and the consumer was not", and
    the second is a catastrophic bug a count assertion reads as success.
    """
    recorded: list[tuple[type, Any]] = []

    def _capture(actor_cls: type, config: Any = None) -> MagicMock:
        recorded.append((actor_cls, config))
        return MagicMock()

    orch_proxy = MagicMock()
    orch_proxy.getChildrenOrCreate.side_effect = _capture
    observer = MagicMock()
    observer.orchestrator = MagicMock()
    observer.proxy_ask.return_value = orch_proxy

    card.observer(observer)
    return recorded


def _consumer_cards() -> list[tuple[type, type]]:
    """``(card class, the consumer actor class it creates)`` for both consumers."""
    from akgentic.tool.knowledge_graph.kg_actor import KnowledgeGraphActor
    from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool
    from akgentic.tool.planning.planning import PlanningTool
    from akgentic.tool.planning.planning_actor import PlanActor

    return [(PlanningTool, PlanActor), (KnowledgeGraphTool, KnowledgeGraphActor)]


class TestADisabledCardCreatesNoStoreActor:
    """The negative and its positive, over one double, for both consumer cards."""

    @pytest.mark.parametrize(("card_cls", "consumer_cls"), _consumer_cards())
    def test_the_store_actor_appears_only_for_the_enabled_card(
        self, card_cls: type, consumer_cls: type
    ) -> None:
        enabled = _run_card_observer(card_cls(vector_store=True))
        disabled = _run_card_observer(card_cls(vector_store=False))

        # The positive: an enabled card creates the store, then its consumer.
        assert [cls for cls, _config in enabled] == [VectorStoreActor, consumer_cls]
        # The negative, and the proof it is not vacuous: the disabled card
        # created no store actor *and still created its own consumer*, so it
        # bound rather than failing early for an unrelated reason.
        assert [cls for cls, _config in disabled] == [consumer_cls]

    @pytest.mark.parametrize(("card_cls", "consumer_cls"), _consumer_cards())
    def test_the_config_carries_the_resolved_value(
        self, card_cls: type, consumer_cls: type
    ) -> None:
        """``None`` reaches the actor, because the config is its only channel."""
        enabled = _run_card_observer(card_cls(vector_store=True))
        disabled = _run_card_observer(card_cls(vector_store=False))

        enabled_config = next(cfg for cls, cfg in enabled if cls is consumer_cls)
        disabled_config = next(cfg for cls, cfg in disabled if cls is consumer_cls)

        assert isinstance(enabled_config.vector_store, VectorStoreParam)
        assert disabled_config.vector_store is None

    @pytest.mark.parametrize(("card_cls", "consumer_cls"), _consumer_cards())
    def test_an_explicit_param_reaches_the_config_by_identity(
        self, card_cls: type, consumer_cls: type
    ) -> None:
        """The third shape is handed on untouched, not copied."""
        param = VectorStoreParam(tenant="team-7")
        recorded = _run_card_observer(card_cls(vector_store=param))

        config = next(cfg for cls, cfg in recorded if cls is consumer_cls)
        assert config.vector_store is param


class TestTheConsumerConfigsAcceptBothShapes:
    """Both configs take ``None``, and both still default to a param."""

    def test_both_configs_accept_none_and_default_to_a_param(self) -> None:
        from akgentic.tool.knowledge_graph.kg_actor import KnowledgeGraphConfig
        from akgentic.tool.planning.planning_actor import PlanConfig

        for config_cls in (PlanConfig, KnowledgeGraphConfig):
            disabled = config_cls(name="#Tool", role="ToolActor", vector_store=None)
            assert disabled.vector_store is None

            # An absent key is unchanged: a config persisted before the opt-out
            # returned still gets a param, so this is not a migration.
            absent = config_cls.model_validate({"name": "#Tool", "role": "ToolActor"})
            assert absent.vector_store == VectorStoreParam()

            explicit_none = config_cls.model_validate(
                {"name": "#Tool", "role": "ToolActor", "vector_store": None}
            )
            assert explicit_none.vector_store is None


class TestADisabledCardSkipsTheThreeStoreObligations:
    """A disabled card owes nothing to a backend it will never contact."""

    @pytest.mark.parametrize(("card_cls", "consumer_cls"), _consumer_cards())
    def test_the_backend_probe_fires_when_enabled_and_is_skipped_when_disabled(
        self, card_cls: type, consumer_cls: type
    ) -> None:
        """A stored card naming an unprovisioned cluster must not fail its team.

        ``conftest`` hides any ambient cluster, so ``weaviate`` here is a backend
        the environment has not provisioned. The positive half — the raise — is
        what proves the guard is reachable; without it "it did not raise" would
        only mean nothing ran.
        """
        unprovisioned = VectorStoreParam(backend="weaviate")

        with pytest.raises(ValueError, match="AKGENTIC_WEAVIATE_URL"):
            _run_card_observer(card_cls(vector_store=unprovisioned))

        recorded = _run_card_observer(card_cls(vector_store=False))
        assert [cls for cls, _config in recorded] == [consumer_cls]

    @pytest.mark.parametrize(("card_cls", "consumer_cls"), _consumer_cards())
    def test_the_dimension_check_fires_when_enabled_and_is_skipped_when_disabled(
        self, card_cls: type, consumer_cls: type
    ) -> None:
        """A dimension contradicting an embedding model nobody will run."""
        contradictory = VectorStoreParam(
            embedding_model="text-embedding-3-small", dimension=3072
        )

        with pytest.raises(ValueError, match="dimension"):
            _run_card_observer(card_cls(vector_store=contradictory))

        recorded = _run_card_observer(card_cls(vector_store=False))
        assert [cls for cls, _config in recorded] == [consumer_cls]


class TestTheKnowledgeGraphExtraIsStillRequired:
    """``_check_kg_dependencies()`` is not moved inside the disabled guard.

    It is ``_check_vector_search_dependencies`` under an alias, so it guards the
    ``[vector_search]`` extra for the whole knowledge-graph package rather than
    for the store. Whether a store-less graph should still need it is an open
    question on ADR-049; this pins today's answer so it cannot drift by accident.
    """

    def test_it_runs_for_a_disabled_card_exactly_as_for_an_enabled_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import akgentic.tool.knowledge_graph as kg_package
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool

        calls: list[None] = []
        monkeypatch.setattr(
            kg_package, "_check_kg_dependencies", lambda: calls.append(None)
        )

        _run_card_observer(KnowledgeGraphTool(vector_store=True))
        assert len(calls) == 1

        _run_card_observer(KnowledgeGraphTool(vector_store=False))
        assert len(calls) == 2


class TestTheWorkspaceCardIsUntouched:
    """Decision 4, pinned as behaviour rather than assumed."""

    def test_the_workspace_card_refuses_a_bool(self) -> None:
        """Its field is narrow, so the second switch cannot be written at all."""
        from pydantic import ValidationError

        from akgentic.tool.workspace.card import WorkspaceTool

        with pytest.raises(ValidationError):
            WorkspaceTool(vector_store=False)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            WorkspaceTool(vector_store=True)  # type: ignore[arg-type]

    def test_its_backend_is_readable_without_resolving_anything(self) -> None:
        """``derived_document_caps`` reads ``.backend`` outside the rag guard.

        A bool there would be an ``AttributeError`` on every workspace card, rag
        enabled or not — which is the concrete cost decision 4 refuses to pay.
        """
        from akgentic.tool.workspace.card import WorkspaceTool

        assert WorkspaceTool().vector_store.backend == "inmemory"


class TestADisabledActorStaysDegradedAndSaysSoOnce:
    """The bind-time half, paired: the enabled actor does what the disabled one skips."""

    def test_the_plan_actor_never_looks_up_and_logs_one_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger_name = "akgentic.tool.planning.planning_actor"

        # The positive: an enabled actor looks the store actor up and binds it.
        enabled, enabled_orch = _plan_actor(VectorStoreParam(backend="inmemory"))
        with caplog.at_level(logging.WARNING, logger=logger_name):
            enabled._acquire_vs_proxy()
        assert enabled_orch.get_team_member.call_count == 1
        assert enabled._vs_proxy is enabled._store_double
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

        # The negative, in the same shape: no lookup, no backend, one line.
        caplog.clear()
        disabled, disabled_orch = _plan_actor(None)
        with caplog.at_level(logging.WARNING, logger=logger_name):
            disabled._acquire_vs_proxy()

        assert disabled_orch.get_team_member.call_count == 0
        assert disabled._vs_proxy is None
        assert disabled._embedder is None
        assert disabled._store_double.calls == []
        # The count, not merely the presence: caplog accumulates.
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "off by configuration" in warnings[0].getMessage()

    def test_the_kg_actor_never_looks_up_and_logs_one_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger_name = "akgentic.tool.knowledge_graph.kg_actor"

        enabled, enabled_orch = _kg_actor(VectorStoreParam(backend="inmemory"))
        with caplog.at_level(logging.WARNING, logger=logger_name):
            enabled._acquire_vs_proxy()
        assert enabled_orch.get_team_member.call_count == 1
        assert enabled._vs_proxy is enabled._store_double
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

        caplog.clear()
        disabled, disabled_orch = _kg_actor(None)
        with caplog.at_level(logging.WARNING, logger=logger_name):
            disabled._acquire_vs_proxy()

        assert disabled_orch.get_team_member.call_count == 0
        assert disabled._vs_proxy is None
        assert disabled._embedder is None
        assert disabled._store_double.calls == []
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "off by configuration" in warnings[0].getMessage()

    def test_a_disabled_actor_never_builds_a_cluster_backend_either(self) -> None:
        """The other branch of ``_resolve_store`` is not reached either.

        A card can only be disabled *or* name a backend, never both, so the pair
        here is the factory itself: it is invoked for an enabled cluster param
        and never for a disabled one.
        """
        enabled, _orch = _plan_actor(VectorStoreParam(backend="weaviate"))
        with _factory_for("weaviate", lambda _c: _RecordingStore()) as enabled_contexts:
            enabled._acquire_vs_proxy()
        assert len(enabled_contexts) == 1

        disabled, _orch = _plan_actor(None)
        with _factory_for("weaviate", lambda _c: _RecordingStore()) as disabled_contexts:
            disabled._acquire_vs_proxy()
        assert disabled_contexts == []
