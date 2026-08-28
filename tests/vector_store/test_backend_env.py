"""The environment decides the backend, and disagreeing with it fails loudly.

Two rules, both about `AKGENTIC_WEAVIATE_URL`:

- a collection that names no backend follows the environment, so a deployment with
  a cluster does not silently get a process-local index;
- a collection that *names* `weaviate` where there is no cluster is a configuration
  error, raised while the team is being built rather than degraded at the first search.
"""

from __future__ import annotations

import pytest

from akgentic.tool.vector_store.protocol import (
    WEAVIATE_API_KEY_ENV,
    WEAVIATE_URL_ENV,
    CollectionConfig,
    default_backend,
    require_weaviate_configured,
    weaviate_api_key,
    weaviate_url,
)

CLUSTER = "http://localhost:8080"


@pytest.fixture(autouse=True)
def _no_ambient_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise a developer's exported cluster so the tests read the same everywhere."""
    monkeypatch.delenv(WEAVIATE_URL_ENV, raising=False)
    monkeypatch.delenv(WEAVIATE_API_KEY_ENV, raising=False)


# ---------------------------------------------------------------------------
# Reading the environment
# ---------------------------------------------------------------------------


class TestEnvironmentReading:
    """`weaviate_url` / `weaviate_api_key`."""

    def test_url_is_none_when_unset(self) -> None:
        assert weaviate_url() is None

    def test_url_is_none_when_exported_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A template that always exports the name must not read as a cluster at ''."""
        monkeypatch.setenv(WEAVIATE_URL_ENV, "")
        assert weaviate_url() is None

    def test_url_is_returned_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEAVIATE_URL_ENV, CLUSTER)
        assert weaviate_url() == CLUSTER

    def test_api_key_is_none_when_unset(self) -> None:
        assert weaviate_api_key() is None

    def test_api_key_is_none_when_exported_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEAVIATE_API_KEY_ENV, "")
        assert weaviate_api_key() is None

    def test_api_key_is_returned_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEAVIATE_API_KEY_ENV, "secret")
        assert weaviate_api_key() == "secret"


# ---------------------------------------------------------------------------
# The default follows the environment
# ---------------------------------------------------------------------------


class TestDefaultBackend:
    """`CollectionConfig()` with no backend named."""

    def test_defaults_to_inmemory_without_a_cluster(self) -> None:
        assert default_backend() == "inmemory"
        assert CollectionConfig().backend == "inmemory"

    def test_defaults_to_weaviate_with_a_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEAVIATE_URL_ENV, CLUSTER)
        assert default_backend() == "weaviate"
        assert CollectionConfig().backend == "weaviate"

    def test_an_empty_url_does_not_switch_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(WEAVIATE_URL_ENV, "")
        assert CollectionConfig().backend == "inmemory"

    def test_resolved_per_instantiation_not_at_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A process that exports the variable late must still see it."""
        assert CollectionConfig().backend == "inmemory"
        monkeypatch.setenv(WEAVIATE_URL_ENV, CLUSTER)
        assert CollectionConfig().backend == "weaviate"

    def test_an_explicit_backend_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEAVIATE_URL_ENV, CLUSTER)
        assert CollectionConfig(backend="inmemory").backend == "inmemory"

    def test_the_default_survives_a_serialisation_round_trip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(WEAVIATE_URL_ENV, CLUSTER)
        dumped = CollectionConfig().model_dump()
        assert dumped["backend"] == "weaviate"
        assert CollectionConfig.model_validate(dumped).backend == "weaviate"


# ---------------------------------------------------------------------------
# Asking for a cluster that is not there
# ---------------------------------------------------------------------------


class TestRequireWeaviateConfigured:
    """The guard consumer cards call at `observer()` time."""

    def test_raises_when_weaviate_is_named_without_a_cluster(self) -> None:
        with pytest.raises(ValueError, match=WEAVIATE_URL_ENV):
            require_weaviate_configured(CollectionConfig(backend="weaviate"), "PlanningTool")

    def test_the_error_names_the_offending_card(self) -> None:
        with pytest.raises(ValueError, match="KnowledgeGraphTool"):
            require_weaviate_configured(CollectionConfig(backend="weaviate"), "KnowledgeGraphTool")

    def test_the_error_names_the_way_out(self) -> None:
        """A configuration error should say what to change, not only what is wrong."""
        with pytest.raises(ValueError, match="in-memory"):
            require_weaviate_configured(CollectionConfig(backend="weaviate"), "PlanningTool")

    def test_an_empty_url_counts_as_no_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEAVIATE_URL_ENV, "")
        with pytest.raises(ValueError, match=WEAVIATE_URL_ENV):
            require_weaviate_configured(CollectionConfig(backend="weaviate"), "PlanningTool")

    def test_passes_when_the_cluster_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEAVIATE_URL_ENV, CLUSTER)
        require_weaviate_configured(CollectionConfig(backend="weaviate"), "PlanningTool")

    def test_passes_for_inmemory_without_a_cluster(self) -> None:
        require_weaviate_configured(CollectionConfig(backend="inmemory"), "PlanningTool")

    def test_a_defaulted_collection_never_trips_the_guard(self) -> None:
        """Without a cluster the default is already inmemory, so there is nothing to catch."""
        require_weaviate_configured(CollectionConfig(), "PlanningTool")

    def test_an_api_key_alone_is_not_a_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEAVIATE_API_KEY_ENV, "secret")
        with pytest.raises(ValueError, match=WEAVIATE_URL_ENV):
            require_weaviate_configured(CollectionConfig(backend="weaviate"), "PlanningTool")


# ---------------------------------------------------------------------------
# The guard fires where it matters — building the team
# ---------------------------------------------------------------------------


class TestCardsFailAtWiring:
    """`observer()` must refuse, so the failure lands at team creation."""

    @staticmethod
    def _observer() -> object:
        from unittest.mock import MagicMock

        observer = MagicMock()
        observer.orchestrator = MagicMock()
        observer.proxy_ask.return_value = MagicMock()
        return observer

    def test_planning_tool_refuses_to_wire(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool

        tool = PlanningTool(collection=CollectionConfig(backend="weaviate"))
        with pytest.raises(ValueError, match="PlanningTool"):
            tool.observer(self._observer())  # type: ignore[arg-type]

    def test_knowledge_graph_tool_refuses_to_wire(self) -> None:
        from akgentic.tool.knowledge_graph.kg_tool import KnowledgeGraphTool

        tool = KnowledgeGraphTool(collection=CollectionConfig(backend="weaviate"))
        with pytest.raises(ValueError, match="KnowledgeGraphTool"):
            tool.observer(self._observer())  # type: ignore[arg-type]

    def test_it_raises_before_any_actor_is_created(self) -> None:
        """A half-built team is worse than none: nothing may be spawned first."""
        from akgentic.tool.planning.planning import PlanningTool

        observer = self._observer()
        tool = PlanningTool(collection=CollectionConfig(backend="weaviate"))
        with pytest.raises(ValueError):
            tool.observer(observer)  # type: ignore[arg-type]
        observer.proxy_ask.assert_not_called()  # type: ignore[attr-defined]

    def test_cards_wire_normally_once_the_cluster_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from akgentic.tool.planning.planning import PlanningTool

        monkeypatch.setenv(WEAVIATE_URL_ENV, CLUSTER)
        tool = PlanningTool(collection=CollectionConfig(backend="weaviate"))
        tool.observer(self._observer())  # type: ignore[arg-type]
