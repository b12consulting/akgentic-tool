"""The card side of retrieval: the derived caps, the announcement, the registration.

The caps are asserted on the ``WorkspaceConfig`` the card actually hands to
``getResourceOrCreate``, never on the helper in isolation — the helper being right
while the call site ignores it is precisely the failure this file exists to catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from akgentic.core.agent_config import BaseConfig

from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL
from akgentic.tool.vector_store import registry
from akgentic.tool.vector_store.protocol import VectorStoreParam
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.card.params import (
    WorkspaceRagIndex,
    WorkspaceRagList,
    WorkspaceRagSearch,
    WorkspaceRead,
)
from akgentic.tool.workspace.card.rag import RagFactories
from akgentic.tool.workspace.documents.models import (
    DEFAULT_MAX_DOCUMENT_CHARS,
    DEFAULT_MAX_DOCUMENTS,
    IN_MEMORY_MAX_DOCUMENT_CHARS,
    IN_MEMORY_MAX_DOCUMENTS,
    RagFile,
    RagStatus,
)
from akgentic.tool.workspace.models import WorkspaceConfig
from akgentic.tool.workspace.readers import DocumentReader
from akgentic.tool.workspace.tool import WorkspaceTool
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    factory_for,
    seed_row,
)


def workspace_config_of(orchestrator_proxy: FakeOrchestratorProxy) -> WorkspaceConfig:
    """The ``WorkspaceConfig`` the card handed to ``getChildrenOrCreate``."""
    for actor_class, config in orchestrator_proxy.create_calls:
        if actor_class is WorkspaceActor:
            assert isinstance(config, WorkspaceConfig)
            return config
    raise AssertionError("the card never bound a workspace actor")


def bind(
    orchestrator_proxy: FakeOrchestratorProxy,
    tell_proxy: object | None = None,
    **card_kwargs: Any,
) -> tuple[WorkspaceTool, FakeActorToolObserver]:
    """Wire a card onto the test workspace and return it with its live observer."""
    observer = FakeActorToolObserver(orchestrator_proxy, workspace_tell_proxy=tell_proxy)
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, **card_kwargs)
    card.observer(observer)
    return card, observer


class RecordingTell:
    """A tell proxy that records every announcement the card makes."""

    def __init__(self) -> None:
        self.enable_calls: list[tuple[Any, ...]] = []
        self.calls: list[str] = []
        """Every announcement's name, in the order the bind made it."""

    def enable_rag(
        self,
        agent_id: str,
        params: WorkspaceRagIndex,
        reader: DocumentReader,
        collection: VectorStoreParam,
    ) -> None:
        self.calls.append("enable_rag")
        self.enable_calls.append((agent_id, params, reader, collection))

    def __getattr__(self, name: str) -> Any:
        def announced(*args: Any, **kwargs: Any) -> None:
            self.calls.append(name)

        return announced


class TestTheCardsRetrievalFields:
    """Five new fields, and what their defaults mean."""

    def test_both_capabilities_are_off_by_default(self) -> None:
        """They reach the vector store and can spend embedding credits on a tree."""
        card = WorkspaceTool()

        assert card.workspace_rag_index is False
        assert card.workspace_rag_list is False

    def test_the_caps_default_to_derive_rather_than_to_a_number(self) -> None:
        """``None`` is not zero and not "use the default" — it is "derive it"."""
        card = WorkspaceTool()

        assert card.max_documents is None
        assert card.max_document_chars is None

    def test_the_collection_field_is_named_for_the_workspace(self) -> None:
        """A bare ``collection`` reads as the workspace's collection of files."""
        assert "vector_store" in WorkspaceTool.model_fields
        assert "collection" not in WorkspaceTool.model_fields

    def test_a_payload_carrying_the_new_fields_round_trips(self) -> None:
        """Compare the models, never two dumps — ``expose`` is a ``set``."""
        payload = {
            "workspace_rag_index": {"chunk_chars": 800},
            "workspace_rag_list": {"max_pending_shown": 5},
            "vector_store": {"backend": "inmemory", "dimension": 512},
            "max_documents": 99,
        }
        card = WorkspaceTool.model_validate(payload)

        again = WorkspaceTool.model_validate(card.model_dump())

        assert again == card


class TestTheSearchCapability:
    """Story 45-8's field, its channel, and the third term it adds to enablement."""

    def test_it_is_off_by_default_like_its_two_siblings(self) -> None:
        """It reaches the vector store and spends an embedding call per query."""
        assert WorkspaceTool().workspace_rag_search is False

    def test_it_is_a_tool_call_and_nothing_else(self) -> None:
        """ADR-045 §5 gives the three capabilities three different channel sets."""
        assert WorkspaceRagSearch().expose == {TOOL_CALL}

    def test_it_is_a_read_side_tool_and_survives_read_only(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Retrieval derives from the tree and writes nothing into it."""
        card, _ = bind(orchestrator_proxy, workspace_rag_search=True, read_only=True)

        assert "workspace_rag_search" in {tool.__name__ for tool in card.get_tools()}

    def test_it_is_absent_when_the_capability_is_off(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card, _ = bind(orchestrator_proxy)

        assert "workspace_rag_search" not in {tool.__name__ for tool in card.get_tools()}

    def test_it_reaches_no_other_channel(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A search is something the model does, not something it is shown."""
        card, _ = bind(orchestrator_proxy, workspace_rag_search=True)

        assert WorkspaceRagSearch not in card.get_commands()
        assert card.get_context_states() == []

    def test_a_search_only_card_still_enables_retrieval_on_the_actor(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The AC most likely to be missed, driven end to end.

        Without the third term in ``_rag_enabled`` there is no ``enable_rag``, so
        no proxy, no collection and no chunking parameters — and every search
        answers that retrieval is unavailable with nothing anywhere saying why.
        """
        tell = RecordingTell()

        bind(
            orchestrator_proxy,
            tell_proxy=tell,
            workspace_rag_index=False,
            workspace_rag_list=False,
            workspace_rag_search=True,
        )

        [(_, announced, _, _)] = tell.enable_calls
        assert announced == WorkspaceRagIndex()

    def test_a_search_only_card_also_derives_the_small_caps(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The second of the three sites that read the predicate.

        The card names no backend, which is the only way its declaration is still
        the in-actor one: naming it explicitly is refused at bind now (see
        ``TestAWorkspaceMayNotRunOnTheInActorBackend``).
        """
        bind(orchestrator_proxy, workspace_rag_search=True)

        assert workspace_config_of(orchestrator_proxy).max_documents == IN_MEMORY_MAX_DOCUMENTS

    def test_a_search_only_card_naming_weaviate_with_no_cluster_fails_at_wiring(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The third site — a card that asked for shared storage must not get a local index."""
        with pytest.raises(ValueError, match="AKGENTIC_WEAVIATE_URL"):
            bind(
                orchestrator_proxy,
                workspace_rag_search=True,
                vector_store=VectorStoreParam(backend="weaviate"),
            )

    def test_a_retrieval_card_with_a_mismatched_dimension_fails_at_wiring(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Same severity as the Weaviate check three lines above it."""
        with pytest.raises(ValueError, match="WorkspaceTool.*dimension=3072"):
            bind(
                orchestrator_proxy,
                workspace_rag_index=True,
                vector_store=VectorStoreParam(dimension=3072),
            )

    def test_a_card_with_retrieval_off_never_inherits_the_dimension_rule(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Most cards never create the collection and must not be constrained by it."""
        card, _ = bind(orchestrator_proxy, vector_store=VectorStoreParam(dimension=3072))

        assert card.vector_store.dimension == 3072

    def test_a_payload_carrying_the_search_capability_round_trips(self) -> None:
        """Compare the models, never two dumps — ``expose`` is a ``set``."""
        card = WorkspaceTool.model_validate(
            {"workspace_rag_search": {"top_k": 3, "alpha": 0.4, "score_threshold": 0.2}}
        )

        assert WorkspaceTool.model_validate(card.model_dump()) == card


class TestTheSearchCallable:
    """A thin ask, and the card's configured values travel with it."""

    def test_it_forwards_the_cards_knobs_to_the_actor(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The whole search runs on the actor; the card supplies configuration."""
        seen: list[dict[str, Any]] = []

        class Recording:
            def attach(self, agent: Any, agent_name: str) -> None:
                """The bind-time holder registration — the actor was alive then."""

            def rag_search(self, query: str, **kwargs: Any) -> str:
                seen.append({"query": query, **kwargs})
                return "ok"

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=Recording())
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME,
            workspace_rag_search=WorkspaceRagSearch(top_k=3, alpha=0.4, score_threshold=0.2),
        )
        card.observer(observer)

        assert self._tool(card, "workspace_rag_search")("terms", path_prefix="docs/") == "ok"
        assert seen == [
            {
                "query": "terms",
                "top_k": 3,
                "path_prefix": "docs/",
                "alpha": 0.4,
                "score_threshold": 0.2,
            }
        ]

    def test_the_callables_own_top_k_overrides_the_cards(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The budget is the one knob the model may set per call."""
        seen: list[int] = []

        class Recording:
            def attach(self, agent: Any, agent_name: str) -> None:
                """The bind-time holder registration — the actor was alive then."""

            def rag_search(self, query: str, **kwargs: Any) -> str:
                seen.append(int(kwargs["top_k"]))
                return "ok"

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=Recording())
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME, workspace_rag_search=WorkspaceRagSearch(top_k=3)
        )
        card.observer(observer)

        self._tool(card, "workspace_rag_search")("terms", top_k=9)

        assert seen == [9]

    def test_it_degrades_to_a_sentence_when_the_actor_raises(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """It is an LLM-facing callable; a traceback is not an answer it can use."""

        class Gone:
            def attach(self, agent: Any, agent_name: str) -> None:
                """The bind-time holder registration — the actor was alive then."""

            def rag_search(self, query: str, **kwargs: Any) -> str:
                raise RuntimeError("actor is dead")

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=Gone())
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_search=True)
        card.observer(observer)

        assert self._tool(card, "workspace_rag_search")("terms") == (
            "Retrieval indexing is not available for this workspace."
        )

    def test_an_unbound_card_answers_the_sentence_rather_than_raising(self) -> None:
        """A harness that wires a bare observer binds no proxy at all."""
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_search=True)

        assert card._rag_search_factory(WorkspaceRagSearch())("terms") == (
            "Retrieval indexing is not available for this workspace."
        )

    def test_the_docstring_carries_the_cards_extra_instructions(self) -> None:
        """``format_docstring`` is what puts a team's configuration in front of the model."""
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        params = WorkspaceRagSearch(instructions="Prefer the reports/ directory.")

        assert "Prefer the reports/ directory." in (card._rag_search_factory(params).__doc__ or "")

    @staticmethod
    def _tool(card: WorkspaceTool, name: str) -> Any:
        for tool in card.get_tools():
            if tool.__name__ == name:
                return tool
        raise AssertionError(f"{name} is not exposed by this card")


class TestTheDerivedCaps:
    """AC10: all three outcomes and the override, at the one construction site."""

    def test_in_memory_with_retrieval_on_shrinks_the_cache(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The caps are derived from the **author's** declaration, which is the
        in-actor backend for a card that names none.

        That the store the card then resolves is the file-backed one is a real
        loose end rather than a property being asserted here: an index on disk is
        not the ~44 MB re-serialisation the small caps exist for (ADR-045 §7). It
        is out of this story's scope and recorded in epic 52's deferred findings.
        """
        bind(orchestrator_proxy, workspace_rag_index=True)

        config = workspace_config_of(orchestrator_proxy)
        assert (config.max_documents, config.max_document_chars) == (
            IN_MEMORY_MAX_DOCUMENTS,
            IN_MEMORY_MAX_DOCUMENT_CHARS,
        )

    def test_weaviate_with_retrieval_on_keeps_the_large_cache(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A card naming Weaviate with no cluster fails at wiring, which is its own
        # spec below; here the cluster exists so the caps are what is under test.
        monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "https://cluster.example")
        bind(
            orchestrator_proxy,
            workspace_rag_index=True,
            vector_store=VectorStoreParam(backend="weaviate"),
        )

        config = workspace_config_of(orchestrator_proxy)
        assert (config.max_documents, config.max_document_chars) == (
            DEFAULT_MAX_DOCUMENTS,
            DEFAULT_MAX_DOCUMENT_CHARS,
        )

    def test_retrieval_off_keeps_the_large_cache_on_an_in_memory_backend(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """No vectors exist, so nothing is derived from the document cap.

        A retrieval-off card may still *declare* the in-actor backend: the refusal
        is inside the same ``_rag_enabled()`` guard as the two ``require_*`` calls,
        because a card that will never open a store owes it no obligation.
        """
        bind(orchestrator_proxy, vector_store=VectorStoreParam(backend="inmemory"))

        config = workspace_config_of(orchestrator_proxy)
        assert (config.max_documents, config.max_document_chars) == (
            DEFAULT_MAX_DOCUMENTS,
            DEFAULT_MAX_DOCUMENT_CHARS,
        )

    def test_an_explicit_card_value_beats_every_derivation(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """ "An explicit catalog value always wins" is what the two fields are for."""
        bind(
            orchestrator_proxy,
            workspace_rag_index=True,
            max_documents=99,
            max_document_chars=12345,
        )

        config = workspace_config_of(orchestrator_proxy)
        assert (config.max_documents, config.max_document_chars) == (99, 12345)

    def test_the_list_capability_alone_also_derives_the_small_caps(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Either capability turns retrieval on, and the caps follow retrieval."""
        bind(orchestrator_proxy, workspace_rag_list=True)

        config = workspace_config_of(orchestrator_proxy)
        assert config.max_documents == IN_MEMORY_MAX_DOCUMENTS


class TestTheWeaviateCheck:
    """It is imposed on the cards that asked for a cluster, and on no others."""

    def test_a_retrieval_card_naming_qdrant_with_no_cluster_fails_at_wiring(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The guard is backend-agnostic now, so qdrant fails the build like weaviate.

        Before this it only tested ``backend == "weaviate"``, so a qdrant card with
        no URL passed the check and degraded silently at ``enable_rag``.
        """
        monkeypatch.delenv("AKGENTIC_QDRANT_URL", raising=False)

        with pytest.raises(ValueError, match="AKGENTIC_QDRANT_URL") as excinfo:
            bind(
                orchestrator_proxy,
                workspace_rag_search=True,
                vector_store=VectorStoreParam(backend="qdrant"),
            )
        assert "WorkspaceTool" in str(excinfo.value)

    def test_a_retrieval_off_card_naming_qdrant_does_not_raise(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A retrieval-off card inherits no collection constraint at all."""
        monkeypatch.delenv("AKGENTIC_QDRANT_URL", raising=False)

        card, _ = bind(orchestrator_proxy, vector_store=VectorStoreParam(backend="qdrant"))

        assert card.vector_store.backend == "qdrant"

    def test_a_retrieval_card_naming_weaviate_with_no_cluster_fails_at_wiring(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A card that asked for durable shared storage must not get a local index."""
        with pytest.raises(ValueError, match="AKGENTIC_WEAVIATE_URL"):
            bind(
                orchestrator_proxy,
                workspace_rag_index=True,
                vector_store=VectorStoreParam(backend="weaviate"),
            )

    def test_a_card_with_retrieval_off_is_untouched_by_the_check(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The overwhelming majority of ``WorkspaceTool()`` instances never enable it."""
        card, _ = bind(orchestrator_proxy, vector_store=VectorStoreParam(backend="weaviate"))

        assert card.vector_store.backend == "weaviate"

    def test_a_plain_card_binds_with_no_collection_configuration_at_all(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """``id_workspace.yaml`` ships ``payload: {}``; defaults must suffice."""
        card, _ = bind(orchestrator_proxy)

        assert card.vector_store.backend == "inmemory"


class TestTheCardCreatesNoStoreActor:
    """**Premise reversed for the actor-backed case** — see the class below.

    Story 51-1 moved the store to the workspace actor's own child, which is what
    a *hosted* actor required (core ADR-022 Decision 3). Nothing is hosted after
    epic 52, and Decision 2 records that the prohibition was always on the actor:
    "a card keeps talking to its orchestrator". So a backend that needs an actor
    is bound by the card again, through the team's one ``#VectorStore``.

    What survives unchanged is the **cluster** case: there is nothing for an actor
    to hold, so no store actor is created for one — and the retrieval-off case,
    which creates nothing at all.
    """

    def test_a_cluster_retrieval_card_creates_no_store_actor(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from akgentic.tool.vector_store.actor import VectorStoreActor

        monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "http://localhost:8080")
        bind(
            orchestrator_proxy,
            workspace_rag_index=True,
            vector_store=VectorStoreParam(backend="weaviate"),
        )

        # Kept as a positive with a negative beside it: the bind happened, and
        # nothing it did named a store actor — on the child path or through a
        # host, of which this process runs none.
        created = [cls for cls, _config in orchestrator_proxy.create_calls]
        assert created == [WorkspaceActor]
        assert VectorStoreActor not in created
        assert orchestrator_proxy.resource_calls == []

    def test_a_retrieval_off_card_creates_no_store_actor(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        from akgentic.tool.vector_store.actor import VectorStoreActor

        bind(orchestrator_proxy, vector_store=VectorStoreParam(backend="inmemory"))

        created = [cls for cls, _config in orchestrator_proxy.create_calls]
        assert created == [WorkspaceActor]
        assert VectorStoreActor not in created
        assert orchestrator_proxy.resource_calls == []


class TestTheBindTimeAnnouncement:
    """The first bind fixes the config; a capable card announces itself."""

    def test_a_retrieval_card_announces_itself(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        tell = RecordingTell()
        params = WorkspaceRagIndex(chunk_chars=900)

        bind(
            orchestrator_proxy,
            tell_proxy=tell,
            workspace_rag_index=params,
            vector_store=VectorStoreParam(tenant="acme"),
        )

        [(agent_id, announced, reader, collection)] = tell.enable_calls
        assert agent_id
        assert announced == params
        assert isinstance(reader, DocumentReader)
        assert collection.tenant == "acme"

    def test_the_document_store_is_announced_before_retrieval_is_enabled(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The actor may never enable retrieval under a store it has not been given.

        Story 52-3's ordering clause, and the reason ``_announce_document_store``
        sits between the bind and ``_announce_rag`` rather than after it: the
        moment retrieval is on, the actor reads and writes index rows, and a
        window where it does that with no store is a window where the index looks
        empty and every row it writes is dropped.
        """
        tell = RecordingTell()

        bind(orchestrator_proxy, tell_proxy=tell, workspace_rag_index=True)

        assert "configure_document_store" in tell.calls
        assert "enable_rag" in tell.calls
        assert tell.calls.index("configure_document_store") < tell.calls.index("enable_rag")

    def test_a_card_with_retrieval_off_announces_nothing(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A workspace with retrieval off must never create a collection."""
        tell = RecordingTell()

        bind(orchestrator_proxy, tell_proxy=tell)

        assert tell.enable_calls == []

    def test_a_list_only_card_still_contributes_chunking_parameters(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The actor needs the splitter's parameters whatever turned retrieval on."""
        tell = RecordingTell()

        bind(orchestrator_proxy, tell_proxy=tell, workspace_rag_list=True)

        [(_, announced, _, _)] = tell.enable_calls
        assert announced == WorkspaceRagIndex()

    def test_the_cards_document_reader_travels_with_the_announcement(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The worker extracts, and extraction configuration lives on the card."""
        tell = RecordingTell()
        reader = DocumentReader(llm_client=None, llm_model="chosen-model")

        bind(
            orchestrator_proxy,
            tell_proxy=tell,
            workspace_rag_index=True,
            workspace_read=WorkspaceRead(document_reader=reader),
        )

        [(_, _, announced_reader, _)] = tell.enable_calls
        assert announced_reader == reader

    def test_a_card_that_disabled_the_reader_still_contributes_a_default_one(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """An indexer with no extractor could not index a PDF at all."""
        tell = RecordingTell()

        bind(
            orchestrator_proxy,
            tell_proxy=tell,
            workspace_rag_index=True,
            workspace_read=WorkspaceRead(document_reader=False),
        )

        [(_, _, reader, _)] = tell.enable_calls
        assert reader == DocumentReader()

    def test_the_announcement_never_takes_the_binding_down(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A stand-in proxy without the method must not stop a card binding."""

        class NoSuchMethod:
            def __getattr__(self, name: str) -> Any:
                raise AttributeError(name)

        card, _ = bind(orchestrator_proxy, tell_proxy=NoSuchMethod(), workspace_rag_index=True)

        assert card.workspace_rag_index is True


class TestRegistration:
    """Where each capability shows up, and on which channel."""

    def test_the_indexer_is_a_read_side_tool(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Indexing derives from the tree and writes nothing into it."""
        card, _ = bind(orchestrator_proxy, workspace_rag_index=True)

        assert "workspace_rag_index" in {tool.__name__ for tool in card.get_tools()}

    def test_the_indexer_survives_read_only(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card, _ = bind(orchestrator_proxy, workspace_rag_index=True, read_only=True)

        assert "workspace_rag_index" in {tool.__name__ for tool in card.get_tools()}

    def test_the_indexer_is_absent_when_the_capability_is_off(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card, _ = bind(orchestrator_proxy)

        assert "workspace_rag_index" not in {tool.__name__ for tool in card.get_tools()}

    def test_the_list_is_never_a_tool_call(self) -> None:
        """Deliberate: the model sees it as context, not as something to call."""
        assert TOOL_CALL not in WorkspaceRagList().expose
        assert WorkspaceRagList().expose == {COMMAND, LLM_CONTEXT}

    def test_both_capabilities_reach_the_command_channel(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card, _ = bind(orchestrator_proxy, workspace_rag_index=True, workspace_rag_list=True)

        commands = card.get_commands()
        assert WorkspaceRagIndex in commands
        assert WorkspaceRagList in commands

    def test_the_command_channel_carries_nothing_when_retrieval_is_off(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card, _ = bind(orchestrator_proxy)

        commands = card.get_commands()
        assert WorkspaceRagIndex not in commands
        assert WorkspaceRagList not in commands

    def test_the_context_state_provider_is_returned_only_when_enabled(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        with_list, _ = bind(orchestrator_proxy, workspace_rag_list=True)
        assert len(with_list.get_context_states()) == 1

    def test_no_provider_without_the_capability(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card, _ = bind(orchestrator_proxy)

        assert card.get_context_states() == []

    def test_no_provider_when_the_capability_is_off_the_context_channel(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A card may expose the command and withhold the per-turn state."""
        card, _ = bind(orchestrator_proxy, workspace_rag_list=WorkspaceRagList(expose={COMMAND}))

        assert card.get_context_states() == []


class TestTheProvider:
    """It never raises, and it is what the model actually sees each turn."""

    def _actor(self, orchestrator_proxy: FakeOrchestratorProxy) -> WorkspaceActor:
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        return actor

    def test_it_renders_the_rows_the_actor_holds(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        from datetime import UTC, datetime

        card, _ = bind(orchestrator_proxy, workspace_rag_list=True)
        actor = self._actor(orchestrator_proxy)
        seed_row(
            actor,
            "notes.md",
            RagFile(
                path="notes.md",
                status=RagStatus.EMBEDDED,
                chunk_count=4,
                updated_at=datetime.now(UTC),
            ),
        )

        [provider] = card.get_context_states()
        state = provider()

        assert state is not None
        assert "notes.md" in state.render_full()

    def test_an_empty_index_renders_a_sentence_rather_than_nothing(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A real state, distinct from a provider returning ``None``."""
        card, _ = bind(orchestrator_proxy, workspace_rag_list=True)

        [provider] = card.get_context_states()
        state = provider()

        assert state is not None
        assert state.render_full() == "No workspace files are indexed for retrieval."

    def test_it_returns_none_when_the_actor_is_unreachable(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The ``ContextState`` contract: never raise, answer ``None`` instead."""

        class Gone:
            def attach(self, agent: Any, agent_name: str) -> None:
                """The bind-time holder registration — the actor was alive then."""

            def rag_snapshot(self, max_pending_shown: int) -> Any:
                raise RuntimeError("actor is dead")

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=Gone())
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_list=True)
        card.observer(observer)

        [provider] = card.get_context_states()

        assert provider() is None

    def test_the_card_cap_reaches_the_snapshot(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """``max_pending_shown`` is captured at ``get_context_states`` time."""
        seen: list[int] = []

        class Recording:
            def attach(self, agent: Any, agent_name: str) -> None:
                """The bind-time holder registration — the actor was alive then."""

            def rag_snapshot(self, max_pending_shown: int) -> Any:
                seen.append(max_pending_shown)
                return None

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=Recording())
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME,
            workspace_rag_list=WorkspaceRagList(max_pending_shown=7),
        )
        card.observer(observer)

        [provider] = card.get_context_states()
        provider()

        assert seen == [7]


class TestTheMixinRules:
    """45-2's rules, applied to the fifth mixin."""

    def test_rag_factories_declares_no_pydantic_field(self) -> None:
        """Its annotations sit under ``if TYPE_CHECKING:`` and never reach Pydantic."""
        assert not getattr(RagFactories, "__annotations__", {})

    def test_no_two_card_mixins_define_the_same_name(self) -> None:
        """A real definition on two bases lets the MRO pick a winner in silence."""
        from akgentic.tool.workspace.card.execution import ExecFactories
        from akgentic.tool.workspace.card.read import ReadFactories
        from akgentic.tool.workspace.card.write import WriteFactories

        owners: dict[str, str] = {}
        for mixin in (ReadFactories, WriteFactories, ExecFactories, RagFactories):
            for name in vars(mixin):
                if name.startswith("__"):
                    continue
                owner = owners.setdefault(name, mixin.__name__)
                assert owner == mixin.__name__, (
                    f"{name} is defined on both {owner} and {mixin.__name__}"
                )

    def test_the_card_still_carries_every_field_itself(self) -> None:
        """The mixins contribute none, which is what keeps the frozen set meaningful."""
        for mixin in (RagFactories,):
            assert not set(mixin.__dict__.get("model_fields", {}))


class TestTheCallablesThemselves:
    """What an agent — or a person typing a command — actually gets back."""

    def _tool(self, card: WorkspaceTool, name: str) -> Any:
        for tool in card.get_tools():
            if tool.__name__ == name:
                return tool
        raise AssertionError(f"{name} is not exposed by this card")

    def test_the_indexer_returns_the_actors_counts(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The counts are the answer, which is why this leg is an ask."""

        class Counting:
            def attach(self, agent: Any, agent_name: str) -> None:
                """The bind-time holder registration — the actor was alive then."""

            def index_paths(self, path: str, force: bool) -> str:
                return f"queued {path!r} force={force}"

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=Counting())
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_index=True)
        card.observer(observer)

        assert self._tool(card, "workspace_rag_index")("docs", True) == ("queued 'docs' force=True")

    def test_the_indexer_degrades_to_a_sentence_when_the_actor_raises(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """It is an LLM-facing callable; a traceback is not an answer it can use."""

        class Gone:
            def attach(self, agent: Any, agent_name: str) -> None:
                """The bind-time holder registration — the actor was alive then."""

            def index_paths(self, path: str, force: bool) -> str:
                raise RuntimeError("actor is dead")

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=Gone())
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_index=True)
        card.observer(observer)

        assert self._tool(card, "workspace_rag_index")("") == (
            "Retrieval indexing is not available for this workspace."
        )

    def test_the_list_command_renders_the_full_table(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A person asking for the list wants the list, not what changed."""
        from datetime import UTC, datetime

        card, _ = bind(orchestrator_proxy, workspace_rag_list=True)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        seed_row(
            actor,
            "notes.md",
            RagFile(
                path="notes.md",
                status=RagStatus.EMBEDDED,
                chunk_count=4,
                updated_at=datetime.now(UTC),
            ),
        )

        rendered = card.get_commands()[WorkspaceRagList]()

        assert "notes.md" in rendered
        assert "4 chunk(s)" in rendered

    def test_the_list_command_degrades_when_the_actor_raises(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        class Gone:
            def attach(self, agent: Any, agent_name: str) -> None:
                """The bind-time holder registration — the actor was alive then."""

            def rag_snapshot(self, max_pending_shown: int) -> Any:
                raise RuntimeError("actor is dead")

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=Gone())
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_list=True)
        card.observer(observer)

        assert card.get_commands()[WorkspaceRagList]() == (
            "Retrieval indexing is not available for this workspace."
        )

    def test_an_unbound_card_answers_the_sentence_rather_than_raising(self) -> None:
        """A harness that wires a bare observer binds no proxy at all."""
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_index=True)

        indexer = card._rag_index_factory(WorkspaceRagIndex())
        lister = card._rag_list_factory(WorkspaceRagList())
        [provider] = [card._rag_list_state_factory(WorkspaceRagList())]

        assert indexer("") == "Retrieval indexing is not available for this workspace."
        assert lister() == "Retrieval indexing is not available for this workspace."
        assert provider() is None

    def test_an_unbound_card_announces_nothing(self) -> None:
        """``_announce_rag`` runs before any proxy exists in some harness shapes."""
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_index=True)

        card._announce_rag()  # must not raise


##
## Story 52-4 — the card resolves the team's store again
##


def store_configs_of(orchestrator_proxy: FakeOrchestratorProxy) -> list[BaseConfig]:
    """Every ``VectorStoreActor`` config the card asked the orchestrator to create."""
    from akgentic.tool.vector_store.actor import VectorStoreActor

    return [
        config
        for actor_class, config in orchestrator_proxy.create_calls
        if actor_class is VectorStoreActor
    ]


class TestTheCardBindsTheTeamsStore:
    """AC 7. Exactly one ``#VectorStore``, created here and looked up by name."""

    def test_a_retrieval_card_creates_the_teams_store_and_resolves_a_proxy_to_it(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        from akgentic.tool.vector_store.actor import VS_ACTOR_NAME, VS_ACTOR_ROLE

        card, _ = bind(orchestrator_proxy, workspace_rag_index=True)

        [config] = store_configs_of(orchestrator_proxy)
        assert (config.name, config.role) == (VS_ACTOR_NAME, VS_ACTOR_ROLE)
        assert orchestrator_proxy.member_lookups == [VS_ACTOR_NAME]
        # The card holds the object the lookup returned, never one it built.
        assert card._vector_store is orchestrator_proxy.children[VS_ACTOR_NAME][1]

    def test_two_retrieval_cards_in_one_team_share_the_one_store(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """``getChildrenOrCreate`` is idempotent, so a team holds one store however
        many cards ask for one — which is the whole point of going back to it."""
        from akgentic.tool.vector_store.actor import VS_ACTOR_NAME, VectorStoreActor

        first, _ = bind(orchestrator_proxy, workspace_rag_index=True)
        second, _ = bind(orchestrator_proxy, workspace_rag_search=True)

        created = [cls for cls, _config in orchestrator_proxy.create_calls]
        assert created.count(VectorStoreActor) == 2  # asked twice
        assert len(store_configs_of(orchestrator_proxy)) == 2
        # Created once, and the workspace beside it once: two cards on one tree
        # in one team share both actors, which is what get-or-create buys.
        assert created.count(VectorStoreActor) == 2
        assert set(orchestrator_proxy.children) == {
            VS_ACTOR_NAME,
            workspace_actor_name(WORKSPACE_PATH),
        }
        assert first._vector_store is second._vector_store

    def test_a_plain_card_creates_no_store_resolves_nothing_and_announces_nothing(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """AC 7's second half, and the one most likely to pass vacuously.

        Asserted on the **absence of the calls**, never on ``_vector_store`` being
        ``None``: a card that made every call and then failed to keep the result
        would leave the slot ``None`` too, and the whole point is that a bare
        ``WorkspaceTool()`` — the overwhelming majority of them — costs nothing.
        """
        from akgentic.tool.vector_store.actor import VectorStoreActor

        tell = RecordingTell()

        card, _ = bind(orchestrator_proxy, tell_proxy=tell)

        # The workspace's own bind is on this list since 52-5; nothing else is,
        # which is what the absence of a store costs.
        created = [cls for cls, _config in orchestrator_proxy.create_calls]
        assert created == [WorkspaceActor]
        assert VectorStoreActor not in created
        assert orchestrator_proxy.member_lookups == []
        assert "configure_vector_store" not in tell.calls
        assert card._vector_store is None
        assert card._resolved_store is None

    def test_the_card_passes_its_real_team_and_nothing_passes_none(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Story 51-4 neutralised the team; the row identity carries it no more.

        The cluster branch is where the team is visible, because that is the one
        that builds a backend directly from a ``BackendContext``.
        """
        from akgentic.tool.vector_store.protocol import VectorStoreService

        contexts: list[Any] = []

        class _Client:
            def create_collection(self, name: str, config: VectorStoreParam) -> None: ...
            def add(self, collection: str, entries: list[Any]) -> None: ...
            def remove(self, collection: str, ref_ids: list[str], **kwargs: Any) -> None: ...
            def search(self, *args: Any, **kwargs: Any) -> Any: ...

        def _factory(context: Any) -> VectorStoreService:
            contexts.append(context)
            return _Client()

        monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "https://cluster.example")
        with factory_for("weaviate", _factory):
            _card, observer = bind(
                orchestrator_proxy,
                workspace_rag_index=True,
                vector_store=VectorStoreParam(backend="weaviate"),
            )

        assert [context.team_id for context in contexts] == [str(observer.team_id)]
        assert None not in [context.team_id for context in contexts]


class TestTheStoreIsAnnouncedBeforeRetrievalIsEnabled:
    """AC 8. The ordering, pinned — moving the announcement later must break this."""

    def test_the_store_announcement_precedes_enable_rag(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        tell = RecordingTell()

        bind(orchestrator_proxy, tell_proxy=tell, workspace_rag_index=True)

        assert "configure_vector_store" in tell.calls
        assert tell.calls.index("configure_vector_store") < tell.calls.index("enable_rag")

    def test_the_actor_is_handed_the_object_the_card_resolved(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Not an equivalent one it built itself: identity, on the announced object."""
        announced: list[Any] = []

        class _Watching(RecordingTell):
            def configure_vector_store(self, store: Any) -> None:
                self.calls.append("configure_vector_store")
                announced.append(store)

        tell = _Watching()
        card, _ = bind(orchestrator_proxy, tell_proxy=tell, workspace_rag_index=True)

        assert announced == [card._vector_store]

    def test_a_lost_announcement_does_not_fail_the_bind(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """It degrades: the actor answers its unavailable sentence and rebinding fixes it."""

        class _Broken(RecordingTell):
            def configure_vector_store(self, store: Any) -> None:
                raise RuntimeError("the actor died between the bind and this line")

        card, _ = bind(orchestrator_proxy, tell_proxy=_Broken(), workspace_rag_index=True)

        assert card._vector_store is not None


class TestTheResolvedParamIsDerivedNotDeclared:
    """AC 9, 11 and 12: what the card sends, and what it leaves the author's field."""

    def test_a_card_naming_no_backend_indexes_into_the_file_backed_one(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        tell = RecordingTell()

        card, _ = bind(orchestrator_proxy, tell_proxy=tell, workspace_rag_index=True)

        [(_, _, _, collection)] = tell.enable_calls
        assert collection.backend == "local"
        # The author's declaration is untouched by the bind.
        assert card.vector_store.backend == "inmemory"
        assert card.vector_store.root is None

    def test_the_root_is_this_trees_metadata_directory(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        from akgentic.tool.workspace.workspace import meta_dir_for

        tell = RecordingTell()

        bind(orchestrator_proxy, tell_proxy=tell, workspace_rag_index=True)

        [(_, _, _, collection)] = tell.enable_calls
        assert collection.root == str(meta_dir_for(WORKSPACE_PATH))

    def test_an_explicitly_named_backend_is_untouched(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nothing about the cluster path changes: same backend, same collection."""
        tell = RecordingTell()

        monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "https://cluster.example")
        with factory_for("weaviate", lambda _ctx: object()):
            card, _ = bind(
                orchestrator_proxy,
                tell_proxy=tell,
                workspace_rag_index=True,
                vector_store=VectorStoreParam(
                    backend="weaviate",
                    dimension=3072,
                    embedding_model="text-embedding-3-large",
                ),
            )

        [(_, _, _, collection)] = tell.enable_calls
        assert (collection.backend, collection.dimension) == ("weaviate", 3072)
        assert card.vector_store.backend == "weaviate"

    def test_a_root_declared_in_a_catalog_is_inert(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """AC 12. Nothing an author writes can point one tree's index at another's."""
        from akgentic.tool.workspace.workspace import meta_dir_for

        tell = RecordingTell()

        card, _ = bind(
            orchestrator_proxy,
            tell_proxy=tell,
            workspace_rag_index=True,
            vector_store=VectorStoreParam(root="/somebody/elses/tree"),
        )

        [(_, _, _, collection)] = tell.enable_calls
        assert collection.root == str(meta_dir_for(WORKSPACE_PATH))
        assert collection.root != "/somebody/elses/tree"
        # And the author's own record still says what they wrote.
        assert card.vector_store.root == "/somebody/elses/tree"

    def test_the_derived_param_is_copied_never_rebuilt(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Golden Rule 12, in the only formulation that works.

        A whole-model comparison passes green against a rebuild that names every
        field existing today — which is the failure this rule is about, since the
        field added tomorrow is the one that disappears. So the param carries a
        field the write path has never heard of, and the assertion is that the
        subclass **and its sentinel** come out the other side.
        """

        class _VectorStoreParamWithExtraField(VectorStoreParam):
            extra_field: str = "sentinel"

        tell = RecordingTell()

        bind(
            orchestrator_proxy,
            tell_proxy=tell,
            workspace_rag_index=True,
            vector_store=_VectorStoreParamWithExtraField(),
        )

        [(_, _, _, collection)] = tell.enable_calls
        assert isinstance(collection, _VectorStoreParamWithExtraField)
        assert collection.extra_field == "sentinel"


class TestAWorkspaceMayNotRunOnTheInActorBackend:
    """Ruling E. An index that dies with the process, under rows that do not."""

    def test_a_retrieval_card_declaring_the_in_actor_backend_fails_the_bind(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """It fails at wiring, in front of the admin who wrote it, and names the fix."""
        with pytest.raises(ValueError) as excinfo:
            bind(
                orchestrator_proxy,
                workspace_rag_index=True,
                vector_store=VectorStoreParam(backend="inmemory"),
            )

        message = str(excinfo.value)
        assert "WorkspaceTool" in message
        assert "local" in message
        assert orchestrator_proxy.create_calls == []

    def test_a_stored_payload_declaring_it_is_refused_the_same_way(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A catalog is where this is most likely to be written, so it is checked there."""
        observer = FakeActorToolObserver(orchestrator_proxy)
        card = WorkspaceTool.model_validate(
            {
                "workspace_id": WORKSPACE_NAME,
                "workspace_rag_index": True,
                "vector_store": {"backend": "inmemory"},
            }
        )

        with pytest.raises(ValueError, match="not a workspace backend"):
            card.observer(observer)

    def test_a_card_that_named_nothing_still_binds_after_a_serialisation_round_trip(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The refusal must not fire on a value the author never wrote.

        ``SerializableBaseModel`` declares a whole-model serializer that emits
        **every** field, so one ``model_dump()`` / ``model_validate()`` round trip
        makes every field look explicitly set — and the agent-card store performs
        exactly that round trip on every team resume. Discriminating on
        ``model_fields_set`` therefore turned a card that named no backend into one
        that had "declared" the in-actor backend, and refused it at the next bind
        with a message blaming its author. The record is a field now, so the trip
        is invisible.
        """
        tell = RecordingTell()
        stored = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_index=True).model_dump()
        card = WorkspaceTool.model_validate(stored)

        # The trip really did inflate the naive discriminator — without this the
        # spec could pass while proving nothing.
        assert "backend" in card.vector_store.model_fields_set
        assert card.vector_store.backend_declared is False

        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_tell_proxy=tell))

        [(_, _, _, collection)] = tell.enable_calls
        assert collection.backend == "local"

    def test_a_declaration_survives_the_round_trip_and_is_still_refused(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The other half: the trip must not launder a declaration away either."""
        stored = WorkspaceTool(
            workspace_id=WORKSPACE_NAME,
            workspace_rag_index=True,
            vector_store=VectorStoreParam(backend="inmemory"),
        ).model_dump()
        card = WorkspaceTool.model_validate(stored)

        assert card.vector_store.backend_declared is True
        with pytest.raises(ValueError, match="not a workspace backend"):
            card.observer(FakeActorToolObserver(orchestrator_proxy))

    def test_a_retrieval_card_naming_a_backend_nobody_registered_fails_the_bind(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """At the bind, not merely in the resolver — the gap story 52-3 shipped.

        ``require_backend_configured`` looks the name up in the registry, so an
        unregistered one raises before anything is created. Asserting it on the
        resolver alone would leave the card free to stop calling it.
        """
        with pytest.raises((ValueError, KeyError)):
            bind(
                orchestrator_proxy,
                workspace_rag_index=True,
                vector_store=VectorStoreParam(backend="no-such-backend"),
            )

        assert orchestrator_proxy.create_calls == []

    def test_a_retrieval_off_card_declaring_it_binds_normally(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The refusal is inside the same guard as the two ``require_*`` calls."""
        card, _ = bind(orchestrator_proxy, vector_store=VectorStoreParam(backend="inmemory"))

        assert card.vector_store.backend == "inmemory"

    def test_the_knowledge_graph_and_the_plan_still_get_the_in_actor_backend(self) -> None:
        """AC 14. They keep their rows in actor state beside the index, so it fits.

        The mismatch this rule exists for is a *persisted* row over an empty
        engine; a consumer whose rows live in the same actor state as its index
        loses and restores both together and never had one.
        """
        from akgentic.tool.vector_store.protocol import needs_store_actor

        assert needs_store_actor(VectorStoreParam(backend="inmemory")) is True
        assert registry.get_backend_spec("inmemory").persists_in_actor_state is True


class TestTheStoreResolutionDegradesRatherThanFailingTheBind:
    """A retrieval capability is one of twenty on a card whose others are file ops."""

    def test_a_store_the_team_does_not_hold_after_creation_degrades(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The lookup answering ``None`` is a real answer — core's, for a miss.

        It should not happen after ``ensure_store_actor`` returns, which is why it
        is a WARNING and not a raise: the bind carries on and the workspace's
        other capabilities are unaffected.
        """
        from akgentic.tool.vector_store.actor import VS_ACTOR_NAME

        tell = RecordingTell()
        orchestrator_proxy.get_team_member = lambda _name: None  # type: ignore[method-assign]

        card, _ = bind(orchestrator_proxy, tell_proxy=tell, workspace_rag_index=True)

        assert card._vector_store is None
        assert "configure_vector_store" not in tell.calls
        assert card.workspace is not None  # the rest of the bind completed
        assert VS_ACTOR_NAME

    def test_a_factory_that_cannot_reach_its_cluster_degrades(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """One WARNING naming the backend, no store, and the bind still returns."""
        import logging

        def _factory(_context: Any) -> Any:
            raise ValueError("cluster unreachable")

        monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "https://cluster.example")
        with (
            factory_for("weaviate", _factory),
            caplog.at_level(logging.WARNING, logger="akgentic.tool.workspace.card"),
        ):
            card, _ = bind(
                orchestrator_proxy,
                workspace_rag_index=True,
                vector_store=VectorStoreParam(backend="weaviate"),
            )

        assert card._vector_store is None
        warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and record.name == "akgentic.tool.workspace.card"
        ]
        assert len(warnings) == 1
        assert "weaviate" in warnings[0].getMessage()
        assert "cluster unreachable" in warnings[0].getMessage()
