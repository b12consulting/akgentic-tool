"""The card side of retrieval: the derived caps, the announcement, the registration.

The caps are asserted on the ``WorkspaceConfig`` the card actually hands to
``getChildrenOrCreate``, never on the helper in isolation — the helper being right
while the call site ignores it is precisely the failure this file exists to catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL
from akgentic.tool.vector_store.protocol import CollectionConfig
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
    FakeActorToolObserver,
    FakeOrchestratorProxy,
)


def workspace_config_of(orchestrator_proxy: FakeOrchestratorProxy) -> WorkspaceConfig:
    """The ``WorkspaceConfig`` the card handed to ``getChildrenOrCreate``."""
    for actor_class, config in orchestrator_proxy.create_calls:
        if actor_class is WorkspaceActor:
            assert isinstance(config, WorkspaceConfig)
            return config
    raise AssertionError("the card never created a workspace actor")


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

    def enable_rag(
        self,
        agent_id: str,
        params: WorkspaceRagIndex,
        reader: DocumentReader,
        collection: CollectionConfig,
    ) -> None:
        self.enable_calls.append((agent_id, params, reader, collection))

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


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
        assert "rag_collection" in WorkspaceTool.model_fields
        assert "collection" not in WorkspaceTool.model_fields

    def test_a_payload_carrying_the_new_fields_round_trips(self) -> None:
        """Compare the models, never two dumps — ``expose`` is a ``set``."""
        payload = {
            "workspace_rag_index": {"chunk_chars": 800},
            "workspace_rag_list": {"max_pending_shown": 5},
            "rag_collection": {"backend": "inmemory", "dimension": 512},
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
        """The second of the three sites that read the predicate."""
        bind(
            orchestrator_proxy,
            workspace_rag_search=True,
            rag_collection=CollectionConfig(backend="inmemory"),
        )

        assert workspace_config_of(orchestrator_proxy).max_documents == IN_MEMORY_MAX_DOCUMENTS

    def test_a_search_only_card_naming_weaviate_with_no_cluster_fails_at_wiring(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The third site — a card that asked for shared storage must not get a local index."""
        with pytest.raises(ValueError, match="AKGENTIC_WEAVIATE_URL"):
            bind(
                orchestrator_proxy,
                workspace_rag_search=True,
                rag_collection=CollectionConfig(backend="weaviate"),
            )

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
        bind(
            orchestrator_proxy,
            workspace_rag_index=True,
            rag_collection=CollectionConfig(backend="inmemory"),
        )

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
            rag_collection=CollectionConfig(backend="weaviate"),
        )

        config = workspace_config_of(orchestrator_proxy)
        assert (config.max_documents, config.max_document_chars) == (
            DEFAULT_MAX_DOCUMENTS,
            DEFAULT_MAX_DOCUMENT_CHARS,
        )

    def test_retrieval_off_keeps_the_large_cache_on_an_in_memory_backend(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """No vectors exist, so nothing is derived from the document cap."""
        bind(orchestrator_proxy, rag_collection=CollectionConfig(backend="inmemory"))

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
            rag_collection=CollectionConfig(backend="inmemory"),
            max_documents=99,
            max_document_chars=12345,
        )

        config = workspace_config_of(orchestrator_proxy)
        assert (config.max_documents, config.max_document_chars) == (99, 12345)

    def test_the_list_capability_alone_also_derives_the_small_caps(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Either capability turns retrieval on, and the caps follow retrieval."""
        bind(
            orchestrator_proxy,
            workspace_rag_list=True,
            rag_collection=CollectionConfig(backend="inmemory"),
        )

        config = workspace_config_of(orchestrator_proxy)
        assert config.max_documents == IN_MEMORY_MAX_DOCUMENTS


class TestTheWeaviateCheck:
    """It is imposed on the cards that asked for Weaviate, and on no others."""

    def test_a_retrieval_card_naming_weaviate_with_no_cluster_fails_at_wiring(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A card that asked for durable shared storage must not get a local index."""
        with pytest.raises(ValueError, match="AKGENTIC_WEAVIATE_URL"):
            bind(
                orchestrator_proxy,
                workspace_rag_index=True,
                rag_collection=CollectionConfig(backend="weaviate"),
            )

    def test_a_card_with_retrieval_off_is_untouched_by_the_check(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The overwhelming majority of ``WorkspaceTool()`` instances never enable it."""
        card, _ = bind(orchestrator_proxy, rag_collection=CollectionConfig(backend="weaviate"))

        assert card.rag_collection.backend == "weaviate"

    def test_a_plain_card_binds_with_no_collection_configuration_at_all(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """``id_workspace.yaml`` ships ``payload: {}``; defaults must suffice."""
        card, _ = bind(orchestrator_proxy)

        assert card.rag_collection.backend == "inmemory"


class TestTheBindTimeAnnouncement:
    """``getChildrenOrCreate`` fixes the config; a capable card announces itself."""

    def test_a_retrieval_card_announces_itself(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        tell = RecordingTell()
        params = WorkspaceRagIndex(chunk_chars=900)

        bind(
            orchestrator_proxy,
            tell_proxy=tell,
            workspace_rag_index=params,
            rag_collection=CollectionConfig(backend="inmemory", tenant="acme"),
        )

        [(agent_id, announced, reader, collection)] = tell.enable_calls
        assert agent_id
        assert announced == params
        assert isinstance(reader, DocumentReader)
        assert collection.tenant == "acme"

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
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_NAME)]
        assert isinstance(actor, WorkspaceActor)
        return actor

    def test_it_renders_the_rows_the_actor_holds(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        from datetime import UTC, datetime

        card, _ = bind(orchestrator_proxy, workspace_rag_list=True)
        actor = self._actor(orchestrator_proxy)
        actor.state.rag_index["notes.md"] = RagFile(
            path="notes.md",
            status=RagStatus.EMBEDDED,
            chunk_count=4,
            updated_at=datetime.now(UTC),
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
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_NAME)]
        assert isinstance(actor, WorkspaceActor)
        actor.state.rag_index["notes.md"] = RagFile(
            path="notes.md",
            status=RagStatus.EMBEDDED,
            chunk_count=4,
            updated_at=datetime.now(UTC),
        )

        rendered = card.get_commands()[WorkspaceRagList]()

        assert "notes.md" in rendered
        assert "4 chunk(s)" in rendered

    def test_the_list_command_degrades_when_the_actor_raises(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        class Gone:
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
