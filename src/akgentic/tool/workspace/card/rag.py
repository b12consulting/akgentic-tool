"""The retrieval capabilities' wiring: the three factories, the provider, the announcement.

:class:`RagFactories` declares **no Pydantic field**. Every field stays on
:class:`~akgentic.tool.workspace.card.WorkspaceTool` in ``card/__init__.py``,
which is what keeps that card's frozen field set meaningful; the annotations
below are inside ``if TYPE_CHECKING:``, so they never reach ``__annotations__``
and Pydantic never collects them.

Putting the wiring here rather than in the façade is deliberate rather than
tidy-minded: ``card/__init__.py`` is already the longest module in the package,
and the exec pair is the precedent for what a capability's wiring looks like —
not a rule that it must live in the façade.

**The import edge runs one way.** This module imports from ``card/params.py``,
which imports from no sibling at all; nothing here may be imported back into it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from akgentic.tool.core import ContextState, _resolve
from akgentic.tool.workspace.card.params import (
    WorkspaceRagIndex,
    WorkspaceRagList,
    WorkspaceRagSearch,
    WorkspaceRead,
)
from akgentic.tool.workspace.readers import DocumentReader

if TYPE_CHECKING:
    from collections.abc import Callable

    from akgentic.tool.vector_store.protocol import CollectionConfig
    from akgentic.tool.workspace.actor import WorkspaceActor

logger = logging.getLogger(__name__)

_UNAVAILABLE = "Retrieval indexing is not available for this workspace."
"""What all three callables answer when the actor cannot be reached.

Deliberately the same sentence the actor itself returns in degraded mode: an
agent should not have to tell "no vector store is wired" apart from "the proxy is
gone", because its next step is the same in both.
"""


class RagFactories:
    """The three retrieval factories and their binding.

    Declares no Pydantic field: the five names below are what the card supplies,
    declared under ``if TYPE_CHECKING:`` so mypy sees them and Pydantic does not.
    """

    if TYPE_CHECKING:
        workspace_rag_index: WorkspaceRagIndex | bool
        workspace_rag_list: WorkspaceRagList | bool
        workspace_rag_search: WorkspaceRagSearch | bool
        workspace_read: WorkspaceRead | bool
        rag_collection: CollectionConfig

        _workspace_proxy: WorkspaceActor | None
        _workspace_tell: WorkspaceActor | None
        _agent_id: str

    ##
    ## Enablement — one predicate, because three sites have to agree on it
    ##
    def _rag_enabled(self) -> bool:
        """Whether this card carries any retrieval capability at all.

        Read by three sites that must never disagree: the backend-derived document
        caps, the Weaviate configuration check, and the bind-time announcement. A
        card with retrieval off must create no collection, impose no Weaviate
        requirement, and shrink no cache.

        **All three capabilities are terms, and the third is the one that is easy
        to forget.** A card enabling only ``workspace_rag_search`` would otherwise
        send no ``enable_rag``, so the actor would acquire no proxy, create no
        collection and hold no chunking parameters — and every search would answer
        that retrieval is unavailable with nothing anywhere explaining why.
        """
        return (
            _resolve(self.workspace_rag_index, WorkspaceRagIndex) is not None
            or _resolve(self.workspace_rag_list, WorkspaceRagList) is not None
            or _resolve(self.workspace_rag_search, WorkspaceRagSearch) is not None
        )

    def _rag_params(self) -> WorkspaceRagIndex:
        """The chunking configuration this card contributes.

        A card that enables only ``workspace_rag_list`` still contributes one: the
        actor needs the splitter's parameters whatever made retrieval turn on, and
        the defaults are what ``WorkspaceRagIndex()`` already means.
        """
        return _resolve(self.workspace_rag_index, WorkspaceRagIndex) or WorkspaceRagIndex()

    def _rag_reader(self) -> DocumentReader:
        """The extraction configuration this card contributes.

        It has to travel to the actor, because the worker extracts and the
        extraction configuration lives on the **card**: it is nested inside
        ``workspace_read``, and a card whose ``document_reader`` is ``False`` or
        absent contributes a plain :class:`DocumentReader` rather than nothing —
        an indexer with no extractor could not index a PDF at all.
        """
        read = _resolve(self.workspace_read, WorkspaceRead)
        configured = read.document_reader if read is not None else True
        return configured if isinstance(configured, DocumentReader) else DocumentReader()

    ##
    ## Bind time — one fire-and-forget announcement
    ##
    def _announce_rag(self) -> None:
        """Tell the actor to turn retrieval on for this tree — fire and forget.

        The same shape as ``_announce_exec`` and ``_register_agent_name``, and for
        the identical reason: ``getChildrenOrCreate`` fixes ``WorkspaceConfig`` at
        creation, and the card that creates the actor for a workspace is routinely
        one with no retrieval capability at all. **The actor does not inspect
        cards** — it cannot, it has no handle on one — so a card tells it.

        **It never raises.** A stand-in proxy that does not carry the method, or an
        actor that died between the get-or-create and this line, must not take the
        whole card binding down. The degradation is a workspace whose
        ``workspace_rag_index`` answers that retrieval is unavailable: visible, and
        recoverable by rebinding.
        """
        if not self._rag_enabled():
            return
        tell = self._workspace_tell
        if tell is None:
            return
        try:
            tell.enable_rag(
                self._agent_id, self._rag_params(), self._rag_reader(), self.rag_collection
            )
        except Exception:
            logger.debug("Could not enable retrieval on #Workspace", exc_info=True)

    ##
    ## The three callables
    ##
    def _rag_search_factory(self, params: WorkspaceRagSearch) -> Callable[..., Any]:
        """Create the ``workspace_rag_search`` callable.

        A thin **ask**, and the whole of the search runs on the actor — every one
        of its four inputs lives there and none of them lives here. The extraction
        bodies the keyword leg scans, the chunk offsets it maps them through, the
        vector-store proxy and the workspace name that scopes every query are all
        actor state; a card holds a proxy and a configuration.

        Args:
            params: The result budget and the two fusion knobs, captured here so
                a team's configured values travel with the call.

        Returns:
            The callable, which never raises.
        """
        proxy = self._workspace_proxy
        top_k, alpha, threshold = params.top_k, params.alpha, params.score_threshold

        def workspace_rag_search(query: str, top_k: int = top_k, path_prefix: str = "") -> str:
            """Search the indexed workspace documents for passages about *query*.

            Combines meaning-based and word-based matching over the files that
            ``workspace_rag_index`` has indexed. Use it to find *where* something
            is said before reading a whole document.

            Args:
                query: What to look for, in natural language.
                path_prefix: Restrict the search to paths starting with this, e.g.
                    "reports/". Wildcards are not accepted. Defaults to the whole
                    workspace.
                top_k: How many passages to return.

            Returns:
                The matching passages with their file, heading path and score, or
                a sentence saying that nothing matched.
            """
            if proxy is None:
                return _UNAVAILABLE
            try:
                return str(
                    proxy.rag_search(
                        query,
                        top_k=top_k,
                        path_prefix=path_prefix,
                        alpha=alpha,
                        score_threshold=threshold,
                    )
                )
            except Exception:
                logger.debug("Could not search the retrieval index", exc_info=True)
                return _UNAVAILABLE

        workspace_rag_search.__doc__ = params.format_docstring(workspace_rag_search.__doc__)
        return workspace_rag_search

    def _rag_index_factory(self, params: WorkspaceRagIndex) -> Callable[..., Any]:
        """Create the ``workspace_rag_index`` callable.

        The closure captures the **ask** proxy, because the counts are the answer.
        What is behind that ask is bounded: a tree walk through the backend, one
        read per candidate to hash it, and up to four actor spawns. No extraction,
        no split and no embedding happens on the calling agent's thread or on the
        actor's.

        Args:
            params: The chunking configuration — captured for its docstring only;
                the actor already holds the parameters it will chunk with.

        Returns:
            The callable, which never raises and never blocks on the pipeline.
        """
        proxy = self._workspace_proxy

        def workspace_rag_index(path: str = "", force: bool = False) -> str:
            """Queue workspace files for retrieval indexing, and return immediately.

            Indexing runs in the background. Use ``workspace_rag_list`` to see
            where each file got to.

            Args:
                path: A file, a directory, or "" for the whole workspace.
                force: Re-index files that are already indexed at their current
                    content. Defaults to False.

            Returns:
                How many files were queued, were already current, and were of a
                type that cannot be indexed.
            """
            if proxy is None:
                return _UNAVAILABLE
            try:
                return str(proxy.index_paths(path, force))
            except Exception:
                logger.debug("Could not queue %r for indexing", path, exc_info=True)
                return _UNAVAILABLE

        workspace_rag_index.__doc__ = params.format_docstring(workspace_rag_index.__doc__)
        return workspace_rag_index

    def _rag_list_factory(self, params: WorkspaceRagList) -> Callable[..., Any]:
        """Create the ``workspace_rag_list`` command callable.

        COMMAND channel only. The same snapshot the context-state provider takes,
        rendered in full rather than as a delta — a person asking for the list
        wants the list, not what changed since last turn.

        Args:
            params: The render cap.

        Returns:
            The callable, which never raises.
        """
        proxy = self._workspace_proxy
        cap = params.max_pending_shown

        def workspace_rag_list() -> str:
            """Show where every workspace file stands in the retrieval index.

            Returns:
                One line per file, with a tail counting the pending files the cap
                left out.
            """
            if proxy is None:
                return _UNAVAILABLE
            try:
                return str(proxy.rag_snapshot(cap).render_full())
            except Exception:
                logger.debug("Could not render the retrieval index", exc_info=True)
                return _UNAVAILABLE

        workspace_rag_list.__doc__ = params.format_docstring(workspace_rag_list.__doc__)
        return workspace_rag_list

    def _rag_list_state_factory(
        self, params: WorkspaceRagList
    ) -> Callable[[], ContextState | None]:
        """Create the ``LLM_CONTEXT`` provider for the retrieval index.

        **One bounded ask, and no I/O of any kind behind it.** This runs on every
        turn of every agent carrying the card, so a ``stat`` per candidate — let
        alone a tree walk — would put the filesystem on the hot path for a
        display. ``rag_snapshot`` is O(n) dict work over rows that already exist.

        Args:
            params: The render cap, captured here at ``get_context_states`` time.

        Returns:
            A provider that returns ``None`` — never raises — when the actor is
            unavailable, which is the ``ContextState`` contract.
        """
        proxy = self._workspace_proxy
        cap = params.max_pending_shown

        def provider() -> ContextState | None:
            if proxy is None:
                return None  # harness shapes that wire a bare observer never bind one
            try:
                return proxy.rag_snapshot(cap)
            except Exception:
                logger.debug("Could not read the retrieval index", exc_info=True)
                return None

        return provider
