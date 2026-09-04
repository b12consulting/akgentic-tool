"""The extraction cache's lookup and fill, and the actor's one notify (ADR-045 §1).

**``notify_state_change()`` is called in exactly one place in the whole
``workspace/`` package, and it is :meth:`DocumentsMixin.cache_document`.** Every
other tool actor calls it freely; this one did not call it at all before this
module existed, which is how epic 29 kept the event-store write off the read
path (ADR-036 §NFR1). A second call site added anywhere under ``workspace/`` is
a defect until a decision says otherwise — most of all on a lookup, which a read
performs and which therefore must stay free.

Both methods are O(1)/O(n) dict work on the actor thread with no I/O, so the
mailbox is the lock and there is nothing here to serialise. Neither raises: an
exception in a document handler would kill the actor that owns the write gate,
so a document path degrades — a miss, or a cache that did not grow — and never
propagates.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from akgentic.tool.workspace.documents.models import DocumentExtract, evict_document_bodies
from akgentic.tool.workspace.models import WorkspaceConfig, WorkspaceState

logger = logging.getLogger(__name__)


class DocumentsMixin:
    """The cached-extract lookup and fill over ``WorkspaceState.documents``.

    Declares no Pydantic field, no state field and no sibling method: it
    consumes the two attributes below, which the actor owns.
    """

    config: WorkspaceConfig
    state: WorkspaceState

    ##
    ## The lookup — reached through the card's **ask** proxy
    ##
    def document_extract(self, path: str, source_sha: str, extractor_version: int) -> str | None:
        """Return the cached Markdown for *path*, or ``None`` on any miss.

        A hit requires all four of: the entry exists, it was produced from these
        source bytes, it was produced by this extractor, and its body is still
        present. Four distinct reasons to miss, one answer — the caller
        re-extracts, which is correct in every one of them.

        On a hit the entry moves to the end of the LRU. **That reorder is an
        in-memory mutation with no notify**, deliberately: this is the read
        path. Persisted recency therefore lags live recency until the next fill,
        which costs at most one extra re-extraction after a resume and is not to
        be "fixed" into a notify.

        Args:
            path: Workspace-relative path of the source file.
            source_sha: Digest of the source bytes the caller just read.
            extractor_version: The extractor the caller would run on a miss.

        Returns:
            The cached Markdown, or ``None``.
        """
        entry = self.state.documents.get(path)
        if (
            entry is None
            or entry.source_sha != source_sha
            or entry.extractor_version != extractor_version
            or entry.markdown is None
        ):
            return None
        # LRU: most recently used moves to the end. In-memory only — no notify
        # on a hit, because a hit is what a read does.
        self.state.documents[path] = self.state.documents.pop(path)
        return entry.markdown

    ##
    ## The fill — reached through the card's **tell** proxy
    ##
    def cache_document(
        self, path: str, source_sha: str, extractor_version: int, markdown: str
    ) -> None:
        """Cache *markdown* as the extraction of *path*, evict, and notify once.

        The notify follows the insert **and** the eviction, so one fill is one
        event however many entries it displaced — never twice, never once per
        evicted entry. It is the one write this package puts on the actor's
        state, and it is amortised against the seconds of extraction that
        preceded it.

        ``char_count`` is computed here rather than taken as a parameter, so it
        cannot disagree with the body it describes. Re-filling a known path
        refreshes its recency instead of leaving it where it was.

        Args:
            path: Workspace-relative path of the source file.
            source_sha: Digest of the source bytes this body was extracted from.
            extractor_version: The extractor that produced this body.
            markdown: The extracted body.
        """
        self.state.documents.pop(path, None)  # a re-fill refreshes recency
        self.state.documents[path] = DocumentExtract(
            path=path,
            source_sha=source_sha,
            extractor_version=extractor_version,
            markdown=markdown,
            char_count=len(markdown),
            extracted_at=datetime.now(UTC),
        )
        evicted = evict_document_bodies(
            self.state.documents,
            max_documents=self.config.max_documents,
            max_document_chars=self.config.max_document_chars,
        )
        if evicted:
            # One line per fill, never one per path, and DEBUG rather than INFO:
            # on a workspace sitting at either cap this fires on every fill, and
            # the question it answers — "why does this document keep
            # re-extracting?" — is a debugging question. It is also the only
            # evidence an entry-cap eviction ever happened, since the row is
            # gone by the time anything else could look.
            #
            # "evicted" covers both remedies deliberately: the return is a flat
            # list of paths and cannot say whether a path lost its whole entry
            # or only its body (45-3's frozen shape).
            logger.debug(
                "Filling the document cache for %s evicted (entry removed or body dropped): %s",
                path,
                ", ".join(evicted),
            )
        self.state.notify_state_change()
