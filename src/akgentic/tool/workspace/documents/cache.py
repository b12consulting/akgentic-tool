"""One tree's document records, reached by the card and by the actor alike.

**This object is the extraction cache.** It holds the tree key every record is
addressed by, the two caps the eviction pass applies, and the
:class:`~akgentic.tool.workspace.documents.store.DocumentStore` the records are
written through — and it is where the read-modify-write of a record is spelled
once.

**Why it is here and not on either caller.** Both halves of a document — the
cached extraction and the retrieval row — are one
:class:`~akgentic.tool.workspace.documents.store.DocumentEntry`, and two
different callers reach them:

- the **card**, on the ordinary read path. A binary read asks for a cached
  extraction and fills it on a miss, and an accepted mutation marks the paths it
  changed ``STALE``. None of that is dispatch, so none of it goes through the
  actor any more (ADR-053 Decision 6) — and it must keep working for a card that
  creates no actor at all, which is every card with neither exec nor retrieval
  enabled.
- the **actor**, inside the retrieval pipeline. An ``#index-`` worker's report
  fills the same cache, and the pipeline reads it back when it re-queues a file
  it has already extracted.

A card-side copy and an actor-side copy of the same read-modify-write would be
two chances to break Golden Rule 12 and two places a cap could be applied
differently. ``documents/`` is already this package's home for what the read and
the retrieval capabilities share (ADR-053 Decision 1), so the one implementation
lives here and both sides hold one of these.

**The card builds it and announces it**, exactly as it builds and announces the
exec hold and the vector store. The actor resolves nothing: it is handed this
object through ``configure_document_cache`` and holds it in a slot, so the caps
travel *with* the cache rather than through the actor's config — which is what
retired ``WorkspaceConfig.max_documents`` and ``max_document_chars``. Under
get-or-create those two were fixed for a team by whichever card bound first; an
announcement is last-writer-wins, like every other one.

**Nothing here holds a lock, and that is stated rather than discovered.**
:meth:`DocumentCache.hold` exists for the callers that want one; the extraction
half deliberately writes without it, and the Protocol docstring in
:mod:`akgentic.tool.workspace.documents.store` already says so and says what it
costs — a re-extract, or a lost ``STALE`` a later write re-applies. What moving
these calls off the actor widened is the *intra-team* interleaving: two agents of
one team used to be serialised by the actor's mailbox and now are not. The
cross-process case never was. Do not answer it with a lock here; it is the same
shape as the unheld-row window the Protocol already records, and it takes the
same fix when that one is taken.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from akgentic.tool.workspace.documents.models import (
    DocumentExtract,
    RagFile,
    RagStatus,
    evict_document_bodies,
)
from akgentic.tool.workspace.documents.store import DocumentEntry

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from akgentic.tool.workspace.documents.store import DocumentStore

logger = logging.getLogger(__name__)

__all__ = ["DocumentCache"]


def _extracted_at(entry: DocumentEntry) -> datetime:
    """Sort key for the eviction pass — when *entry*'s body was last extracted.

    **This is where recency lives.** A map's insertion order used to be the LRU,
    refreshed by a fill and by a hit; on disk there is no order at all, so the
    stamp the fill already wrote is the only honest one. An entry with no body
    sorts oldest, which is correct: it has nothing left for either cap to
    reclaim, so it must never displace one that has.
    """
    if entry.extract is None:
        return datetime.min.replace(tzinfo=UTC)
    return entry.extract.extracted_at


class DocumentCache:
    """The extraction cache and the index rows of one tree, over one store.

    A plain object, not a Pydantic model and not an actor: it is runtime state
    holding a store with open file handles behind it, so it is a ``PrivateAttr``
    on the card and a plain slot on the actor (Golden Rule 1b).

    ``store`` may be ``None`` and every path degrades through it rather than
    raising — a lost announcement means the cache misses and the index looks
    empty, which is visible and recoverable by rebinding. That is the same rule
    the actor's own document paths have always followed, moved here with them.
    """

    def __init__(
        self,
        store: DocumentStore | None,
        tree_key: str,
        max_documents: int,
        max_document_chars: int,
    ) -> None:
        """Bind this cache to one tree's records.

        Args:
            store: Where the records live, or ``None`` before one is resolved.
            tree_key: The resolved three-segment workspace path — the same string
                the exec hold is taken on, never the actor's ``#Workspace-``
                prefixed name. Spelled once here rather than at every call site.
            max_documents: Cap on the number of bodied entries.
            max_document_chars: Cap on the characters held across them.
        """
        self.store = store
        self.tree_key = tree_key
        self.max_documents = max_documents
        self.max_document_chars = max_document_chars

    ##
    ## The records — read, write, list, hold
    ##
    def entry(self, path: str) -> DocumentEntry:
        """Return *path*'s stored record, or a fresh empty one.

        **A miss and an unannounced store are the same answer**, deliberately: a
        fresh record with both halves absent is what every caller already
        handles, so neither case needs a branch of its own and neither can raise.
        """
        store = self.store
        if store is None:
            return DocumentEntry(path=path)
        return store.get_document(self.tree_key, path) or DocumentEntry(path=path)

    def save(self, entry: DocumentEntry) -> None:
        """Write *entry* whole, or do nothing when no store has been announced."""
        store = self.store
        if store is None:
            return
        store.put_document(self.tree_key, entry)

    def entries(self) -> list[DocumentEntry]:
        """Every stored record for this tree, in no guaranteed order.

        **One listing serves all its callers** — the drain, the render, the
        keyword leg and the eviction pass — because one file carries both halves
        of a document.

        Callers that need an order sort for themselves: a directory glob's order
        is the file system's, and leaning on it is how a render stops being
        stable across runs.
        """
        store = self.store
        if store is None:
            return []
        return store.list_documents(self.tree_key)

    def hold(self, path: str) -> AbstractContextManager[None]:
        """Serialise *path*'s record against every other process, for the block.

        An **unannounced store yields without a hold**, deliberately and for
        :meth:`entry`'s reason: every document path already degrades to a miss
        when no card has announced a store, and a raise here would be the one
        that took the actor down.

        Args:
            path: Workspace-relative path of the source document.
        """
        store = self.store
        if store is None:
            return nullcontext()
        return store.hold(self.tree_key, path)

    def next_pending(self, exclude: frozenset[str]) -> DocumentEntry | None:
        """One record waiting to be claimed, or ``None`` — never a whole listing."""
        store = self.store
        if store is None:
            return None
        return store.next_pending(self.tree_key, exclude)

    ##
    ## The extraction cache — lookup, fill, eviction
    ##
    def lookup(self, path: str, source_sha: str, extractor_version: int) -> str | None:
        """Return the cached Markdown for *path*, or ``None`` on any miss.

        A hit requires all four of: the entry exists, it was produced from these
        source bytes, it was produced by this extractor, and its body is still
        present. Four distinct reasons to miss, one answer — the caller
        re-extracts, which is correct in every one of them.

        **One ``get_document`` and no listing, and a hit writes nothing at all.**
        Recency is ``extract.extracted_at``, stamped at the fill and read off the
        disk by the eviction pass, so there is nothing for a hit to record. That
        keeps the read path free, which is the rule this method has always been
        the load-bearing case of.

        Args:
            path: Workspace-relative path of the source file.
            source_sha: Digest of the source bytes the caller just read.
            extractor_version: The extractor the caller would run on a miss.

        Returns:
            The cached Markdown, or ``None``.
        """
        extract = self.entry(path).extract
        if (
            extract is None
            or extract.source_sha != source_sha
            or extract.extractor_version != extractor_version
            or extract.markdown is None
        ):
            return None
        return extract.markdown

    def fill(self, path: str, source_sha: str, extractor_version: int, markdown: str) -> None:
        """Cache *markdown* as the extraction of *path*, then apply both caps.

        **The write reaches the disk on this turn**, and so does every eviction
        it causes. There is no dirty set to ride on and no delta to amortise
        against: the fill costs one read-modify-write of one file, which is
        nothing beside the seconds of extraction that preceded it.

        The **row half is preserved by construction** —
        ``model_copy(update={"extract": ...})`` over the stored record — so a
        re-fill can never de-index a file it was only re-reading, and a field
        added to :class:`~akgentic.tool.workspace.documents.store.DocumentEntry`
        tomorrow survives (Golden Rule 12).

        ``char_count`` is computed here rather than taken as a parameter, so it
        cannot disagree with the body it describes. Recency needs no bookkeeping:
        ``extracted_at`` is stamped here, and the eviction pass sorts on it.

        Args:
            path: Workspace-relative path of the source file.
            source_sha: Digest of the source bytes this body was extracted from.
            extractor_version: The extractor that produced this body.
            markdown: The extracted body.
        """
        extract = DocumentExtract(
            path=path,
            source_sha=source_sha,
            extractor_version=extractor_version,
            markdown=markdown,
            char_count=len(markdown),
            extracted_at=datetime.now(UTC),
        )
        self.save(self.entry(path).model_copy(update={"extract": extract}))
        self.apply_caps(path)

    def forget_extract(self, path: str) -> None:
        """Drop *path*'s cached extraction, **keeping its index row**.

        The record is removed outright only when there is no row to keep. That is
        the on-disk form of the rule the two maps used to hold structurally: the
        caps bound the *extraction cache*, and
        :func:`~akgentic.tool.workspace.documents.models.evict_document_bodies`
        states it directly — a dropped body must not de-index its file. With both
        halves in one file, unlinking on the row cap would de-index every file it
        evicted, so the row cap drops the half it is a cap on and the file
        survives for the half it is not.
        """
        entry = self.entry(path)
        if entry.row is None:
            store = self.store
            if store is not None:
                store.evict(self.tree_key, path)
            return
        self.save(entry.model_copy(update={"extract": None}))

    def apply_caps(self, filled: str) -> None:
        """Bring the cache back under both caps, least recently extracted first.

        The recency order is ``extract.extracted_at`` read off the disk, **not**
        whatever order the directory glob returned — a glob's order is the file
        system's, and evicting on it would evict an arbitrary document while
        looking exactly like an LRU.

        :func:`~akgentic.tool.workspace.documents.models.evict_document_bodies`
        is consumed **unchanged**: the same signature, the same two caps and the
        same flat return. What changed is only how its verdict is applied — a
        path it removed from the mapping lost its whole entry to the row cap and
        goes through :meth:`forget_extract`; a path still present whose
        ``markdown`` it nulled is written back with the body dropped and its row
        untouched.

        **The caps are this object's, not the actor's config's.** They are
        derived by the card from the backend the collection really resolves to,
        which is what an announcement carries and a get-or-create config could
        not: the first card of a team to bind used to fix them for every later
        one.

        Args:
            filled: The path whose fill triggered this, for the log line.
        """
        entries = {entry.path: entry for entry in sorted(self.entries(), key=_extracted_at)}
        documents = {
            path: entry.extract for path, entry in entries.items() if entry.extract is not None
        }
        evicted = evict_document_bodies(
            documents,
            max_documents=self.max_documents,
            max_document_chars=self.max_document_chars,
        )
        for path in evicted:
            remaining = documents.get(path)
            if remaining is None:
                self.forget_extract(path)
            else:
                self.save(entries[path].model_copy(update={"extract": remaining}))
        if evicted:
            # One line per fill, never one per path, and DEBUG rather than INFO:
            # on a workspace sitting at either cap this fires on every fill, and
            # the question it answers — "why does this document keep
            # re-extracting?" — is a debugging question. It is also the only
            # evidence an entry-cap eviction ever happened, since the body is
            # gone by the time anything else could look.
            #
            # "evicted" covers both remedies deliberately: the return is a flat
            # list of paths and cannot say whether a path lost its whole entry
            # or only its body (45-3's frozen shape).
            # The paths are passed through rather than joined here: at either cap
            # this line runs on every fill, and an eager join would build the
            # string even with DEBUG off. ``%s`` over the list defers all of it.
            logger.debug(
                "Filling the document cache for %s evicted (entry removed or body dropped): %s",
                filled,
                evicted,
            )

    ##
    ## The one row writer, and the one status bump a mutation causes
    ##
    def put_row(self, path: str, row: RagFile) -> None:
        """Write *row* as *path*'s index half, on this turn.

        **The only writer of a row**, which is what makes the write inventory
        structural rather than remembered: a write that bypasses this is a write
        no reader of the store can attribute, and the suite's ``ast`` canary
        refuses one.

        The extraction half is preserved by construction —
        ``model_copy(update={"row": ...})`` over the stored record — so a status
        transition can never blank a cached body, and a field added to
        :class:`~akgentic.tool.workspace.documents.store.DocumentEntry` tomorrow
        survives (Golden Rule 12).

        Args:
            path: Workspace-relative path the row describes.
            row: The row, already derived by ``model_copy(update=...)`` or built
                fresh — never rebuilt by naming fields.
        """
        self.save(self.entry(path).model_copy(update={"row": row}))

    def mark_paths_stale(self, paths: list[str]) -> None:
        """Mark every indexed path in *paths* ``STALE`` — and re-index none of them.

        Called **directly on this object** from the one point the six mutations
        converge on. It used to be a fire-and-forget ``tell`` to the actor, which
        made it a duty of an actor a read/write card no longer has: two teams on
        one tree is the normal case, so a write card on a tree another team
        indexes must keep invalidating it whether or not this card dispatches
        anything.

        **It marks and returns.** An agent mid-task rewrites the same file
        repeatedly, and auto-indexing every accepted write would spend embedding
        credits on every save and queue workers behind a file that is about to
        change again. Gate writes mark stale; uploads index.

        It writes **only when it actually changes a status**, so a tree that has
        never been indexed pays one lookup per mutated path and no write at all —
        which is the common case, and must stay cheap.

        **It takes no hold, and losing the mailbox took none away.** The
        stale-marking was already outside the record hold on the actor, which the
        ``DocumentStore`` Protocol states outright; what the direct call widens is
        the *intra-team* interleaving, from "two processes" to "two processes or
        two agents of one team". The cost is bounded the way that docstring
        already bounds it — a lost ``STALE`` that the next accepted mutation
        re-applies.

        Args:
            paths: The mutation's own write set.
        """
        now = datetime.now(UTC)
        for path in paths:
            row = self.entry(path).row
            if row is None or row.status is RagStatus.STALE:
                continue
            self.put_row(
                path, row.model_copy(update={"status": RagStatus.STALE, "updated_at": now})
            )
