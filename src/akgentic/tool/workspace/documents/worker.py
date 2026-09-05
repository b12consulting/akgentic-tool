"""``#index-``: the off-thread half of indexing one file (ADR-045 §5).

The workspace actor owns the tree, the write gate and the extraction cache on a
single thread. Extracting an 800-page PDF and splitting it takes seconds, so
neither may ever happen on that thread — every mutation in the team queues behind
it. This module is where they happen instead: one short-lived actor per file,
which reads, extracts, splits, composes the chunk texts, reports once and stops
itself.

**It is a plain :class:`~akgentic.core.agent.Akgent`, not a ``DeferredWorker``,
and that is a correctness requirement rather than a preference.**
``DeferredWorker`` reports through ``parent.deliver(key, value)`` /
``parent.fail(key, error)``, and on ``WorkspaceActor`` those belong to a
``DeferredResultActor[…, str, ExecOutcome]`` — the **exec** result cache. An
index result delivered that way would evict a running agent's exec outcome and
mis-type the cache's value. ``EmbeddingActor`` is the package's existing idiom
for this shape and is what this follows.

**The worker extracts and splits; the actor batches and adds.** ADR §5's literal
wording puts the ``add()`` calls here, and it cannot be here: the vector store
delivers ``EmbeddingCompleted`` to the *requester* address, and this worker stops
itself the instant it reports — so a worker-issued ``add()`` would name a
requester that is dead before the first batch embeds, and the ``EMBEDDED``
transition the whole design turns on would be delivered to nothing. The actor is
also the only place that can hold ``batches_expected`` consistently with the row
it counts for.

**The three payloads are ``SerializableBaseModel`` and never ``Message``.**
``Akgent.on_receive`` emits the ``ReceivedMessage`` / ``ProcessedMessage``
telemetry sandwich only for ``Message`` instances, and consumers derive "who is
working" from exactly those two — so a ``Message`` payload would surface every
transient worker as a busy team member.

**This module is deliberately absent from ``documents/__init__.py``.** It imports
:class:`~akgentic.tool.workspace.card.params.WorkspaceRagIndex` at runtime,
because that class is a Pydantic *field type* here and a string annotation cannot
serve; importing ``card.params`` executes ``card/__init__.py``, which imports
``workspace.actor``, which imports ``actor/documents.py``, which imports this
package. Re-exporting it from the package façade would close that cycle. The one
production caller imports it inside the method that spawns — the shape
``vector_store/actor.py`` already uses for ``EmbeddingActor``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.workspace.card.params import WorkspaceRagIndex
from akgentic.tool.workspace.documents.models import RagChunk, chunk_id
from akgentic.tool.workspace.documents.splitter import BlockSplitter
from akgentic.tool.workspace.readers import DocumentReader
from akgentic.tool.workspace.workspace import get_workspace

__all__ = [
    "EMBED_BATCH_SIZE",
    "INDEX_WORKER_NAME_PREFIX",
    "MAX_CONCURRENT_INDEX_WORKERS",
    "IndexError",
    "IndexRequest",
    "IndexResult",
    "IndexWorker",
    "compose_chunk_text",
    "index_worker_name",
]

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64
"""How many chunks one ``add()`` call carries.

**Not a card parameter**: it is a provider limit, not a corpus property.
``EmbeddingService.embed`` sends every text it is given in **one**
``client.embeddings.create(input=texts)`` call, and the vector store spawns one
``EmbeddingActor`` per ``add()`` carrying every entry it was handed. An 800-page
document is ~1,900 chunks; one request of that size fails whole. Batching is what
turns that into thirty requests of which one can fail.
"""

MAX_CONCURRENT_INDEX_WORKERS = 4
"""How many files may have a live worker at once.

Derived, not taken from the ADR, which caps nothing. ``workspace_rag_index("")``
over a tree of 500 candidates would otherwise spawn 500 actors in one mailbox
turn — every one of which appears in ``Orchestrator.get_team()`` and every one of
which holds the workspace actor's teardown open. Files past the cap stay
:attr:`~akgentic.tool.workspace.documents.models.RagStatus.PENDING`, which is
what ``PENDING`` means and what ``workspace_rag_list`` renders, and the next one
is spawned when a result or an error settles.
"""

INDEX_WORKER_NAME_PREFIX = "#index-"
"""Prefix of every index worker's actor name.

**Only the leading ``#`` is load-bearing** — it is what classifies the actor as a
tool actor during the orchestrator's two-phase stop. The rest is a readability
aid, mirroring ``#embed-{collection}-{request_id}``. Deliberately not
``WORKER_NAME_PREFIX``, which belongs to the deferred mechanism this worker is
not part of.
"""

_NAME_DIGEST_CHARS = 12
"""How much of the path digest rides in the worker name — readability only."""


def index_worker_name(scope: str, path: str) -> str:
    """Return the actor name of the worker indexing *path* in *scope*.

    The path is digested rather than embedded: a workspace path may contain
    anything a filesystem allows, and an actor name is looked up by string.

    Args:
        scope: The workspace the file belongs to.
        path: Workspace-relative path of the file.

    Returns:
        ``#index-<scope>-<12 hex characters of the path digest>``.
    """
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:_NAME_DIGEST_CHARS]
    return f"{INDEX_WORKER_NAME_PREFIX}{scope}-{digest}"


def _delimiter_row(header: str) -> str:
    """Build the GFM delimiter line for a table whose header row is *header*.

    A cut table's continuation piece carries its header row so it stays
    independently readable — but the delimiter line (``|---|---|``) is not a table
    row, so it falls outside the header's offsets and the composed text would be a
    header plus body rows, which is **not a parseable table**. It is derivable
    from the header's column count, so it needs no new field on
    :class:`~akgentic.tool.workspace.documents.models.RagChunk`, and it costs one
    generated line.

    The count is ``header.count("|") - 1``, which is exact for the pipe-delimited
    form MarkItDown emits (``| a | b |`` — three pipes, two columns). A header
    written without its outer pipes would be undercounted by one; the floor of one
    keeps the result a table rather than a crash, and a wrong column count still
    re-parses.

    Args:
        header: The table's header row, verbatim.

    Returns:
        A delimiter row with one ``---`` cell per column.
    """
    columns = max(1, header.strip().count("|") - 1)
    return "| " + " | ".join(["---"] * columns) + " |"


def compose_chunk_text(markdown: str, chunk: RagChunk, prepend_heading_path: bool) -> str:
    """Build the text that is embedded for *chunk*, from *markdown* and nothing else.

    Composition is what keeps a stored chunk a pair of offsets rather than a copy
    of the document: the heading prefix and a cut table's header are re-derived
    here, at embed time, and never written back into
    :class:`~akgentic.tool.workspace.documents.models.RagChunk`.

    The order is heading prefix, header row, generated delimiter, slice. The blank
    line after the prefix is load-bearing rather than cosmetic — a table cannot
    interrupt a paragraph in GFM, so a prefix on the line immediately above would
    absorb the whole table into a paragraph and the piece would stop being a table
    at all.

    Args:
        markdown: The document the chunk's offsets index into.
        chunk: The chunk to compose.
        prepend_heading_path: Whether to lead with the enclosing headings.

    Returns:
        The composed text.
    """
    body = markdown[chunk.start : chunk.end]
    if chunk.header_start is not None and chunk.header_end is not None:
        header = markdown[chunk.header_start : chunk.header_end]
        body = f"{header}\n{_delimiter_row(header)}\n{body}"
    if prepend_heading_path and chunk.heading_path:
        return " > ".join(chunk.heading_path) + "\n\n" + body
    return body


class IndexRequest(SerializableBaseModel):
    """One file handed to an :class:`IndexWorker`.

    Attributes:
        path: Workspace-relative path of the source file.
        scope: The workspace name. It is called ``scope`` because that is what it
            becomes — in every :func:`~akgentic.tool.workspace.documents.models.chunk_id`
            and in every ``VectorEntry.scope``.
        source_sha: Digest of the source bytes this run is indexing. Echoed back
            so the actor can drop a report for a file that has since moved.
        markdown: The actor's cached extraction when it had one. ``None`` means
            the worker extracts — including when the body was evicted under the
            character cap, which is an ordinary state and never a failure.
        params: The chunking configuration, already validated.
        reader: The extraction configuration. It travels with the request because
            it lives on the **card**, not on the actor: the card that created the
            actor for a workspace is routinely one with no retrieval capability.
    """

    path: str
    scope: str
    source_sha: str
    markdown: str | None
    params: WorkspaceRagIndex
    reader: DocumentReader


class IndexResult(SerializableBaseModel):
    """What an :class:`IndexWorker` reports when it succeeded.

    Attributes:
        path: Workspace-relative path of the source file.
        scope: The workspace name, as it was requested.
        source_sha: The digest this run indexed, for attribution.
        markdown: The body the chunks index into — returned so the actor can fill
            its extraction cache when the worker was the one that extracted.
        extracted: ``False`` when the actor supplied the body, so the actor knows
            whether the cache would learn anything from filling it.
        chunks: The chunks, in document order, already carrying their ids.
        texts: The composed chunk texts, index-aligned with :attr:`chunks`.
    """

    path: str
    scope: str
    source_sha: str
    markdown: str
    extracted: bool
    chunks: list[RagChunk]
    texts: list[str]


class IndexError(SerializableBaseModel):
    """What an :class:`IndexWorker` reports when it could not produce chunks.

    Deliberately not an exception and deliberately not named ``…Error`` for the
    sake of it: it mirrors ``EmbeddingError``, the payload the vector store's
    worker reports failure with.

    Attributes:
        path: Workspace-relative path of the source file.
        scope: The workspace name, as it was requested.
        source_sha: The digest this run was indexing, for attribution.
        reason: What went wrong, in the words the index row will carry.
    """

    path: str
    scope: str
    source_sha: str
    reason: str


class IndexWorker(Akgent[BaseConfig, BaseState]):
    """Reads, extracts, splits and composes one file, then stops itself.

    **It makes no ask and holds no proxy but the one it reports through**, so
    there is no I/O client here to hand a budget to. The one call that can be
    slow without bound is the optional LLM pass inside
    :meth:`~akgentic.tool.workspace.readers.DocumentReader.extract_text`, whose
    client is built inside that class and takes no budget today; a timeout field
    on :class:`IndexRequest` would reach nothing, and a timeout that does not
    reach the I/O client is decoration.
    """

    def on_start(self) -> None:
        """Initialise the empty state this worker never writes to."""
        self.state = BaseState()
        self.state.observer(self)

    def receiveMsg_IndexRequest(self, msg: IndexRequest) -> None:  # noqa: N802
        """Produce *msg*'s chunks and report exactly once, then stop.

        Every failure — a path that escaped, a file that vanished, an extractor
        that raised, bytes that are not UTF-8 — becomes an :class:`IndexError`
        rather than an exception out of this actor. The workspace actor is the
        only party that can record it against the file, and it must be told.

        Args:
            msg: The file to index.
        """
        try:
            markdown, extracted = self._body(msg)
            chunks, texts = self._chunks(markdown, msg)
            self._report(
                IndexResult(
                    path=msg.path,
                    scope=msg.scope,
                    source_sha=msg.source_sha,
                    markdown=markdown,
                    extracted=extracted,
                    chunks=chunks,
                    texts=texts,
                )
            )
        except Exception as exc:
            self._report(
                IndexError(
                    path=msg.path,
                    scope=msg.scope,
                    source_sha=msg.source_sha,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            self.stop()

    def _body(self, msg: IndexRequest) -> tuple[str, bool]:
        """Return the Markdown to split, and whether this worker produced it.

        Reads **through** :class:`~akgentic.tool.workspace.workspace.Filesystem`,
        never by joining onto its private root: every read there goes through the
        path validation, and re-implementing the join is how a traversal gets
        back in.

        Args:
            msg: The request being served.

        Returns:
            The body and ``True`` when it was extracted here, ``False`` when the
            actor supplied it from its cache.
        """
        if msg.markdown is not None:
            return msg.markdown, False
        data = get_workspace(msg.scope).read(msg.path)
        if Path(msg.path).suffix.lower() in DocumentReader.extensions:
            return msg.reader.extract_text(data, msg.path), True
        return data.decode("utf-8"), True

    @staticmethod
    def _chunks(markdown: str, msg: IndexRequest) -> tuple[list[RagChunk], list[str]]:
        """Split *markdown* and compose one text per chunk.

        Args:
            markdown: The body to split.
            msg: The request, for the scope, path, digest and chunking parameters.

        Returns:
            The chunks and their composed texts, index-aligned.
        """
        spans = BlockSplitter().split(markdown, msg.params)
        chunks = [
            RagChunk(
                chunk_id=chunk_id(msg.scope, msg.path, msg.source_sha, ordinal),
                ordinal=ordinal,
                start=span.start,
                end=span.end,
                heading_path=span.heading_path,
                header_start=span.header_start,
                header_end=span.header_end,
            )
            for ordinal, span in enumerate(spans)
        ]
        texts = [
            compose_chunk_text(markdown, chunk, msg.params.prepend_heading_path) for chunk in chunks
        ]
        return chunks, texts

    def _report(self, payload: IndexResult | IndexError) -> None:
        """Tell the parent what happened — fire and forget, and never raising.

        A parent that has stopped between the spawn and this line must not turn a
        finished extraction into a traceback: there is nobody left to record it
        against, and this worker is about to stop either way.

        Args:
            payload: The result or the failure.
        """
        from akgentic.tool.workspace.actor import WorkspaceActor  # noqa: PLC0415 — cycle

        parent = self._parent
        if parent is None:
            logger.warning("[%s] no parent address — the index report is dropped", self.config.name)
            return
        try:
            proxy = self.proxy_tell(parent, WorkspaceActor)
            if isinstance(payload, IndexResult):
                proxy.receiveMsg_IndexResult(payload)
            else:
                proxy.receiveMsg_IndexError(payload)
        except Exception:
            logger.warning(
                "[%s] could not report the index outcome for %s",
                self.config.name,
                payload.path,
                exc_info=True,
            )
