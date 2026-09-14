"""The workspace's extracted-document cache: what one entry holds, and what bounds it.

The records themselves are files under the tree's sibling metadata directory,
one per source document, written through a
:class:`~akgentic.tool.workspace.documents.store.DocumentStore`; what lives here
is what describes one entry of each and what bounds them. The actor-side lookup,
fill and indexing pipeline are in :mod:`akgentic.tool.workspace.actor.documents`.

**This package is shared by two capabilities, which is why it is not one.** The
extraction cache is filled and read on the **read** path — ``card/__init__.py``
resolves the store unconditionally at bind and the read closures call
``document_extract`` / ``cache_document`` — while the chunk records and the
collection name serve retrieval, and ``workspace/models.py`` takes two cap
constants from :mod:`akgentic.tool.workspace.documents.models`. Code genuinely
shared by two capabilities stays shared (ADR-053 Decision 1), so it stays here
rather than being forced under one owner.

**The retrieval-only halves have left.** The splitter, the retrieval context
state and the index worker are the retrieval capability's and live in
:mod:`akgentic.tool.workspace.rag` — which also dissolved the cycle that used to
keep the worker out of this façade: it needed
:class:`~akgentic.tool.workspace.rag.params.WorkspaceRagIndex` as a Pydantic field
type, and while that class lived under ``card/`` importing it from here would
have closed a cycle back through the actor. The parameter is a capability sibling
now, and nothing in this package names it at all.

A handful of names here are deliberately **not** re-exported, in two groups:

- **Internal helpers**, reached by their full module path:
  :func:`~akgentic.tool.workspace.documents.models.evict_document_bodies`.
- **The chunk-id machinery** —
  :data:`~akgentic.tool.workspace.documents.models.CHUNK_ID_NAMESPACE` and
  :func:`~akgentic.tool.workspace.documents.models.chunk_id`. They are the
  package's own minting rule, called from the index worker and from nowhere
  outside; the ids themselves travel as ``RagChunk.chunk_id``.
"""

from akgentic.tool.workspace.documents.models import (
    DEFAULT_MAX_DOCUMENT_CHARS,
    DEFAULT_MAX_DOCUMENTS,
    EMBEDDING_STALE_AFTER_S,
    EXTRACTOR_VERSION,
    IN_MEMORY_MAX_DOCUMENT_CHARS,
    IN_MEMORY_MAX_DOCUMENTS,
    RAG_COLLECTION,
    DocumentExtract,
    NewFileMessage,
    RagChunk,
    RagFile,
    RagStatus,
    derived_document_caps,
)

__all__ = [
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_MAX_DOCUMENT_CHARS",
    "EMBEDDING_STALE_AFTER_S",
    "EXTRACTOR_VERSION",
    "IN_MEMORY_MAX_DOCUMENTS",
    "IN_MEMORY_MAX_DOCUMENT_CHARS",
    "RAG_COLLECTION",
    "DocumentExtract",
    "NewFileMessage",
    "RagChunk",
    "RagFile",
    "RagStatus",
    "derived_document_caps",
]
