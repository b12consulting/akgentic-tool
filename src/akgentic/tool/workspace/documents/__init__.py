"""The workspace's extracted-document cache, its splitter, and its retrieval index.

The two maps themselves are fields on
:class:`~akgentic.tool.workspace.models.WorkspaceState`; what lives here is what
describes one entry of each and what bounds them. The actor-side lookup, fill and
indexing pipeline are in :mod:`akgentic.tool.workspace.actor.documents`.
:class:`~akgentic.tool.workspace.documents.splitter.BlockSplitter` turns a cached
body into spans, and story 45-7's index turns those spans into identified,
persisted chunks.

Several names here are deliberately **not** re-exported, in three groups:

- **Internal helpers**, reached by their full module path:
  :func:`~akgentic.tool.workspace.documents.models.evict_document_bodies`, and the
  splitter's two phases
  :func:`~akgentic.tool.workspace.documents.splitter.parse_blocks` and
  :func:`~akgentic.tool.workspace.documents.splitter.pack_blocks`.
- **The chunk-id machinery** —
  :data:`~akgentic.tool.workspace.documents.models.CHUNK_ID_NAMESPACE` and
  :func:`~akgentic.tool.workspace.documents.models.chunk_id`. They are the
  package's own minting rule, called from the worker and from nowhere outside;
  the ids themselves travel as ``RagChunk.chunk_id``.
- **The whole of :mod:`akgentic.tool.workspace.documents.worker`**, and that one is
  structural rather than editorial. It imports
  :class:`~akgentic.tool.workspace.card.params.WorkspaceRagIndex` at runtime,
  because that class is a Pydantic field type there; importing ``card.params``
  executes ``card/__init__.py``, which imports ``workspace.actor``, which imports
  ``actor/documents.py``, which imports this package. A re-export here would close
  that cycle at import time. Its module docstring says the same from the other
  side.
"""

from akgentic.tool.workspace.documents.context import RagFileRow, RagIndexState
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
from akgentic.tool.workspace.documents.splitter import BlockSplitter, Span, TextSplitter

__all__ = [
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_MAX_DOCUMENT_CHARS",
    "EMBEDDING_STALE_AFTER_S",
    "EXTRACTOR_VERSION",
    "IN_MEMORY_MAX_DOCUMENTS",
    "IN_MEMORY_MAX_DOCUMENT_CHARS",
    "RAG_COLLECTION",
    "BlockSplitter",
    "DocumentExtract",
    "NewFileMessage",
    "RagChunk",
    "RagFile",
    "RagFileRow",
    "RagIndexState",
    "RagStatus",
    "Span",
    "TextSplitter",
    "derived_document_caps",
]
