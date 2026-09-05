"""The workspace's extracted-document cache: the model, the version, the caps.

The cache itself is a field on
:class:`~akgentic.tool.workspace.models.WorkspaceState`; what lives here is what
describes one entry and what bounds the map. The actor-side lookup and fill are
in :mod:`akgentic.tool.workspace.actor.documents`. **The splitter has now
arrived** — :class:`~akgentic.tool.workspace.documents.splitter.BlockSplitter`
and the :class:`~akgentic.tool.workspace.documents.splitter.TextSplitter`
Protocol turn a cached body into chunks — and the index that 45-7 adds joins this
package rather than the actor's.

Three helpers here are deliberately **not** re-exported:
:func:`~akgentic.tool.workspace.documents.models.evict_document_bodies`, and the
splitter's two phases
:func:`~akgentic.tool.workspace.documents.splitter.parse_blocks` and
:func:`~akgentic.tool.workspace.documents.splitter.pack_blocks`. They are
internal helpers reached by their full module path, and the package façade
carries only what a caller outside the package is meant to name — for the
splitter, that is ``BlockSplitter``.
"""

from akgentic.tool.workspace.documents.models import (
    DEFAULT_MAX_DOCUMENT_CHARS,
    DEFAULT_MAX_DOCUMENTS,
    EXTRACTOR_VERSION,
    DocumentExtract,
)
from akgentic.tool.workspace.documents.splitter import BlockSplitter, Span, TextSplitter

__all__ = [
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_MAX_DOCUMENT_CHARS",
    "EXTRACTOR_VERSION",
    "BlockSplitter",
    "DocumentExtract",
    "Span",
    "TextSplitter",
]
