"""The workspace's extracted-document cache: the model, the version, the caps.

The cache itself is a field on
:class:`~akgentic.tool.workspace.models.WorkspaceState`; what lives here is what
describes one entry and what bounds the map. The actor-side lookup and fill are
in :mod:`akgentic.tool.workspace.actor.documents`, and the splitter and the
index that 45-5 and 45-7 add join this package rather than the actor's.

:func:`~akgentic.tool.workspace.documents.models.evict_document_bodies` is
deliberately **not** re-exported: it is an internal helper reached by its full
module path, and the package façade carries only what a caller outside the
package is meant to name.
"""

from akgentic.tool.workspace.documents.models import (
    DEFAULT_MAX_DOCUMENT_CHARS,
    DEFAULT_MAX_DOCUMENTS,
    EXTRACTOR_VERSION,
    DocumentExtract,
)

__all__ = [
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_MAX_DOCUMENT_CHARS",
    "EXTRACTOR_VERSION",
    "DocumentExtract",
]
