"""The read capability's parameters — read, list, glob, grep, view, media refs.

Configuration only. Every class here is a :class:`~akgentic.tool.core.BaseToolParam`
subclass carrying what a capability is configured *with*, never what its callable
is *called* with (ADR-020).

**They live here rather than in ``card/params.py``, and the reason is structural
rather than aesthetic.** Importing ``akgentic.tool.workspace.card.params``
executes ``card/__init__.py`` first — that is how Python imports a submodule —
and that module imports the journal, the exec machinery, the RAG mixin, the
vector-store registry and the actor. A capability module that took its parameters
from there would depend on every other capability by construction, and the
property this package's capability modules exist to have would be unobtainable.

``card/params.py`` keeps **re-exporting** these six names, and that is not
tidiness either: :class:`~akgentic.tool.core.BaseToolParam` is a
:class:`~akgentic.core.utils.SerializableBaseModel`, so a card a deployment
persisted with one of them set explicitly carries
``__model__: akgentic.tool.workspace.card.params.<Name>`` and resolves it with
``import_module`` plus ``getattr``. A path that has gone turns a stored team's
tool card into a bad record rather than a card.
"""

from __future__ import annotations

from akgentic.tool.core import COMMAND, TOOL_CALL, BaseToolParam, Channels
from akgentic.tool.workspace.readers import DocumentReader


class WorkspaceRead(BaseToolParam):
    """Read a file from the team workspace with pagination support."""

    expose: set[Channels] = {TOOL_CALL}
    default_limit: int = 2000

    force_document_regeneration: bool = False
    """Default for the callable's parameter of the same name: ignore a **valid**
    cached extraction and re-extract the document.

    A forced read still fills the cache with what it extracted, so it costs one
    extraction rather than turning caching off for that path.

    The meaning is new in ADR-045. It used to mean "ignore a file that happens to
    sit beside the source", which no notion of validity governed at all — the
    thing it bypassed could never say whether it described the current bytes.
    A cache entry can, so forcing now means overriding a *correct* answer, which
    is a coherent thing to ask for and a rare thing to need."""

    document_reader: DocumentReader | bool = True


class WorkspaceList(BaseToolParam):
    """List immediate children of a directory in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
    max_depth: int = 1  # 1 = flat list (default), 0 = unlimited, N = N levels deep


class WorkspaceGlob(BaseToolParam):
    """Find files matching a glob pattern in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
    max_results: int = 100


class WorkspaceGrep(BaseToolParam):
    """Search file contents by regex in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
    max_results: int = 100
    max_line_length: int = 2000


class ExpandMediaRefs(BaseToolParam):
    """Expand ``!!glob_pattern`` tokens in a prompt into binary image content.

    COMMAND channel only — never exposed as an LLM tool.
    """

    expose: set[Channels] = {COMMAND}


class WorkspaceView(BaseToolParam):
    """View an image file from the team workspace as binary content for LLM vision."""

    expose: set[Channels] = {TOOL_CALL}
    max_dimension: int = 1568
    """Longest-side pixel cap. Images exceeding this are resized (aspect-ratio preserved, LANCZOS).
    Set to 0 to disable resizing and return raw bytes."""
