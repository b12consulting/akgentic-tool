"""Capability parameters and the seeded-resource models for :class:`WorkspaceTool`.

Configuration only. Every class here is a :class:`~akgentic.tool.core.BaseToolParam`
subclass — or, for :class:`Resource`, a plain
:class:`~akgentic.core.utils.SerializableBaseModel` — carrying what a capability
is configured *with*, never what its callable is *called* with (ADR-020).

No helper and no factory lives in this module: it is the leaf of ``card/``'s
import graph, imported by every sibling and importing none of them (ADR-045 §1).
"""

from __future__ import annotations

import base64
from enum import StrEnum

from pydantic import Field, model_validator

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import COMMAND, TOOL_CALL, BaseToolParam, Channels
from akgentic.tool.sandbox.actor import CardMode
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_POLL_ATTEMPTS,
    DEFAULT_EXEC_POLL_DELAY_S,
    DEFAULT_EXEC_TIMEOUT_S,
)
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


class WorkspaceWrite(BaseToolParam):
    """Write content to a file in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceDelete(BaseToolParam):
    """Delete a file from the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceEdit(BaseToolParam):
    """Apply a surgical find-and-replace edit to a workspace file."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceMultiEdit(BaseToolParam):
    """Apply a sequence of find-and-replace edits to workspace files."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspacePatch(BaseToolParam):
    """Apply a unified diff patch to the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceMkdir(BaseToolParam):
    """Create a directory (and parents) in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceExec(BaseToolParam):
    """Run a sandboxed shell command against the team workspace.

    Configuration only — nothing here duplicates an argument of the callables it
    enables. The two budgets it carries are two different things and are easy to
    conflate:

    - ``timeout_s`` bounds the **subprocess**, and reaches
      ``subprocess.run(timeout=...)`` in the backend. It is clamped to the
      worker's own budget, which sits below the orchestrator's stop backstop.
    - ``poll_attempts`` × ``poll_delay_seconds`` bounds how long the **agent's
      own thread** waits inside the tool call. It cannot extend the first:
      raising it buys more looking, never more running.

    ``poll_attempts`` has three settings, and each is bounded by a different
    thing:

    - ``-1`` (the default) — **wait out the run.** Resolved at wiring time to
      the count whose wait is the longest still fitting the *effective run
      budget* (``min(timeout_s, DEFAULT_WORKER_TIMEOUT_S)``) plus
      :data:`~akgentic.tool.workspace.execution.EXEC_REPORT_MARGIN_S`, so the
      wait covers the worker's report and not merely the command. The common
      case then returns the command's own output and the model never sees a run
      id.
    - a **positive count** — a bounded look of ``count × poll_delay_seconds``,
      clamped to the effective run budget and **without** the margin. Exhausting
      it hands back a run id.
    - ``0`` — no polling at all: the run id comes back immediately.

    Anything below ``-1`` is a validation error rather than a second spelling of
    the sentinel.

    A run that outlives the wait is collected with ``workspace_exec_result``; it
    never outlives ``timeout_s``.
    """

    expose: set[Channels] = {TOOL_CALL}
    mode: CardMode = "auto"
    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S
    poll_attempts: int = Field(default=DEFAULT_EXEC_POLL_ATTEMPTS, ge=-1)
    poll_delay_seconds: float = DEFAULT_EXEC_POLL_DELAY_S


class WorkspaceRagIndex(BaseToolParam):
    """Chunking configuration for indexing a workspace document for retrieval.

    Configuration only, as every class in this module is: the *path* to index and
    whether to force a re-index are arguments of the callable, never fields here
    (ADR-020).

    The five numbers describe how a document's extracted Markdown is cut into
    chunks. Four of them bound the splitter; the fifth is read by the embedder
    and never by the splitter — see :attr:`prepend_heading_path`.

    Nothing consumes this class yet. Declaring it surfaces **no capability**: a
    capability exists only when :class:`~akgentic.tool.workspace.card.WorkspaceTool`
    declares a field of this type, and no field is declared. It lives in this
    module from the first day rather than beside the splitter because it is a
    ``SerializableBaseModel``, so ``serialize_type()`` will stamp this module path
    into every persisted ``__model__`` marker the moment a card carries one —
    and nothing may move afterwards.
    """

    expose: set[Channels] = {TOOL_CALL, COMMAND}

    chunk_chars: int = 1200
    """**Target** chunk size, in characters. Soft: packing stops at the first
    block that would take the chunk past it, so a chunk lands near this size from
    below and a single block larger than it is emitted whole."""

    chunk_overlap_chars: int = 150
    """Overlap **budget**, in characters, honoured in whole blocks.

    A chunk begins with as many of the previous chunk's trailing blocks as fit
    inside this budget, so a chunk never starts mid-sentence. It is not a cut
    point: the number is in characters because that is the familiar unit, but the
    unit of carriage is a block. ``0`` disables overlap."""

    max_chunk_chars: int = 4000
    """**Hard ceiling**, in characters, and the only point at which an atomic
    block — a table, a fenced or indented code block, an html block, a list — is
    ever cut. It is what keeps a chunk inside the embedding model's input limit;
    sizes are in characters rather than tokens so the splitter stays uncoupled
    from any one model, at roughly four characters per token."""

    min_chunk_chars: int = 200
    """Below this, a chunk merges **forward** with the next sibling, and only
    under the same heading path — a chunk whose next sibling sits under a
    different heading stays small, because "a chunk never crosses a heading
    boundary" outranks this. A merge that would breach :attr:`max_chunk_chars`
    also does not happen. Without both exceptions, a document of many tiny
    sections merges into one oversized chunk the embedding model then rejects."""

    prepend_heading_path: bool = True
    """Embed ``"Invoice > Payment terms > Late fees"`` ahead of a chunk's slice.

    Read by the **embedder**, never by the splitter, and that is deliberate: the
    heading context is composed at embed time from ``Span.heading_path`` and is
    never stored, which is what keeps a stored chunk a pair of offsets rather
    than a copy of the document. Nothing branches on this field inside
    ``documents/splitter.py``, and a reader should not read it as dangling."""

    @model_validator(mode="after")
    def _check_chunk_bounds(self) -> WorkspaceRagIndex:
        """Reject a configuration the splitter could not honour.

        Two inequalities, checked here so an operator reads the message at
        configuration time rather than meeting the consequence at index time:

        - ``min_chunk_chars <= chunk_chars <= max_chunk_chars`` — a target
          outside its own bounds has no meaning.
        - ``chunk_overlap_chars < chunk_chars`` — an overlap at or above the
          target does not converge. Every chunk would begin with the whole of the
          previous one, which is the classic way a splitter emits the same text
          for ever.

        Raises:
            ValueError: If either inequality fails, naming both offending values.
        """
        if self.min_chunk_chars > self.chunk_chars:
            raise ValueError(
                f"min_chunk_chars ({self.min_chunk_chars}) must not exceed "
                f"chunk_chars ({self.chunk_chars})"
            )
        if self.chunk_chars > self.max_chunk_chars:
            raise ValueError(
                f"chunk_chars ({self.chunk_chars}) must not exceed "
                f"max_chunk_chars ({self.max_chunk_chars})"
            )
        if self.chunk_overlap_chars >= self.chunk_chars:
            raise ValueError(
                f"chunk_overlap_chars ({self.chunk_overlap_chars}) must be below "
                f"chunk_chars ({self.chunk_chars}); an overlap at or above the "
                f"target does not converge"
            )
        return self


class ResourceType(StrEnum):
    """Encoding of a seeded resource's ``content`` field.

    Acts as the explicit encoding discriminator for a :class:`Resource`: it
    decides how ``content`` is decoded into bytes (see :meth:`Resource.to_bytes`).
    Encoding is always explicit — never inferred from the filename extension.
    """

    TEXT = "text"  # content is UTF-8 text, written verbatim
    IMAGE = "image"  # content is base64-encoded binary, decoded before write


class Resource(SerializableBaseModel):
    """A file seeded into the team workspace at team-creation time.

    Fully Pydantic-serializable: primitive fields plus a :class:`ResourceType`
    ``StrEnum`` only, so it round-trips cleanly through ``model_dump`` /
    ``model_validate``. The file extension lives in ``file_name`` (e.g.
    ``logo.png``); ``file_type`` carries the encoding discriminator, not a MIME
    type.
    """

    file_name: str
    file_type: ResourceType = ResourceType.TEXT
    content: str

    def to_bytes(self) -> bytes:
        """Decode ``content`` into the bytes to write to the workspace.

        Returns:
            ``base64.b64decode(content)`` when ``file_type`` is
            :attr:`ResourceType.IMAGE`, else ``content.encode("utf-8")``.

        Raises:
            binascii.Error: If ``file_type`` is :attr:`ResourceType.IMAGE` and
                ``content`` is not valid base64.
        """
        if self.file_type is ResourceType.IMAGE:
            return base64.b64decode(self.content)
        return self.content.encode("utf-8")
