"""Capability parameters and the seeded-resource models for :class:`WorkspaceTool`.

Configuration only. Every class here is a :class:`~akgentic.tool.core.BaseToolParam`
subclass — or, for :class:`Resource`, a plain
:class:`~akgentic.core.utils.SerializableBaseModel` — carrying what a capability
is configured *with*, never what its callable is *called* with (ADR-020).

No helper and no factory lives in this module: it is the leaf of ``card/``'s
import graph, imported by every sibling and importing none of them (ADR-045 §1).

**Two capabilities' parameters are re-exported, not defined here.** The read
capability's six live in :mod:`akgentic.tool.workspace.read.params` and the write
capability's six in :mod:`akgentic.tool.workspace.write.params`, beside the
closures they configure, because importing anything under ``card/`` executes
``card/__init__.py`` and therefore every other capability — so a capability
module that took its parameters from here would depend on all of them.

The re-export is what keeps stored records readable. ``serialize_type`` stamps
``f"{cls.__module__}.{cls.__name__}"`` into ``__model__`` on every
``SerializableBaseModel``, and ``BaseToolParam`` is one, so every card a
deployment persisted with a read or write parameter set explicitly carries
``akgentic.tool.workspace.card.params.<Name>``. Reading it back is
``import_module`` plus ``getattr`` on exactly that path
(:func:`akgentic.core.utils.deserializer.import_class`); a path that has gone
raises ``UnresolvableClassError``, which is how a stored team's tool card becomes
a bad record rather than a card. ``workspace/tool.py`` is on disk for the same
reason after an earlier decomposition.

``Resource`` and ``ResourceType`` stay defined here: they are the card's own
seeding vocabulary and belong to no capability.
"""

from __future__ import annotations

import base64
from enum import StrEnum

from pydantic import Field, model_validator

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL, BaseToolParam, Channels
from akgentic.tool.sandbox.backend import CardMode
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_POLL_ATTEMPTS,
    DEFAULT_EXEC_POLL_DELAY_S,
    DEFAULT_EXEC_TIMEOUT_S,
)
from akgentic.tool.workspace.read.params import (
    ExpandMediaRefs,
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceRead,
    WorkspaceView,
)
from akgentic.tool.workspace.write.params import (
    WorkspaceDelete,
    WorkspaceEdit,
    WorkspaceMkdir,
    WorkspaceMultiEdit,
    WorkspacePatch,
    WorkspaceWrite,
)

# mypy strict implies ``no_implicit_reexport``, so the twelve re-exported names
# need an explicit export to keep serving the module path stored records name.
# ``__all__`` does it in twelve lines where ``X as X`` aliases would cost an
# import statement each — the spelling ``workspace/tool.py`` already uses, and for
# the same reason.
#
# **Nothing in this package imports the twelve from here.** ``card/__init__.py``
# and ``card/rag.py`` take them from ``read/params.py`` and ``write/params.py``,
# where they are defined, so this re-export exists for exactly one purpose — the
# stored ``__model__`` markers described above — and ``test_read_capability.py``
# and ``test_write_capability.py`` are what hold it in place. Were production to
# import them from here instead, deleting the re-export would break the package
# loudly and the compatibility path would have no guard of its own.
__all__ = [
    "ExpandMediaRefs",
    "Resource",
    "ResourceType",
    "WorkspaceDelete",
    "WorkspaceEdit",
    "WorkspaceExec",
    "WorkspaceGlob",
    "WorkspaceGrep",
    "WorkspaceList",
    "WorkspaceMkdir",
    "WorkspaceMultiEdit",
    "WorkspacePatch",
    "WorkspaceRagIndex",
    "WorkspaceRagList",
    "WorkspaceRagSearch",
    "WorkspaceRead",
    "WorkspaceView",
    "WorkspaceWrite",
]


class WorkspaceExec(BaseToolParam):
    """Run a sandboxed shell command against the team workspace.

    Configuration only — nothing here duplicates an argument of the callables it
    enables. The two budgets it carries are two different things and are easy to
    conflate:

    - ``timeout_s`` bounds the **subprocess**, and reaches
      ``subprocess.run(timeout=...)`` in the backend. It is clamped to the
      :data:`~akgentic.tool.workspace.execution.MAX_EXEC_BUDGET_S`, which sits
      below the orchestrator's stop backstop.
    - ``poll_attempts`` × ``poll_delay_seconds`` bounds how long the **agent's
      own thread** waits inside the tool call. It cannot extend the first:
      raising it buys more looking, never more running.

    ``poll_attempts`` has three settings, and each is bounded by a different
    thing:

    - ``-1`` (the default) — **wait out the run.** Resolved at wiring time to
      the count whose wait is the longest still fitting the *effective run
      budget* (``effective_budget(timeout_s)``) plus
      :data:`~akgentic.tool.workspace.execution.EXEC_REPORT_MARGIN_S`, so the
      wait covers the sandbox's report and not merely the command. The common
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


class WorkspaceRagList(BaseToolParam):
    """Render where every workspace file stands in the retrieval index.

    Configuration only, as every class in this module is: the render cap is the
    one field, and there is nothing to pass at the call.

    **Deliberately not on ``TOOL_CALL``** (ADR-045 §5). The index is something the
    model should *see*, not something it should decide to look up: it is pushed
    into the context tail as a delta on every turn — one line per file that
    actually moved — and a tool call for it would be a round trip for information
    the model already has.
    """

    expose: set[Channels] = {COMMAND, LLM_CONTEXT}

    max_pending_shown: int = 20
    """How many ``pending`` rows the render may carry.

    Everything that is **not** pending is always shown, because each of those rows
    says something different. Pending rows all say the same thing, so a
    10,000-file tree would otherwise flood the context window with them; past this
    count they collapse into a single ``…and N more pending`` line.
    """


class WorkspaceRagSearch(BaseToolParam):
    """Hybrid retrieval over the workspace's indexed chunks.

    Configuration only, as every class in this module is: the query, the result
    budget and the path filter are arguments of the callable, never fields here
    (ADR-020).

    **``TOOL_CALL`` only, and deliberately.** The three retrieval capabilities
    carry three different channel sets (ADR-045 §5): indexing is
    ``{TOOL_CALL, COMMAND}``, listing is ``{COMMAND, LLM_CONTEXT}``, and search is
    ``{TOOL_CALL}`` alone. A search is something the model *does* with a question
    it has just formed, not something it is *shown* on every turn.
    """

    expose: set[Channels] = {TOOL_CALL}

    top_k: int = 5
    """How many fused hits the render may carry.

    The backend is asked for more than this — fusion reorders, and a hit whose
    chunk no longer resolves consumes no result slot — but the render is cut
    here."""

    alpha: float = 0.7
    """Weight of the vector leg in the fused score; the keyword leg gets ``1 - alpha``.

    The literal mirrors :data:`~akgentic.tool.vector_store.hybrid.DEFAULT_ALPHA`,
    which is the value ``weaviate-client`` sends for ``hybrid(alpha=...)``.
    **Written out rather than imported**: this module imports from no sibling and
    from no other subpackage, and reaching into ``vector_store/hybrid.py`` for a
    float would open a new edge on the package's longest-lived import chain. The
    two values are pinned together by a spec instead.

    ``1.0`` is pure vector search, ``0.0`` pure keyword."""

    score_threshold: float = 0.0
    """Minimum **raw** cosine score for a vector hit, applied before fusion.

    Raw and not fused, so the number keeps its absolute meaning: a fused score is
    normalised against the rest of one result set and is comparable only within
    it. ``0.0`` keeps everything the backend returned."""


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
