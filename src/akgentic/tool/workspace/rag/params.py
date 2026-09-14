"""The retrieval capability's parameters — index, list, search.

Configuration only. Every class here is a :class:`~akgentic.tool.core.BaseToolParam`
subclass carrying what a capability is configured *with*, never what its callable
is *called* with (ADR-020).

**They live here rather than in ``card/params.py``, and the reason is structural
rather than aesthetic.** Importing ``akgentic.tool.workspace.card.params``
executes ``card/__init__.py`` first — that is how Python imports a submodule —
and that module imports the journal, the exec machinery, the vector-store
registry and the actor. A capability module that took its parameters from there
would depend on every other capability by construction, and the property this
package's capability modules exist to have would be unobtainable.

``card/params.py`` keeps **re-exporting** these three names, and that is not
tidiness either: :class:`~akgentic.tool.core.BaseToolParam` is a
:class:`~akgentic.core.utils.SerializableBaseModel`, so a card a deployment
persisted with one of them set explicitly carries
``__model__: akgentic.tool.workspace.card.params.<Name>`` and resolves it with
``import_module`` plus ``getattr``. A path that has gone turns a stored team's
tool card into a bad record rather than a card.
"""

from __future__ import annotations

from pydantic import model_validator

from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL, BaseToolParam, Channels


class WorkspaceRagIndex(BaseToolParam):
    """Chunking configuration for indexing a workspace document for retrieval.

    Configuration only, as every class in this module is: the *path* to index and
    whether to force a re-index are arguments of the callable, never fields here
    (ADR-020).

    The five numbers describe how a document's extracted Markdown is cut into
    chunks. Four of them bound the splitter; the fifth is read by the embedder
    and never by the splitter — see :attr:`prepend_heading_path`.

    **It lives beside the capability it configures, and moving it here cost one
    compatibility path.** It sat in ``card/params.py`` from the first day because
    it is a ``SerializableBaseModel``, so ``serialize_type()`` stamps its module
    path into every persisted ``__model__`` marker the moment a card carries one
    — and that argument was read as "nothing may move afterwards". What it
    actually requires is that the **old path keep resolving**, which a re-export
    from ``card/params.py`` provides; nothing in ``src/`` or ``tests/`` imports
    through it, so the compatibility path has a guard of its own rather than
    being propped up by production code.
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
    ``rag/splitter.py``, and a reader should not read it as dangling."""

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
