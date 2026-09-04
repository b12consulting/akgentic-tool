"""The extraction cache's model, its version stamp, and its two caps (ADR-045 §3).

**This module is the final home of :class:`DocumentExtract`, chosen now on
purpose.** ``serialize()`` stamps ``__model__ = "<module>.<name>"`` into every
nested ``SerializableBaseModel``, and the deserializer resolves that literal
string with ``import_module`` plus ``getattr``. An extract persisted inside a
``WorkspaceState`` snapshot therefore pins this module path in deployments this
repository cannot see — the failure mode that forced ``workspace/tool.py`` to
stay on disk as a shim. Nothing here moves afterwards.

**This module adds no digest.** ``content_sha`` in
:mod:`akgentic.tool.workspace.models` is the one definition of the digest in the
package, and ``source_sha`` below is produced by the *caller* and only compared
here. A second digest expression is how a gate fails closed while looking
healthy; the same reasoning applies to a cache that would then never hit.

The import edge runs one way — :mod:`akgentic.tool.workspace.models` imports
from here — so nothing in this module may import from it.
"""

from __future__ import annotations

from datetime import datetime

from akgentic.core.utils.serializer import SerializableBaseModel

__all__ = [
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_MAX_DOCUMENT_CHARS",
    "EXTRACTOR_VERSION",
    "DocumentExtract",
    "evict_document_bodies",
]

EXTRACTOR_VERSION = 1
"""Stamp on every cached extract, and the whole of the invalidation protocol.

A cached extract is served only while its stamp equals this constant, so
**bumping it invalidates every cached extract in every workspace on the next
read** — no sweep, no migration, no code that has to find the stale rows. The
bodies simply stop being hits and are re-extracted once each, on demand.

That is what makes an extractor defect repairable after the fact. The standing
example is a ``U+0000`` a reader wrote verbatim into every sidecar it produced:
once the sanitiser lands, one increment here is the entire remediation.

Nothing in story 45-3 bumps it. A bump belongs to whoever changes what the
extractor *produces* from unchanged source bytes.
"""

DEFAULT_MAX_DOCUMENTS = 32
"""Bound on the number of rows in ``WorkspaceState.documents``.

Answers the **metadata** dimension: ~200 bytes per row, re-serialised in full by
``model_dump_json()`` on every notify, growing for the life of the team. An
uncapped map on a team singleton leaks exactly that way — the same reasoning
:data:`~akgentic.tool.workspace.models.DEFAULT_MAX_TRACKED_WRITERS` records.

This is the Weaviate / RAG-off default. A backend-derived override belongs at
the one ``WorkspaceConfig`` construction site and is 45-7's, not this story's.
"""

DEFAULT_MAX_DOCUMENT_CHARS = 2_000_000
"""Bound on the characters held across every entry that still has a body.

Answers the **bytes** dimension, which is a different pressure from the row
count and therefore has a different remedy — see :func:`evict_document_bodies`.

This is the Weaviate / RAG-off default, as above.
"""


class DocumentExtract(SerializableBaseModel):
    """One document's extracted Markdown, keyed in the cache by its workspace path.

    Every field is a primitive, so the model round-trips through Pydantic
    unaided: no ``arbitrary_types_allowed`` of its own, and no ``PrivateAttr`` —
    there is no runtime state here to keep out of a snapshot.

    Attributes:
        path: Workspace-relative path of the **source** file, not of a sidecar.
        source_sha: :func:`~akgentic.tool.workspace.models.content_sha` of the
            source bytes this extract was produced from. Supplied by the caller
            and only ever compared here, never recomputed.
        extractor_version: The value of :data:`EXTRACTOR_VERSION` in force when
            the body was produced. An entry stamped with anything else is a miss.
        markdown: The extracted body, or ``None`` when it was dropped under the
            character cap.

            **Nothing may slice this without first checking it is present.** The
            ``str | None`` type is the enforcement and mypy ``--strict`` is the
            gate: no ``# type: ignore`` and no ``assert x is not None`` on this
            field anywhere in ``src/``. A ``None`` body is an ordinary,
            *expected* state — the entry is metadata awaiting a re-extraction,
            not a broken row.

            It also says nothing about whether the file is indexed. See
            :func:`evict_document_bodies`.
        char_count: ``len(markdown)`` as produced, computed at the fill site so
            it cannot disagree with the body it describes. It **survives a body
            drop**, which is what lets a dropped body stop pressing on the
            character cap while the row still records how large it was.
        extracted_at: When the extraction ran, in UTC.
    """

    path: str
    source_sha: str
    extractor_version: int
    markdown: str | None
    char_count: int
    extracted_at: datetime


def _bodied_chars(documents: dict[str, DocumentExtract]) -> int:
    """Sum ``char_count`` over the entries that still carry a body."""
    return sum(entry.char_count for entry in documents.values() if entry.markdown is not None)


def evict_document_bodies(
    documents: dict[str, DocumentExtract], *, max_documents: int, max_document_chars: int
) -> list[str]:
    """Bring *documents* back under both caps, least-recently-used first.

    Pure: it takes a dict and two ints, and touches no actor, no ``self`` and no
    chunk structure. *documents* is mutated in place and iterated in insertion
    order, which is the whole of the LRU — the fill site re-inserts on every
    write and the lookup moves an entry to the end on every hit.

    **Two caps, two different remedies**, because they answer two different
    pressures:

    - Over *max_document_chars* — the least-recently-used entry that still has a
      body **keeps its metadata and drops its body**. The bytes are what press
      on the snapshot, and the row that remains is what lets a later read know
      the file was seen at all.
    - Over *max_documents* — the least-recently-used **entry is removed
      outright**. Dropping only bodies here would leave the row count unbounded,
      and an uncapped map on a team singleton leaks for the life of the team.

    Both are safe: every byte is regenerable from the tree, so an eviction costs
    one re-extraction and never a wrong answer. **A single document larger than
    *max_document_chars* drops its own body on insert** and becomes a permanent
    miss costing one re-extraction per read — correct, and deliberately not
    special-cased.

    **A dropped body must not de-index its file.** Nothing anywhere may infer
    that a file is indexed, searchable, or absent from the index from the
    presence, the absence, or the body-state of an entry in this map. Index
    membership lives in its own map and is keyed off that alone.

    Args:
        documents: The cache, mutated in place.
        max_documents: Cap on the number of entries.
        max_document_chars: Cap on the characters held across bodied entries.

    Returns:
        The paths whose entry was removed or whose body was dropped, in the
        order it happened, for the caller's log line.
    """
    evicted: list[str] = []
    while len(documents) > max_documents:
        oldest = next(iter(documents))
        del documents[oldest]
        evicted.append(oldest)
    while _bodied_chars(documents) > max_document_chars:
        victim = next(
            (path for path, entry in documents.items() if entry.markdown is not None), None
        )
        if victim is None:
            # Every body is already gone and the cap is still exceeded, which a
            # cap below one document's size makes reachable. Nothing is left to
            # drop, so the loop must exit rather than spin.
            break
        # Golden Rule #12: copy and override the one field that changes. A
        # field-by-field rebuild is correct on the day it is written and
        # silently destroys the seventh field the day one is added.
        documents[victim] = documents[victim].model_copy(update={"markdown": None})
        evicted.append(victim)
    return evicted
