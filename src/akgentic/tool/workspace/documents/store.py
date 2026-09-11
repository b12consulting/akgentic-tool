"""What the workspace knows about a document — one file per document, not one map in memory.

The extraction cache and the retrieval index were two mappings on one actor's
persisted state, reachable only from the process holding that actor. They are one
file per source document under ``<meta>/rag/`` instead, so a second process over
the same mount reads the same cache with **no shared memory** (ADR-051 Decision
6).

**The two halves live in one file, deliberately.** ``documents`` and ``rag_index``
were keyed identically — the workspace-relative path — and every settle point
touched both: an index result caches the extraction *and* writes the row, and the
keyword search leg joins them on ``source_sha == indexed_sha``. Two files per
document would double every write and reintroduce the possibility of the halves
disagreeing about which bytes they describe. One file, two optional halves,
written whole.

**The file is named by a digest of the path, not of the content.** ADR-051
Decision 6 sketches ``<meta>/rag/<content_hash>.yaml``; that is not implementable
against the shipped code, and the divergence is deliberate. ``mark_paths_stale``
runs *because* the bytes changed, so at the moment it needs the row the old
digest is gone; a content-keyed name would also leak one file per historical
version of every file an agent edits, with no caller holding an old digest to
evict it. The digest here is a **safe, length-bounded file name** for a path that
contains ``/`` — never an identity. The readable ``path`` is stored in plain text
inside the file so a human debugging a tree can still find a document by grepping
the directory. Content versioning already lives *inside* the row:
``DocumentExtract.source_sha`` is compared on every hit,
``RagFile.indexed_sha`` on every report, and the digest is inside every chunk id.

**The metadata directory is a sibling of the tree, never a child of it.** It lives
under :func:`~akgentic.tool.workspace.workspace.meta_dir_for`'s directory, so no
read capability can name a cache file and no ``rm -rf`` from inside a sandboxed
run can reach one.
:meth:`~akgentic.tool.workspace.workspace.Filesystem._stage` is **not** reusable
for the write and that is not an oversight: it is root-confined by design and
therefore cannot reach ``<meta>``. This module writes its own atomic write,
exactly as :class:`~akgentic.tool.workspace.lock.FileLockBackend` does.

**There is no lock, and none is to be added.** Two agents contend only when
reprocessing the *same* source document, and a lost update costs one
re-extraction — which ADR-045 §C5 established this cache may cost, every byte
here being derivable from the tree and disposable. What the concurrent case
requires is not exclusion but **atomic replacement**: ``mkstemp`` plus
``Path.replace()`` means a reader never sees a partial file, whichever writer
wins. ``<meta>/exec.lock`` is the exec lock and is not for this.

**Imports run one way only** — this module imports ``documents/models.py`` for
the two shipped record models and ``workspace/models.py`` / ``workspace.py`` for
the digest and :func:`meta_dir_for`. None of those may import this one, or the
pair becomes a cycle. This module is also deliberately **not** re-exported from
``documents/__init__.py``: that package is imported by ``workspace/models.py``,
so a re-export there would close the cycle at import time. The five public names
are re-exported from ``workspace/__init__.py`` instead.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml
from pydantic import ValidationError

from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.workspace.documents.models import DocumentExtract, RagFile
from akgentic.tool.workspace.models import content_sha
from akgentic.tool.workspace.workspace import meta_dir_for

logger = logging.getLogger(__name__)

RAG_DIR_NAME = "rag"
"""The directory under ``<meta>`` holding one file per source document.

Spelled once, and only here. A second spelling is a second directory over one
tree, which is how a cache silently stops hitting.
"""

DOCUMENT_FILE_SUFFIX = ".yaml"
"""Suffix of every file this store writes — and what :meth:`list_documents` globs."""

DEFAULT_DOCUMENT_STORE = "yaml"
"""What ``AKGENTIC_DOCUMENT_STORE`` resolves to when it is unset or empty."""

DOCUMENT_STORE_ENV = "AKGENTIC_DOCUMENT_STORE"
"""Environment variable naming the registry entry to build."""


class DocumentEntry(SerializableBaseModel):
    """Everything the workspace knows about one source document, in one record.

    It **composes the two shipped models and invents nothing**. An earlier sketch
    proposed a ``ChunkEntry(chunk_id, text, status)``, which contradicts the
    shipped code twice over: :class:`~akgentic.tool.workspace.documents.models.RagChunk`
    is offsets and never text — deliberately, with the render path depending on
    it — and :class:`~akgentic.tool.workspace.documents.models.RagStatus` describes
    a **file**'s journey through the pipeline and means nothing applied to one
    chunk. A story that needed a new field on either half would be a different
    story.

    **Two ``None``s are ordinary, not a broken row.** A file queued for indexing
    but never read has a ``row`` and no ``extract``; a file read but never
    indexed has an ``extract`` and no ``row``.

    Attributes:
        path: Workspace-relative path of the source document. The key, always
            set, and stored in plain text so the directory stays greppable even
            though the file names are digests.
        extract: The cached extraction, or ``None`` when the document has never
            been read — or when the row cap dropped the entry and only the
            indexing half was written back.
        row: Where the document stands in the retrieval pipeline, or ``None``
            when it has never been queued.
    """

    path: str
    extract: DocumentExtract | None = None
    row: RagFile | None = None


@runtime_checkable
class DocumentStore(Protocol):
    """Where one tree's document records live, and the whole of that surface.

    Four methods: read one, write one, remove one, list them all. There is
    deliberately no "write many" and no transaction — every caller writes one
    document at a time on its own turn, which is what makes a torn write cost one
    document rather than a batch.

    ``@runtime_checkable`` buys an ``isinstance`` check on **method names only**
    — not a signature, not an argument count, not a return type — exactly as
    :class:`~akgentic.tool.workspace.lock.LockBackend`'s does. The real
    conformance check is mypy over ``src/``.

    Every registered backend is constructed with **no arguments**, so the registry
    can build any entry with no type switch; everything a backend needs arrives
    per call, in *tree_key*.
    """

    def get_document(self, tree_key: str, path: str) -> DocumentEntry | None:
        """Return the record for *path*, or ``None`` when there is none to read."""
        ...

    def put_document(self, tree_key: str, entry: DocumentEntry) -> None:
        """Write *entry*, replacing whatever was stored for its path."""
        ...

    def evict(self, tree_key: str, path: str) -> None:
        """Remove *path*'s record, if there is one."""
        ...

    def list_documents(self, tree_key: str) -> list[DocumentEntry]:
        """Return every readable record for *tree_key*, in no guaranteed order."""
        ...


class YamlDocumentStore:
    """One YAML file per source document, under ``<meta>/rag/``.

    **Stateless by construction.** It holds nothing per tree — everything comes
    from *tree_key* on each call — so one instance serves every tree, and two
    instances in one process are exactly as separate as two processes over one
    mount. That is what makes the cross-process guard meaningful: it puts through
    one *object* and reads through another, so nothing in-process can be doing
    the sharing.

    **Nothing here raises on a file it cannot read.** A file that does not parse,
    or that :class:`DocumentEntry` rejects, is one WARNING and a miss, and is
    **left in place**: removing a file we cannot read is how a cache turns a bad
    parse into data loss, and the entry is regenerable from the tree anyway. One
    corrupt file therefore costs one re-extraction rather than taking the
    retrieval capability down.
    """

    def get_document(self, tree_key: str, path: str) -> DocumentEntry | None:
        """Return the stored record for *path*, or ``None`` on any miss.

        Three ways to miss, one answer, because the caller re-derives in all
        three: no file, a file that is not YAML, and a mapping this model
        rejects.

        Args:
            tree_key: The three-segment ``<scope>/<kind>/<leaf>`` path
                :func:`~akgentic.tool.workspace.workspace.get_workspace` takes.
            path: Workspace-relative path of the source document.

        Returns:
            The record, or ``None``.
        """
        return self._read(self._file(tree_key, path))

    def put_document(self, tree_key: str, entry: DocumentEntry) -> None:
        """Write *entry* whole, atomically, replacing any previous record.

        The ``rag/`` directory is created **lazily, here** rather than at
        construction: a store is built for every card that binds, and a workspace
        that never reads a document must provision nothing. It is the reasoning
        :meth:`~akgentic.tool.workspace.lock.FileLockBackend.acquire` gives for
        its own lazy ``mkdir``.

        The write is temp-then-``replace()``: a reader either sees the previous
        file or the new one, never a half of either, and a serialisation or write
        that raises part-way leaves the previous file untouched and no ``.tmp``
        behind.

        Args:
            tree_key: The tree the document belongs to.
            entry: The record to store. Its ``path`` is the key.

        Raises:
            OSError: Whatever creating the directory or writing the file raised.
                The caller degrades — a cache that did not grow — and never lets
                it reach an agent.
        """
        target = self._file(tree_key, entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(target, entry)

    def evict(self, tree_key: str, path: str) -> None:
        """Remove *path*'s file, and return quietly when there is nothing to remove.

        A missing file and a missing directory are both ordinary: eviction runs
        over a cap on a tree that may never have been written to.
        """
        self._file(tree_key, path).unlink(missing_ok=True)

    def list_documents(self, tree_key: str) -> list[DocumentEntry]:
        """Return every readable record under *tree_key*'s ``rag/`` directory.

        A missing directory is an **empty list**, not an error — a workspace that
        has never read a document has no directory, and that is the ordinary
        state rather than a failure. Unreadable files are skipped exactly as
        :meth:`get_document` misses on them.

        **The order is not guaranteed** and no caller may lean on it: a directory
        glob's order is the file system's. Every caller that needs an order sorts
        for itself — eviction by ``extract.extracted_at``, the render by ``path``.
        """
        directory = self._rag_dir(tree_key)
        try:
            files = sorted(directory.glob(f"*{DOCUMENT_FILE_SUFFIX}"))
        except OSError:
            return []
        entries = [self._read(file) for file in files]
        return [entry for entry in entries if entry is not None]

    def _rag_dir(self, tree_key: str) -> Path:
        """The directory holding *tree_key*'s records — a sibling of the tree."""
        return meta_dir_for(tree_key) / RAG_DIR_NAME

    def _file(self, tree_key: str, path: str) -> Path:
        """The file *path*'s record lives in, named by a digest of the path.

        ``content_sha`` is reused rather than respelled: this package has one
        definition of a digest on purpose, and a second expression of one is how
        two callers end up disagreeing about a name.
        """
        return self._rag_dir(tree_key) / f"{content_sha(path.encode())}{DOCUMENT_FILE_SUFFIX}"

    def _read(self, file: Path) -> DocumentEntry | None:
        """Parse one record file, or answer ``None`` and leave it where it is."""
        try:
            raw = file.read_text()
        except OSError:
            return None
        try:
            return DocumentEntry.model_validate(yaml.safe_load(raw))
        except (yaml.YAMLError, ValidationError, TypeError):
            logger.warning(
                "Document record %s does not parse — treating it as a miss and leaving it "
                "in place; the entry is regenerable from the tree",
                file,
            )
            return None

    @staticmethod
    def _atomic_write(target: Path, entry: DocumentEntry) -> None:
        """Dump *entry* to a temp file beside *target*, then replace it in one step.

        Mirrors ``YamlEventStore._atomic_write`` in ``akgentic-team`` — the same
        shape, not an import: that is another submodule, and a cross-submodule
        import is what Golden Rule 4 forbids.

        The temp file is created **in the destination directory** so the replace
        is a same-filesystem rename, which is what makes it atomic. Any
        ``BaseException`` — a serialisation failure, a full disk, a
        ``KeyboardInterrupt`` between the two — unlinks it before re-raising, so
        a failed write leaves no debris for :meth:`list_documents` to trip over.
        """
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                yaml.dump(entry.model_dump(mode="json"), handle, default_flow_style=False)
            Path(tmp).replace(target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


DOCUMENT_STORE_CLASSES: dict[str, type[DocumentStore]] = {"yaml": YamlDocumentStore}
"""The registry, with the one entry this story ships.

Shaped like ``LOCK_BACKEND_CLASSES``: a mutable dict a deployment assigns its own
store into, resolved **at call time** so an entry registered after this module was
imported is still found. A Mongo or Postgres document store is a second entry here
and nothing else.
"""


def resolve_document_store() -> DocumentStore:
    """Build the store ``AKGENTIC_DOCUMENT_STORE`` names, defaulting to ``yaml``.

    Called at bind time, unconditionally, beside ``resolve_lock_backend`` — so a
    typo in the variable fails the bind in front of the admin who set it rather
    than at the first document read.

    An **empty** value falls back rather than being honoured: a compose file
    interpolating an unset variable and a bare ``FOO=`` in an env file both
    arrive here as ``""``, and neither is somebody asking for a store called the
    empty string.

    Returns:
        A fresh store. It holds no per-tree state, so the caller may share one
        instance across every tree it binds.

    Raises:
        KeyError: If the variable names no registered store — a configuration
            error, deliberately at start-up.
    """
    name = os.environ.get(DOCUMENT_STORE_ENV) or DEFAULT_DOCUMENT_STORE
    return DOCUMENT_STORE_CLASSES[name]()
