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

**Half of "there is no lock, and none is to be added" survives; half of it was
false.** The surviving half is the **extraction cache**: two agents contend only
when reprocessing the same source document, a lost update costs one
re-extraction, and ADR-045 §C5 established this cache may cost exactly that,
every byte here being derivable from the tree and disposable. For that half what
the concurrent case requires is not exclusion but **atomic replacement** —
``mkstemp`` plus ``Path.replace()`` means a reader never sees a partial file,
whichever writer wins.

The half that was false is the **index row** in the same file, and the argument
never transferred to it. A row is not derivable and a lost update to one is not
one re-extraction: two processes can both see an absent or ``PENDING`` row and
both spawn an ``IndexWorker`` for one file — **two paid embedding runs** — and a
whole-file replace can lose a ``superseded_chunk_ids`` list, **orphaning vectors
that have already been paid for**, with nothing raised. The mailbox used to
serialise this; nothing replaced it when the records moved to disk.

So there **is** a lock now, and it is per document: :meth:`DocumentStore.hold`,
an exclusive ``flock`` on ``<meta>/locks/record-<digest of path>``. Every
read-modify-write of one document's row runs inside that document's hold; the
extraction cache's writes deliberately do not, because that is the half the
original argument genuinely covers. ``<meta>/exec.lock`` is the exec lock and is
still not for this.

**Imports run one way only** — this module imports ``documents/models.py`` for
the two shipped record models and ``workspace/models.py`` / ``workspace.py`` for
the digest and :func:`meta_dir_for`. None of those may import this one, or the
pair becomes a cycle. This module is also deliberately **not** re-exported from
``documents/__init__.py``: that package is imported by ``workspace/models.py``,
so a re-export there would close the cycle at import time. The five public names
are re-exported from ``workspace/__init__.py`` instead.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import yaml
from pydantic import ValidationError

from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.workspace.documents.models import DocumentExtract, RagFile, RagStatus
from akgentic.tool.workspace.models import content_sha
from akgentic.tool.workspace.workspace import meta_dir_for

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

logger = logging.getLogger(__name__)

RAG_DIR_NAME = "rag"
"""The directory under ``<meta>`` holding one file per source document.

Spelled once, and only here. A second spelling is a second directory over one
tree, which is how a cache silently stops hitting.
"""

DOCUMENT_FILE_SUFFIX = ".yaml"
"""Suffix of every file this store writes — and what :meth:`list_documents` globs."""

LOCKS_DIR_NAME = "locks"
"""The directory under ``<meta>`` holding one lock file per contended thing.

**Spelled here rather than imported**, and it is one of three such spellings —
``write/gate.py`` and ``vector_store/backends/local.py`` hold the others.
Importing ``lock_file_for`` from the gate would make the retrieval capability
import the **write** capability at runtime, which is worse than a third copy and
is the same call this module already makes for its own ``_atomic_write``. A
single spine helper for the family is a cross-capability decision, recorded as an
open question rather than taken here.
"""

RECORD_LOCK_PREFIX = "record-"
"""What a document record's lock file is named, before the digest of its path."""

_LOCK_FILE_MODE = 0o600
"""Owner-only, like every other lock file under ``<meta>``."""

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

    Six methods: read one, write one, remove one, list them all, find the next
    pending one, and hold one. There is deliberately no "write many" and no
    transaction — every caller writes one document at a time on its own turn,
    which is what makes a torn write cost one document rather than a batch.

    **The atomicity contract is :meth:`hold`, and it is a requirement on the
    backend rather than advice to the caller.** A read-modify-write of one
    document's *row* that has to exclude another process runs inside that
    document's hold, with the read taken **inside** it — a hold around a decision
    made on an earlier read serialises nothing. A backend that cannot serialise
    one document's read-modify-write across processes is not a valid registry
    entry: without it two processes both spawn an index worker for one file,
    which is two paid embedding runs, and a ``superseded_chunk_ids`` list is lost
    in a whole-file replace, which orphans vectors already paid for. The two
    contended sequences — the pending-spawn claim and the superseded clear — are
    held; so is ``index_paths``' accounted-for/enqueue pair.

    The **extraction** half is deliberately outside that rule — the batch
    counters, ``DocumentCache.fill``, the cache eviction and the stale-marking
    all write without a hold. Once the spawn is exclusive, one process owns a file's
    in-flight lifecycle, so those have a single writer for the duration; and a
    cached extraction is derivable and disposable, which is the half of ADR-051
    Decision 6's argument that survives.

    **Two row writes are outside it as well, and saying so is the point of
    writing a contract down.** A worker's own report — the settle in
    ``_on_index_result`` and the ``FAILED`` transition in ``_fail`` — reads the
    row and writes it back without a hold, on the single-writer argument above.
    That argument is now *almost* always true rather than always true: deleting
    the reaper's per-process exemption means a row whose worker overruns the
    stale bound can be re-queued and re-claimed elsewhere, so a concurrent
    ``_enqueue`` can append superseded ids between that read and that write and
    lose them. The window is two statements and the cost is orphaned vectors
    rather than a wrong answer. It is not closed here because closing it is
    surgery rather than a wrap: ``_on_index_result`` goes on to call
    ``_drop_superseded``, which takes this hold, and ``_fail`` is itself called
    from inside the claim's hold — so a hold added naively at either site
    deadlocks against a second descriptor on the same file. It wants its own
    red-first cross-process spec, and is recorded as a deferred finding.

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

    def next_pending(self, tree_key: str, exclude: frozenset[str]) -> DocumentEntry | None:
        """Return one record whose row is ``PENDING`` and whose path is not excluded.

        The filtered read the drain needs, so that spawning four workers no longer
        costs four full listings of every record on the tree.

        *exclude* is **required and has no default**: it is what terminates the
        drain's loop after a claim is lost to another process, and a default would
        let a caller omit it and spin. The same principle
        ``resolve_workspace_path``'s ``workspace_sharable`` follows.

        Args:
            tree_key: The tree to look in.
            exclude: Paths this caller has already tried and lost.

        Returns:
            One matching record, in no guaranteed order, or ``None``.
        """
        ...

    def hold(self, tree_key: str, path: str) -> AbstractContextManager[None]:
        """Hold *path*'s record exclusively, across processes, for the block.

        See the class docstring: this is the Protocol's atomicity contract, not a
        convenience. The hold is per **document**, so two files are worked on
        concurrently and only one file's read-modify-write is serialised.
        """
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
        entries = [self._read(file) for file in self._record_files(tree_key)]
        return [entry for entry in entries if entry is not None]

    def next_pending(self, tree_key: str, exclude: frozenset[str]) -> DocumentEntry | None:
        """Return the first ``PENDING`` record not in *exclude*, reading no further.

        **It stops at the first match**, and short-circuiting is the whole of the
        saving — not a bound. A drain that meets a waiting record early reads one
        record on a tree of a thousand; one whose only pending path sorts last, or
        which is offered nothing at all, still walks the directory, exactly as
        :meth:`list_documents` does. What it never does is *build* the thousand,
        and it never reads past the record it answers with. The order is the
        glob's and no caller may lean on it — the drain asks for *a* pending path,
        never a particular one.

        Args:
            tree_key: The tree to look in.
            exclude: Paths the caller has already tried and lost a claim on.

        Returns:
            One matching record, or ``None`` when there is none left.
        """
        for file in self._record_files(tree_key):
            entry = self._read(file)
            if entry is None or entry.path in exclude:
                continue
            if entry.row is not None and entry.row.status is RagStatus.PENDING:
                return entry
        return None

    @contextlib.contextmanager
    def hold(self, tree_key: str, path: str) -> Iterator[None]:
        """Take *path*'s record lock exclusively, and release it whatever happens.

        The eight lines are ``CardGate._hold`` / ``LocalBackend._hold``'s idiom,
        **copied rather than imported** — see :data:`LOCKS_DIR_NAME` for why — on a
        lazily created file that is unlocked and closed in ``finally`` and **never
        unlinked**: unlinking would let a second process create a fresh inode and
        take a hold that excludes nobody, which is the classic way a file lock
        stops locking.

        A ``<meta>`` whose locks directory cannot be created logs one WARNING and
        yields unserialised, which is ``CardGate._hold``'s stated choice copied
        rather than re-decided. Failing closed instead would wedge the whole
        retrieval pipeline on a tree whose ``<meta>`` went read-only mid-session,
        and the degradation is the behaviour that was on offer before this lock
        existed at all.

        Args:
            tree_key: The tree the document belongs to.
            path: Workspace-relative path of the source document.
        """
        handle: int | None = None
        try:
            try:
                lock_path = self._lock_file(tree_key, path)
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, _LOCK_FILE_MODE)
                fcntl.flock(handle, fcntl.LOCK_EX)
            except OSError:
                logger.warning(
                    "Workspace %s: could not take the record lock for %s — proceeding unserialised",
                    tree_key,
                    path,
                    exc_info=True,
                )
            yield
        finally:
            if handle is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle, fcntl.LOCK_UN)
                os.close(handle)

    def _record_files(self, tree_key: str) -> list[Path]:
        """Every record file under *tree_key*'s ``rag/`` directory, sorted.

        A missing directory is an **empty list**, not an error: a workspace that
        has never read a document has no directory, and that is the ordinary state
        rather than a failure.
        """
        try:
            return sorted(self._rag_dir(tree_key).glob(f"*{DOCUMENT_FILE_SUFFIX}"))
        except OSError:
            return []

    def _lock_file(self, tree_key: str, path: str) -> Path:
        """The lock file *path*'s record is contended on — a sibling of the tree.

        Named by the same digest :meth:`_file` uses, **for safety rather than for
        identity**: a workspace-relative path carries slashes and may be longer
        than a filename may be. Two paths colliding would cost a spurious
        serialisation and never a lost update.
        """
        return (
            meta_dir_for(tree_key)
            / LOCKS_DIR_NAME
            / f"{RECORD_LOCK_PREFIX}{content_sha(path.encode())}"
        )

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
