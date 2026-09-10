"""A numpy index over files, so a second process on the same mount sees it.

The in-memory backend keeps its index inside the store actor's serialisable
state: reachable only from the process holding that actor, and lost the moment a
tree is bound from a second worker. This backend keeps the same
:class:`~akgentic.tool.vector_store.vector.VectorIndex` — **this is a persistence
change, not a scoring one** — and writes it under ``<root>/index/<collection>/``,
where ``<root>`` is the workspace's ``<meta>`` sibling
(:func:`~akgentic.tool.workspace.workspace.meta_dir_for`, ADR-051 Decision 9).

**One file, replaced whole, and its ``mtime`` is the version stamp.**
``index.npz`` carries the matrix and, beside it, the per-row metadata in the same
row order; it is written temp-then-:func:`os.replace`, so a reader meets either
the whole previous version or the whole new one and a write interrupted before
the rename leaves the previous version untouched. **One file rather than two is
the decision, not an implementation detail**: a matrix and a metadata file are
replaced by two separate renames, so a crash between them publishes a matrix
whose rows the metadata does not describe — and no ordering of the two removes
that, it only chooses which half is stale.

**Exclusion is a ``flock`` on ``<root>/locks/index-<collection>``, plus a
``threading.Lock`` in process.** The file lock is what two *processes* over one
mount share — a ``threading.Lock`` excludes nothing across a mount, exactly as
:class:`~akgentic.tool.workspace.lock.FileLockBackend` records for the exec
marker — and the thread lock is what keeps two agent threads inside one process
from interleaving a read-modify-publish cycle. Mutations take the file lock
exclusively; a reload takes it shared, so a reader can never observe the window
between the two ``os.replace`` calls.

**Coherence is by ``mtime``, and nothing here promises more.** Two teams on one
tree hold two matrices over one directory: each sees the other's write on its
next operation after the stamp moves, and on a network mount the attribute cache
bounds that further. That is accepted (ADR-051 *Consequences*); this backend does
not poll, does not notify, and must not grow a spec asserting instant
cross-instance visibility.

**There is deliberately no ``get_state`` / ``restore_state``.** The whole point
is that this index does **not** go into ``VectorStoreState``: a no-op pair would
be a false claim the store actor acts on, and a real pair would re-serialise the
whole matrix on every mutation — the row ADR-045 §7 names as the one to avoid.
Its :class:`~akgentic.tool.vector_store.registry.BackendSpec` therefore declares
``persists_in_actor_state=False`` and ``needs_actor=True``.
"""

from __future__ import annotations

import contextlib
import fcntl
import io
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from akgentic.tool.vector_store.backends.inmemory import _entry_matches
from akgentic.tool.vector_store.protocol import (
    CollectionStatus,
    SearchHit,
    SearchResult,
    VectorQuery,
    VectorStoreParam,
    check_path_prefix,
    check_shared_scope,
)
from akgentic.tool.vector_store.registry import BackendContext, BackendSpec, register_backend
from akgentic.tool.vector_store.vector import (
    VectorEntry,
    VectorIndex,
    _check_vector_search_dependencies,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

INDEX_DIR_NAME = "index"
"""The directory under ``<root>`` holding one subdirectory per collection."""

LOCKS_DIR_NAME = "locks"
"""The directory under ``<root>`` holding one lock file per collection."""

LOCK_FILE_PREFIX = "index-"
"""What a collection's lock file is named, before the collection's own name."""

INDEX_FILE = "index.npz"
"""The whole of a collection on disk: the matrix, the row metadata, the config.

Its ``mtime`` is the version stamp every other instance reloads against, and it
can be, precisely because it is the only file: there is no second write for a
stamp to run ahead of.
"""

VECTORS_KEY = "vectors"
"""The archive member holding the ``(N, D)`` matrix, in row order."""

METADATA_KEY = "metadata"
"""The archive member holding the row metadata and config, as UTF-8 JSON bytes.

Stored as a ``uint8`` array rather than a numpy string: a unicode array costs
four bytes per character for text that is ASCII by construction, and
``allow_pickle`` stays off either way.
"""

_LOCAL_ROOT_REQUIRED = (
    "The 'local' vector backend indexes into files and therefore needs a root "
    "directory, but BackendContext.root was None. The consumer that has a "
    "filesystem stamps VectorStoreParam.root at bind time; a consumer with no "
    "filesystem — the knowledge graph, the plan — must name a different backend."
)
"""Why a ``local`` backend built with no root refuses rather than picking one.

Defaulting to the process working directory is the failure this message exists
to prevent: every tree in the deployment would then index into one directory
nobody chose, and the mistake would surface as cross-tree search hits rather
than as an error.
"""


class LocalBackend:
    """One ``VectorIndex`` per collection, persisted under ``<root>/index/``.

    A plain Python class rather than a Pydantic model, for
    :class:`~akgentic.tool.vector_store.backends.inmemory.InMemoryBackend`'s
    reason: it holds numpy arrays. It satisfies ``VectorStoreService``
    structurally and adds no method to that protocol — ``create_collection`` /
    ``add`` / ``remove`` / ``search`` are the whole surface (ADR-051 Decision 7).

    Args:
        root: The directory this backend hangs ``index/`` and ``locks/`` off.
            Never created here — :meth:`create_collection` and the writes below
            create what they need, lazily, exactly as ``FileLockBackend.acquire``
            does.
    """

    def __init__(self, root: str) -> None:
        _check_vector_search_dependencies()
        self._root = Path(root)
        self._lock = threading.Lock()
        self._collections: dict[str, VectorIndex] = {}
        self._configs: dict[str, VectorStoreParam] = {}
        self._stamps: dict[str, tuple[int, int] | None] = {}

    # ------------------------------------------------------------------
    # VectorStoreService protocol methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        """Adopt *name*'s stored index, or publish an empty one for it.

        Idempotent in both directions, which is what a second instance over the
        same root requires: a directory that already holds an index is **loaded**
        rather than replaced, so the instance that arrives second inherits the
        rows the first one wrote instead of blanking them.

        Args:
            name: Unique collection identifier.
            config: Collection configuration, stored beside the rows so a later
                instance can read back what this collection was created with.
        """
        with self._lock, self._hold(name, fcntl.LOCK_EX):
            self._configs[name] = config
            if self._load_locked(name):
                return
            self._collections.setdefault(name, VectorIndex())
            self._publish_locked(name)

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        """Ingest embedding entries and publish the index they land in.

        Reloads first, under the same exclusive hold as the write, so a concurrent
        writer's rows are folded in rather than overwritten: read-modify-publish
        is one critical section, not three.

        Args:
            collection: Target collection name.
            entries: List of vector entries to store.

        Raises:
            ValueError: If the collection does not exist.
        """
        with self._lock, self._hold(collection, fcntl.LOCK_EX):
            self._refresh_locked(collection)
            index = self._get_index(collection)
            for entry in entries:
                index.add(entry)
            self._publish_locked(collection)

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        """Remove entries from a collection by reference ID, then publish.

        The two guards are the two ``InMemoryBackend.remove`` runs, on the same
        lines and in the same order, because the rules they enforce belong to the
        protocol rather than to a backend: a ``path_prefix`` carrying ``*`` or
        ``?`` means two different things on two backends, and a shared collection
        has no boundary but ``scope``.

        Args:
            collection: Target collection name.
            ref_ids: List of reference IDs to remove.
            scope: Restrict removal to entries carrying this ``scope``.
            path_prefix: Restrict removal to entries whose ``path`` starts with this.

        Raises:
            ValueError: If the collection does not exist, if ``path_prefix``
                contains ``*`` or ``?``, or if the collection is shared across
                teams and no ``scope`` was given.
        """
        check_path_prefix(path_prefix)
        check_shared_scope(collection, scope)
        with self._lock, self._hold(collection, fcntl.LOCK_EX):
            self._refresh_locked(collection)
            index = self._get_index(collection)
            if scope is None and path_prefix is None:
                index.remove(set(ref_ids))
            else:
                index.remove(
                    set(ref_ids),
                    matches=lambda entry: _entry_matches(entry, scope, path_prefix),
                )
            self._publish_locked(collection)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
        query: VectorQuery | None = None,
    ) -> SearchResult:
        """Search a collection by cosine similarity, over whatever is on disk now.

        The stored pair is re-read only when its stamp has moved; a search that
        follows no write re-reads nothing and scores the matrix already in
        memory.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            scope: Restrict the search to entries carrying this ``scope``.
            path_prefix: Restrict the search to entries whose ``path`` starts with this.
            query: Optional filters / score threshold.

        Returns:
            Search results with hits ranked by cosine similarity and collection
            status ``READY``.

        Raises:
            ValueError: If the collection does not exist, if ``path_prefix``
                contains ``*`` or ``?``, or if the collection is shared across
                teams and no ``scope`` was given. One team may hold two
                ``WorkspaceTool`` cards on two trees, and two teams may hold one,
                so an unscoped query crosses a boundary here as much as on a
                cluster.
        """
        check_path_prefix(path_prefix)
        check_shared_scope(collection, scope)
        with self._lock, self._hold(collection, fcntl.LOCK_SH):
            self._refresh_locked(collection)
            index = self._get_index(collection)
            scoped = scope is not None or path_prefix is not None
            refined = query is not None and (
                bool(query.filters) or query.score_threshold is not None
            )
            if not scoped and not refined:
                hits = _map_search_hits(index, index.search_cosine(query_vector, top_k))
            else:
                hits = _filtered_search(
                    index, query_vector, top_k, query, scope=scope, path_prefix=path_prefix
                )
        return SearchResult(hits=hits, status=CollectionStatus.READY)

    # ------------------------------------------------------------------
    # Locking — the file hold two processes share, lazily created
    # ------------------------------------------------------------------

    def _lock_file(self, collection: str) -> Path:
        """The lock file *collection*'s writers contend on."""
        return self._root / LOCKS_DIR_NAME / f"{LOCK_FILE_PREFIX}{collection}"

    @contextlib.contextmanager
    def _hold(self, collection: str, operation: int) -> Iterator[None]:
        """Hold *collection*'s ``flock`` for the duration of the block.

        *operation* is ``fcntl.LOCK_EX`` for a mutation and ``fcntl.LOCK_SH`` for
        a reload — one generator rather than two, because the only difference
        between a writer's hold and a reader's is that constant.

        **Taken at the four public methods and nowhere below them, deliberately.**
        A ``flock`` belongs to the open file description, not to the process, so a
        nested ``_hold`` on a second descriptor would block against this process's
        own exclusive hold and deadlock a single-threaded caller. The private
        helpers therefore assume a hold and never take one.

        The lock file is created lazily and **never unlinked**: unlinking it would
        let a second process create a fresh inode and take a hold that excludes
        nobody, which is the classic way a file lock stops locking. It is an
        empty file whose only content is its identity.
        """
        path = self._lock_file(collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(handle, operation)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)

    # ------------------------------------------------------------------
    # Persistence — the stamp, the load, the publish
    # ------------------------------------------------------------------

    def _collection_dir(self, collection: str) -> Path:
        """Where *collection*'s two files live."""
        return self._root / INDEX_DIR_NAME / collection

    def _stamp(self, collection: str) -> tuple[int, int] | None:
        """*collection*'s on-disk version, or ``None`` when nothing is published.

        The modification time of :data:`INDEX_FILE`, in nanoseconds, paired with
        its size. The ``mtime`` is the signal; the size is there because two
        publications inside one clock tick would otherwise be indistinguishable on
        a file system whose timestamps are coarser than the write.
        """
        try:
            stat = (self._collection_dir(collection) / INDEX_FILE).stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _refresh_locked(self, collection: str) -> None:
        """Re-read *collection* when its stamp has moved. Caller holds both locks.

        A stamp equal to the one this instance last saw means the files have not
        been republished since, so nothing is read at all — which is what keeps a
        search that follows no write off the disk entirely.
        """
        if collection not in self._collections:
            self._load_locked(collection)
            return
        if self._stamp(collection) == self._stamps.get(collection):
            return
        self._load_locked(collection)

    def _load_locked(self, collection: str) -> bool:
        """Read *collection*'s stored file into memory. Caller holds both locks.

        **A file that does not load is a miss, and is left where it is.** Removing
        it would turn a bad read into data loss for rows that are still perfectly
        good to some other reader, and every chunk here is regenerable from the
        source document it was cut from — the rule
        ``YamlDocumentStore._read`` already follows one directory over.

        Returns:
            ``True`` when a stored index was adopted, ``False`` when there is
            nothing published to adopt.
        """
        import numpy as np

        stamp = self._stamp(collection)
        if stamp is None:
            return False
        stored = self._collection_dir(collection) / INDEX_FILE
        try:
            with np.load(stored, allow_pickle=False) as archive:
                matrix = archive[VECTORS_KEY]
                payload = json.loads(bytes(archive[METADATA_KEY]).decode("utf-8"))
            index = VectorIndex()
            # ``strict`` because a row count that disagrees with its metadata is a
            # corrupt archive rather than something to truncate silently: zipping
            # to the shorter of the two would drop rows and say nothing.
            for row, data in zip(matrix, payload["entries"], strict=True):
                index.add(VectorEntry.model_validate({**data, "vector": [float(v) for v in row]}))
            stored_config = payload["config"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Vector index %s does not load (%s) — treating it as empty and leaving "
                "it in place; the chunks are regenerable from their source documents",
                stored,
                exc,
            )
            return False
        self._collections[collection] = index
        if stored_config is not None and collection not in self._configs:
            self._configs[collection] = VectorStoreParam.model_validate(stored_config)
        self._stamps[collection] = stamp
        return True

    def _publish_locked(self, collection: str) -> None:
        """Write *collection*'s whole index in one replacement. Caller holds both locks.

        The vector lives in the matrix and **only** there — ``exclude={"vector"}``
        on the metadata — so the two halves cannot disagree about a row's
        embedding and the archive is not twice the size it needs to be.
        """
        import numpy as np

        index = self._collections[collection]
        directory = self._collection_dir(collection)
        directory.mkdir(parents=True, exist_ok=True)
        entries = list(index._entries)
        width = len(entries[0].vector) if entries else 0
        matrix = np.array([entry.vector for entry in entries], dtype=np.float64).reshape(
            (len(entries), width)
        )
        config = self._configs.get(collection)
        payload = {
            "config": config.model_dump(mode="json") if config is not None else None,
            "entries": [_metadata_of(entry) for entry in entries],
        }
        metadata = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        buffer = io.BytesIO()
        # Named through the two constants rather than as literal keywords, so the
        # writer and :meth:`_load_locked` cannot drift on what the members are
        # called — which would be a silently unreadable archive, not an error.
        members: dict[str, NDArray[np.generic]] = {
            VECTORS_KEY: matrix,
            METADATA_KEY: np.frombuffer(metadata, dtype=np.uint8),
        }
        np.savez(buffer, allow_pickle=False, **members)
        _atomic_write(directory / INDEX_FILE, buffer.getvalue())
        self._stamps[collection] = self._stamp(collection)

    def _get_index(self, collection: str) -> VectorIndex:
        """Return the ``VectorIndex`` for *collection* or raise.

        Args:
            collection: Collection name to look up.

        Returns:
            The ``VectorIndex`` instance.

        Raises:
            ValueError: If the collection does not exist.
        """
        try:
            return self._collections[collection]
        except KeyError:
            msg = f"Collection '{collection}' does not exist"
            raise ValueError(msg) from None


def _metadata_of(entry: VectorEntry) -> dict[str, Any]:
    """One row's stored metadata: everything but the vector, which is in the matrix.

    The key is **popped rather than excluded**, because ``SerializableBaseModel``
    declares a ``@model_serializer`` and a whole-model serializer takes precedence
    over ``exclude=`` — which Pydantic applies silently, so an ``exclude`` here
    would look right and store every embedding twice.
    """
    data = entry.model_dump(mode="json")
    data.pop("vector", None)
    return data


def _atomic_write(target: Path, payload: bytes) -> None:
    """Write *payload* to a temp file beside *target*, then replace it in one step.

    The same shape ``YamlDocumentStore._atomic_write`` and ``FileLockBackend``
    use — a copy rather than an import, because those live in the workspace
    package and this one may not import it. The temp file is created **in the
    destination directory** so the replace is a same-filesystem rename, which is
    what makes it atomic; any ``BaseException`` unlinks it before re-raising, so
    an interrupted write leaves the previous file intact and no debris behind.
    """
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _map_search_hits(index: VectorIndex, results: list[tuple[str, float]]) -> list[SearchHit]:
    """Convert raw ``(ref_id, score)`` tuples to ``SearchHit`` models."""
    entries_by_id: dict[str, VectorEntry] = {entry.ref_id: entry for entry in index._entries}
    hits: list[SearchHit] = []
    for ref_id, score in results:
        entry = entries_by_id.get(ref_id)
        if entry is None:
            logger.warning("Search returned ref_id '%s' not found in entries", ref_id)
            continue
        hits.append(
            SearchHit(
                ref_type=entry.ref_type,
                ref_id=entry.ref_id,
                text=entry.text,
                score=score,
                scope=entry.scope,
                path=entry.path,
                ordinal=entry.ordinal,
            )
        )
    return hits


def _filtered_search(
    index: VectorIndex,
    query_vector: list[float],
    top_k: int,
    query: VectorQuery | None = None,
    *,
    scope: str | None = None,
    path_prefix: str | None = None,
) -> list[SearchHit]:
    """Apply predicates / ``filters`` / ``score_threshold``, then cut to ``top_k``.

    Over-fetches the whole index so that constraints never starve the result the
    way filtering a pre-cut top-k would — ``InMemoryBackend._filtered_search``'s
    reasoning, over the same index type.
    """
    candidates = index.search_cosine(query_vector, len(index) or top_k)
    entries_by_id: dict[str, VectorEntry] = {entry.ref_id: entry for entry in index._entries}
    threshold = query.score_threshold if query is not None else None
    filters = query.filters if query is not None else None
    kept: list[tuple[str, float]] = []
    for ref_id, score in candidates:
        if threshold is not None and score < threshold:
            continue
        entry = entries_by_id.get(ref_id)
        if entry is None:
            continue
        if not _entry_matches(entry, scope, path_prefix):
            continue
        if filters and not _matches(entry, filters):
            continue
        kept.append((ref_id, score))
        if len(kept) >= top_k:
            break
    return _map_search_hits(index, kept)


def _matches(entry: VectorEntry, filters: dict[str, Any]) -> bool:
    """Return whether *entry* satisfies every exact-match *filter*."""
    for key, expected in filters.items():
        actual = getattr(entry, key, None)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _make_local_backend(context: BackendContext) -> LocalBackend:
    """Build a :class:`LocalBackend` over ``context.root``, or refuse loudly.

    Raises:
        ValueError: When the context carries no root. The store actor turns that
            into one WARNING and a degraded collection, which is where a
            misconfiguration belongs — not into an index silently written beside
            the process.
    """
    if context.root is None:
        raise ValueError(_LOCAL_ROOT_REQUIRED)
    return LocalBackend(root=context.root)


register_backend(
    BackendSpec(
        name="local",
        factory=_make_local_backend,
        # The index is on disk, so the actor must never snapshot it into
        # ``VectorStoreState`` — but one actor still has to own the matrix, which
        # is the second flag. See ``BackendSpec.needs_actor``.
        persists_in_actor_state=False,
        needs_actor=True,
        # ``resolve_default_backend`` must never hand this to a consumer with no
        # filesystem: ``PlanActor`` and ``KnowledgeGraphActor`` have no ``<meta>``
        # to point it at, and would get a factory that refuses. The workspace
        # names it explicitly.
        selectable_as_default=False,
    ),
    replace=True,
)
