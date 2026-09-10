"""The file-backed vector backend: what it writes, what it re-reads, and what it refuses.

**Every cross-instance assertion here goes through two objects**, never one, for
``FileLockBackend``'s reason: two instances in one process are exactly as separate
as two processes over one mount, so a spec that put through one object and read
through another has proved the bytes reached the disk. One that reused a single
instance would prove nothing at all — the in-memory matrix would answer.

The backend is testable with no cluster and no actor, which is the point: the
shared-scope rule is exercised against a real implementation here rather than
only against the two paths a cluster-less CI never runs.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("numpy", reason="the [vector_search] extra is not installed")

from akgentic.tool.vector_store import registry
from akgentic.tool.vector_store.backends.local import (
    INDEX_DIR_NAME,
    INDEX_FILE,
    LOCK_FILE_PREFIX,
    LOCKS_DIR_NAME,
    METADATA_KEY,
    VECTORS_KEY,
    LocalBackend,
)
from akgentic.tool.vector_store.protocol import (
    SHARED_COLLECTIONS,
    CollectionStatus,
    VectorQuery,
    VectorStoreConfig,
    VectorStoreParam,
    VectorStoreService,
    collection_is_team_scoped,
)
from akgentic.tool.vector_store.registry import BackendContext
from akgentic.tool.vector_store.vector import VectorEntry

if TYPE_CHECKING:
    from collections.abc import Iterator

SHARED = "workspace_chunks"
"""The workspace's collection — shared across every team that indexes one tree."""

TEAM_SCOPED = "planning"
"""A collection that is not shared, so its rules are the opposite ones."""

SCOPE = "u-alice/notes"
"""The workspace a chunk belongs to. Mandatory on the shared collection."""


def entry(
    ref_id: str,
    vector: list[float] | None = None,
    *,
    text: str = "body",
    scope: str | None = SCOPE,
    path: str | None = "a.md",
    ordinal: int | None = 0,
) -> VectorEntry:
    """One embedded chunk, with the three workspace fields set by default."""
    return VectorEntry(
        ref_type="chunk",
        ref_id=ref_id,
        text=text,
        vector=vector if vector is not None else [1.0, 0.0],
        scope=scope,
        path=path,
        ordinal=ordinal,
    )


def backend_over(root: Path, *collections: str) -> LocalBackend:
    """A fresh backend over *root* that has created each named collection."""
    built = LocalBackend(root=str(root))
    for name in collections:
        built.create_collection(name, VectorStoreParam(dimension=2))
    return built


def index_dir(root: Path, collection: str = SHARED) -> Path:
    """Where *collection*'s stored index lives under *root*."""
    return root / INDEX_DIR_NAME / collection


def stored_metadata(root: Path, collection: str = SHARED) -> dict[str, Any]:
    """The metadata half of the stored archive, read straight off the disk."""
    import numpy as np

    with np.load(index_dir(root, collection) / INDEX_FILE, allow_pickle=False) as archive:
        return dict(json.loads(bytes(archive[METADATA_KEY]).decode("utf-8")))


##
## AC 1 — registered, and never the default
##


class TestTheBackendIsRegistered:
    """One registry entry, satisfying the shipped protocol and no wider surface."""

    def test_the_registry_lists_it(self) -> None:
        assert "local" in registry.available_backends()
        assert registry.is_registered("local")

    def test_the_factory_builds_something_that_satisfies_the_service_protocol(
        self, tmp_path: Path
    ) -> None:
        """The four methods ``VectorStoreService`` declares, with its signatures.

        ``isinstance`` against a non-runtime-checkable ``Protocol`` is not
        available, so the check is the one mypy makes structurally: the object is
        assigned to a ``VectorStoreService`` slot and each method is called with
        the protocol's own arguments.
        """
        spec = registry.get_backend_spec("local")
        built = spec.factory(
            BackendContext(
                config=VectorStoreConfig(name="#VectorStore", role="ToolActor"),
                team_id="team-a",
                root=str(tmp_path),
            )
        )

        service: VectorStoreService = built
        service.create_collection(TEAM_SCOPED, VectorStoreParam(dimension=2))
        service.add(TEAM_SCOPED, [entry("c1", scope=None, path=None, ordinal=None)])
        result = service.search(TEAM_SCOPED, [1.0, 0.0], 5)
        service.remove(TEAM_SCOPED, ["c1"])

        assert isinstance(built, LocalBackend)
        assert result.status is CollectionStatus.READY
        assert [hit.ref_id for hit in result.hits] == ["c1"]

    def test_it_adds_no_method_to_the_protocol_it_implements(self) -> None:
        """No fifth verb: the registry's contract is four methods (ADR-051 Decision 7)."""
        public = {
            name
            for name in vars(LocalBackend)
            if not name.startswith("_") and callable(getattr(LocalBackend, name, None))
        }

        assert public == {"create_collection", "add", "remove", "search"}

    def test_it_declares_no_actor_state_and_implements_no_snapshot(self) -> None:
        """A ``get_state`` pair would be a claim the store actor acts on every mutation."""
        spec = registry.get_backend_spec("local")

        assert spec.persists_in_actor_state is False
        assert spec.needs_actor is True
        assert not hasattr(LocalBackend, "get_state")
        assert not hasattr(LocalBackend, "restore_state")

    def test_the_default_resolver_never_returns_it_on_any_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``PlanActor`` and ``KnowledgeGraphActor`` have no root to give it.

        Every combination of the two cluster variables is tried, because
        ``resolve_default_backend`` walks the registry in order and the answer
        depends on which probes report provisioned.
        """
        for weaviate_url, qdrant_url in (
            (None, None),
            ("http://localhost:8080", None),
            (None, "http://localhost:6333"),
            ("http://localhost:8080", "http://localhost:6333"),
        ):
            with monkeypatch.context() as patched:
                for name, value in (
                    ("AKGENTIC_WEAVIATE_URL", weaviate_url),
                    ("AKGENTIC_QDRANT_URL", qdrant_url),
                ):
                    if value is None:
                        patched.delenv(name, raising=False)
                    else:
                        patched.setenv(name, value)

                assert registry.resolve_default_backend() != "local"

    def test_a_factory_with_no_root_refuses_rather_than_choosing_one(self) -> None:
        """Defaulting to the process cwd would put every tree in one directory."""
        spec = registry.get_backend_spec("local")

        with pytest.raises(ValueError, match="root"):
            spec.factory(
                BackendContext(
                    config=VectorStoreConfig(name="#VectorStore", role="ToolActor"),
                    team_id="team-a",
                )
            )


##
## AC 2 — the index is files, and it survives the process
##


class TestTheIndexIsFilesUnderTheRoot:
    """What lands on disk, and that a second object over the same root finds it."""

    def test_the_archive_lands_under_index_slash_collection(self, tmp_path: Path) -> None:
        written = backend_over(tmp_path, SHARED)

        written.add(SHARED, [entry("c1")])

        assert (tmp_path / INDEX_DIR_NAME / SHARED / INDEX_FILE).is_file()

    def test_a_second_independent_instance_searches_what_the_first_wrote(
        self, tmp_path: Path
    ) -> None:
        """Two objects, no shared memory — which is what makes this worth asserting."""
        written = backend_over(tmp_path, SHARED)
        written.add(SHARED, [entry("c1", text="net thirty")])

        reader = backend_over(tmp_path, SHARED)
        hits = reader.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits

        assert [(hit.ref_id, hit.text, hit.path, hit.ordinal) for hit in hits] == [
            ("c1", "net thirty", "a.md", 0)
        ]

    def test_a_second_instance_adopts_the_stored_rows_rather_than_blanking_them(
        self, tmp_path: Path
    ) -> None:
        """``create_collection`` is idempotent in the direction that matters.

        The instance that arrives second must inherit the rows the first wrote —
        a ``create_collection`` that published an empty index over them would
        destroy a whole tree's embeddings on the next process start, silently.
        """
        first = backend_over(tmp_path, SHARED)
        first.add(SHARED, [entry("c1")])

        second = backend_over(tmp_path, SHARED)

        assert [hit.ref_id for hit in second.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits] == [
            "c1"
        ]
        # And the first instance's own view is not damaged by the second arriving.
        assert [hit.ref_id for hit in first.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits] == [
            "c1"
        ]

    def test_the_vector_is_stored_once_in_the_matrix_and_not_in_the_metadata(
        self, tmp_path: Path
    ) -> None:
        """Two copies of an embedding are two things that can disagree about a row."""
        import numpy as np

        written = backend_over(tmp_path, SHARED)
        written.add(SHARED, [entry("c1", [0.6, 0.8])])

        payload = stored_metadata(tmp_path)
        with np.load(index_dir(tmp_path) / INDEX_FILE, allow_pickle=False) as archive:
            matrix = archive[VECTORS_KEY]

        assert [row.get("vector") for row in payload["entries"]] == [None]
        assert matrix.shape == (1, 2)
        assert list(matrix[0]) == [0.6, 0.8]

    def test_the_collection_config_travels_with_the_rows(self, tmp_path: Path) -> None:
        """So a later instance knows the width and the tenant it was created with."""
        written = LocalBackend(root=str(tmp_path))
        written.create_collection(SHARED, VectorStoreParam(dimension=2, tenant="acme"))

        payload = stored_metadata(tmp_path)

        assert payload["config"]["dimension"] == 2
        assert payload["config"]["tenant"] == "acme"


##
## AC 3 — atomic, and lock-guarded
##


class TestAWriteIsAtomicAndLockGuarded:
    """Nothing observes half an index, and two threads do not interleave."""

    def test_a_write_that_fails_before_the_rename_leaves_the_previous_index_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The temp file is removed and the published archive is untouched."""
        from akgentic.tool.vector_store.backends import local as module

        written = backend_over(tmp_path, SHARED)
        written.add(SHARED, [entry("c1")])
        before = (index_dir(tmp_path) / INDEX_FILE).read_bytes()

        def _boom(target: Path, payload: bytes) -> None:
            raise OSError("no space left on device")

        monkeypatch.setattr(module, "_atomic_write", _boom)
        with pytest.raises(OSError, match="no space"):
            written.add(SHARED, [entry("c2", [0.0, 1.0])])
        monkeypatch.undo()

        assert (index_dir(tmp_path) / INDEX_FILE).read_bytes() == before
        reader = backend_over(tmp_path, SHARED)
        assert [hit.ref_id for hit in reader.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits] == [
            "c1"
        ]

    def test_an_interrupted_write_leaves_no_temp_file_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Debris in the directory is what a later listing would trip over.

        A ``KeyboardInterrupt`` rather than an ``OSError``, because the cleanup is
        on ``BaseException``: a story-writer who narrowed it to ``Exception``
        would leave a temp file behind on exactly the interrupt that is most
        likely to land mid-write.
        """
        written = backend_over(tmp_path, SHARED)
        written.add(SHARED, [entry("c1")])

        def _boom(source: Any, destination: Any) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("akgentic.tool.vector_store.backends.local.os.replace", _boom)
        with pytest.raises(KeyboardInterrupt):
            written.add(SHARED, [entry("c2")])
        monkeypatch.undo()

        assert sorted(p.name for p in index_dir(tmp_path).iterdir()) == [INDEX_FILE]

    def test_the_lock_file_is_created_for_the_collection_it_guards(self, tmp_path: Path) -> None:
        """One hold per collection, named for it, under ``<root>/locks/``."""
        written = backend_over(tmp_path, SHARED, TEAM_SCOPED)

        written.add(SHARED, [entry("c1")])

        assert (tmp_path / LOCKS_DIR_NAME / f"{LOCK_FILE_PREFIX}{SHARED}").is_file()
        assert (tmp_path / LOCKS_DIR_NAME / f"{LOCK_FILE_PREFIX}{TEAM_SCOPED}").is_file()

    def test_a_mutation_holds_the_file_lock_for_the_whole_read_modify_publish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hold spans the publish, not just the rename.

        Asserted by watching the hold's own enter/exit against the publish: a
        ``flock`` taken around the rename alone would let a second writer read the
        index, be pre-empted, and then publish over the first writer's rows.
        """
        events: list[str] = []
        written = backend_over(tmp_path, SHARED)
        original_hold = LocalBackend._hold
        original_publish = LocalBackend._publish_locked

        def _watched_hold(self: LocalBackend, collection: str, operation: int) -> Iterator[None]:
            import contextlib

            @contextlib.contextmanager
            def _inner() -> Iterator[None]:
                events.append("hold-enter")
                with original_hold(self, collection, operation):
                    yield
                events.append("hold-exit")

            return _inner()

        def _watched_publish(self: LocalBackend, collection: str) -> None:
            events.append("publish")
            original_publish(self, collection)

        monkeypatch.setattr(LocalBackend, "_hold", _watched_hold)
        monkeypatch.setattr(LocalBackend, "_publish_locked", _watched_publish)

        written.add(SHARED, [entry("c1")])

        assert events == ["hold-enter", "publish", "hold-exit"]

    def test_a_second_process_cannot_publish_while_the_lock_is_held(self, tmp_path: Path) -> None:
        """The exclusion is a ``flock``, and only a second **process** can prove it.

        **This spec exists because the obvious one is inert.** Watching the hold
        enter and exit around the publish, or contending two threads, both stay
        green with ``fcntl.flock`` deleted outright: the thread lock still
        serialises everything inside one interpreter, and the context manager
        still enters and exits. A ``threading.Lock`` excludes nothing across a
        mount — which is the whole reason the file lock is there — so the
        falsifying case has to be a real second interpreter.

        The parent takes the collection's hold **directly**, by the same path the
        backend derives, and watches a child that has been started and has said so
        fail to publish until the hold is released.
        """
        import subprocess
        import sys
        import textwrap
        import time

        backend_over(tmp_path, SHARED).add(SHARED, [entry("c1")])
        before = (index_dir(tmp_path) / INDEX_FILE).stat().st_mtime_ns
        script = tmp_path / "other_process.py"
        script.write_text(
            textwrap.dedent(f"""
                from akgentic.tool.vector_store.backends.local import LocalBackend
                from akgentic.tool.vector_store.protocol import VectorStoreParam
                from akgentic.tool.vector_store.vector import VectorEntry

                print("started", flush=True)
                backend = LocalBackend(root={str(tmp_path)!r})
                backend.create_collection({SHARED!r}, VectorStoreParam(dimension=2))
                backend.add(
                    {SHARED!r},
                    [
                        VectorEntry(
                            ref_type="chunk",
                            ref_id="from-the-other-process",
                            text="body",
                            vector=[0.0, 1.0],
                            scope={SCOPE!r},
                        )
                    ],
                )
                print("published", flush=True)
            """),
            encoding="utf-8",
        )

        held = os.open(tmp_path / LOCKS_DIR_NAME / f"{LOCK_FILE_PREFIX}{SHARED}", os.O_RDWR)
        try:
            fcntl.flock(held, fcntl.LOCK_EX)
            child = subprocess.Popen(
                [sys.executable, str(script)], stdout=subprocess.PIPE, text=True
            )
            try:
                assert child.stdout is not None
                assert child.stdout.readline().strip() == "started"
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    assert (index_dir(tmp_path) / INDEX_FILE).stat().st_mtime_ns == before, (
                        "the other process published while the lock was held"
                    )
                    time.sleep(0.02)
            finally:
                fcntl.flock(held, fcntl.LOCK_UN)
        finally:
            os.close(held)

        assert child.wait(timeout=30) == 0
        reader = backend_over(tmp_path, SHARED)
        found = {hit.ref_id for hit in reader.search(SHARED, [0.0, 1.0], 5, scope=SCOPE).hits}
        assert found == {"c1", "from-the-other-process"}

    def test_two_processes_writing_at_once_lose_no_update(self, tmp_path: Path) -> None:
        """The hold spans read-modify-publish, and is **exclusive** — both, or updates vanish.

        **Three inert formulations were tried before this one.** Watching the
        hold's enter and exit around the publish stays green when the lock is
        deleted; contending two threads stays green for the same reason; and even
        the blocked-child spec above stays green when the hold is released
        immediately after it is taken, or downgraded to a shared one, because the
        *acquire* still waits behind an externally held exclusive lock.

        What none of those survive is two real writers. Each ``add`` re-reads
        under its own hold and publishes the union, so with the hold correct every
        row lands; with the hold released early, or taken shared, two processes
        read the same version and the second publication overwrites the first.
        """
        import subprocess
        import sys
        import textwrap

        rounds = 40
        backend_over(tmp_path, SHARED)
        script = tmp_path / "writer.py"
        script.write_text(
            textwrap.dedent(f"""
                import sys

                from akgentic.tool.vector_store.backends.local import LocalBackend
                from akgentic.tool.vector_store.protocol import VectorStoreParam
                from akgentic.tool.vector_store.vector import VectorEntry

                prefix = sys.argv[1]
                backend = LocalBackend(root={str(tmp_path)!r})
                backend.create_collection({SHARED!r}, VectorStoreParam(dimension=2))
                for number in range({rounds}):
                    backend.add(
                        {SHARED!r},
                        [
                            VectorEntry(
                                ref_type="chunk",
                                ref_id=f"{{prefix}}{{number}}",
                                text="body",
                                vector=[1.0, 0.0],
                                scope={SCOPE!r},
                            )
                        ],
                    )
            """),
            encoding="utf-8",
        )

        children = [
            subprocess.Popen([sys.executable, str(script), prefix]) for prefix in ("a", "b")
        ]
        for child in children:
            assert child.wait(timeout=120) == 0

        reader = backend_over(tmp_path, SHARED)
        found = {hit.ref_id for hit in reader.search(SHARED, [1.0, 0.0], 500, scope=SCOPE).hits}
        assert found == {f"{prefix}{number}" for prefix in "ab" for number in range(rounds)}

    def test_two_threads_mutating_one_instance_do_not_lose_a_write(self, tmp_path: Path) -> None:
        """Both batches survive, and the archive describes exactly what is in it.

        A lost update here is not a lost row on one path — it is a matrix and a
        metadata list of different lengths, which the loader refuses outright. So
        reading it back through a second object is the assertion.
        """
        written = backend_over(tmp_path, SHARED)
        start = threading.Barrier(2)

        def _add(prefix: str) -> None:
            start.wait(timeout=5)
            for number in range(20):
                written.add(SHARED, [entry(f"{prefix}{number}")])

        threads = [threading.Thread(target=_add, args=(name,)) for name in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        reader = backend_over(tmp_path, SHARED)
        found = {hit.ref_id for hit in reader.search(SHARED, [1.0, 0.0], 100, scope=SCOPE).hits}
        assert found == {f"{prefix}{number}" for prefix in "ab" for number in range(20)}


##
## AC 4 — a reload is triggered by mtime movement, and only by it
##


class TestAReloadFollowsTheStamp:
    """The stored file is re-read when its stamp moves, and not otherwise."""

    def test_a_search_sees_an_entry_another_instance_added(self, tmp_path: Path) -> None:
        """The instance that has already loaded picks the new row up on its next search."""
        first = backend_over(tmp_path, SHARED)
        second = backend_over(tmp_path, SHARED)
        assert first.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits == []

        second.add(SHARED, [entry("c1")])

        assert [hit.ref_id for hit in first.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits] == [
            "c1"
        ]

    def test_a_search_that_follows_no_movement_does_not_re_read_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counting the loads is the direct question; a hit count would not answer it."""
        reader = backend_over(tmp_path, SHARED)
        backend_over(tmp_path, SHARED).add(SHARED, [entry("c1")])
        assert reader.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits  # the one reload

        loads: list[str] = []
        original = LocalBackend._load_locked

        def _counted(self: LocalBackend, collection: str) -> bool:
            loads.append(collection)
            return original(self, collection)

        monkeypatch.setattr(LocalBackend, "_load_locked", _counted)
        for _ in range(3):
            reader.search(SHARED, [1.0, 0.0], 5, scope=SCOPE)

        assert loads == []

    def test_a_stamp_that_moves_is_what_triggers_the_re_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half: the same three searches after a write do reload, once."""
        reader = backend_over(tmp_path, SHARED)
        writer = backend_over(tmp_path, SHARED)
        reader.search(SHARED, [1.0, 0.0], 5, scope=SCOPE)

        loads: list[str] = []
        original = LocalBackend._load_locked

        def _counted(self: LocalBackend, collection: str) -> bool:
            loads.append(collection)
            return original(self, collection)

        writer.add(SHARED, [entry("c1")])
        monkeypatch.setattr(LocalBackend, "_load_locked", _counted)
        for _ in range(3):
            reader.search(SHARED, [1.0, 0.0], 5, scope=SCOPE)

        assert loads == [SHARED]

    def test_a_removal_by_another_instance_is_seen_too(self, tmp_path: Path) -> None:
        """Not only additions: the stamp moves on every publication."""
        first = backend_over(tmp_path, SHARED)
        second = backend_over(tmp_path, SHARED)
        first.add(SHARED, [entry("c1"), entry("c2", [0.0, 1.0])])
        assert len(second.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits) == 2

        first.remove(SHARED, ["c1"], scope=SCOPE)

        assert [hit.ref_id for hit in second.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits] == [
            "c2"
        ]

    def test_an_unreadable_archive_is_a_miss_and_is_left_in_place(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Removing a file we cannot read is how a cache turns a bad parse into data loss."""
        import logging

        written = backend_over(tmp_path, SHARED)
        written.add(SHARED, [entry("c1")])
        stored = index_dir(tmp_path) / INDEX_FILE
        stored.write_bytes(b"not an npz archive at all")

        reader = LocalBackend(root=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="akgentic.tool.vector_store.backends.local"):
            reader.create_collection(SHARED, VectorStoreParam(dimension=2))

        assert "does not load" in caplog.text
        assert stored.is_file()

    def test_a_damaged_archive_is_a_miss_too_rather_than_raising(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A truncated ``.npz``, not a file of unrelated bytes — a different exception.

        The spec above overwrites the archive with bytes that were never a zip, so
        ``np.load`` refuses on the magic and raises ``ValueError``. An archive that
        *was* written by this backend and then damaged — the shape a network mount
        produces, which is the mount this backend exists for — reaches
        ``zipfile.ZipFile`` and raises ``BadZipFile``, which derives from
        ``Exception`` alone and so is caught by no broader clause. Left out of the
        tuple it propagated out of ``search``, which is the one thing this method
        promises never to do.
        """
        import logging

        written = backend_over(tmp_path, SHARED)
        written.add(SHARED, [entry("c1")])
        stored = index_dir(tmp_path) / INDEX_FILE
        stored.write_bytes(stored.read_bytes()[:40])

        reader = LocalBackend(root=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="akgentic.tool.vector_store.backends.local"):
            reader.create_collection(SHARED, VectorStoreParam(dimension=2))
            hits = reader.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits

        assert hits == []
        assert "does not load" in caplog.text


##
## AC 5 — the shared-collection rule, enforced here exactly as on a cluster
##


class TestTheSharedCollectionRule:
    """A shared collection has no boundary but ``scope``, so it refuses without one."""

    def test_the_workspace_collection_is_the_shared_one(self) -> None:
        """The premise every spec in this class rests on."""
        assert SHARED in SHARED_COLLECTIONS
        assert not collection_is_team_scoped(SHARED)
        assert collection_is_team_scoped(TEAM_SCOPED)

    @pytest.mark.parametrize("method", ["search", "remove"], ids=["search", "remove"])
    def test_an_unscoped_call_on_the_shared_collection_raises(
        self, tmp_path: Path, method: str
    ) -> None:
        """It refuses: it does not default, and it does not answer empty."""
        built = backend_over(tmp_path, SHARED)
        built.add(SHARED, [entry("c1")])

        with pytest.raises(ValueError, match="shared across teams"):
            if method == "search":
                built.search(SHARED, [1.0, 0.0], 5)
            else:
                built.remove(SHARED, ["c1"])

        # And it really did refuse rather than silently doing the work anyway.
        assert [hit.ref_id for hit in built.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits] == [
            "c1"
        ]

    @pytest.mark.parametrize("method", ["search", "remove"], ids=["search", "remove"])
    def test_a_team_scoped_collection_is_unaffected(self, tmp_path: Path, method: str) -> None:
        built = backend_over(tmp_path, TEAM_SCOPED)
        built.add(TEAM_SCOPED, [entry("c1", scope=None, path=None, ordinal=None)])

        if method == "search":
            assert [hit.ref_id for hit in built.search(TEAM_SCOPED, [1.0, 0.0], 5).hits] == ["c1"]
        else:
            built.remove(TEAM_SCOPED, ["c1"])
            assert built.search(TEAM_SCOPED, [1.0, 0.0], 5).hits == []

    @pytest.mark.parametrize("method", ["search", "remove"], ids=["search", "remove"])
    def test_the_prefix_check_runs_before_the_scope_check(
        self, tmp_path: Path, method: str
    ) -> None:
        """Same two lines, same order as the other backends, so the three cannot drift.

        A call that violates **both** rules must answer the prefix one, which is
        what pins the order rather than merely the presence of the two checks.
        """
        built = backend_over(tmp_path, SHARED)

        with pytest.raises(ValueError, match="cannot contain"):
            if method == "search":
                built.search(SHARED, [1.0, 0.0], 5, path_prefix="a*")
            else:
                built.remove(SHARED, ["c1"], path_prefix="a*")

    def test_a_scoped_search_never_crosses_into_another_scope(self, tmp_path: Path) -> None:
        """The boundary the mandatory scope exists to draw."""
        built = backend_over(tmp_path, SHARED)
        built.add(
            SHARED,
            [entry("mine"), entry("theirs", [0.9, 0.1], scope="u-bob/notes")],
        )

        hits = built.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits

        assert [hit.ref_id for hit in hits] == ["mine"]

    def test_a_scoped_removal_never_takes_another_scopes_row(self, tmp_path: Path) -> None:
        """A colliding ``ref_id`` across two scopes is exactly what the scalpel is for."""
        built = backend_over(tmp_path, SHARED)
        built.add(SHARED, [entry("c1"), entry("c1", [0.9, 0.1], scope="u-bob/notes")])

        built.remove(SHARED, ["c1"], scope=SCOPE)

        assert built.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits == []
        assert [
            hit.ref_id for hit in built.search(SHARED, [1.0, 0.0], 5, scope="u-bob/notes").hits
        ] == ["c1"]

    def test_a_query_refinement_is_applied_before_top_k_is_taken(self, tmp_path: Path) -> None:
        """The over-fetch rule the in-memory backend states; the same index type."""
        built = backend_over(tmp_path, SHARED)
        built.add(
            SHARED,
            [entry("near", [1.0, 0.0]), entry("far", [0.0, 1.0], path="b.md")],
        )

        hits = built.search(
            SHARED,
            [1.0, 0.0],
            1,
            scope=SCOPE,
            query=VectorQuery(filters={"path": "b.md"}),
        ).hits

        assert [hit.ref_id for hit in hits] == ["far"]

    def test_a_missing_collection_raises_rather_than_answering_empty(self, tmp_path: Path) -> None:
        """An empty answer would be an assertion about the data, not the configuration."""
        built = LocalBackend(root=str(tmp_path))

        with pytest.raises(ValueError, match="does not exist"):
            built.search(TEAM_SCOPED, [1.0, 0.0], 5)


##
## AC 13 — two trees in one team are two roots and two indexes
##


class TestTwoRootsAreTwoIndexes:
    """A tree's index is its root's, and nothing crosses between two of them."""

    def test_an_entry_under_one_root_is_not_found_under_the_other(self, tmp_path: Path) -> None:
        first_root = tmp_path / "notes.akgentic"
        second_root = tmp_path / "cases.akgentic"
        first = backend_over(first_root, SHARED)
        second = backend_over(second_root, SHARED)

        first.add(SHARED, [entry("only-in-notes")])

        assert [hit.ref_id for hit in first.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits] == [
            "only-in-notes"
        ]
        assert second.search(SHARED, [1.0, 0.0], 5, scope=SCOPE).hits == []
        assert (
            not (second_root / INDEX_DIR_NAME / SHARED / INDEX_FILE).read_bytes()
            == (first_root / INDEX_DIR_NAME / SHARED / INDEX_FILE).read_bytes()
        )

    def test_each_root_holds_its_own_archive_and_its_own_lock(self, tmp_path: Path) -> None:
        first_root = tmp_path / "notes.akgentic"
        second_root = tmp_path / "cases.akgentic"

        backend_over(first_root, SHARED).add(SHARED, [entry("c1")])
        backend_over(second_root, SHARED).add(SHARED, [entry("c2")])

        for root in (first_root, second_root):
            assert (root / INDEX_DIR_NAME / SHARED / INDEX_FILE).is_file()
            assert (root / LOCKS_DIR_NAME / f"{LOCK_FILE_PREFIX}{SHARED}").is_file()


##
## AC 14 — the knowledge graph and the plan are untouched
##


class TestTheOtherConsumersAreUntouched:
    """``PlanActor`` and ``KnowledgeGraphActor`` keep the backend they had.

    They have no filesystem to hang an index off — no ``<meta>``, no tree — so a
    file-backed backend is meaningless to them, and their rows live in the same
    ``VectorStoreState`` as their index, which is why they never had the
    persisted-row-over-an-empty-engine mismatch the workspace did.

    The three specs are behavioural rather than a diff check: "no file under
    ``planning/`` changed" is not something a test can assert, but "neither one
    can reach the new backend" is.
    """

    @pytest.mark.parametrize(
        ("module", "field"),
        [
            ("akgentic.tool.planning.planning", "PlanningTool"),
            ("akgentic.tool.knowledge_graph.kg_tool", "KnowledgeGraphTool"),
        ],
        ids=["planning", "knowledge-graph"],
    )
    def test_their_card_still_defaults_to_the_in_actor_backend(
        self, module: str, field: str
    ) -> None:
        """No substitution runs for them: the workspace card is the only one that derives."""
        import importlib

        from akgentic.tool.vector_store.protocol import resolve_store_param

        card_class = getattr(importlib.import_module(module), field)
        param = resolve_store_param(card_class().vector_store)

        assert param is not None
        assert param.backend == "inmemory"
        assert param.root is None

    def test_they_still_need_the_teams_store_actor(self) -> None:
        """Both resolve ``#VectorStore`` and hold their index in its state."""
        from akgentic.tool.vector_store.protocol import needs_store_actor

        assert needs_store_actor(VectorStoreParam(backend="inmemory")) is True
        assert registry.get_backend_spec("inmemory").persists_in_actor_state is True

    @pytest.mark.parametrize(
        "package",
        ["akgentic.tool.planning", "akgentic.tool.knowledge_graph"],
        ids=["planning", "knowledge-graph"],
    )
    def test_neither_package_reaches_the_file_backed_backend(self, package: str) -> None:
        """An import of it anywhere under either package is out of this story's scope.

        The sweep floors its own size, for the reason every canary in this suite
        does: an empty or moved package would find nothing and pass.
        """
        import ast
        import importlib

        module = importlib.import_module(package)
        assert module.__file__ is not None
        modules = sorted(Path(module.__file__).parent.rglob("*.py"))
        assert len(modules) >= 2, f"the sweep found {len(modules)} module(s) under {package}"

        hits = []
        for source in modules:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                    "backends.local"
                ):
                    hits.append(f"{source.name}:{node.lineno}")
                if isinstance(node, ast.Import) and any(
                    alias.name.endswith("backends.local") for alias in node.names
                ):
                    hits.append(f"{source.name}:{node.lineno}")

        assert hits == [], f"{package} imports the file-backed backend: {hits}"
