"""Story 55-5: the record file's read-modify-write is serialised across processes.

ADR-051 Decision 6's *"there is no lock, and none is to be added"* holds for the
**extraction** half of a record — derivable, disposable, atomic replacement is
enough — and never transferred to the **index row** in the same file. Two
processes can both see a ``PENDING`` row and both spawn an ``IndexWorker`` for
one file (**two paid embedding runs**), and can lose a ``superseded_chunk_ids``
list in a whole-file replace (**paid vectors orphaned**, with nothing raised).
The mailbox used to serialise this; nothing replaced it.

**The cross-process specs spawn real second interpreters, and they must.** A
``threading.Lock``, the GIL and one mailbox all provide exclusion inside one
interpreter — story 52-4 deleted ``fcntl.flock`` outright and watched three
separate single-process formulations stay green. The one-process specs below are
the ones whose property is genuinely not about exclusion: the drain's filtering
and termination, and a per-process rule being deleted.

**The observable is the spawn, not the embedding.** The paid unit is the
``IndexWorker`` run — extraction, split and embed all live behind it. Counting
embeddings would need a network double inside a child and an ordering decided by
timing; counting spawns is a line appended to a shared file by a ``createActor``
stand-in, ``O_APPEND``, one line per spawn, counted by the parent. Deterministic,
and exactly what "two paid embedding runs" means.

**Determinism comes from sequencing, never from timing.** The parent takes the
record's hold **by the production derivation**, starts a child that has printed
that it is under way, and then decides when the child may proceed. No spec here
waits and hopes.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from akgentic.tool.workspace.documents.models import RagChunk, RagFile, RagStatus
from akgentic.tool.workspace.documents.store import DocumentEntry, YamlDocumentStore
from akgentic.tool.workspace.models import content_sha
from tests.workspace.conftest import (
    CHILD_TIMEOUT_S,
    WORKSPACE_PATH,
    start_child,
    watch_store,
    write_script,
)
from tests.workspace.test_rag_pipeline import RagHarness, _started_actor, write


@pytest.fixture
def harness(workspace_tree: Path, monkeypatch: pytest.MonkeyPatch) -> RagHarness:
    """The in-process pipeline rig, installed on a started actor over the test tree.

    The same pair ``test_rag_pipeline.py`` builds — its ``createActor`` is
    recorded rather than real, which is what lets a spec count spawns without
    starting a worker. The one-process specs at the foot of this module use it;
    the cross-process ones above build no actor at all, because their children do.
    """
    built = RagHarness(_started_actor(WORKSPACE_PATH))
    built.install(monkeypatch)
    return built


EXCLUSION_WINDOW_S = 0.5
"""How long the parent watches a blocked child make no progress.

A failure budget expressed as a window, exactly as ``test_gate_locks.py``'s is:
without the lock the child gets through in milliseconds, so this is generous by
two orders of magnitude and never a source of flakiness.
"""

DECOY = "decoy.md"
"""A file the child's ``workspace_rag_index`` call names, already accounted for.

The child has to reach ``_drain`` **without** contending on the target's record
first: ``index_paths`` holds each candidate's record around its
accounted-for/enqueue pair, so a child that walked the whole tree would block
there rather than inside the claim, and the guard would stop saying anything
about where the claim's re-read happens.
"""

TARGET = "target.md"
"""The file whose record the parent holds, and whose claim the child attempts."""


def _seed(path: str, row: RagFile) -> None:
    """Put *row* on disk through the shipped store, as a previous turn would have."""
    store = YamlDocumentStore()
    existing = store.get_document(WORKSPACE_PATH, path) or DocumentEntry(path=path)
    store.put_document(WORKSPACE_PATH, existing.model_copy(update={"row": row}))


def _row_on_disk(path: str) -> RagFile:
    """Read *path*'s row back through a second store object."""
    entry = YamlDocumentStore().get_document(WORKSPACE_PATH, path)
    assert entry is not None and entry.row is not None, f"{path} has no row on disk"
    return entry.row


def _spawns(log: Path) -> list[str]:
    """Every worker the child recorded, in order — an empty list when none."""
    try:
        return [line for line in log.read_text(encoding="utf-8").splitlines() if line]
    except FileNotFoundError:
        return []


_CLAIMING_CHILD = """
from akgentic.core.agent_config import BaseConfig
from akgentic.tool.workspace.actor import WorkspaceActor

spawn_log = Path(sys.argv[1])


class _Silent:
    \"\"\"Stands in for a worker's tell proxy: it accepts anything and does nothing.\"\"\"

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _recording_create(self, actor_class, config=None, **kwargs):
    # O_APPEND, one line per spawn, so the parent can count them while the child
    # is still running and two writers could never interleave a line.
    handle = os.open(str(spawn_log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(handle, (getattr(config, "name", "?") + "\\n").encode("utf-8"))
    finally:
        os.close(handle)

    class _Address:
        name = "spawned"

    return _Address()


WorkspaceActor.createActor = _recording_create
WorkspaceActor.proxy_tell = lambda self, actor, actor_type=None: _Silent()

card = bind("child", workspace_rag_index=True)
print("started", flush=True)
for tool in card.get_tools():
    if tool.__name__ == "workspace_rag_index":
        print("ANSWER " + tool(sys.argv[2]), flush=True)
        break
"""
"""A child that drives the **real card** through the shipped claim path.

``workspace_rag_index`` is the agent-facing callable, so everything below it is
production: ``index_paths`` → the per-candidate hold → ``_drain`` →
``next_pending`` → ``_spawn``'s hold and its re-read. Only ``createActor`` and the
worker's tell proxy are stood in for, which is ``RagHarness``'s model — a child
that built a ``WorkspaceTool`` and poked its private attributes would be testing
this file.
"""


class TestTwoProcessesClaimOneFile:
    """B4's first half: one spawn for one file, whichever process gets there first.

    Without the hold both processes see ``PENDING``, both decide they may claim
    it and both start a worker — **two paid embedding runs** for one document,
    with nothing raised and nothing in a log to say so.
    """

    def _prepare(self, workspace_tree: Path) -> None:
        """One indexable file with a ``PENDING`` row, and one already accounted for."""
        decoy_sha = write(workspace_tree, DECOY)
        write(workspace_tree, TARGET)
        now = datetime.now(UTC)
        # ``EMBEDDED`` at its current bytes, so the child's candidate loop counts
        # it as current and enqueues nothing — see :data:`DECOY`.
        _seed(
            DECOY,
            RagFile(path=DECOY, status=RagStatus.EMBEDDED, indexed_sha=decoy_sha, updated_at=now),
        )
        _seed(
            TARGET,
            RagFile(path=TARGET, status=RagStatus.PENDING, indexed_sha="seeded", updated_at=now),
        )

    def test_a_child_cannot_claim_the_file_while_the_record_is_held(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """Both halves of AC 16, in the order the sequence runs them.

        **The exclusion half**: while the parent holds ``target.md``'s record, the
        child — which has printed that it is under way — records no spawn at all.
        Without the ``flock`` it spawns within milliseconds.

        **The outcome half**: still holding, the parent writes the claimed row
        through the real store, releases, and the child records **zero** spawns.
        This is the half that catches a hold taken around a decision made on an
        *earlier* read: such a claim has already decided the row is ``PENDING``,
        so it spawns the moment it gets the lock.

        The hold is taken through :meth:`YamlDocumentStore.hold`, the production
        derivation, never a re-spelled path — a spec that locked a different file
        would let the child through and could never say why.
        """
        self._prepare(workspace_tree)
        log = tmp_path / "spawns.log"
        script = write_script(tmp_path, "claimer.py", _CLAIMING_CHILD)

        with YamlDocumentStore().hold(WORKSPACE_PATH, TARGET):
            child = start_child(script, workspaces_root, str(log), DECOY)
            try:
                assert child.stdout is not None
                assert child.stdout.readline().strip() == "started"

                deadline = time.monotonic() + EXCLUSION_WINDOW_S
                while time.monotonic() < deadline:
                    assert _spawns(log) == [], (
                        "the other process claimed the file while its record was held"
                    )
                    time.sleep(0.02)

                # Still holding: the claim the parent's own process made.
                _seed(
                    TARGET,
                    _row_on_disk(TARGET).model_copy(
                        update={"status": RagStatus.EXTRACTION, "updated_at": datetime.now(UTC)}
                    ),
                )
            except BaseException:
                child.kill()
                raise

        assert child.wait(timeout=CHILD_TIMEOUT_S) == 0, child.communicate()
        assert _spawns(log) == [], "the child spawned a worker for a file already claimed"

    def test_the_same_child_does_spawn_when_the_record_is_free(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """The positive control, and it is mandatory rather than decoration.

        A child that could never spawn — a broken patch, a card that bound
        degraded, a tool name that no longer exists — would make both halves above
        pass with the lock deleted outright. So the same child, on the same tree,
        with nothing held, must record exactly one spawn.
        """
        self._prepare(workspace_tree)
        log = tmp_path / "spawns.log"
        script = write_script(tmp_path, "claimer.py", _CLAIMING_CHILD)

        child = start_child(script, workspaces_root, str(log), DECOY)
        code = child.wait(timeout=CHILD_TIMEOUT_S)

        assert code == 0, child.communicate()
        assert len(_spawns(log)) == 1, _spawns(log)
        assert _row_on_disk(TARGET).status is not RagStatus.PENDING


_ENQUEUE_CHILD = """
card = bind("child", workspace_rag_index=True)
print("started", flush=True)
for tool in card.get_tools():
    if tool.__name__ == "workspace_rag_index":
        print("ANSWER " + tool(sys.argv[1]), flush=True)
        break
"""
"""A child whose ``index_paths`` re-queues one file, appending to its superseded ids.

No spawn recorder: what this child is here for is ``_enqueue``'s append, which
runs inside the candidate's record hold. The worker it goes on to start is real
and harmless — the file is tiny and the child exits.
"""


class TestASupersededListSurvivesAConcurrentClear:
    """B4's second half: the list cannot be lost, so paid vectors are not orphaned.

    ``_enqueue`` moves the previous chunk set's ids into ``superseded_chunk_ids``
    so a later removal can still find them; ``_drop_superseded`` clears the ones
    it has just removed. Both are read-modify-writes of one record, and the two
    interleaved as whole-file replaces lose one of them — with nothing raised,
    and with the vectors already paid for left in the index with nothing pointing
    at them.
    """

    OLD = "chunk-old"
    NEW = "chunk-new"

    def _prepare(self, workspace_tree: Path) -> str:
        """An ``EMBEDDED`` row owing one removal, whose file has since changed."""
        write(workspace_tree, TARGET, "# Changed\n\nnew bytes\n")
        _seed(
            TARGET,
            RagFile(
                path=TARGET,
                status=RagStatus.EMBEDDED,
                indexed_sha=content_sha(b"the bytes it was indexed at"),
                chunks=[RagChunk(chunk_id=self.NEW, ordinal=0, start=0, end=4)],
                chunk_count=1,
                superseded_chunk_ids=[self.OLD],
                updated_at=datetime.now(UTC),
            ),
        )
        return TARGET

    def test_the_childs_append_cannot_land_inside_the_clears_hold(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """**The exclusion half is the red-before one**, and it is deterministic.

        The parent holds ``target.md``'s record and performs the clear a settled
        file's ``_drop_superseded`` performs — re-read inside the hold, drop the
        ids it has removed, write. While it holds, the child's ``_enqueue`` cannot
        append; unlocked, the append lands within milliseconds and the poll below
        fails on its first iteration.

        The outcome half is the positive control beside it: once released, the
        child's own id **is** on disk, so the clear did not simply blank the list
        it was sharing.
        """
        self._prepare(workspace_tree)
        script = write_script(tmp_path, "enqueuer.py", _ENQUEUE_CHILD)

        with YamlDocumentStore().hold(WORKSPACE_PATH, TARGET):
            child = start_child(script, workspaces_root, TARGET)
            try:
                assert child.stdout is not None
                assert child.stdout.readline().strip() == "started"

                deadline = time.monotonic() + EXCLUSION_WINDOW_S
                while time.monotonic() < deadline:
                    assert _row_on_disk(TARGET).superseded_chunk_ids == [self.OLD], (
                        "the other process rewrote the record while the clear held it"
                    )
                    time.sleep(0.02)

                current = _row_on_disk(TARGET)
                removed = {self.OLD}
                _seed(
                    TARGET,
                    current.model_copy(
                        update={
                            "superseded_chunk_ids": [
                                stale
                                for stale in current.superseded_chunk_ids
                                if stale not in removed
                            ]
                        }
                    ),
                )
            except BaseException:
                child.kill()
                raise

        assert child.wait(timeout=CHILD_TIMEOUT_S) == 0, child.communicate()
        # Both writes survived: the parent's removal of the old id, and the
        # child's append of the id it has just superseded.
        assert _row_on_disk(TARGET).superseded_chunk_ids == [self.NEW]


##
## AC 11 and AC 12 — the drain's filtered read, one process and honestly so
##


class _ClaimStealer:
    """Delegates to a real store, and claims a path as another process would.

    The claim is made **inside** ``hold``, before the caller's own re-read — which
    is the window a second process genuinely occupies. It needs no subprocess:
    what is under test here is the drain's *reaction* to a lost claim, not the
    exclusion that produces one, and exclusion is guarded above with real
    interpreters.
    """

    def __init__(self, steal_every: bool = False) -> None:
        self._inner = YamlDocumentStore()
        self.stolen: list[str] = []
        self.pending_calls: list[frozenset[str]] = []
        self.listings: list[str] = []
        self._steal_every = steal_every

    def get_document(self, tree_key: str, path: str) -> DocumentEntry | None:
        return self._inner.get_document(tree_key, path)

    def put_document(self, tree_key: str, entry: DocumentEntry) -> None:
        self._inner.put_document(tree_key, entry)

    def evict(self, tree_key: str, path: str) -> None:
        self._inner.evict(tree_key, path)

    def list_documents(self, tree_key: str) -> list[DocumentEntry]:
        self.listings.append(tree_key)
        return self._inner.list_documents(tree_key)

    def next_pending(self, tree_key: str, exclude: frozenset[str]) -> DocumentEntry | None:
        self.pending_calls.append(exclude)
        return self._inner.next_pending(tree_key, exclude)

    @contextmanager
    def hold(self, tree_key: str, path: str) -> Iterator[None]:
        if (self._steal_every or not self.stolen) and self._steal(tree_key, path):
            self.stolen.append(path)
        yield

    def _steal(self, tree_key: str, path: str) -> bool:
        """Move *path* out of ``PENDING``, or answer that there was nothing to take."""
        entry = self._inner.get_document(tree_key, path)
        if entry is None or entry.row is None or entry.row.status is not RagStatus.PENDING:
            return False
        self._inner.put_document(
            tree_key,
            entry.model_copy(
                update={"row": entry.row.model_copy(update={"status": RagStatus.EXTRACTION})}
            ),
        )
        return True


def _queued_rows(paths: list[str]) -> None:
    """Seed a ``PENDING`` row per path, with **no file** behind it.

    Nothing here is a candidate for ``index_paths``' tree walk, so the drain is
    what these specs exercise and the candidate loop cannot enqueue or claim
    anything of its own. ``_spawn`` never stats the tree — it works from the row
    and the extraction cache — so a row with no file spawns exactly as one with a
    file does.
    """
    now = datetime.now(UTC)
    for path in paths:
        _seed(
            path,
            RagFile(path=path, status=RagStatus.PENDING, indexed_sha="seeded", updated_at=now),
        )


class TestTheDrainReadsOneRecordAtATime:
    """W2: the sweep stops fetching every record to find one waiting path."""

    def test_it_performs_no_full_listing(self, harness: RagHarness, workspace_tree: Path) -> None:
        """A tree of a thousand documents must cost one record read, not a thousand.

        ``reap_abandoned_rows`` and the render legitimately need every record and
        keep ``list_documents``; the drain does not, which is the whole finding.
        The reaper's listing is taken **before** the recorder is installed, so what
        is counted is the drain's own reads.
        """
        harness.enable()
        _queued_rows(["alpha.md", "beta.md"])
        recorder = watch_store(harness.actor)

        harness.actor._drain()

        assert recorder.listings == [], "the drain listed every record on the tree"
        assert recorder.pending_calls, "the drain read no record at all"

    def test_a_lost_claim_continues_to_the_next_pending_path(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """ "Somebody else claimed it" is ordinary once two processes drain one tree.

        The steal fires for whichever path the drain reaches first, so the spec
        does not depend on the glob's order — it asserts that the one path that
        was *not* stolen is the one that got a worker.
        """
        harness.enable()
        _queued_rows(["alpha.md", "beta.md"])
        stealer = _ClaimStealer()
        harness.actor.configure_document_store(stealer)

        harness.actor._drain()

        assert len(stealer.stolen) == 1, stealer.stolen
        [claimed] = [path for path in ("alpha.md", "beta.md") if path not in stealer.stolen]
        assert [request.path for request in harness.requests] == [claimed]

    @pytest.mark.timeout(60)
    def test_losing_every_claim_terminates_instead_of_spinning(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """What ``exclude`` is for, and why it has no default.

        A drain that re-offered a path whose claim it had just lost would be
        handed the same still-``PENDING``-looking record for ever. The failure
        mode is a **hang**, not an assertion, so this spec carries its own timeout
        — a run that reaches it has found the defect even though the report will
        say "timeout" rather than "assert".
        """
        harness.enable()
        _queued_rows(["alpha.md", "beta.md"])
        stealer = _ClaimStealer(steal_every=True)
        harness.actor.configure_document_store(stealer)

        harness.actor._drain()

        assert harness.requests == []
        assert sorted(stealer.stolen) == ["alpha.md", "beta.md"]
        # Every pass after the first carries what the previous one lost, which is
        # the only thing that can terminate the loop.
        assert stealer.pending_calls[-1] == frozenset({"alpha.md", "beta.md"})

    def test_a_spawn_that_raised_still_stops_the_pass(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A lost claim continues; a process that cannot start actors stops.

        Two waiting rows and a ``createActor`` that raises: the first attempt
        fails the file and the drain returns rather than trying the second.
        """
        harness.enable()
        _queued_rows(["alpha.md", "beta.md"])
        harness.spawn_error = RuntimeError("can't start new thread")

        harness.actor._drain()

        failed = [
            path
            for path in ("alpha.md", "beta.md")
            if _row_on_disk(path).status is RagStatus.FAILED
        ]
        still_pending = [
            path
            for path in ("alpha.md", "beta.md")
            if _row_on_disk(path).status is RagStatus.PENDING
        ]
        assert len(failed) == 1, failed
        assert len(still_pending) == 1, still_pending


class TestTheSlotCountIsAResourceBoundNotADeDuplicator:
    """AC 10 — what survives of ``_index_active``, and what it is allowed to mean."""

    def test_the_count_bounds_the_pass(self, harness: RagHarness, workspace_tree: Path) -> None:
        """A process already at its cap spawns nothing more."""
        from akgentic.tool.workspace.documents.worker import MAX_CONCURRENT_INDEX_WORKERS

        harness.enable()
        _queued_rows(["alpha.md"])
        harness.actor._index_workers = MAX_CONCURRENT_INDEX_WORKERS

        harness.actor._drain()

        assert harness.requests == []
        assert _row_on_disk("alpha.md").status is RagStatus.PENDING

    def test_a_report_gives_its_slot_back(self, harness: RagHarness, workspace_tree: Path) -> None:
        """One spawn, one report, one slot — balanced by construction."""
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        assert harness.actor._index_workers == 1

        harness.report("a.md")

        assert harness.actor._index_workers == 0
