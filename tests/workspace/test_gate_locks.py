"""The locks the gate is made of, proved by real second interpreters.

**Every spec in this module spawns a subprocess, and that is the point.** The
three properties it guards — the per-path write lock, the journal's commit lock,
and the exec busy refusal — are each about a thing a ``threading.Lock``, one
actor's mailbox or the GIL *already* provides inside one interpreter. A spec that
contends two threads therefore passes whether or not the file lock exists, which
is not a hypothesis: story 52-4 deleted ``fcntl.flock`` outright and watched three
separate single-process formulations stay green, including one that asserted the
hold's enter and exit around the publish. Only real second interpreters bit.

So the shape here is 52-4's, deliberately copied rather than reinvented
(``tests/vector_store/test_local.py``): the parent either takes a hold directly,
by the same path the production code derives, and watches a child that has
*started and said so* fail to make progress until it is released — or it starts
two children and checks that what they produced together is possible only under
exclusion.

**What the children are is a real card.** They import this suite's own observer
and orchestrator fakes and call ``observer()``, so every child drives the shipped
bind, the shipped gate and the shipped journal. A child that constructed a
``WorkspaceTool`` and poked its private attributes would be testing this file.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from akgentic.tool.workspace.card.gate import lock_file_for
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_TIMEOUT_S,
    LEASE_GRACE_S,
    effective_budget,
    mutation_busy,
)
from akgentic.tool.workspace.lock import FileLockBackend, LockTicket
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import meta_dir_for
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    mutate,
    read,
    requires_git,
)

CHILD_TIMEOUT_S = 180.0
"""Upper bound on a child — a failure budget, never a delay."""

_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])
"""The package root, so a child can import ``tests.workspace.conftest``."""

_PRELUDE = f"""
import os, sys, time
sys.path.insert(0, {_PACKAGE_ROOT!r})
from pathlib import Path
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.errors import RetriableError
from tests.workspace.conftest import FakeActorToolObserver, FakeOrchestratorProxy


def bind(name, **kwargs):
    proxy = FakeOrchestratorProxy()
    card = WorkspaceTool(workspace_id={WORKSPACE_NAME!r}, **kwargs)
    card.observer(FakeActorToolObserver(proxy, name=name))
    return card


def barrier(meta, tag, count):
    ready = Path(meta) / ("ready-" + tag)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("x")
    deadline = time.time() + 60
    while time.time() < deadline:
        if len(list(Path(meta).glob("ready-*"))) >= count:
            return
        time.sleep(0.005)
    raise SystemExit("the barrier never completed")
"""


@dataclass
class ChildReport:
    """One child's exit code and what it printed."""

    code: int
    out: str
    err: str


def run_child(script: Path, workspaces_root: Path, *args: str) -> ChildReport:
    """Run *script* in a fresh interpreter, with this suite's workspaces root."""
    env = dict(os.environ)
    env["AKGENTIC_WORKSPACES_ROOT"] = str(workspaces_root)
    env.pop("AKGENTIC_WORKSPACE_META_ROOT", None)
    done = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        env=env,
        check=False,
    )
    return ChildReport(done.returncode, done.stdout.strip(), done.stderr.strip())


def start_child(script: Path, workspaces_root: Path, *args: str) -> subprocess.Popen[str]:
    """Start *script* in a fresh interpreter without waiting for it."""
    env = dict(os.environ)
    env["AKGENTIC_WORKSPACES_ROOT"] = str(workspaces_root)
    env.pop("AKGENTIC_WORKSPACE_META_ROOT", None)
    return subprocess.Popen(
        [sys.executable, str(script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def write_script(tmp_path: Path, name: str, body: str) -> Path:
    """Write a child script made of the shared prelude plus *body*."""
    script = tmp_path / name
    script.write_text(_PRELUDE + textwrap.dedent(body), encoding="utf-8")
    return script


# ---------------------------------------------------------------------------
# AC 7 — the check→write window is closed by a real cross-process flock
# ---------------------------------------------------------------------------


class TestThePathLockIsCrossProcess:
    """The window is closed by an ``fcntl.flock``, and only a second process shows it."""

    def test_a_second_process_cannot_publish_while_the_path_lock_is_held(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """The parent holds the path's lock directly; a started child cannot write.

        The hold is taken on the file :func:`lock_file_for` names — the same
        derivation the gate uses, never a re-spelling — so a gate that locked a
        *different* file would let the child through and redden this.

        **Mutation (a), delete the ``flock`` call**: the child publishes
        immediately and the loop below fails on its first iteration.
        **Mutation (c), take ``LOCK_SH`` instead of ``LOCK_EX``**: a shared hold
        does not exclude the parent's exclusive one, and the child publishes just
        the same.
        """
        target = workspace_tree / "held.md"
        script = write_script(
            tmp_path,
            "writer.py",
            """
            card = bind("child")
            print("started", flush=True)
            print(card.apply_write("held.md", "from the other process\\n").message, flush=True)
            """,
        )

        lock_path = lock_file_for(meta_dir_for(WORKSPACE_PATH), "held.md")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        held = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(held, fcntl.LOCK_EX)
            child = start_child(script, workspaces_root)
            try:
                assert child.stdout is not None
                assert child.stdout.readline().strip() == "started"
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    assert not target.exists(), (
                        "the other process published while the path lock was held"
                    )
                    time.sleep(0.02)
            finally:
                fcntl.flock(held, fcntl.LOCK_UN)
        finally:
            os.close(held)

        assert child.wait(timeout=CHILD_TIMEOUT_S) == 0
        assert target.read_text(encoding="utf-8") == "from the other process\n"

    def test_two_processes_creating_one_path_produce_one_winner(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """The cross-process twin of ``test_two_agents_creating_one_path_produce_one_winner``.

        Two real interpreters, two real cards, one path, released together by a
        filesystem barrier. **Without the lock both win**: each reads the live
        file, finds nothing, is allowed to create, and publishes — so the loser's
        bytes silently replace the winner's and *no refusal is produced at all*.
        That is the lost update the gate exists to prevent, and it is invisible
        to every single-process spec because a mailbox or a ``threading.Lock``
        already serialises the pair.
        """
        meta = meta_dir_for(WORKSPACE_PATH)
        meta.mkdir(parents=True, exist_ok=True)
        script = write_script(
            tmp_path,
            "racer.py",
            """
            tag = sys.argv[1]
            card = bind(tag)
            barrier(sys.argv[2], tag, 2)
            try:
                print("OK " + card.apply_write("race.md", tag + " was here\\n").message, flush=True)
            except Exception as exc:
                print("RAISED " + repr(exc), flush=True)
            """,
        )

        children = [start_child(script, workspaces_root, tag, str(meta)) for tag in ("a", "b")]
        reports = [ChildReport(c.wait(CHILD_TIMEOUT_S), *c.communicate()) for c in children]

        assert [report.code for report in reports] == [0, 0], reports
        messages = [report.out.strip() for report in reports]
        written = [line for line in messages if "Written: race.md" in line]
        refused = [line for line in messages if "read it before overwriting" in line]
        assert len(written) == 1, f"exactly one create must survive, got {messages}"
        assert len(refused) == 1, f"the loser must be refused, got {messages}"
        assert (workspace_tree / "race.md").read_text(encoding="utf-8") in (
            "a was here\n",
            "b was here\n",
        )

    def test_the_lock_file_is_a_sibling_of_the_tree_and_unnameable_from_inside_it(
        self, workspaces_root: Path, workspace_tree: Path
    ) -> None:
        """Placement is a containment rule, not a naming preference.

        A lock inside the tree would be listable, greppable, readable — and
        deletable by an ``rm -rf`` from a sandboxed run, which is the lock
        guarding that very run.
        """
        lock_path = lock_file_for(meta_dir_for(WORKSPACE_PATH), "notes.md")

        assert not lock_path.is_relative_to(workspace_tree.resolve())
        assert lock_path.parent.parent == meta_dir_for(WORKSPACE_PATH)
        # And two paths get two files, or the lock would serialise the tree.
        other = lock_file_for(meta_dir_for(WORKSPACE_PATH), "other.md")
        assert other != lock_path


class TestTheWritersRefreshHappensInsideTheHeldLock:
    """AC 9, and the obvious guard for it is **inert** — which is why this one exists.

    ``_accept`` refreshes the writing agent's own observation, because an agent
    that has just written a file has by definition observed it in full. It has to
    happen *inside* the lock the write happened under: released first, another
    writer can land between the two, and the agent's next write to the path is
    then refused against content that is no longer there — a failure that looks
    like a gate bug and surfaces one turn after the mistake.

    **The behavioural guards do not catch that, and were measured not to.**
    Moving the ``_accept`` call out of ``_write`` and into the convergence point,
    after the ``with`` block — the edit a developer would actually make — left
    ``test_write_gate`` and ``test_live_hash`` entirely green. In one interpreter
    nothing can occupy the window, so no spec written in terms of what an agent
    is *told* can see it. Only this class went red.

    So the property is asserted where it lives. A **second process** tries to
    take the path's lock, without blocking, at the moment ``_accept`` runs. Inside
    the region it cannot have it; outside, it can — and that is the whole
    difference, stated as the thing it is rather than as a consequence no
    single-process test can reach.

    **What this class does not claim**, so that nobody reads more into a green
    run than is there: it pins *where the call happens*, not what the call does.
    A variant that kept the call inside the region and deferred only the map
    write stays green here, and correctly — the observation is still recorded
    before the mutation returns, so no window is opened by it.
    """

    _PROBE = """
        import fcntl, os, sys

        handle = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("ACQUIRED", flush=True)
        except OSError:
            print("BLOCKED", flush=True)
        """

    def _probe_during_accept(
        self,
        card: WorkspaceTool,
        workspaces_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[str]:
        """Run the non-blocking probe from a second interpreter inside every ``_accept``."""
        probe = tmp_path / "probe.py"
        probe.write_text(textwrap.dedent(self._PROBE), encoding="utf-8")
        seen: list[str] = []
        real = WorkspaceTool._accept

        def probing(this: WorkspaceTool, path: str, data: bytes) -> None:
            lock_path = lock_file_for(meta_dir_for(WORKSPACE_PATH), path)
            seen.append(run_child(probe, workspaces_root, str(lock_path)).out)
            real(this, path, data)

        monkeypatch.setattr(WorkspaceTool, "_accept", probing)
        return seen

    def test_no_other_process_can_take_the_path_lock_while_accept_runs(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A write: the probe is blocked, so the refresh is inside the region."""
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
        seen = self._probe_during_accept(card, workspaces_root, tmp_path, monkeypatch)

        assert card.apply_write("notes.md", "mine\n").message == "Written: notes.md"

        assert seen == ["BLOCKED"]

    def test_the_probe_can_take_the_lock_once_the_mutation_has_finished(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """The positive control, without which ``BLOCKED`` proves nothing.

        A probe that could never acquire — a bad path, a mis-spelled flag, a
        child that always failed — would make the spec above pass with the lock
        deleted outright. So the same probe, on the same file, after the
        mutation has released it, must come back ``ACQUIRED``.
        """
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
        card.apply_write("notes.md", "mine\n")
        probe = tmp_path / "probe.py"
        probe.write_text(textwrap.dedent(self._PROBE), encoding="utf-8")

        lock_path = lock_file_for(meta_dir_for(WORKSPACE_PATH), "notes.md")
        assert run_child(probe, workspaces_root, str(lock_path)).out == "ACQUIRED"

    def test_every_publishing_mutation_refreshes_inside_the_region(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``edit``, ``multi_edit`` and ``patch`` publish through their own paths.

        Three publication points reach ``_accept``, and a lock held around one of
        them says nothing about the other two.
        """
        from akgentic.tool.workspace.edit import EditItem

        for name, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
        read(card, "a.py")
        read(card, "b.py")
        seen = self._probe_during_accept(card, workspaces_root, tmp_path, monkeypatch)

        assert card.apply_edit("a.py", "x = 1", "x = 10").status is not None
        assert (
            card.apply_multi_edit(
                [
                    EditItem(path="a.py", old_string="x = 10", new_string="x = 100"),
                    EditItem(path="b.py", old_string="y = 2", new_string="y = 20"),
                ]
            ).status
            is not None
        )
        assert (
            card.apply_patch("--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-y = 20\n+y = 200\n").status
            is not None
        )

        assert seen == ["BLOCKED"] * 4, seen


# ---------------------------------------------------------------------------
# AC 18 — the journal survives two processes committing at once
# ---------------------------------------------------------------------------


_JOURNAL_CHILD = """
import contextlib
from akgentic.tool.workspace.journal import GitJournal, Identity

root, meta, tag, mode, rounds = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
)

if mode == "unlocked":
    @contextlib.contextmanager
    def _no_lock(self):
        yield
    GitJournal._holding = _no_lock

journal = GitJournal(root, enabled=True, timeout_s=60.0, meta_dir=meta)
if not journal.initialise():
    print("INIT-FAILED", flush=True)
    raise SystemExit(0)
print("inited", flush=True)

go = meta / "go"
deadline = time.time() + 60
while time.time() < deadline and not go.exists():
    time.sleep(0.005)

for index in range(rounds):
    name = tag + "-" + str(index) + ".md"
    (root / name).write_text(tag + " " + str(index) + "\\n", encoding="utf-8")
    journal.commit_paths([name], Identity(tag, tag), "write")

print("enabled=" + str(journal.enabled), flush=True)
"""

_JOURNAL_ROUNDS = 25
"""Commits per child. Two children at this depth collided on the first attempt."""


def _drive_journal_race(
    tmp_path: Path, workspaces_root: Path, tree: Path, mode: str
) -> tuple[list[ChildReport], set[str]]:
    """Run two interpreters committing into one repository; return what they said and landed.

    The two ``initialise`` calls are **staggered** — the parent waits for each
    child's ``inited`` before starting the next — so the only thing being raced
    is the commit path. (Repository creation races too, on git's ``config.lock``,
    which is why the lock covers it; that is a different failure and mixing the
    two would make neither observable.)
    """
    meta = meta_dir_for(WORKSPACE_PATH)
    meta.mkdir(parents=True, exist_ok=True)
    script = tmp_path / f"journal_{mode}.py"
    script.write_text(
        "import sys, time\nfrom pathlib import Path\n" + textwrap.dedent(_JOURNAL_CHILD),
        encoding="utf-8",
    )

    children = []
    for tag in ("alpha", "beta"):
        child = start_child(
            script, workspaces_root, str(tree), str(meta), tag, mode, str(_JOURNAL_ROUNDS)
        )
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "inited", "a child failed to initialise"
        children.append(child)
    (meta / "go").write_text("x", encoding="utf-8")
    reports = [ChildReport(c.wait(CHILD_TIMEOUT_S), *c.communicate()) for c in children]

    from tests.workspace.conftest import journal_log

    landed = {commit.subject for commit in journal_log(tree)}
    return reports, landed


def _expected_subjects() -> set[str]:
    """Every commit subject the two children between them must produce."""
    return {
        f"write: {tag}-{index}.md" for tag in ("alpha", "beta") for index in range(_JOURNAL_ROUNDS)
    }


@requires_git
class TestTheJournalSurvivesTwoProcesses:
    """One ``flock`` on ``<meta>/locks/journal``, and the proof that it is load-bearing.

    **The failure was observed before the fix was written**, which AC 18 makes
    binding because the hazard was originally *reasoned* from ``GitJournal._run``
    and the reasoning was wrong in its detail. What actually happens is worse
    than what was predicted: git's ``index.lock`` collision makes ``git add``
    exit non-zero, ``_commit`` reports it through ``_warn`` and returns — so the
    journal **stays enabled** and simply loses the commit. Nothing is disabled,
    nothing raises, and the only trace is one warning nobody reads.

    Measured, twice, on this machine: unlocked, 10 of 50 commits landed and 40
    vanished; locked, 50 of 50 landed with both journals still enabled.
    """

    def test_two_processes_committing_at_once_lose_no_commit(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """The guard. Every commit both children make is in the log afterwards.

        Mandatory mutations, each recorded with what it does to this spec:
        **(a) delete the journal ``flock``** — commits vanish and the assertion
        below fails by dozens; **(b) release it before the commit runs** — the
        same, because the pair ``add``-then-``commit`` is what is not atomic.
        """
        reports, landed = _drive_journal_race(tmp_path, workspaces_root, workspace_tree, "locked")

        assert [report.code for report in reports] == [0, 0], reports
        assert all("enabled=True" in report.out for report in reports), reports
        missing = sorted(_expected_subjects() - landed)
        assert missing == [], f"{len(missing)} commit(s) lost under the lock: {missing[:5]}"

    def test_the_same_run_without_the_lock_loses_what_it_committed(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """The demonstration, and it is not decoration.

        A guard that only asserted the happy path would prove nothing here,
        because the defect's symptom is **silence**: two children that never
        collided would produce the same green. So the unlocked run is driven too,
        and the hazard has to be visible in it — as a lost commit, a journal that
        turned itself off, or git saying so on stderr.

        The disjunction is deliberate rather than loose. Which of the three
        shows up depends on where in ``add``/``commit`` the collision lands, and
        pinning one of them would be narrowing the check until it agreed with a
        particular interleaving.
        """
        reports, landed = _drive_journal_race(tmp_path, workspaces_root, workspace_tree, "unlocked")

        missing = _expected_subjects() - landed
        disabled = any("enabled=False" in report.out for report in reports)
        complained = any("index.lock" in report.err for report in reports)
        assert missing or disabled or complained, (
            "two unlocked processes committed into one repository with no trace of a "
            "collision — if git has stopped using index.lock, this module's lock and "
            "the reasoning behind it need revisiting rather than deleting"
        )


# ---------------------------------------------------------------------------
# AC 5 — the busy refusal is answered from the tree, not from one process
# ---------------------------------------------------------------------------


class TestTheBusyRefusalIsCrossProcess:
    """A run held by a **second** process refuses this one's mutations.

    This is the property the actor's ``_running`` could never have: it is an
    attribute of one instance in one interpreter, so two workers over one mounted
    tree held two of them and neither saw the other's run. The marker at
    ``<meta>/exec.lock`` is the only thing both of them can read.
    """

    def _hold_the_tree(self, tmp_path: Path, workspaces_root: Path, budget_s: float) -> str:
        """Take the tree's exec hold from a **second interpreter**, and leave it taken.

        The child exits once the marker is written: an ``O_EXCL`` marker is state
        on disk, not a held file descriptor, so the hold outlives the process
        that took it — which is exactly why it can fence a worker that has since
        crashed, and exactly why this spec can be written at all.
        """
        script = write_script(
            tmp_path,
            "holder.py",
            f"""
            from akgentic.tool.workspace.lock import FileLockBackend, LockTicket

            grant = FileLockBackend().acquire(
                {WORKSPACE_PATH!r},
                LockTicket(
                    agent_id="other-process-id",
                    agent_name="the-other-agent",
                    cmd="sleep 1",
                    budget_s={budget_s!r},
                ),
            )
            assert grant.run_id, grant.refusal
            print(grant.run_id, flush=True)
            """,
        )
        report = run_child(script, workspaces_root)
        assert report.code == 0, report
        return report.out.strip()

    def test_a_mutation_is_refused_under_another_processs_run(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """And the text is byte-identical to the one the actor produced.

        The name comes from the marker, which is why it survives the process
        boundary at all: this card has never heard of ``the-other-agent`` and has
        no name map to look it up in. Degrading to the id was refused — the
        reader of this text is a model deciding what to do next, and a UUID gives
        it nothing to act on.
        """
        from akgentic.tool.errors import RetriableError

        run_id = self._hold_the_tree(tmp_path, workspaces_root, budget_s=600.0)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))

        with pytest.raises(RetriableError) as refusal:
            mutate(card, "workspace_write", "fresh.md", "body\n")

        assert str(refusal.value) == mutation_busy(run_id, "the-other-agent")
        assert "workspace busy" in str(refusal.value)
        assert not (workspace_tree / "fresh.md").exists()

    def test_every_one_of_the_six_mutations_is_refused(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """All six, because the busy check lives at the one point they converge on.

        ``mkdir`` included: it takes no path lock and is gated by neither table,
        and it is still refused — which is what says the check is at the
        convergence point rather than in the five places that happened to need it.
        """
        from akgentic.tool.workspace.edit import EditItem

        (workspace_tree / "notes.md").write_text("alpha\nbravo\n", encoding="utf-8")
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
        read(card, "notes.md")
        self._hold_the_tree(tmp_path, workspaces_root, budget_s=600.0)

        calls = [
            lambda: card.apply_write("notes.md", "mine\n"),
            lambda: card.apply_delete("notes.md"),
            lambda: card.apply_edit("notes.md", "alpha", "ALPHA"),
            lambda: card.apply_multi_edit(
                [EditItem(path="notes.md", old_string="a", new_string="b")]
            ),
            lambda: card.apply_patch(
                "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n-alpha\n+ALPHA\n bravo\n"
            ),
            lambda: card.apply_mkdir("sub"),
        ]
        outcomes = [call() for call in calls]

        assert all("workspace busy" in outcome.message for outcome in outcomes), outcomes
        assert (workspace_tree / "notes.md").read_text(encoding="utf-8") == "alpha\nbravo\n"
        assert not (workspace_tree / "sub").exists()

    def test_a_refusal_reads_no_file_and_makes_no_commit(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The busy check is ahead of the gate's file read, and ahead of the journal.

        A refusal that first opened the file would be doing work it is about to
        throw away, once per refused mutation on a busy tree — which is the state
        a tree is in for the whole of a long run.
        """
        from akgentic.tool.workspace.workspace import Filesystem

        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, git_journal=True)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
        self._hold_the_tree(tmp_path, workspaces_root, budget_s=600.0)

        reads: list[str] = []
        real_read = Filesystem.read

        def watched(self: Filesystem, path: str) -> bytes:
            reads.append(path)
            return real_read(self, path)

        commits: list[str] = []
        monkeypatch.setattr(Filesystem, "read", watched)
        monkeypatch.setattr(
            type(card._journal),
            "commit_out_of_band",
            lambda self: commits.append("out-of-band"),
        )

        assert "workspace busy" in card.apply_write("fresh.md", "body\n").message

        assert reads == []
        assert commits == []

    def test_past_the_budget_and_the_grace_the_hold_is_not_live(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """A marker older than ``budget_s + LEASE_GRACE_S`` fences nobody.

        The window is **derived** from the one predicate ``FileLockBackend``
        already uses rather than respelled here, so a mutation is never refused
        against a hold the very next ``request_exec`` would take over — which is
        what the second half asserts, on the same marker.

        **The budget is the *asker's*, not the holder's**, which is inherited
        rather than introduced: ``acquire`` has always measured staleness against
        the ticket in front of it, and a mutation gate that used a different
        number would be the second spelling of one window that this story exists
        to avoid. So the ageing below is measured against **this card's** budget,
        which is the default one because the card enables no exec of its own.
        """
        self._hold_the_tree(tmp_path, workspaces_root, budget_s=1.0)
        marker = meta_dir_for(WORKSPACE_PATH) / "exec.lock"
        window = effective_budget(DEFAULT_EXEC_TIMEOUT_S) + LEASE_GRACE_S
        aged = time.time() - (window + 1.0)
        os.utime(marker, (aged, aged))

        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))

        assert card.apply_write("fresh.md", "body\n").message == "Written: fresh.md"
        # And the two clocks agree: the same marker is takeover-stale for exec.
        grant = FileLockBackend().acquire(
            WORKSPACE_PATH,
            LockTicket(
                agent_id="next",
                agent_name="next",
                cmd="echo",
                budget_s=effective_budget(DEFAULT_EXEC_TIMEOUT_S),
            ),
        )
        assert grant.run_id

    def test_a_live_hold_from_the_same_process_refuses_too(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """The positive control for the ageing above.

        Without it, a gate that answered ``None`` for *every* marker — because it
        read the wrong path, or swallowed the read — would pass the staleness
        spec for entirely the wrong reason.
        """
        self._hold_the_tree(tmp_path, workspaces_root, budget_s=600.0)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))

        assert "workspace busy" in card.apply_write("fresh.md", "body\n").message
