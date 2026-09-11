"""The exec lock, on its own — no actor, no card, no sandbox.

Story 52-2. Everything here is the backend and the filesystem: a ``tree_key``, a
temporary workspaces root, and one or two :class:`FileLockBackend` objects. That
is deliberate rather than convenient — the hold has to be exclusive between two
things that share nothing but the volume, so a spec that reached through the
actor could not tell an ``O_EXCL`` create from an in-process flag.

Nothing here sleeps. The staleness specs move the marker's mtime with
``os.utime`` instead of waiting out a window, and the concurrency spec
synchronises two threads on a ``threading.Barrier`` rather than on a delay.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import ValidationError

from akgentic.tool.workspace import (
    EXEC_LOCK_FILENAME,
    LEASE_GRACE_S,
    META_DIR_SUFFIX,
    FileLockBackend,
    LockBackend,
    LockGrant,
    LockMarker,
    LockTicket,
    resolve_lock_backend,
)
from akgentic.tool.workspace.workspace import meta_dir_for
from tests.workspace.conftest import HANDSHAKE_TIMEOUT_S

TREE = "alice/_id/notes"
"""The three-segment ``<scope>/<kind>/<leaf>`` key every spec here locks."""

AGENT = "agent-1"
AGENT_B = "agent-2"

BUDGET_S = 1.0
"""A short run budget, so the staleness window is ``1 + LEASE_GRACE_S`` seconds.

Never waited out: every spec that needs a stale marker sets its mtime directly.
"""


@pytest.fixture
def tree_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the workspaces root at a temporary directory, and make the tree.

    ``monkeypatch`` for every environment variable, without exception: all three
    of ``AKGENTIC_WORKSPACES_ROOT``, ``AKGENTIC_WORKSPACE_META_ROOT`` and
    ``AKGENTIC_LOCK_BACKEND`` are read at call time, so a leaked value silently
    changes another module's tests.
    """
    root = tmp_path / "workspaces"
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(root))
    monkeypatch.delenv("AKGENTIC_WORKSPACE_META_ROOT", raising=False)
    tree = root / TREE
    tree.mkdir(parents=True)
    return tree


def ticket(agent_id: str = AGENT, cmd: str = "echo hi") -> LockTicket:
    """A ticket for *agent_id*, under the short budget every spec here uses."""
    return LockTicket(agent_id=agent_id, cmd=cmd, budget_s=BUDGET_S)


def marker_path() -> Path:
    """Where the marker for :data:`TREE` belongs, derived exactly as production does."""
    return meta_dir_for(TREE) / EXEC_LOCK_FILENAME


def held() -> LockMarker:
    """Parse the marker currently on disk."""
    return LockMarker.model_validate_json(marker_path().read_text())


def age_marker(seconds: float) -> None:
    """Move the marker's mtime *seconds* into the past."""
    path = marker_path()
    when = time.time() - seconds
    os.utime(path, (when, when))


# ---------------------------------------------------------------------------
# AC1 — the protocol, the three models, and their serialization
# ---------------------------------------------------------------------------


class TestTheProtocolAndItsModels:
    def test_the_three_models_are_serializable(self) -> None:
        # Everything crossing a boundary is a Pydantic model, and the marker
        # crosses the widest one there is: it is written by one process and read
        # by another.
        for model in (LockTicket, LockGrant, LockMarker):
            assert issubclass(model, SerializableBaseModel)

    def test_every_model_round_trips(self) -> None:
        for instance in (
            ticket(),
            LockGrant(run_id="abc123"),
            LockGrant(refusal="busy"),
            LockMarker(run_id="abc123", agent_id=AGENT),
        ):
            again = type(instance).model_validate_json(instance.model_dump_json())
            assert again == instance

    def test_a_grant_carrying_both_is_rejected(self) -> None:
        # The same rule ``ExecStart`` enforces, for the same reason: every
        # caller branches on ``if not grant.run_id``, so an answer carrying both
        # would run a command the backend had decided to refuse.
        with pytest.raises(ValidationError):
            LockGrant(run_id="abc123", refusal="busy")

    def test_a_grant_carrying_neither_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LockGrant()

    def test_the_file_backend_satisfies_the_protocol_at_runtime(self) -> None:
        assert isinstance(FileLockBackend(), LockBackend)

    def test_the_backend_takes_no_constructor_arguments(self) -> None:
        # What lets the registry build any entry with no type switch, exactly as
        # ``SANDBOX_BACKEND_CLASSES[resolved]()`` does.
        assert FileLockBackend() is not None


class TestTheBackendRegistry:
    def test_an_unset_variable_resolves_the_file_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AKGENTIC_LOCK_BACKEND", raising=False)
        assert isinstance(resolve_lock_backend(), FileLockBackend)

    def test_the_file_name_resolves_the_same_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AKGENTIC_LOCK_BACKEND", "file")
        assert isinstance(resolve_lock_backend(), FileLockBackend)

    def test_an_empty_value_falls_back_rather_than_being_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A compose file interpolating an unset variable and a bare ``FOO=`` in
        # an env file both arrive as "", and neither is somebody asking for a
        # backend called the empty string.
        monkeypatch.setenv("AKGENTIC_LOCK_BACKEND", "")
        assert isinstance(resolve_lock_backend(), FileLockBackend)

    def test_an_unregistered_name_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AKGENTIC_LOCK_BACKEND", "nope")
        with pytest.raises(KeyError):
            resolve_lock_backend()


# ---------------------------------------------------------------------------
# AC2 — the marker is a sibling, and this is what first creates <meta>
# ---------------------------------------------------------------------------


class TestTheMarkerIsASiblingOfTheTree:
    def test_the_metadata_directory_does_not_exist_before_the_first_acquire(
        self, tree_root: Path
    ) -> None:
        # 52-1 deliberately created nothing; this story is the first writer.
        assert not meta_dir_for(TREE).exists()

    def test_an_acquire_creates_the_directory_and_the_marker(self, tree_root: Path) -> None:
        grant = FileLockBackend().acquire(TREE, ticket())

        assert grant.run_id
        assert marker_path().is_file()
        assert held() == LockMarker(run_id=grant.run_id, agent_id=AGENT)

    def test_the_marker_is_outside_the_tree_the_capabilities_can_reach(
        self, tree_root: Path
    ) -> None:
        # The containment rule, not a naming preference: a path beside the tree
        # is one no read capability can name and no sandboxed ``rm -rf`` can
        # delete — including the lock guarding that very run.
        FileLockBackend().acquire(TREE, ticket())

        assert not marker_path().is_relative_to(tree_root)
        assert list(tree_root.rglob(EXEC_LOCK_FILENAME)) == []

    def test_the_tree_key_is_the_same_string_get_workspace_takes(self, tree_root: Path) -> None:
        # ``meta_dir_for`` is consumed exactly as 52-1 shipped it: the
        # workspace path, never a resolved root.
        FileLockBackend().acquire(TREE, ticket())

        assert marker_path().parent.name == f"{Path(TREE).name}{META_DIR_SUFFIX}"
        assert marker_path().parent.parent == tree_root.parent


# ---------------------------------------------------------------------------
# AC3 — two concurrent acquires, one grant, one refusal
# ---------------------------------------------------------------------------


class TestTwoConcurrentAcquires:
    def test_exactly_one_wins_and_the_winner_is_the_one_on_disk(self, tree_root: Path) -> None:
        # Two *objects*, so nothing in-process is doing the excluding, and both
        # calls are inside the same barrier so the create really does race. A
        # sequential pair would pass even if the exclusion were a Python flag —
        # which is what an ``exists()`` check-then-write would be.
        rounds = 5
        for index in range(rounds):
            tree = f"alice/tree-{index}"
            backends = [FileLockBackend(), FileLockBackend()]
            barrier = threading.Barrier(len(backends))
            answers: list[LockGrant] = []
            answers_lock = threading.Lock()

            def acquire(backend: FileLockBackend, tree: str = tree) -> None:
                barrier.wait(timeout=HANDSHAKE_TIMEOUT_S)
                grant = backend.acquire(tree, ticket())
                with answers_lock:
                    answers.append(grant)

            threads = [threading.Thread(target=acquire, args=(backend,)) for backend in backends]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=HANDSHAKE_TIMEOUT_S)
                assert not thread.is_alive(), "an acquiring thread never finished"

            granted = [answer for answer in answers if answer.run_id]
            refused = [answer for answer in answers if answer.refusal]
            assert len(granted) == 1, f"round {index}: {answers}"
            assert len(refused) == 1, f"round {index}: {answers}"

            on_disk = LockMarker.model_validate_json(
                (meta_dir_for(tree) / EXEC_LOCK_FILENAME).read_text()
            )
            assert on_disk.run_id == granted[0].run_id


# ---------------------------------------------------------------------------
# AC4, AC5 — staleness, from both sides, and the crashed holder
# ---------------------------------------------------------------------------


class TestAStaleMarkerIsTakenOver:
    def test_past_the_budget_and_the_grace_the_next_acquirer_is_granted(
        self, tree_root: Path
    ) -> None:
        backend = FileLockBackend()
        first = backend.acquire(TREE, ticket())
        age_marker(BUDGET_S + LEASE_GRACE_S + 1)

        second = FileLockBackend().acquire(TREE, ticket(AGENT_B))

        assert second.run_id
        assert second.run_id != first.run_id
        assert held() == LockMarker(run_id=second.run_id, agent_id=AGENT_B)

    def test_inside_the_window_the_next_acquirer_is_refused(self, tree_root: Path) -> None:
        # The near side of the boundary, asserted so that dropping the grace
        # from the predicate — or inverting the comparison — cannot pass.
        first = FileLockBackend().acquire(TREE, ticket())
        age_marker(BUDGET_S + LEASE_GRACE_S - 1)

        second = FileLockBackend().acquire(TREE, ticket(AGENT_B))

        assert second.refusal
        assert held().run_id == first.run_id

    def test_a_marker_that_vanishes_mid_check_is_refused_not_taken_over(
        self, tree_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The race between the failed create and the stat: the holder released
        # in between. There is nothing left to take over, so the honest answer
        # is a refusal the caller retries through — not a takeover of a file
        # that no longer exists, and not an exception either.
        FileLockBackend().acquire(TREE, ticket())
        real_stat = Path.stat

        def vanished(self: Path, *args: object, **kwargs: object) -> os.stat_result:
            if self.name == EXEC_LOCK_FILENAME:
                raise FileNotFoundError(2, "No such file or directory", str(self))
            return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "stat", vanished)

        answer = FileLockBackend().acquire(TREE, ticket(AGENT_B))

        assert answer.refusal
        assert not answer.run_id

    def test_a_write_that_fails_leaves_no_marker_behind(
        self, tree_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A half-written marker parses as nothing, so ``release`` would leave it
        # for the staleness window — locking the tree for a run that never
        # started. We created it, so removing it is the one safe cleanup there
        # is, and the failure still reaches the caller.
        def explode(fd: int, mode: str) -> object:
            os.close(fd)  # never leak the descriptor the backend opened
            raise OSError(28, "No space left on device")

        # A ``context()`` rather than ``monkeypatch.undo()``: the fixture's own
        # ``setenv`` of the workspaces root is on the SAME function-scoped
        # monkeypatch, so an ``undo()`` here reverts that too — and every
        # assertion below would then read a path under the developer's checkout
        # rather than under ``tmp_path``. (Measured, not assumed: the first
        # draft of this spec did exactly that, passed vacuously, and wrote a
        # marker into the repository.)
        with monkeypatch.context() as patched:
            patched.setattr("akgentic.tool.workspace.lock.os.fdopen", explode)
            with pytest.raises(OSError, match="No space left"):
                FileLockBackend().acquire(TREE, ticket())

        assert marker_path().is_relative_to(tree_root.parent)
        assert not marker_path().exists()
        assert FileLockBackend().acquire(TREE, ticket(AGENT_B)).run_id

    def test_a_crashed_holder_does_not_wedge_the_tree_for_ever(self, tree_root: Path) -> None:
        # No actor, no process bookkeeping, no release — the holder simply
        # vanishes, which is what a killed worker looks like from the volume.
        # This is the property the whole file design rests on.
        crashed = FileLockBackend()
        crashed.acquire(TREE, ticket())
        del crashed

        age_marker(BUDGET_S + LEASE_GRACE_S + 1)

        assert FileLockBackend().acquire(TREE, ticket(AGENT_B)).run_id


# ---------------------------------------------------------------------------
# AC6 — release is never a steal
# ---------------------------------------------------------------------------


class TestReleaseOnlyReleasesWhatItHolds:
    def test_a_foreign_run_id_leaves_the_marker_untouched(self, tree_root: Path) -> None:
        backend = FileLockBackend()
        mine = backend.acquire(TREE, ticket())
        before = marker_path().read_text()

        backend.release(TREE, "not-my-run")

        assert marker_path().read_text() == before
        # And the genuine holder can still give it back afterwards.
        backend.release(TREE, mine.run_id)
        assert not marker_path().exists()

    def test_releasing_a_tree_with_no_marker_does_not_raise(self, tree_root: Path) -> None:
        FileLockBackend().release(TREE, "nothing-holds-this")

    def test_an_unparseable_marker_is_left_for_staleness(self, tree_root: Path) -> None:
        # Unlinking a marker we cannot read is how a release steals a hold it
        # does not own: the bytes may belong to a writer this code never met.
        FileLockBackend().acquire(TREE, ticket())
        marker_path().write_text("{ this is not a marker")

        FileLockBackend().release(TREE, "any-run-at-all")

        assert marker_path().exists()

    def test_an_unparseable_marker_is_still_reclaimed_by_staleness(self, tree_root: Path) -> None:
        # "Left in place" must not mean "wedged for ever" — the mtime window is
        # what clears it, exactly as it clears a crashed holder's.
        FileLockBackend().acquire(TREE, ticket())
        marker_path().write_text("{ this is not a marker")
        age_marker(BUDGET_S + LEASE_GRACE_S + 1)

        assert FileLockBackend().acquire(TREE, ticket(AGENT_B)).run_id


# ---------------------------------------------------------------------------
# AC7 — the exec refusal names nobody
# ---------------------------------------------------------------------------


class TestTheRefusalNamesNobody:
    def test_it_carries_neither_the_holders_run_id_nor_its_agent(self, tree_root: Path) -> None:
        # The defect ADR-047 removed: a loser reading the winner's id out of a
        # refusal and collecting it as its own answer. A refused exec caller has
        # no id of its own to be given instead, so it is given nobody's.
        holder = FileLockBackend().acquire(TREE, ticket(AGENT))

        refused = FileLockBackend().acquire(TREE, ticket(AGENT_B))

        assert refused.refusal
        assert holder.run_id not in refused.refusal
        assert AGENT not in refused.refusal

    def test_it_opens_with_the_wording_every_other_refusal_shares(self, tree_root: Path) -> None:
        # One prefix across all seven refused operations, so an agent recognises
        # the family without learning a second protocol.
        FileLockBackend().acquire(TREE, ticket())

        refused = FileLockBackend().acquire(TREE, ticket(AGENT_B))

        assert refused.refusal.startswith("workspace busy")
