"""The strategy contract: the Protocol, the filter, ``kill``, ``stop``, the registry.

``kill`` is the centre of this file and the one piece of genuinely new behaviour.
The obvious spec for it is inert:

    def test_kill_does_not_raise() -> None:
        LocalBackend().kill()

That passes identically whether ``kill`` works or its body is ``pass``, so it is
written here as the *idempotence* spec and never as the guard. The guard is
:func:`test_kill_ends_a_command_that_is_actually_running`, which observes the
effect: a real 30-second command, killed from another thread, has to make
``exec()`` return far inside its own 25-second budget carrying a signal exit
code. A ``kill`` that does nothing, signals the wrong process, or whose effect
``_run`` fails to report all turn that red.

The budget is deliberately long. Shortening it to make the spec quick would let
the *budget* end the run, and the spec would then pass with ``kill`` stubbed out.

**The group kill has a guard of its own, and it is not those two specs.** On
macOS ``sh -c 'sleep 30'`` execs ``sleep`` in place, so the direct child *is*
the sleeping process and the specs above pass whether or not the process group
is signalled — which is exactly how CI stayed red on Linux for three stories
while the local gate stayed green. The guards in *Decision 4b* below make the
shell genuinely fork: a background job whose pid the shell writes to a file,
asserted **dead** after the kill, never merely by elapsed time.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core.agent import Akgent

from akgentic.tool.sandbox import SANDBOX_BACKEND_CLASSES
from akgentic.tool.sandbox.backend import (
    CommandNotAllowedError,
    CommandParseError,
    ExecResult,
    ProcessBackend,
    SandboxBackend,
)
from akgentic.tool.sandbox.bwrap import BwrapBackend
from akgentic.tool.sandbox.docker import DockerBackend
from akgentic.tool.sandbox.local import LocalBackend
from akgentic.tool.sandbox.seatbelt import SeatbeltBackend
from akgentic.tool.workspace.execution import resolve_mode

POPEN = "akgentic.tool.sandbox.backend.subprocess.Popen"
KILLPG = "akgentic.tool.sandbox.backend.os.killpg"

#: The command's own budget. Long on purpose — see the module docstring.
KILL_SPEC_BUDGET_S = 25.0

#: How long the killed run may take before the spec calls it a failure. Generous:
#: the signal should land in milliseconds.
KILL_SPEC_BOUND_S = 10.0

WORKSPACE = "u-alice/notes"


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalBackend:
    """A started :class:`LocalBackend` rooted at *tmp_path*.

    ``LocalBackend`` carries the ``kill`` specs because it is the one backend that
    runs everywhere: bwrap and seatbelt are platform-gated and docker needs a
    daemon.
    """
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path))
    started = LocalBackend()
    started.start(WORKSPACE)
    return started


def _kill_once_running(target: SandboxBackend, holder: ProcessBackend) -> None:
    """Wait until *holder* has a process in flight, then kill *target*.

    Polls the handle rather than sleeping a fixed interval, so the spec neither
    races the fork nor pads itself with wall clock it does not need. If the handle
    never appears the kill is simply not sent, and the elapsed-time assertion is
    what reports it.
    """
    deadline = time.monotonic() + KILL_SPEC_BOUND_S
    while time.monotonic() < deadline:
        if holder._running is not None:
            target.kill()
            return
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# AC 8 — kill() ends a command that is actually running
# ---------------------------------------------------------------------------


def test_kill_ends_a_command_that_is_actually_running(backend: LocalBackend) -> None:
    """AC8: a real long-running command, killed from a worker, returns early and signalled.

    The fork happens on the **main** thread and the kill comes from the worker,
    not the reverse: ``preexec_fn`` is documented as unsafe in the presence of
    threads, and ``LocalBackend`` installs one.

    Name what would have to be true for this to fail to catch a broken ``kill``:
    the call would have to return inside ten seconds *and* report a signal, with
    nothing having killed anything. Neither is reachable — the command sleeps for
    thirty seconds and ``sh`` exits 0 when left alone.

    Where ``sh`` forks ``sleep`` rather than exec'ing it, the kill reaches the
    whole process group; the guards under *Decision 4b* below are what prove
    that, because on this host this spec cannot tell the difference.
    """
    killer = threading.Thread(target=_kill_once_running, args=(backend, backend), daemon=True)
    killer.start()

    started_at = time.monotonic()
    result = backend.exec("sh -c 'sleep 30'", "", KILL_SPEC_BUDGET_S)
    elapsed = time.monotonic() - started_at
    killer.join(timeout=KILL_SPEC_BOUND_S)

    assert elapsed < KILL_SPEC_BOUND_S, (
        f"exec() ran for {elapsed:.1f}s of its {KILL_SPEC_BUDGET_S}s budget — "
        "kill() did not end the command."
    )
    # The exit code, not merely "not zero": a kill that ended the wrong process
    # would also return fast if the command happened to finish on its own.
    assert result.exit_code == -signal.SIGKILL


def test_stop_ends_a_command_that_is_actually_running(backend: LocalBackend) -> None:
    """AC10: ``stop()`` ends the run in flight, observed the same way as AC 8.

    ``stop`` kills before it releases, so the effect is the same and the
    ordering is asserted separately on docker, where the release does something.
    """
    stopper = threading.Thread(
        target=lambda: _kill_once_running(_Stopper(backend), backend), daemon=True
    )
    stopper.start()

    started_at = time.monotonic()
    result = backend.exec("sh -c 'sleep 30'", "", KILL_SPEC_BUDGET_S)
    elapsed = time.monotonic() - started_at
    stopper.join(timeout=KILL_SPEC_BOUND_S)

    assert elapsed < KILL_SPEC_BOUND_S
    assert result.exit_code == -signal.SIGKILL


class _Stopper:
    """Adapts ``stop()`` onto the ``kill()`` the polling helper calls."""

    def __init__(self, target: ProcessBackend) -> None:
        self._target = target

    def kill(self) -> None:
        self._target.stop()

    def start(self, workspace_path: str) -> None: ...

    def exec(self, cmd: str, cwd: str, timeout: float | None) -> ExecResult:
        raise NotImplementedError

    def stop(self) -> None: ...


# ---------------------------------------------------------------------------
# AC 9 — kill() is idempotent and raises nothing (valid only beside AC 8)
# ---------------------------------------------------------------------------


def test_kill_raises_nothing_before_any_run(backend: LocalBackend) -> None:
    """AC9: nothing is in flight, so there is nothing to signal.

    Inert on its own — it would pass with a ``pass`` body. It is here for the
    exit paths AC 8 does not reach, and is worth nothing without it.
    """
    backend.kill()
    assert backend._running is None


def test_kill_raises_nothing_after_a_completed_run(backend: LocalBackend) -> None:
    """AC9: the handle is gone by the time the run returns, so a late kill is a no-op."""
    result = backend.exec("sh -c 'exit 0'", "", 10.0)
    assert result.exit_code == 0

    backend.kill()
    backend.kill()

    assert backend._running is None


def test_kill_survives_a_child_that_exits_between_the_read_and_the_signal(
    backend: LocalBackend,
) -> None:
    """AC9: ``ProcessLookupError`` from ``Popen.kill`` is the outcome kill asked for.

    The race is real and not otherwise reachable from a test: the handle is read
    under the lock and signalled outside it, so the child may reap in between.
    """
    proc = MagicMock()
    proc.kill.side_effect = ProcessLookupError(3, "No such process")
    backend._running = proc

    backend.kill()  # must not raise

    proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# Decision 4b — a kill and a timeout end the process group, not only the shell
# ---------------------------------------------------------------------------

#: Far past every bound in this file, so a grandchild that survived is
#: unmistakable and never "finished on its own" inside the spec's window.
GRANDCHILD_LIFETIME_S = 300


def _pid_written_to(path: Path) -> int:
    """Wait for the shell to publish its background job's pid, and return it."""
    deadline = time.monotonic() + KILL_SPEC_BOUND_S
    while time.monotonic() < deadline:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            text = ""
        if text:
            return int(text)
        time.sleep(0.01)
    raise AssertionError("the shell never wrote its background job's pid")


def _is_alive(pid: int) -> bool:
    """Whether *pid* still exists — a zombie not yet reaped counts as alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_dead(pid: int, bound: float = 2.0) -> bool:
    """Whether *pid* is gone inside *bound* seconds.

    An orphaned grandchild is reparented and reaped by init in milliseconds; the
    bound is for that reaping, never for a sleep to run out.
    """
    deadline = time.monotonic() + bound
    while time.monotonic() < deadline:
        if not _is_alive(pid):
            return True
        time.sleep(0.01)
    return not _is_alive(pid)


def _reap(pid: int) -> None:
    """Never leave a five-minute ``sleep`` behind on a red run."""
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _kill_once_the_grandchild_exists(target: ProcessBackend, pid_file: Path) -> None:
    """Wait for the job's pid to be published and the handle to be held, then kill *target*.

    The shell writes the pid from inside the child; ``_run`` records the handle
    on the calling thread after ``Popen`` returns. Nothing orders the two, so
    both are waited for — a kill issued between them finds no handle, sends
    nothing, and the spec would then run out its whole budget before going red.
    """
    _pid_written_to(pid_file)
    deadline = time.monotonic() + KILL_SPEC_BOUND_S
    while target._running is None and time.monotonic() < deadline:
        time.sleep(0.01)
    target.kill()


def test_kill_ends_the_grandchild_the_shell_forked(backend: LocalBackend, tmp_path: Path) -> None:
    """Decision 4b: what the shell forked dies with the shell, observed on the grandchild.

    The command makes ``sh`` genuinely fork — ``&`` is a fork on every shell,
    exec-in-place or not — and keeps ``sh`` alive at ``wait`` so the kill has a
    target. The job's output goes to ``/dev/null`` on purpose: it must **not**
    hold the pipes, so that a kill reaching only the shell would still let
    ``exec()`` return promptly and this spec would go red on the grandchild
    being alive, not on a timeout it could be confused with.

    Name what would have to be true for this to pass with the group kill
    deleted: ``sleep`` would have to die of something else inside two seconds,
    when it was told to live for five minutes. Not reachable.
    """
    pid_file = tmp_path / "grandchild.pid"
    assert " " not in str(pid_file) and "'" not in str(pid_file)
    cmd = f"sh -c 'sleep {GRANDCHILD_LIFETIME_S} >/dev/null 2>&1 & echo $! > {pid_file}; wait'"
    killer = threading.Thread(
        target=_kill_once_the_grandchild_exists, args=(backend, pid_file), daemon=True
    )
    killer.start()

    started_at = time.monotonic()
    result = backend.exec(cmd, "", KILL_SPEC_BUDGET_S)
    elapsed = time.monotonic() - started_at
    killer.join(timeout=KILL_SPEC_BOUND_S)
    grandchild = _pid_written_to(pid_file)
    try:
        assert elapsed < KILL_SPEC_BOUND_S
        assert result.exit_code == -signal.SIGKILL
        assert _wait_dead(grandchild), f"sleep {grandchild} outlived the kill of its shell"
    finally:
        _reap(grandchild)


def test_a_timeout_ends_the_grandchild_and_the_drain_returns_promptly(
    backend: LocalBackend, tmp_path: Path
) -> None:
    """Decision 4b's other half: the expiry path signals the group, so the drain is bounded.

    Here the job **does** inherit the pipes, which is the shape that held
    ``_run``'s second ``communicate()`` open: a timeout that killed only the
    shell would drain until the eight-second sleep exited on its own. Both
    facts are asserted — the grandchild is dead, and the call returned in far
    less than the sleep's lifetime — because either alone could be satisfied by
    the wrong mechanism.
    """
    pid_file = tmp_path / "grandchild.pid"
    assert " " not in str(pid_file) and "'" not in str(pid_file)
    cmd = f"sh -c 'sleep 8 & echo $! > {pid_file}; wait'"

    started_at = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        backend.exec(cmd, "", 0.5)
    elapsed = time.monotonic() - started_at
    grandchild = _pid_written_to(pid_file)
    try:
        assert elapsed < 3.0, f"the drain took {elapsed:.1f}s — the grandchild held the pipes"
        assert _wait_dead(grandchild), f"sleep {grandchild} outlived the timeout of its shell"
    finally:
        _reap(grandchild)


def test_signal_ends_the_group_where_the_backend_declared_one() -> None:
    """``_signal(proc, True)`` is one ``killpg`` on the child's pid — its group id."""
    proc = MagicMock()
    proc.pid = 4242

    with patch(KILLPG) as killpg:
        ProcessBackend._signal(proc, True)

    killpg.assert_called_once_with(4242, signal.SIGKILL)
    proc.kill.assert_not_called()


def test_signal_falls_back_to_the_child_when_the_group_is_already_gone() -> None:
    """A vanished group still gets the direct-child kill, and nothing raises."""
    proc = MagicMock()
    proc.pid = 4242

    with patch(KILLPG, side_effect=ProcessLookupError(3, "No such process")):
        ProcessBackend._signal(proc, True)  # must not raise

    proc.kill.assert_called_once()


def test_signal_ends_only_the_child_where_no_group_was_declared() -> None:
    """``_signal(proc, False)`` never reaches ``killpg`` — that would signal the caller's group."""
    proc = MagicMock()
    proc.pid = 4242

    with patch(KILLPG) as killpg:
        ProcessBackend._signal(proc, False)

    killpg.assert_not_called()
    proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# AC 10 — stop() kills before it releases
# ---------------------------------------------------------------------------


def test_docker_stop_kills_the_run_then_removes_the_container() -> None:
    """AC10: ``kill`` precedes the release, and the release is one ``docker rm -f``.

    Ordering matters where the release does something: removing a container
    around a command still writing to the tree is the failure this rules out.
    The container is ephemeral compute, so the release removes rather than stops
    — there is no filesystem here worth preserving between runs, and the only
    writes that outlive it are on the bind mount.
    """
    docker = DockerBackend("t1")
    docker.container_name = "akgentic-sandbox-0123456789ab"
    proc = MagicMock()
    order: list[str] = []
    proc.kill.side_effect = lambda: order.append("kill")
    docker._running = proc

    def record(argv: list[str], **_kwargs: object) -> MagicMock:
        order.append(" ".join(argv[:3]))
        return MagicMock(stdout="", stderr="", returncode=0)

    with patch("akgentic.tool.sandbox.docker.subprocess.run") as mock_run:
        mock_run.side_effect = record
        docker.stop()

    assert order == ["kill", "docker rm -f"]
    assert mock_run.call_args_list[0][0][0] == [
        "docker",
        "rm",
        "-f",
        "akgentic-sandbox-0123456789ab",
    ]
    assert mock_run.call_count == 1
    for call_item in mock_run.call_args_list:
        assert "stop" not in call_item[0][0]


def test_stop_releases_after_killing_on_a_backend_with_nothing_to_release() -> None:
    """AC10: the base ``stop`` is ``kill`` then ``_release``, in that order."""
    order: list[str] = []

    class Recording(ProcessBackend):
        def kill(self) -> None:
            order.append("kill")
            super().kill()

        def _release(self) -> None:
            order.append("release")

    Recording().stop()

    assert order == ["kill", "release"]


# ---------------------------------------------------------------------------
# AC 11 — the handle is released on every exit
# ---------------------------------------------------------------------------


def test_the_handle_is_released_after_a_successful_run(backend: LocalBackend) -> None:
    """AC11: success is an exit like any other."""
    assert backend.exec("sh -c 'exit 0'", "", 10.0).exit_code == 0
    assert backend._running is None


def test_the_handle_is_released_after_a_non_zero_exit(backend: LocalBackend) -> None:
    """AC11: a non-zero exit is a result, and it still clears the handle."""
    assert backend.exec("sh -c 'exit 3'", "", 10.0).exit_code == 3
    assert backend._running is None


def test_the_handle_is_released_after_a_timeout(backend: LocalBackend) -> None:
    """AC11: the timeout path kills, drains, re-raises — and still clears the handle."""
    with pytest.raises(subprocess.TimeoutExpired):
        backend.exec("sh -c 'sleep 30'", "", 0.5)

    assert backend._running is None


def test_the_handle_is_released_after_a_raise(backend: LocalBackend) -> None:
    """AC11: an exception nobody anticipated is the exit the ``finally`` is for.

    A retained handle to an exited process is not untidiness: a later ``kill()``
    would then signal a pid the OS may have handed to somebody else.
    """
    with patch(POPEN) as mock_popen:
        mock_popen.return_value.communicate.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            backend.exec("echo hi", "", 10.0)

    assert backend._running is None


# ---------------------------------------------------------------------------
# AC 1 / AC 4 — the four classes, and what runtime_checkable does not check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend_class", [LocalBackend, BwrapBackend, SeatbeltBackend, DockerBackend]
)
def test_the_four_classes_expose_the_four_names(backend_class: type[Any]) -> None:
    """AC1: ``isinstance`` against the Protocol — which checks names, and only names.

    ``@runtime_checkable`` verifies that four attributes exist and no more: not a
    signature, not an argument count, not a return type. A backend whose
    ``exec()`` took the wrong arguments would pass this. mypy over ``src/`` is the
    real conformance check, and it does not see anything defined in ``tests/``.
    """
    instance = backend_class()

    assert isinstance(instance, SandboxBackend)
    for name in ("start", "exec", "kill", "stop"):
        assert callable(getattr(instance, name))


@pytest.mark.parametrize(
    "backend_class", [LocalBackend, BwrapBackend, SeatbeltBackend, DockerBackend]
)
def test_the_four_classes_are_plain_classes(backend_class: type[Any]) -> None:
    """AC4: not actors and not abstract — a strategy has no mailbox and no lifecycle."""
    assert not issubclass(backend_class, Akgent)
    assert not getattr(backend_class, "__abstractmethods__", frozenset())
    assert issubclass(backend_class, ProcessBackend)


# ---------------------------------------------------------------------------
# AC 3 — the filter behaves identically through a backend's exec()
# ---------------------------------------------------------------------------


def test_an_unbalanced_quote_is_refused_by_the_backend(backend: LocalBackend) -> None:
    """AC3: a quoting mistake, with the message unchanged and the remedy in it."""
    with pytest.raises(CommandParseError) as caught:
        backend.exec('echo "unbalanced', "", 10.0)

    message = str(caught.value)
    assert "No closing quotation" in message
    assert "bash -c" in message


def test_an_empty_command_is_refused_by_the_backend(backend: LocalBackend) -> None:
    """AC3: no binary to validate is not a parse failure."""
    with pytest.raises(CommandNotAllowedError, match="empty") as caught:
        backend.exec("", "", 10.0)

    assert not isinstance(caught.value, CommandParseError)


def test_a_binary_off_the_allowlist_is_refused_by_the_backend(backend: LocalBackend) -> None:
    """AC3: the filter is the thing the move must not lose.

    ``ssh`` is the exemplar because it has to stay off the list for this to mean
    anything; a binary the sandbox might one day want would be added and quietly
    turn this into a test of nothing.
    """
    with pytest.raises(CommandNotAllowedError, match="ssh"):
        backend.exec("ssh nowhere", "", 10.0)


def test_a_permitted_binary_reaches_the_process(backend: LocalBackend) -> None:
    """AC3: the other half — the filter lets a permitted command through, for real."""
    result = backend.exec("echo hi", "", 10.0)

    assert result.exit_code == 0
    assert "hi" in result.stdout


# ---------------------------------------------------------------------------
# AC 12 — the one registry
# ---------------------------------------------------------------------------


def test_the_backend_registry_maps_the_four_modes_to_the_four_strategies() -> None:
    """AC12: the shipped mapping, pinned entry by entry."""
    assert SANDBOX_BACKEND_CLASSES == {
        "local": LocalBackend,
        "bwrap": BwrapBackend,
        "seatbelt": SeatbeltBackend,
        "docker": DockerBackend,
    }


# ---------------------------------------------------------------------------
# AC 13 — resolve_mode returns a live strategy
# ---------------------------------------------------------------------------


def test_resolve_mode_returns_a_live_backend_instance() -> None:
    """AC13: an instance, not a class — inert until something calls ``start()``."""
    mode, resolved = resolve_mode("local")

    assert mode == "local"
    assert isinstance(resolved, LocalBackend)
    assert resolved.workspace_path is None


def test_resolve_mode_hands_out_a_fresh_instance_each_time() -> None:
    """AC13: a shared instance would give two cards one process handle."""
    _, first = resolve_mode("local")
    _, second = resolve_mode("local")

    assert first is not second


def test_auto_still_warns_when_it_degrades_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC13: the warning is the whole reason every wiring goes through here."""
    monkeypatch.setattr("akgentic.tool.sandbox._resolve_auto_mode", lambda: "local")

    with pytest.warns(DeprecationWarning, match="no isolation backend found"):
        mode, resolved = resolve_mode("auto")

    assert mode == "local"
    assert isinstance(resolved, LocalBackend)


def test_auto_resolving_to_an_isolating_backend_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC13: the warning fires on the degradation, not on ``auto`` itself."""
    monkeypatch.setattr("akgentic.tool.sandbox._resolve_auto_mode", lambda: "docker")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mode, resolved = resolve_mode("auto")

    assert mode == "docker"
    assert isinstance(resolved, DockerBackend)


def test_an_unregistered_mode_raises_at_wiring_time() -> None:
    """AC13: a typo in a card is a configuration error, and those belong at start-up."""
    with pytest.raises(KeyError):
        resolve_mode("e2b")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC 17 — an injected backend is what the resolution path hands out
# ---------------------------------------------------------------------------


class _StubBackend:
    """A backend that runs nothing and says so in its result."""

    def __init__(self, team_id: str = "") -> None:
        self.team_id = team_id

    def start(self, workspace_path: str) -> None: ...

    def exec(self, cmd: str, cwd: str, timeout: float | None) -> ExecResult:
        return ExecResult(stdout="from the stub", stderr="", exit_code=7)

    def kill(self) -> None: ...

    def stop(self) -> None: ...


def test_an_injected_backend_is_what_resolve_mode_hands_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC17: the registry is the injection point, and the swap is total.

    ``echo hi`` really would print ``hi`` and exit 0. Getting the stub's answer
    from the instance ``resolve_mode`` built is the proof that resolution reads
    the registry at call time and that nothing between the registry and the
    caller keeps an execution path of its own. The end-to-end form — the same
    injection reaching ``#Workspace``'s worker — is ``InjectedBackend`` in
    ``tests/workspace/test_exec.py``.
    """
    monkeypatch.setitem(SANDBOX_BACKEND_CLASSES, "local", _StubBackend)

    mode, backend = resolve_mode("local")
    result = backend.exec("echo hi", "", None)

    assert mode == "local"
    assert isinstance(backend, _StubBackend)
    assert result.stdout == "from the stub"
    assert result.exit_code == 7


# ---------------------------------------------------------------------------
# 50-2 — one uniform constructor, and docker's two asserts become guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend_class", [LocalBackend, BwrapBackend, SeatbeltBackend, DockerBackend]
)
def test_every_backend_accepts_a_team_id(backend_class: type[Any]) -> None:
    """All four take it, and it is stored, so the caller needs no type switch.

    The registry holds classes and ``resolve_mode`` constructs from it, so a
    backend that refused the keyword would make that one call site branch on the
    mode it had just resolved — which is the thing having a registry is for.
    """
    backend = backend_class(team_id="team-42")

    assert backend.team_id == "team-42"


@pytest.mark.parametrize(
    "backend_class", [LocalBackend, BwrapBackend, SeatbeltBackend, DockerBackend]
)
def test_every_backend_is_still_constructible_with_no_arguments(
    backend_class: type[Any],
) -> None:
    """The default is what keeps ``resolve_mode`` able to build one from the registry."""
    assert backend_class().team_id == ""


def test_resolve_mode_carries_the_team_to_the_backend() -> None:
    """The team reaches the backend by this path, and by no other.

    **It no longer names the container.** The name became opaque and per
    lifetime, because ``stop()`` removes the container and a name derived from
    the team collides with its own predecessor on the next ``start()``. What is
    asserted here is the wiring — ``resolve_mode``'s keyword reaches the
    constructor — not a consequence the name no longer has.
    """
    _mode, backend = resolve_mode("docker", team_id="team-42")

    assert isinstance(backend, DockerBackend)
    assert backend.team_id == "team-42"


def test_resolve_mode_without_a_team_leaves_the_backend_teamless() -> None:
    """The card's wiring call names no team and must not invent one."""
    _mode, backend = resolve_mode("docker")

    assert isinstance(backend, DockerBackend)
    assert backend.team_id == ""


def test_an_unstarted_docker_release_returns_instead_of_raising() -> None:
    """``stop()`` on a backend that never ran must not raise into a teardown.

    It used to ``assert self.container_name is not None``. That is an
    ``AssertionError`` raised inside ``on_stop`` — where an exception is worse
    than any error it could report — and under ``python -O`` it is worse still:
    the assert is stripped and the docker command runs carrying a literal
    ``None`` as the container name.
    """
    backend = DockerBackend(team_id="team-42")
    assert backend.container_name is None

    with patch("akgentic.tool.sandbox.docker.subprocess.run") as run:
        backend.stop()  # must not raise

    assert run.call_count == 0  # and it reached no docker CLI


def test_an_unstarted_docker_exec_raises_a_readable_error() -> None:
    """The other assert. A refusal a reader can act on, and one ``-O`` cannot strip."""
    backend = DockerBackend(team_id="team-42")

    with pytest.raises(RuntimeError, match="before start"):
        backend.exec("echo hi", "", None)
