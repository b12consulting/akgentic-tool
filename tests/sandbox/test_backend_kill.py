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
"""

from __future__ import annotations

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

from akgentic.tool.sandbox import SANDBOX_ACTOR_CLASSES, SANDBOX_BACKEND_CLASSES
from akgentic.tool.sandbox.actor import SandboxConfig, SandboxState
from akgentic.tool.sandbox.backend import (
    CommandNotAllowedError,
    CommandParseError,
    ExecResult,
    ProcessBackend,
    SandboxBackend,
)
from akgentic.tool.sandbox.bwrap import BwrapBackend
from akgentic.tool.sandbox.docker import DockerBackend
from akgentic.tool.sandbox.local import LocalBackend, LocalSandboxActor
from akgentic.tool.sandbox.seatbelt import SeatbeltBackend
from akgentic.tool.workspace.execution import resolve_mode

POPEN = "akgentic.tool.sandbox.backend.subprocess.Popen"

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

    Killing ``sh`` may orphan the ``sleep`` it started; ``Popen.kill`` signals the
    direct child only. That is noted rather than chased — it is pre-existing and
    changes nothing this spec asserts.
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
# AC 10 — stop() kills before it releases
# ---------------------------------------------------------------------------


def test_docker_stop_kills_the_run_then_stops_the_container() -> None:
    """AC10: ``kill`` precedes the release, and the release is ``docker stop`` alone.

    Ordering matters where the release does something: stopping a container
    around a command still writing to the tree is the failure this rules out.
    """
    docker = DockerBackend("t1")
    docker.container_name = "sandbox-t1"
    proc = MagicMock()
    order: list[str] = []
    proc.kill.side_effect = lambda: order.append("kill")
    docker._running = proc

    with patch("akgentic.tool.sandbox.docker.subprocess.run") as mock_run:
        mock_run.side_effect = lambda argv, **kwargs: order.append(" ".join(argv[:2]))
        docker.stop()

    assert order == ["kill", "docker stop"]
    assert mock_run.call_args_list[0][0][0] == ["docker", "stop", "sandbox-t1"]
    # As today: the container filesystem is preserved between restarts.
    assert mock_run.call_count == 1
    for call_item in mock_run.call_args_list:
        assert "rm" not in call_item[0][0]


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
# AC 12 — two registries, kept in step
# ---------------------------------------------------------------------------


def test_the_backend_registry_maps_the_four_modes_to_the_four_strategies() -> None:
    """AC12: the shipped mapping, pinned entry by entry."""
    assert SANDBOX_BACKEND_CLASSES == {
        "local": LocalBackend,
        "bwrap": BwrapBackend,
        "seatbelt": SeatbeltBackend,
        "docker": DockerBackend,
    }


def test_the_two_registries_carry_the_same_keys() -> None:
    """AC12: two dicts that must agree is a drift hazard for as long as both exist.

    One line, and it dies with ``SANDBOX_ACTOR_CLASSES``.
    """
    assert set(SANDBOX_BACKEND_CLASSES) == set(SANDBOX_ACTOR_CLASSES)


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
# AC 17 — the actor holds no second execution path
# ---------------------------------------------------------------------------


class _StubBackend:
    """A backend that runs nothing and says so in its result."""

    def start(self, workspace_path: str) -> None: ...

    def exec(self, cmd: str, cwd: str, timeout: float | None) -> ExecResult:
        return ExecResult(stdout="from the stub", stderr="", exit_code=7)

    def kill(self) -> None: ...

    def stop(self) -> None: ...


def test_replacing_the_backend_changes_what_the_actor_returns(tmp_path: Path) -> None:
    """AC17: no subclass calls ``subprocess`` any more, so swapping the strategy is total.

    ``echo hi`` really would print ``hi`` and exit 0. Getting the stub's answer
    instead is the proof that the actor kept no path of its own — a reintroduced
    ``subprocess`` call in ``_exec`` returns the real output and fails here.
    """
    actor = LocalSandboxActor()
    actor.config = SandboxConfig(
        name="#SandboxActor", role="ToolActor", team_id="t1", workspace_path=WORKSPACE
    )
    actor.state = SandboxState()
    actor.state.observer(actor)
    actor.state.workspace_path = tmp_path
    actor._backend = _StubBackend()  # type: ignore[assignment]

    result = actor._exec("echo hi", "", None)

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


def test_resolve_mode_carries_the_team_to_the_backend_that_needs_it() -> None:
    """The team names the docker container, and this is the only path it travels.

    Asserted through the container name rather than only the attribute: the name
    is the consequence, and a ``team_id`` stored but never used in it would be a
    per-team resource shared across teams.
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
    the assert is stripped and ``docker stop None`` is what actually runs.
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
