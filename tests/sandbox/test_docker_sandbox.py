"""Tests for ``DockerBackend`` — an ephemeral container as compute for one tree.

**Every spec in this file is mocked, and what a mock proves is bounded.** A
whole-argv assertion proves *the code builds this command line*. It does not
prove docker honours it, that a read-only root refuses a write, that git stops
reporting *dubious ownership*, or that ``pip`` can write a cache. Those are facts
about docker and git, they need a daemon, and they live in
``test_docker_container_integration.py`` behind both an ``integration`` marker
and a daemon probe.

The assertions here are deliberately whole-vector rather than membership.
``docker run`` is positional in the way that matters: everything before the image
name is an option to ``docker run`` and everything after it is the container's
own command, so ``assert "--read-only" in argv`` passes for a flag that has
become an argument to ``sleep``.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core.orchestrator import STOP_TIMEOUT

from akgentic.tool.sandbox.backend import DEFAULT_BACKEND_TIMEOUT_S, ExecResult
from akgentic.tool.sandbox.docker import (
    CONTAINER_NAME_PREFIX,
    DOCKER_EXEC_TIMEOUT,
    DOCKER_RM_TIMEOUT_S,
    SANDBOX_IMAGE,
    SANDBOX_IMAGE_BUILD_TIMEOUT_S,
    WORKSPACE_PATH_LABEL,
    DockerBackend,
)
from akgentic.tool.workspace.execution import EXEC_SHUTDOWN_GRACE_S

# The exec path runs through ``ProcessBackend._run``, so exec-path mocks target
# ``backend.subprocess.Popen``. The start, build and stop paths still run
# ``docker.subprocess.run`` and keep that target.
POPEN = "akgentic.tool.sandbox.backend.subprocess.Popen"
KILLPG = "akgentic.tool.sandbox.backend.os.killpg"
DOCKER_LOGGER = "akgentic.tool.sandbox.docker"

TMPFS_OPTIONS = "rw,exec,mode=1777,size=512m"
"""Repeated as a literal rather than imported, so a change to the constant that
nobody meant fails a spec instead of travelling silently into both sides."""


def popen_mock(
    mock_popen: MagicMock, stdout: str = "", stderr: str = "", returncode: int = 0
) -> MagicMock:
    """Shape *mock_popen* like a ``Popen``: ``communicate()`` pair plus ``returncode``."""
    proc = mock_popen.return_value
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    proc.pid = 4242
    return proc


def started_backend(
    team_id: str = "team-1", container_name: str = "akgentic-sandbox-0123456789ab"
) -> DockerBackend:
    """A backend that believes its container is running, without a daemon.

    What ``start()`` would leave behind — the name — is set directly, so the
    ``exec()`` argv can be asserted with no ``docker run`` having happened.
    """
    backend = DockerBackend(team_id)
    backend.container_name = container_name
    return backend


def run_argv(mock_run: MagicMock) -> list[str]:
    """The argv of the single ``subprocess.run`` the start path issued."""
    assert mock_run.call_count == 1, (
        f"start() must issue exactly one docker command, got {mock_run.call_count}: "
        f"{[c[0][0][:3] for c in mock_run.call_args_list]}"
    )
    argv: list[str] = mock_run.call_args_list[0][0][0]
    return argv


def value_after(argv: list[str], flag: str) -> str:
    """The single value following *flag*, asserting the flag appears exactly once."""
    assert argv.count(flag) == 1, f"{flag} must appear exactly once, got {argv.count(flag)}"
    return argv[argv.index(flag) + 1]


def env_values(argv: list[str], prefix: str) -> list[str]:
    """Every ``-e <prefix>=<value>`` value in *argv*, in the order docker reads them."""
    return [
        token[len(prefix) + 1 :]
        for index, token in enumerate(argv)
        if index > 0 and argv[index - 1] == "-e" and token.startswith(f"{prefix}=")
    ]


def start_and_capture(
    mock_run: MagicMock,
    *,
    team_id: str = "team-1",
    workspace_path: str | None = None,
) -> tuple[DockerBackend, list[str]]:
    """Start a backend against a mocked ``docker run`` and hand back its argv."""
    mock_run.return_value = MagicMock(stdout="abc123", stderr="", returncode=0)
    backend = DockerBackend(team_id)
    backend.start(team_id if workspace_path is None else workspace_path)
    return backend, run_argv(mock_run)


def expected_argv(container_name: str, volume: str, workspace_path: str) -> list[str]:
    """The complete vector ``start()`` must build, written out in order.

    A literal rather than a rebuild of the production expression: a helper that
    derived this the way the code does would agree with any change to the code,
    which is the whole failure this spec exists to catch.
    """
    return [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--label",
        f"akgentic.workspace_path={workspace_path}",
        "--read-only",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/home/agent",
        "--tmpfs",
        f"/tmp:{TMPFS_OPTIONS}",
        "--tmpfs",
        f"/home/agent:{TMPFS_OPTIONS}",
        "-e",
        "GIT_CONFIG_COUNT=3",
        "-e",
        "GIT_CONFIG_KEY_0=safe.directory",
        "-e",
        "GIT_CONFIG_VALUE_0=*",
        "-e",
        "GIT_CONFIG_KEY_1=user.name",
        "-e",
        "GIT_CONFIG_VALUE_1=akgentic-sandbox",
        "-e",
        "GIT_CONFIG_KEY_2=user.email",
        "-e",
        "GIT_CONFIG_VALUE_2=sandbox@akgentic",
        "-v",
        volume,
        "-w",
        "/workspace",
        SANDBOX_IMAGE,
        "sleep",
        "infinity",
    ]


# ---------------------------------------------------------------------------
# The constants
# ---------------------------------------------------------------------------


def test_sandbox_image_names_a_tag_the_pre_story_image_cannot_satisfy() -> None:
    """AC14: the tag moved, so a host holding the old image builds the new one.

    ``_ensure_image`` skips the build whenever *any* image carries the tag, so a
    host with an ``akgentic-sandbox:latest`` built before ``uv`` moved off
    ``/root`` would have kept an image whose ``uv`` is unreachable under
    ``--user``. Asserted as a difference from the old tag rather than as a
    literal: what matters is that the check misses, not which tag it misses on.
    """
    assert SANDBOX_IMAGE != "akgentic-sandbox:latest"
    assert SANDBOX_IMAGE.startswith("akgentic-sandbox:")
    assert not SANDBOX_IMAGE.endswith(":latest")


def test_docker_exec_timeout_constant() -> None:
    """Docker's default budget no longer outlives the orchestrator's stop backstop.

    It was 60 s — twice the 30 s backstop — which made docker the one backend
    able to hold a team's teardown open past the point that teardown gives up.
    The assertion is on the relationship, not on a number: what matters is that
    docker stopped being the exception.
    """
    assert DOCKER_EXEC_TIMEOUT == DEFAULT_BACKEND_TIMEOUT_S
    assert DOCKER_EXEC_TIMEOUT <= STOP_TIMEOUT


def test_the_build_budget_sits_far_above_any_exec_budget() -> None:
    """The build blocks one run for minutes; the exec budget bounds seconds.

    A build budget at or below the exec budget would fail every cold start, so
    the relationship is the invariant rather than the figure.
    """
    assert SANDBOX_IMAGE_BUILD_TIMEOUT_S > DOCKER_EXEC_TIMEOUT * 10


def test_the_whole_of_exec_teardown_fits_under_the_orchestrator_backstop() -> None:
    """The teardown arithmetic, asserted rather than stated in a docstring.

    Teardown is the bounded drain followed by the backend's ``stop()``, and the
    slowest ``stop()`` is docker's bounded ``rm -f``. Their sum is what the
    orchestrator's stop backstop has to cover; a docstring that says "~13 s" goes
    stale the day either constant moves, and this does not.
    """
    assert EXEC_SHUTDOWN_GRACE_S + DOCKER_RM_TIMEOUT_S < STOP_TIMEOUT


# ---------------------------------------------------------------------------
# AC 1, 2 — one docker run, and the whole argv in order
# ---------------------------------------------------------------------------


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_start_issues_exactly_one_docker_command_and_it_is_a_run(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """AC1: no ``docker ps``, no ``docker start``, no reuse branch.

    The reuse branch is gone because its precondition cannot hold: ``stop()``
    removes the container, so there is never one to find. ``call_count == 1`` is
    what makes the deletion observable — a reintroduced probe is a second call.
    """
    _backend, argv = start_and_capture(mock_run)

    assert argv[:2] == ["docker", "run"]
    assert "ps" not in argv
    assert "start" not in argv


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_start_builds_the_whole_argv_in_order(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2: the complete vector, asserted in order, against a literal.

    Only the generated name is read back from the backend. A flag that moves
    after the image name becomes an argument to ``sleep``, and this is the
    assertion that can tell the two apart.
    """
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    monkeypatch.delenv("AKGENTIC_SANDBOX_IMAGE", raising=False)
    backend, argv = start_and_capture(mock_run)

    volume = f"{Path('./workspaces/team-1').resolve()}:/workspace"
    assert backend.container_name is not None
    assert argv == expected_argv(backend.container_name, volume, "team-1")


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_every_option_precedes_the_image_and_only_sleep_follows_it(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2: position, stated as the rule rather than as one vector.

    The whole-argv spec above catches any reordering, but it catches it as "the
    list differs". This says what the difference would mean: docker reads
    everything after the image as the container's own command.
    """
    monkeypatch.delenv("AKGENTIC_SANDBOX_IMAGE", raising=False)
    _backend, argv = start_and_capture(mock_run)

    image_index = argv.index(SANDBOX_IMAGE)
    assert argv[image_index + 1 :] == ["sleep", "infinity"]
    for flag in ("-d", "--name", "--label", "--read-only", "--user", "--tmpfs", "-v", "-w"):
        assert argv.index(flag) < image_index, f"{flag} lands after the image name"
    assert argv.count("-e") == 8
    assert all(index < image_index for index, t in enumerate(argv) if t == "-e")


# ---------------------------------------------------------------------------
# AC 3 — the name is opaque and per lifetime
# ---------------------------------------------------------------------------


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_two_starts_on_one_backend_produce_two_different_names(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """AC3: the case that matters, which two instances would not exercise.

    Two instances differ for free if the name comes from anything
    instance-scoped. The reachable collision is ``stop()`` then ``start()`` on
    one backend, which is what ``docker run --name`` refuses outright.
    """
    mock_run.return_value = MagicMock(stdout="abc123", stderr="", returncode=0)
    backend = DockerBackend("team-1")

    backend.start("team-1")
    first = backend.container_name
    backend.start("team-1")
    second = backend.container_name

    assert first != second
    names = [value_after(call[0][0], "--name") for call in mock_run.call_args_list]
    assert names == [first, second]


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_the_name_is_a_function_of_nothing_the_caller_supplied(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """AC3: neither ``team_id`` nor ``workspace_path`` appears in the name.

    A metadata leaf can carry a customer id, and a name that carried it would put
    it in every ``docker ps`` line on the host. The label is where the path goes.
    """
    _backend, argv = start_and_capture(
        mock_run, team_id="acme-team", workspace_path="u-alice/case-42"
    )

    name = value_after(argv, "--name")
    assert name.startswith(CONTAINER_NAME_PREFIX)
    assert "acme-team" not in name
    assert "u-alice" not in name
    assert "case-42" not in name
    assert len(name) == len(CONTAINER_NAME_PREFIX) + 12


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_two_backends_over_one_workspace_do_not_collide(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """AC3: the collision that ``sandbox-{team_id}`` produced for two workspaces."""
    mock_run.return_value = MagicMock(stdout="abc123", stderr="", returncode=0)
    first = DockerBackend("team-1")
    second = DockerBackend("team-1")

    first.start("u-alice/notes")
    second.start("u-alice/notes")

    assert first.container_name != second.container_name


# ---------------------------------------------------------------------------
# AC 4 — the label
# ---------------------------------------------------------------------------


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
@pytest.mark.parametrize(
    "workspace_path",
    ["team-1", "u-alice/notes", "u-alice/customer_id-ACME__case_id-42"],
)
def test_the_label_carries_the_workspace_path_verbatim(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
    workspace_path: str,
) -> None:
    """AC4: the path the reaper keys on, unchanged — slash and metadata leaf alike.

    The name carries nothing, so this label is the only thing a host-side reaper
    can filter on. A path that arrived encoded must leave encoded.
    """
    _backend, argv = start_and_capture(mock_run, workspace_path=workspace_path)

    assert value_after(argv, "--label") == f"{WORKSPACE_PATH_LABEL}={workspace_path}"
    assert WORKSPACE_PATH_LABEL == "akgentic.workspace_path"


# ---------------------------------------------------------------------------
# AC 5 — the bind mount, unchanged and the only one
# ---------------------------------------------------------------------------


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_the_bind_mount_is_the_only_one_and_is_the_path_it_was_handed(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC5: exactly one ``-v``, and it is the resolved root joined to the path.

    The tmpfs mounts are writable and deliberately are *not* ``-v``: only this
    one has writes that outlive the container.
    """
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    _backend, argv = start_and_capture(mock_run, team_id="t1", workspace_path="u-alice/notes")

    assert argv.count("-v") == 1
    assert value_after(argv, "-v") == f"{Path('./workspaces/u-alice/notes').resolve()}:/workspace"


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_the_bind_mount_uses_a_custom_workspaces_root(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Story 6.5, unchanged: ``-v`` uses AKGENTIC_WORKSPACES_ROOT when set."""
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", "/workspaces")
    _backend, argv = start_and_capture(mock_run)

    assert value_after(argv, "-v") == "/workspaces/team-1:/workspace"


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_the_bind_mount_normalizes_a_trailing_slash_in_the_root(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Story 6.5, unchanged: '/workspaces/' must not produce a double slash."""
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", "/workspaces/")
    _backend, argv = start_and_capture(mock_run)

    volume = value_after(argv, "-v")
    assert volume == "/workspaces/team-1:/workspace", (
        f"Volume mount must not contain double slash: got '{volume}'"
    )


# ---------------------------------------------------------------------------
# AC 6, 7, 8 — the three things a read-only root needs, as one unit
# ---------------------------------------------------------------------------


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_user_carries_the_host_uid_and_gid(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """AC6: matched to the host, which is what makes the bind mount writable.

    The mount is owned by the host uid. Any other uid gets *dubious ownership*
    from git on every command and cannot write the tree at all.
    """
    _backend, argv = start_and_capture(mock_run)

    assert value_after(argv, "--user") == f"{os.getuid()}:{os.getgid()}"
    assert "--read-only" in argv


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_home_and_its_tmpfs_name_the_same_directory(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """AC7: asserted as an agreement, never as two presences.

    A spec that asserted ``-e HOME=/home/agent`` and separately asserted
    ``--tmpfs /home/agent:…`` passes when someone changes one literal and not the
    other — which is exactly the failure it exists to catch. Docker gives a
    numeric ``--user`` with no passwd entry ``HOME=/``, which is on the read-only
    root, so a ``HOME`` no tmpfs mounts is a wall.
    """
    _backend, argv = start_and_capture(mock_run)

    home = env_values(argv, "HOME")
    assert len(home) == 1, f"exactly one HOME entry, got {home}"
    mounted = [entry.split(":", 1)[0] for entry in _tmpfs_entries(argv)]
    assert home[0] in mounted, f"HOME={home[0]} is on no tmpfs; mounted: {mounted}"


def _tmpfs_entries(argv: list[str]) -> list[str]:
    """Every value following a ``--tmpfs`` flag."""
    return [token for index, token in enumerate(argv) if index > 0 and argv[index - 1] == "--tmpfs"]


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_both_tmpfs_mounts_carry_rw_exec_a_world_writable_mode_and_a_size(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """AC8: a bare ``--tmpfs /tmp`` is a second wall, not a smaller fix.

    Observed on the daemon rather than assumed: a bare ``--tmpfs`` mounts
    ``rw,nosuid,nodev,noexec`` with no ``size=`` at the kernel's default mode
    ``1777`` — so it *is* writable by a non-root uid, and the wall is ``noexec``
    (``pip``'s wheel builds and build backends run out of ``TMPDIR``) plus a
    size that defaults to half the daemon's RAM. ``mode=1777`` is pinned here
    because it restates the default explicitly, not because the default differs.
    """
    _backend, argv = start_and_capture(mock_run)

    entries = _tmpfs_entries(argv)
    assert len(entries) == 2, f"two tmpfs mounts expected, got {entries}"
    for entry in entries:
        path, _, options = entry.partition(":")
        assert path.startswith("/"), entry
        parsed = options.split(",")
        assert "rw" in parsed, entry
        assert "exec" in parsed, entry
        assert "noexec" not in parsed, entry
        mode = [opt for opt in parsed if opt.startswith("mode=")]
        assert mode == ["mode=1777"], f"a non-root uid needs a world-writable mode: {entry}"
        assert [opt for opt in parsed if opt.startswith("size=")], f"no explicit size: {entry}"


# ---------------------------------------------------------------------------
# AC 9 — the git environment, checked for self-consistency
# ---------------------------------------------------------------------------


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_the_git_environment_is_complete_and_self_consistent(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """AC9: the count agrees with the pairs, and the indices are contiguous from 0.

    A hard-coded expected block goes stale the moment a fifth setting is added.
    A consistency check catches the count that was not incremented — which is the
    real defect, because git reads only the first COUNT pairs and drops the rest
    with no error anywhere.
    """
    _backend, argv = start_and_capture(mock_run)

    count_values = env_values(argv, "GIT_CONFIG_COUNT")
    assert len(count_values) == 1
    count = int(count_values[0])

    pairs: dict[str, str] = {}
    for index in range(count):
        keys = env_values(argv, f"GIT_CONFIG_KEY_{index}")
        values = env_values(argv, f"GIT_CONFIG_VALUE_{index}")
        assert len(keys) == 1, f"GIT_CONFIG_KEY_{index} missing or duplicated: {keys}"
        assert len(values) == 1, f"GIT_CONFIG_VALUE_{index} missing or duplicated: {values}"
        pairs[keys[0]] = values[0]

    # Nothing may sit past the count — git would silently never read it.
    assert env_values(argv, f"GIT_CONFIG_KEY_{count}") == []
    assert len(pairs) == count, f"a key was written twice: {pairs}"
    assert pairs["safe.directory"] == "*"
    assert pairs["user.name"]
    assert "@" in pairs["user.email"]


# ---------------------------------------------------------------------------
# AC 10 — the image comes from _resolved_image()
# ---------------------------------------------------------------------------


def test_resolved_image_defaults_to_sandbox_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """_resolved_image() returns SANDBOX_IMAGE when AKGENTIC_SANDBOX_IMAGE is unset."""
    monkeypatch.delenv("AKGENTIC_SANDBOX_IMAGE", raising=False)
    assert DockerBackend("team-test")._resolved_image() == SANDBOX_IMAGE


def test_resolved_image_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """_resolved_image() returns AKGENTIC_SANDBOX_IMAGE when set."""
    monkeypatch.setenv("AKGENTIC_SANDBOX_IMAGE", "ghcr.io/myorg/akgentic-sandbox:v1.2")
    assert (
        DockerBackend("team-test")._resolved_image() == "ghcr.io/myorg/akgentic-sandbox:v1.2"
    )


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_the_override_image_sits_exactly_once_immediately_before_sleep(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC10: the override still reaches the argv, in the one position that works."""
    monkeypatch.setenv("AKGENTIC_SANDBOX_IMAGE", "ghcr.io/myorg/akgentic-sandbox:v1.2")
    _backend, argv = start_and_capture(mock_run)

    assert argv.count("ghcr.io/myorg/akgentic-sandbox:v1.2") == 1
    assert SANDBOX_IMAGE not in argv
    assert argv[-3:] == ["ghcr.io/myorg/akgentic-sandbox:v1.2", "sleep", "infinity"]


# ---------------------------------------------------------------------------
# AC 11 — docker missing from PATH
# ---------------------------------------------------------------------------


@patch("akgentic.tool.sandbox.docker.subprocess.run")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value=None)
def test_start_raises_before_anything_runs_when_docker_is_not_on_path(
    mock_which: MagicMock, mock_run: MagicMock
) -> None:
    """AC11: the guard precedes the image check, so no docker command is issued."""
    backend = DockerBackend("team-1")

    with pytest.raises(RuntimeError, match="docker CLI not found on PATH"):
        backend.start("team-1")

    assert mock_run.call_count == 0
    assert backend.container_name is None


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_a_failed_docker_run_leaves_the_backend_unstarted(
    mock_run: MagicMock,
    mock_which: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """The name is assigned only after the run succeeds, so ``stop()``'s guard stays live.

    A backend that recorded a name for a container that was never created would
    send a ``docker rm -f`` after a name nothing owns, and would let ``exec``
    address it.
    """
    mock_run.return_value = MagicMock(stdout="", stderr="no such image", returncode=125)
    backend = DockerBackend("team-1")

    with pytest.raises(RuntimeError, match="docker run failed"):
        backend.start("team-1")

    assert backend.container_name is None


# ---------------------------------------------------------------------------
# AC 12, 13 — the image is built, bounded, and never pulled
# ---------------------------------------------------------------------------


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_ensure_image_env_override_skips_all_docker(
    mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC12: AKGENTIC_SANDBOX_IMAGE set → no docker command at all."""
    monkeypatch.setenv("AKGENTIC_SANDBOX_IMAGE", "ghcr.io/myorg/akgentic-sandbox:v1.2")

    DockerBackend("team-test")._ensure_image()

    assert mock_run.call_count == 0


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_ensure_image_skips_the_build_when_the_image_is_present(
    mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC12: image present → only the images check, and no build."""
    monkeypatch.delenv("AKGENTIC_SANDBOX_IMAGE", raising=False)
    mock_run.return_value = MagicMock(stdout="cached-id\n", returncode=0)

    DockerBackend("team-test")._ensure_image()

    assert mock_run.call_count == 1
    assert mock_run.call_args_list[0][0][0] == ["docker", "images", "-q", SANDBOX_IMAGE]


@patch("akgentic.tool.sandbox.docker.importlib.resources.files")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_an_absent_image_builds_under_an_explicit_budget_with_output_captured(
    mock_run: MagicMock, mock_files: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC13: the build carries a timeout and captures its output.

    Unbounded was tolerable behind an actor that swallowed everything. On the
    worker thread it is the hang class this design closes everywhere else, and
    uncaptured output leaves a failure with nothing in its message.
    """
    monkeypatch.delenv("AKGENTIC_SANDBOX_IMAGE", raising=False)
    mock_files.return_value.joinpath.return_value.read_text.return_value = "FROM python:3.12-slim\n"
    mock_run.side_effect = [
        MagicMock(stdout="", returncode=0),  # docker images -q → absent
        MagicMock(stdout="", stderr="", returncode=0),  # docker build → success
    ]

    DockerBackend("team-test")._ensure_image()

    build_argv, build_kwargs = mock_run.call_args_list[1][0][0], mock_run.call_args_list[1][1]
    assert build_argv[:4] == ["docker", "build", "-t", SANDBOX_IMAGE]
    assert build_kwargs["timeout"] == SANDBOX_IMAGE_BUILD_TIMEOUT_S
    assert build_kwargs["capture_output"] is True


@patch("akgentic.tool.sandbox.docker.importlib.resources.files")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_the_image_is_never_pulled(
    mock_run: MagicMock, mock_files: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC13: no ``docker pull`` on any path.

    The image is published to no registry, so a pull reaches Docker Hub for a
    name this project chose and either 404s or fetches a stranger's image. That
    is a supply-chain hazard rather than a fallback.
    """
    monkeypatch.delenv("AKGENTIC_SANDBOX_IMAGE", raising=False)
    mock_files.return_value.joinpath.return_value.read_text.return_value = "FROM python:3.12-slim\n"
    mock_run.side_effect = [
        MagicMock(stdout="", returncode=0),
        MagicMock(stdout="", stderr="", returncode=0),
    ]

    DockerBackend("team-test")._ensure_image()

    for call_item in mock_run.call_args_list:
        assert "pull" not in call_item[0][0]


@patch("akgentic.tool.sandbox.docker.importlib.resources.files")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_a_failed_build_raises_naming_both_remedies_and_the_stderr_tail(
    mock_run: MagicMock, mock_files: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC13: the message an operator reads is the whole of what they get."""
    monkeypatch.delenv("AKGENTIC_SANDBOX_IMAGE", raising=False)
    mock_files.return_value.joinpath.return_value.read_text.return_value = "FROM python:3.12-slim\n"
    mock_run.side_effect = [
        MagicMock(stdout="", returncode=0),
        MagicMock(stdout="", stderr="E: Unable to locate package libreoffice", returncode=1),
    ]

    with pytest.raises(RuntimeError) as exc_info:
        DockerBackend("team-test")._ensure_image()

    message = str(exc_info.value)
    assert SANDBOX_IMAGE in message
    assert "pre-build" in message.lower()
    assert "AKGENTIC_SANDBOX_IMAGE" in message
    assert "Unable to locate package libreoffice" in message


@patch("akgentic.tool.sandbox.docker.importlib.resources.files")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_a_build_timeout_becomes_a_runtime_error_naming_the_budget(
    mock_run: MagicMock, mock_files: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC13: a raw ``TimeoutExpired`` would reach the run's report naming neither remedy."""
    monkeypatch.delenv("AKGENTIC_SANDBOX_IMAGE", raising=False)
    mock_files.return_value.joinpath.return_value.read_text.return_value = "FROM python:3.12-slim\n"
    mock_run.side_effect = [
        MagicMock(stdout="", returncode=0),
        subprocess.TimeoutExpired(
            cmd=["docker", "build"],
            timeout=SANDBOX_IMAGE_BUILD_TIMEOUT_S,
            stderr="Step 4/9 : RUN apt-get install",
        ),
    ]

    with pytest.raises(RuntimeError) as exc_info:
        DockerBackend("team-test")._ensure_image()

    message = str(exc_info.value)
    assert str(int(SANDBOX_IMAGE_BUILD_TIMEOUT_S)) in message
    assert "AKGENTIC_SANDBOX_IMAGE" in message
    assert "Step 4/9" in message


# ---------------------------------------------------------------------------
# AC 15–18 — stop() removes, and nothing raises
# ---------------------------------------------------------------------------


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_stop_removes_the_container(mock_run: MagicMock) -> None:
    """AC15: one ``docker rm -f``, and no ``docker stop`` or ``docker start`` anywhere.

    ``rm -f`` rather than stop-then-rm: one round trip, and it spends none of
    ``docker stop``'s ten-second SIGTERM grace on a container whose only purpose
    was to hold a process the caller has already given up on.
    """
    backend = DockerBackend("team-1")
    backend.container_name = "akgentic-sandbox-0123456789ab"
    mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

    backend.stop()

    assert mock_run.call_count == 1
    assert mock_run.call_args_list[0][0][0] == [
        "docker",
        "rm",
        "-f",
        "akgentic-sandbox-0123456789ab",
    ]
    # Bounded, because this runs on the actor's thread inside ``on_stop``: a
    # daemon that never answers must not hold teardown open to the backstop.
    assert mock_run.call_args_list[0][1]["timeout"] == DOCKER_RM_TIMEOUT_S
    for call_item in mock_run.call_args_list:
        assert "stop" not in call_item[0][0]
        assert "start" not in call_item[0][0]


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_stop_on_a_backend_that_never_started_issues_no_docker_command(
    mock_run: MagicMock,
) -> None:
    """AC16: teardown reaches this on a backend that never had a first command.

    A guard rather than an ``assert``: this runs inside ``on_stop``, and under
    ``python -O`` an assert vanishes and ``docker rm -f None`` is what would run.
    The behaviour is asserted here; the ``-O`` claim rests on the guard being an
    ``if``, which is not something ``-O`` can strip.
    """
    backend = DockerBackend("team-1")
    assert backend.container_name is None

    backend.stop()  # must not raise

    assert mock_run.call_count == 0


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_stop_when_the_container_is_already_gone_does_not_raise_and_does_not_warn(
    mock_run: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """AC17: already gone is the outcome that was asked for, so it is debug.

    ``caplog`` accumulates across a test's lifetime, so it is cleared
    immediately before the call and only the records that call produced are read.
    """
    backend = DockerBackend("team-1")
    backend.container_name = "akgentic-sandbox-0123456789ab"
    mock_run.return_value = MagicMock(
        stdout="",
        stderr="Error response from daemon: No such container: akgentic-sandbox-0123456789ab",
        returncode=1,
    )

    with caplog.at_level(logging.DEBUG, logger=DOCKER_LOGGER):
        caplog.clear()
        backend.stop()  # must not raise

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], f"already-gone must not warn, got {[r.message for r in warnings]}"
    assert [r for r in caplog.records if r.levelno == logging.DEBUG]


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_stop_warns_on_any_other_failure_and_still_does_not_raise(
    mock_run: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """AC17: a daemon that has gone away is a warning, and never a raise."""
    backend = DockerBackend("team-1")
    backend.container_name = "akgentic-sandbox-0123456789ab"
    mock_run.return_value = MagicMock(
        stdout="", stderr="Cannot connect to the Docker daemon", returncode=1
    )

    with caplog.at_level(logging.DEBUG, logger=DOCKER_LOGGER):
        caplog.clear()
        backend.stop()  # must not raise

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "akgentic-sandbox-0123456789ab" in warnings[0].getMessage()
    assert "Cannot connect to the Docker daemon" in warnings[0].getMessage()


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_stop_does_not_raise_when_docker_has_left_the_path_since_start(
    mock_run: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """AC17: ``subprocess.run`` raises ``FileNotFoundError`` there — an OSError."""
    backend = DockerBackend("team-1")
    backend.container_name = "akgentic-sandbox-0123456789ab"
    mock_run.side_effect = FileNotFoundError(2, "No such file or directory: 'docker'")

    with caplog.at_level(logging.DEBUG, logger=DOCKER_LOGGER):
        caplog.clear()
        backend.stop()  # must not raise

    assert [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert backend.container_name is None


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_stop_warns_when_the_removal_outlives_its_budget_and_does_not_raise(
    mock_run: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """AC17's fifth case: a daemon that never answers is a warning, not a hang.

    ``subprocess.run`` raises ``TimeoutExpired`` at the bound, which is not an
    ``OSError`` — a handler that caught only the latter would let it propagate
    into ``on_stop``. The name is cleared regardless, so nothing retries a
    daemon that has already shown it will not answer.
    """
    backend = DockerBackend("team-1")
    backend.container_name = "akgentic-sandbox-0123456789ab"
    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd=["docker", "rm", "-f"], timeout=DOCKER_RM_TIMEOUT_S
    )

    with caplog.at_level(logging.DEBUG, logger=DOCKER_LOGGER):
        caplog.clear()
        backend.stop()  # must not raise

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "akgentic-sandbox-0123456789ab" in warnings[0].getMessage()
    assert backend.container_name is None


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_stop_is_idempotent(mock_run: MagicMock) -> None:
    """AC18: a second stop issues no docker command, because the name was cleared.

    A retained name is a second ``docker rm -f`` after a container that is
    already gone, and — worse — an ``exec`` that can still address it.
    """
    backend = DockerBackend("team-1")
    backend.container_name = "akgentic-sandbox-0123456789ab"
    mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

    backend.stop()
    assert mock_run.call_count == 1
    assert backend.container_name is None

    backend.stop()
    assert mock_run.call_count == 1


@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_an_exec_after_stop_refuses_rather_than_addressing_a_removed_container(
    mock_run: MagicMock,
) -> None:
    """The consequence of clearing the name, stated as behaviour."""
    backend = DockerBackend("team-1")
    backend.container_name = "akgentic-sandbox-0123456789ab"
    mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

    backend.stop()

    with pytest.raises(RuntimeError, match="before start"):
        backend.exec("echo hi", "", None)


# ---------------------------------------------------------------------------
# The name lives on the backend and nowhere else
# ---------------------------------------------------------------------------


@patch.object(DockerBackend, "_ensure_image")
@patch("akgentic.tool.sandbox.docker.shutil.which", return_value="/usr/bin/docker")
@patch("akgentic.tool.sandbox.docker.subprocess.run")
def test_the_backend_knows_its_own_name_after_start(
    mock_run: MagicMock, mock_which: MagicMock, mock_ensure: MagicMock
) -> None:
    """The only record of the container's name is the instance that created it.

    There is no checkpointed state for the name to reach any more — the sandbox
    actor and its ``SandboxState`` are gone — so the positive half is what is
    left to assert: the backend can address what it started, and ``stop()``
    forgets it.
    """
    mock_run.return_value = MagicMock(stdout="abc123", stderr="", returncode=0)
    backend = DockerBackend("team-1")

    backend.start("team-1")

    assert backend.container_name is not None
    assert backend.container_name.startswith(CONTAINER_NAME_PREFIX)


# ---------------------------------------------------------------------------
# exec() — through the backend, with the container's name already known
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_exec_with_cwd_builds_correct_docker_command(mock_popen: MagicMock) -> None:
    """exec('pytest tests/', cwd='src') builds docker exec -w /workspace/src."""
    backend = started_backend()
    popen_mock(mock_popen)

    backend.exec("pytest tests/", "src")

    expected_cmd = [
        "docker",
        "exec",
        "-w",
        "/workspace/src",
        "akgentic-sandbox-0123456789ab",
        "pytest",
        "tests/",
    ]
    mock_popen.assert_called_once_with(
        expected_cmd,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=None,
        preexec_fn=None,
    )
    # The budget moved with the call shape: Popen returns once the child is
    # spawned, so the wall clock is spent in communicate().
    assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == DOCKER_EXEC_TIMEOUT


@patch(POPEN)
def test_exec_without_cwd_uses_workspace_root(mock_popen: MagicMock) -> None:
    """exec('pytest tests/', cwd='') builds docker exec -w /workspace (no trailing slash)."""
    backend = started_backend()
    popen_mock(mock_popen)

    backend.exec("pytest tests/", "")

    expected_cmd = [
        "docker",
        "exec",
        "-w",
        "/workspace",
        "akgentic-sandbox-0123456789ab",
        "pytest",
        "tests/",
    ]
    mock_popen.assert_called_once_with(
        expected_cmd,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=None,
        preexec_fn=None,
    )
    assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == DOCKER_EXEC_TIMEOUT


@patch(POPEN)
def test_exec_returns_exec_result_with_correct_fields(mock_popen: MagicMock) -> None:
    """exec() returns ExecResult with stdout, stderr, exit_code from mocked subprocess."""
    backend = started_backend()
    popen_mock(mock_popen, stdout="test passed", stderr="warning")

    result = backend.exec("pytest tests/", "")

    assert isinstance(result, ExecResult)
    assert result.stdout == "test passed"
    assert result.stderr == "warning"
    assert result.exit_code == 0


@patch(POPEN)
def test_exec_captures_non_zero_exit_code(mock_popen: MagicMock) -> None:
    """exec() correctly captures non-zero exit codes."""
    backend = started_backend()
    popen_mock(mock_popen, stderr="test failed", returncode=1)

    result = backend.exec("pytest tests/", "")

    assert result.exit_code == 1
    assert result.stderr == "test failed"


@patch(KILLPG)
@patch(POPEN)
def test_exec_timeout_propagates_and_signals_the_direct_child(
    mock_popen: MagicMock, mock_killpg: MagicMock
) -> None:
    """exec() propagates subprocess.TimeoutExpired, and the kill is the direct child's.

    This backend passes no ``preexec_fn`` and so makes no process group of its
    own: the host-side tree is the ``docker exec`` client, one process deep. A
    group kill here would be ``killpg`` on the caller's own group, which is why
    the expiry path must reach ``Popen.kill`` and never ``os.killpg``.
    """
    backend = started_backend()
    proc = popen_mock(mock_popen)
    proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=60),
        ("", ""),
    ]

    with pytest.raises(subprocess.TimeoutExpired):
        backend.exec("pytest tests/", "")

    proc.kill.assert_called_once()
    mock_killpg.assert_not_called()


@patch(POPEN)
def test_exec_keeps_a_quoted_argument_whole_after_the_docker_prefix(
    mock_popen: MagicMock,
) -> None:
    """shlex tokens follow ``docker exec -w <workdir> <container>``, unchanged."""
    backend = started_backend()
    popen_mock(mock_popen)

    backend.exec('echo "hello world"', "src")

    docker_cmd: list[str] = mock_popen.call_args[0][0]
    assert docker_cmd[:5] == [
        "docker",
        "exec",
        "-w",
        "/workspace/src",
        "akgentic-sandbox-0123456789ab",
    ]
    assert docker_cmd[5:] == ["echo", "hello world"]
