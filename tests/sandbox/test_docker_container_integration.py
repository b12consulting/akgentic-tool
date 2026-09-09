"""What only a running Docker daemon can prove about the container ``start()`` creates.

**Every spec here needs a daemon, and the mocked suite must never be described as
covering them.** `test_docker_sandbox.py` asserts the argv the backend builds,
which is a claim about this code. Whether a read-only root actually refuses a
write, whether the uid mapping produces a host-owned file, whether git stops
reporting *dubious ownership*, whether a tmpfs is writable at the mode it was
given, and whether a container is really gone — those are facts about docker and
git, and an argv assertion cannot reach any of them.

**The gate needs both a marker and a skip, and the reason is not symmetry.**
``packages/akgentic-tool/pyproject.toml`` declares no ``[tool.pytest.ini_options]``
and registers no markers; ``integration`` exists only in the workspace-root
``pyproject.toml``. The package's own CI runs ``pytest tests/`` with **no ``-m``
filter**, so a marker alone keeps these out of the local gate command and does
**not** keep them out of package CI. The ``skipif`` is what protects a runner
with no daemon.

**And package CI is not such a runner.** GitHub-hosted ``ubuntu-latest`` ships a
running Docker daemon, so these specs *execute* there on every push — building
the image in-run, at about two minutes of CI time, and then running as the
runner's uid. That is the one place the host-side ownership check in the uid spec
below is literal rather than remapped, so it is coverage rather than a hazard;
but nobody should read the ``skipif`` as keeping a daemon-dependent spec out of
CI.

The daemon probe runs once at import under its own timeout, so a host with the
CLI installed but no daemon running skips rather than hangs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from akgentic.tool.sandbox.docker import SANDBOX_HOME, DockerBackend

if TYPE_CHECKING:
    from collections.abc import Iterator


def _daemon_available() -> bool:
    """True when a docker daemon answers, under a bound so a dead socket cannot hang."""
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


_DAEMON = _daemon_available()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _DAEMON, reason="needs a running Docker daemon"),
]


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A two-segment workspace under a root the backend will resolve."""
    root = tmp_path / "workspaces"
    workspace = root / "u-alice" / "notes"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(root))
    return workspace


@pytest.fixture
def started(tree: Path) -> Iterator[DockerBackend]:
    """A real container, removed in a ``finally`` so a failing spec leaks nothing."""
    backend = DockerBackend("team-integration")
    backend.start("u-alice/notes")
    try:
        yield backend
    finally:
        backend.stop()


def test_a_write_outside_the_workspace_is_refused_and_one_inside_succeeds(
    started: DockerBackend, tree: Path
) -> None:
    """AC20: ``--read-only`` is enforced by the kernel, not by the CLI.

    No argv assertion can show that a write is refused, which is the whole reason
    this spec needs a daemon.
    """
    outside = started.exec("sh -c 'echo denied > /etc/akgentic-probe'", "", 60.0)
    assert outside.exit_code != 0
    assert "read-only" in (outside.stderr + outside.stdout).lower()

    inside = started.exec("sh -c 'echo ok > inside.txt'", "", 60.0)
    assert inside.exit_code == 0, inside.stderr
    assert (tree / "inside.txt").read_text().strip() == "ok"


def test_the_container_runs_as_the_host_uid_and_the_host_can_replace_what_it_wrote(
    started: DockerBackend, tree: Path
) -> None:
    """AC21: publication by rename needs the host process able to replace the inode.

    ``Filesystem.write`` publishes by renaming over the target. If the container
    wrote a file the host cannot replace, every later write to that path fails.

    **The container's own view of its uid is asserted first, and that is not
    belt-and-braces.** The host-side ownership check below is *inert on macOS*:
    Docker Desktop runs the daemon in a VM and its file sharing remaps bind-mount
    ownership, so a file written by a container running as **root** still appears
    owned by the host user. Dropping ``--user`` was mutation-tested against a real
    daemon on macOS and left the host-side assertion green, which is the "check
    narrowed until it agreed" shape. ``id -u`` inside the container is what
    ``--user`` actually controls, on every platform, so it is what carries this
    spec here; the host-side half still carries it on Linux, where the mapping is
    literal.
    """
    import os

    whoami = started.exec("sh -c 'id -u; id -g'", "", 60.0)
    assert whoami.exit_code == 0, whoami.stderr
    assert whoami.stdout.split() == [str(os.getuid()), str(os.getgid())]

    result = started.exec("sh -c 'echo from-container > owned.txt'", "", 60.0)
    assert result.exit_code == 0, result.stderr

    written = tree / "owned.txt"
    assert written.stat().st_uid == os.getuid()

    replacement = tree / "owned.txt.tmp"
    replacement.write_text("from-host\n")
    replacement.replace(written)  # must not raise
    assert written.read_text().strip() == "from-host"


def test_git_reports_no_dubious_ownership_and_commits_without_a_written_config(
    started: DockerBackend, tree: Path
) -> None:
    """AC22: git's *protected configuration* rule, observed rather than reasoned about.

    ``safe.directory`` is honoured only in system, global and **command** scope,
    and ``GIT_CONFIG_COUNT``/``KEY_n``/``VALUE_n`` is command scope. The reasoning
    is sound and still needs to be seen once — nothing is written to the
    read-only home for any of it.
    """
    init = started.exec("git init", "", 60.0)
    assert init.exit_code == 0, init.stderr

    status = started.exec("git status", "", 60.0)
    assert status.exit_code == 0, status.stderr
    assert "dubious ownership" not in (status.stdout + status.stderr).lower()

    (tree / "a.txt").write_text("hello\n")
    add = started.exec("git add a.txt", "", 60.0)
    assert add.exit_code == 0, add.stderr
    commit = started.exec("git commit -m probe", "", 60.0)
    assert commit.exit_code == 0, commit.stderr

    log = started.exec("git log -1 --format=%an", "", 60.0)
    assert log.stdout.strip() == "akgentic-sandbox"


@pytest.mark.parametrize("mount", ["/tmp", SANDBOX_HOME])
def test_each_tmpfs_is_mounted_without_noexec_and_with_an_explicit_size(
    started: DockerBackend, mount: str
) -> None:
    """AC23: the options are read back from the kernel, not from the argv.

    **This is the assertion the package-manager smoke checks below could not
    make.** Running ``uv --version`` under a bare ``--tmpfs`` passes, because it
    writes nothing — so those specs stayed green when the options were mutated
    away, which was found by running that mutation rather than by reading them.
    ``/proc/mounts`` is what docker actually did with the flags: a bare
    ``--tmpfs`` mounts ``noexec`` with no size at all on this daemon, and both
    differences are visible here.
    """
    mounts = started.exec("cat /proc/mounts", "", 60.0)
    assert mounts.exit_code == 0, mounts.stderr

    line = next((ln for ln in mounts.stdout.splitlines() if f" {mount} tmpfs " in ln), None)
    assert line is not None, f"{mount} is not a tmpfs; mounts were:\n{mounts.stdout}"

    options = line.split()[3].split(",")
    assert "rw" in options, line
    assert "noexec" not in options, f"noexec breaks pip's wheel builds out of TMPDIR: {line}"
    assert any(opt.startswith("size=") for opt in options), f"no explicit size: {line}"


def test_a_script_can_be_executed_from_the_tmpfs_home(started: DockerBackend) -> None:
    """AC23: what ``noexec`` actually costs, exercised rather than described.

    ``pip`` builds wheels and runs build backends out of ``TMPDIR``. Under a bare
    ``--tmpfs`` this exact sequence fails with *Permission denied*.
    """
    result = started.exec(
        f"sh -c 'printf \"#!/bin/sh\\necho ran\\n\" > {SANDBOX_HOME}/s.sh "
        f"&& chmod +x {SANDBOX_HOME}/s.sh && {SANDBOX_HOME}/s.sh'",
        "",
        60.0,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "ran"


@pytest.mark.parametrize(
    ("cmd", "tool"),
    [
        ("pip download --no-deps --dest /tmp/pipdl six", "pip"),
        ("uv --version", "uv"),
        ("npm config get cache", "npm"),
    ],
)
def test_the_package_managers_run_under_a_tmpfs_home_on_a_read_only_root(
    started: DockerBackend, cmd: str, tool: str
) -> None:
    """AC23: the three tools ADR-050 names, reachable and runnable.

    **A smoke check, and it is worth saying what it does not prove.** These pass
    under a bare ``--tmpfs`` too — the mount options are guarded by the two specs
    above, not by these. What these do carry is reachability: ``uv`` is here
    because the Dockerfile moved it off ``/root``, whose mode 0700 puts it out of
    reach under ``--user`` however the ``PATH`` is set.
    """
    result = started.exec(cmd, "", 120.0)
    assert result.exit_code == 0, f"{tool}: {result.stderr}"


def test_home_is_writable_and_is_the_directory_the_backend_chose(
    started: DockerBackend,
) -> None:
    """AC23: ``$HOME`` is the backend's, and it is writable.

    Docker gives a numeric user with no passwd entry ``HOME=/``, which is on the
    read-only root — so both halves are worth observing together.
    """
    home = started.exec("sh -c 'echo $HOME'", "", 60.0)
    assert home.stdout.strip() == SANDBOX_HOME

    written = started.exec(f"sh -c 'echo ok > {SANDBOX_HOME}/probe && cat {SANDBOX_HOME}/probe'",
                           "", 60.0)
    assert written.exit_code == 0, written.stderr
    assert written.stdout.strip() == "ok"


def test_stop_actually_removes_the_container_and_a_second_stop_still_does_not_raise(
    tree: Path,
) -> None:
    """AC24: removal is a state change in the daemon; the mock only proves it was issued.

    The external ``docker rm`` stands in for a reaper, an operator, or a daemon
    restart — the case where the container is gone before teardown reaches it.
    """
    backend = DockerBackend("team-integration")
    backend.start("u-alice/notes")
    name = backend.container_name
    assert name is not None

    listed = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert name in listed.stdout.splitlines()

    backend.stop()

    gone = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert name not in gone.stdout.splitlines()

    backend.stop()  # must not raise


def test_a_container_removed_from_outside_does_not_make_stop_raise(tree: Path) -> None:
    """AC24: *already gone* is the outcome teardown asked for, not an error."""
    backend = DockerBackend("team-integration")
    backend.start("u-alice/notes")
    name = backend.container_name
    assert name is not None

    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)

    backend.stop()  # must not raise
    assert backend.container_name is None


def test_the_label_is_readable_from_the_daemon(started: DockerBackend) -> None:
    """AC24 neighbour: the label a reaper keys on survives into ``docker inspect``."""
    name = started.container_name
    assert name is not None
    inspected = subprocess.run(
        ["docker", "inspect", "-f", "{{index .Config.Labels \"akgentic.workspace_path\"}}", name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert inspected.stdout.strip() == "u-alice/notes"
