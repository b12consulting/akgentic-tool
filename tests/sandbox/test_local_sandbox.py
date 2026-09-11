"""Tests for ``LocalBackend`` — subprocess-based sandbox execution.

The backend is constructed directly: there is no actor to stand in for, so
``start()`` is called with the three-segment path a card would resolve, and the
``exec()`` specs point ``workspace_path`` at a temp directory when they want a
specific tree rather than one under ``AKGENTIC_WORKSPACES_ROOT``.

Covers, through the backend:
- ``start()`` creates the workspace directory and stores the resolved absolute path
- ``start()`` uses ``AKGENTIC_WORKSPACES_ROOT`` (default ``./workspaces``) and is idempotent
- ``stop()`` spawns nothing and leaves the directory intact
- ``exec()`` resolves ``cwd`` under the workspace and hands the budget to the process
- ``exec()`` returns ``ExecResult`` with the child's stdout, stderr and exit code
- ``subprocess.TimeoutExpired`` propagates, and the timeout path signals the
  process group the backend asked for (story 8.1's ``os.setpgrp()`` made the
  group; the kill now reaches it)
- resource limits and environment stripping (story 8.1 / 8.5)
- ``shlex`` tokenisation (story 46.1)
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.sandbox.backend import ExecResult
from akgentic.tool.sandbox.local import _SAFE_ENV_KEYS, LocalBackend

POPEN = "akgentic.tool.sandbox.backend.subprocess.Popen"
KILLPG = "akgentic.tool.sandbox.backend.os.killpg"

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def popen_mock(
    mock_popen: MagicMock, stdout: str = "", stderr: str = "", returncode: int = 0
) -> MagicMock:
    """Shape *mock_popen* like a ``Popen``: ``communicate()`` pair plus ``returncode``.

    ``subprocess.run`` returned a ``CompletedProcess`` carrying ``stdout`` /
    ``stderr`` / ``returncode`` as attributes. ``Popen`` yields the two streams
    from ``communicate()`` and the exit status from an attribute set once the
    child is reaped, so every migrated mock is built here rather than eleven
    times by hand. The ``pid`` is an integer because the group kill hands it to
    ``os.killpg``, which refuses a ``MagicMock``.
    """
    proc = mock_popen.return_value
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    proc.pid = 4242
    return proc


def backend_at(workspace_path: Path) -> LocalBackend:
    """A backend already pointed at *workspace_path*, without starting one.

    The equivalent of ``start()`` for a test that wants a specific directory
    rather than one derived from ``AKGENTIC_WORKSPACES_ROOT``.
    """
    backend = LocalBackend()
    backend.workspace_path = workspace_path
    return backend


# ---------------------------------------------------------------------------
# AC1: start() creates workspace and stores absolute path
# ---------------------------------------------------------------------------


def test_start_creates_workspace_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC1: start() creates ./workspaces/team-1/ relative to CWD (default root)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()

    backend.start("team-1")

    expected = tmp_path / "workspaces" / "team-1"
    assert expected.exists()
    assert expected.is_dir()


def test_start_stores_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC1: start() stores the resolved absolute path in ``workspace_path``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()

    backend.start("team-1")

    assert backend.workspace_path is not None
    assert backend.workspace_path.is_absolute()
    assert backend.workspace_path == (tmp_path / "workspaces" / "team-1").resolve()


def test_start_uses_custom_workspaces_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Story 6.5: start() uses AKGENTIC_WORKSPACES_ROOT when set."""
    custom_root = tmp_path / "my-custom-root"
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(custom_root))
    backend = LocalBackend()

    backend.start("team-1")

    expected = custom_root / "team-1"
    assert expected.exists()
    assert expected.is_dir()
    assert backend.workspace_path == expected.resolve()


# ---------------------------------------------------------------------------
# AC2: start() is idempotent
# ---------------------------------------------------------------------------


def test_start_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC2: Calling start() twice does not raise."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()

    backend.start("team-1")
    backend.start("team-1")  # Must not raise

    expected = tmp_path / "workspaces" / "team-1"
    assert expected.exists()


# ---------------------------------------------------------------------------
# AC3: stop() releases nothing
# ---------------------------------------------------------------------------


def test_stop_spawns_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: stop() returns None and starts no process — there is nothing to release."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()
    backend.start("team-1")

    with patch(POPEN) as mock_popen:
        result = backend.stop()

    assert result is None
    mock_popen.assert_not_called()


def test_stop_does_not_remove_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: stop() leaves the workspace directory intact."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()
    backend.start("team-1")

    backend.stop()

    expected = tmp_path / "workspaces" / "team-1"
    assert expected.exists()


# ---------------------------------------------------------------------------
# AC4: exec() with empty cwd uses workspace_path
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_exec_no_cwd_uses_workspace_path(
    mock_popen: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC4: exec(cmd, cwd='') passes ``workspace_path`` as cwd to the process."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()
    backend.start("team-1")

    popen_mock(mock_popen, stdout="output")

    backend.exec("pytest tests/", "")

    call_kwargs = mock_popen.call_args
    assert call_kwargs is not None
    assert call_kwargs.args[0] == ["pytest", "tests/"]
    assert call_kwargs.kwargs["cwd"] == str(backend.workspace_path)
    assert call_kwargs.kwargs["stdout"] is subprocess.PIPE
    assert call_kwargs.kwargs["stderr"] is subprocess.PIPE
    assert call_kwargs.kwargs["text"] is True
    # The budget is an argument to communicate(), not to the constructor: Popen
    # returns as soon as the child is spawned, so the wait is where the wait is.
    assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == 30
    assert call_kwargs.kwargs["preexec_fn"] is not None
    assert callable(call_kwargs.kwargs["preexec_fn"])
    env = call_kwargs.kwargs["env"]
    assert "PATH" in env
    unexpected = set(env) - _SAFE_ENV_KEYS
    assert not unexpected, f"Unexpected keys in env: {unexpected}"


# ---------------------------------------------------------------------------
# AC5: exec() with non-empty cwd uses workspace_path / cwd
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_exec_with_cwd_appends_to_workspace_path(
    mock_popen: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC5: exec(cmd, cwd='src') passes ``workspace_path / 'src'`` as cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()
    backend.start("team-1")

    popen_mock(mock_popen, stdout="output")

    backend.exec("pytest tests/", "src")

    assert backend.workspace_path is not None
    expected_cwd = str(backend.workspace_path / "src")
    call_kwargs = mock_popen.call_args
    assert call_kwargs is not None
    assert call_kwargs.args[0] == ["pytest", "tests/"]
    assert call_kwargs.kwargs["cwd"] == expected_cwd
    assert call_kwargs.kwargs["stdout"] is subprocess.PIPE
    assert call_kwargs.kwargs["stderr"] is subprocess.PIPE
    assert call_kwargs.kwargs["text"] is True
    assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == 30
    assert call_kwargs.kwargs["preexec_fn"] is not None
    assert callable(call_kwargs.kwargs["preexec_fn"])
    env = call_kwargs.kwargs["env"]
    assert "PATH" in env
    unexpected = set(env) - _SAFE_ENV_KEYS
    assert not unexpected, f"Unexpected keys in env: {unexpected}"


# ---------------------------------------------------------------------------
# AC4 / AC5: exec() returns ExecResult with correct fields
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_exec_returns_exec_result(mock_popen: MagicMock, tmp_path: Path) -> None:
    """exec() returns ExecResult with stdout, stderr, exit_code from the process."""
    backend = backend_at(tmp_path)
    popen_mock(mock_popen, stdout="test passed", stderr="warning")

    result = backend.exec("pytest tests/", "")

    assert isinstance(result, ExecResult)
    assert result.stdout == "test passed"
    assert result.stderr == "warning"
    assert result.exit_code == 0


@patch(POPEN)
def test_exec_captures_non_zero_exit_code(mock_popen: MagicMock, tmp_path: Path) -> None:
    """exec() correctly captures non-zero exit codes from the process."""
    backend = backend_at(tmp_path)
    popen_mock(mock_popen, stderr="test failed", returncode=1)

    result = backend.exec("pytest tests/", "")

    assert result.exit_code == 1
    assert result.stderr == "test failed"


# ---------------------------------------------------------------------------
# AC6: subprocess.TimeoutExpired propagates, and the group is what gets killed
# ---------------------------------------------------------------------------


@patch(KILLPG)
@patch(POPEN)
def test_exec_timeout_propagates(
    mock_popen: MagicMock, mock_killpg: MagicMock, tmp_path: Path
) -> None:
    """AC6: subprocess.TimeoutExpired propagates out of exec().

    The command is ``sh -c 'sleep 999'`` rather than ``sleep 999`` because the
    backend applies the allowlist inside its own ``exec()``; ``sleep`` was never
    on the list.

    **The kill on expiry is a group kill.** This backend starts the child under
    ``_make_preexec``, which makes it the leader of its own process group, and
    tells ``_run`` so — so the signal goes to ``os.killpg(child.pid, SIGKILL)``
    and never to ``Popen.kill``, which would reach only the shell and leave what
    the shell forked holding the pipes.
    """
    backend = backend_at(tmp_path)
    proc = popen_mock(mock_popen)
    # First communicate() expires; the second is the drain after the kill, which
    # ProcessBackend._run performs to reproduce subprocess.run's semantics.
    proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=["sh", "-c", "sleep 999"], timeout=30),
        ("", ""),
    ]

    with pytest.raises(subprocess.TimeoutExpired):
        backend.exec("sh -c 'sleep 999'", "")

    # The child's group is signalled and drained, not left behind: skipping
    # either is how a timed-out run becomes a zombie and stops answering
    # ``timed_out``.
    mock_killpg.assert_called_once_with(4242, signal.SIGKILL)
    proc.kill.assert_not_called()
    assert proc.communicate.call_count == 2


@patch(POPEN)
def test_exec_declares_the_process_group_while_the_run_is_in_flight(
    mock_popen: MagicMock, tmp_path: Path
) -> None:
    """The flag that routes ``kill()`` to the group is set for the whole run, and only then.

    ``_make_preexec`` creates the group; this is what says so to ``_run``, and a
    backend that passed the ``preexec_fn`` without it would make a group nothing
    ever signals. Read from inside ``communicate()``, which is when a concurrent
    ``kill()`` would read it, and asserted clear again once the run is over.
    """
    backend = backend_at(tmp_path)
    proc = popen_mock(mock_popen)
    seen: list[bool] = []

    def communicate(timeout: float | None = None) -> tuple[str, str]:
        seen.append(backend._leads_a_group)
        return ("", "")

    proc.communicate.side_effect = communicate

    backend.exec("echo hi", "")

    assert seen == [True]
    assert backend._leads_a_group is False


# ---------------------------------------------------------------------------
# AC7: exec() validates, then runs, end to end
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_exec_validates_then_runs_end_to_end(
    mock_popen: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC7: a started backend's exec() reaches the process with the validated tokens."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()
    backend.start("team-1")

    popen_mock(mock_popen, stdout="passed")

    result = backend.exec("pytest tests/")

    assert isinstance(result, ExecResult)
    assert result.stdout == "passed"
    assert result.exit_code == 0
    mock_popen.assert_called_once()
    assert mock_popen.call_args.args[0] == ["pytest", "tests/"]


# ---------------------------------------------------------------------------
# Story 6.5: shared-filesystem invariant — WorkspaceTool and LocalBackend
# resolve to the same absolute path for the same team_id
# ---------------------------------------------------------------------------


def test_workspace_tool_and_local_backend_resolve_same_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Story 6.5 AC: WorkspaceTool (via get_workspace) and LocalBackend share the same
    workspace root when AKGENTIC_WORKSPACES_ROOT is unset.

    Both must resolve to the same absolute path for team-1 to guarantee the
    shared-filesystem invariant (ADR-006).
    """
    from akgentic.tool.workspace.workspace import get_workspace

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)

    workspace = get_workspace("team-1")
    workspace_tool_root = workspace._root.resolve()

    backend = LocalBackend()
    backend.start("team-1")
    sandbox_root = backend.workspace_path

    assert sandbox_root is not None
    assert workspace_tool_root == sandbox_root, (
        f"WorkspaceTool root ({workspace_tool_root}) != LocalBackend root ({sandbox_root})"
    )


def test_workspace_tool_and_local_backend_resolve_same_path_custom_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Story 6.5 AC: Both use custom AKGENTIC_WORKSPACES_ROOT and resolve to the same path."""
    from akgentic.tool.workspace.workspace import get_workspace

    custom_root = tmp_path / "shared-workspaces"
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(custom_root))

    workspace = get_workspace("team-1")
    workspace_tool_root = workspace._root.resolve()

    backend = LocalBackend()
    backend.start("team-1")
    sandbox_root = backend.workspace_path

    assert sandbox_root is not None
    assert workspace_tool_root == sandbox_root, (
        f"WorkspaceTool root ({workspace_tool_root}) != LocalBackend root ({sandbox_root})"
    )


# ---------------------------------------------------------------------------
# The backend joins the path it was handed, and derives nothing (ADR-048)
# ---------------------------------------------------------------------------


def test_start_opens_exactly_the_path_it_was_handed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory comes from the path ``start()`` was handed and from nothing else.

    There is no override, because there is no derivation. A backend that cannot
    re-derive the path cannot derive a *different* one from the card, the write
    gate and the journal — which is what removes the failure mode rather than
    making it less likely.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()

    backend.start("test")

    expected = tmp_path / "workspaces" / "test"
    assert expected.exists()
    assert expected.is_dir()
    assert backend.workspace_path == expected.resolve()


def test_start_opens_a_three_segment_path_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape a card actually hands it: ``<scope>/<kind>/<leaf>``, created in full.

    ``team-1`` appears nowhere in the result — the backend has no notion of who
    owns the tree, and that is the point.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    backend = LocalBackend()

    backend.start("u-alice/_id/notes")

    expected = tmp_path / "workspaces" / "u-alice" / "_id" / "notes"
    assert expected.is_dir()
    assert backend.workspace_path == expected.resolve()
    assert not (tmp_path / "workspaces" / "team-1").exists()


def test_the_sandbox_and_the_workspace_open_one_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One resolved path in, one directory out, on both sides of the mount.

    The agreement used to rest on two copies of ``workspace_id or team_id``
    staying in step. It now rests on there being one value.
    """
    from akgentic.tool.workspace.workspace import get_workspace

    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path / "workspaces"))

    workspace = get_workspace("test")
    workspace_tool_root = workspace._root.resolve()

    backend = LocalBackend()
    backend.start("test")
    sandbox_root = backend.workspace_path

    assert sandbox_root is not None
    assert workspace_tool_root == sandbox_root, (
        f"WorkspaceTool root ({workspace_tool_root}) != LocalBackend root ({sandbox_root})"
    )


# ---------------------------------------------------------------------------
# Story 8.1: Resource limits and environment stripping (AC: 1, 2, 5)
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_exec_passes_preexec_fn_to_subprocess(mock_popen: MagicMock, tmp_path: Path) -> None:
    """AC5 (Story 8.1): exec() passes a non-None callable preexec_fn to the process."""
    backend = backend_at(tmp_path)
    popen_mock(mock_popen, stdout="output")

    backend.exec("echo hello", "")

    call_kwargs = mock_popen.call_args
    assert call_kwargs is not None
    preexec_fn = call_kwargs.kwargs.get("preexec_fn")
    assert preexec_fn is not None
    assert callable(preexec_fn)


@patch(POPEN)
def test_exec_strips_env_to_safe_keys(mock_popen: MagicMock, tmp_path: Path) -> None:
    """AC5 (Story 8.1): exec() passes env with only safe keys to the process."""
    backend = backend_at(tmp_path)
    popen_mock(mock_popen, stdout="output")

    backend.exec("echo hello", "")

    call_kwargs = mock_popen.call_args
    assert call_kwargs is not None
    env = call_kwargs.kwargs.get("env")
    assert isinstance(env, dict)
    assert "PATH" in env
    assert all(k in _SAFE_ENV_KEYS for k in env), f"Unexpected keys: {set(env) - _SAFE_ENV_KEYS}"


@patch(POPEN)
def test_exec_env_excludes_secrets(
    mock_popen: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC5 (Story 8.1): env dict does not contain API keys or secrets."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    backend = backend_at(tmp_path)
    popen_mock(mock_popen, stdout="output")

    backend.exec("echo hello", "")

    call_kwargs = mock_popen.call_args
    assert call_kwargs is not None
    env = call_kwargs.kwargs.get("env")
    assert isinstance(env, dict)
    assert "OPENAI_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


# ---------------------------------------------------------------------------
# Story 8.1: _make_preexec() unit tests — verify actual resource-limit logic
# ---------------------------------------------------------------------------


def test_make_preexec_returns_callable() -> None:
    """AC1 (Story 8.1): _make_preexec() returns a callable (not None)."""
    from akgentic.tool.sandbox.local import _make_preexec

    fn = _make_preexec()
    assert callable(fn)


def test_make_preexec_custom_defaults_returns_callable() -> None:
    """AC1 (Story 8.1): _make_preexec() with custom args returns a callable."""
    from akgentic.tool.sandbox.local import _make_preexec

    fn = _make_preexec(cpu_s=10, mem_mb=256, fsize_mb=50)
    assert callable(fn)


def test_make_preexec_fn_sets_resource_limits() -> None:
    """AC1 / AC3 (Story 8.1): The callable returned by _make_preexec() sets RLIMIT_CPU,
    RLIMIT_AS, and RLIMIT_FSIZE to the expected values, and calls os.setpgrp().

    We mock resource.setrlimit and os.setpgrp to avoid altering the test process's
    resource limits.
    """
    import resource as resource_module
    from unittest.mock import call, patch

    from akgentic.tool.sandbox.local import _make_preexec

    fn = _make_preexec(cpu_s=30, mem_mb=512, fsize_mb=100)

    with (
        patch.object(resource_module, "setrlimit") as mock_setrlimit,
        patch("os.setpgrp") as mock_setpgrp,
        patch("akgentic.tool.sandbox.local.sys") as mock_sys,
    ):
        mock_sys.platform = "linux"
        fn()

    mb = 1024**2
    expected_calls = [
        call(resource_module.RLIMIT_CPU, (30, 30)),
        call(resource_module.RLIMIT_AS, (512 * mb, 512 * mb)),
        call(resource_module.RLIMIT_FSIZE, (100 * mb, 100 * mb)),
    ]
    mock_setrlimit.assert_has_calls(expected_calls, any_order=False)
    mock_setpgrp.assert_called_once()


# ---------------------------------------------------------------------------
# Story 8.5: Darwin platform guard for RLIMIT_AS (AC: 1, 2)
# ---------------------------------------------------------------------------


def test_make_preexec_skips_rlimit_as_on_darwin() -> None:
    """AC1 (Story 8.5): On macOS (Darwin), RLIMIT_AS is NOT set,
    while RLIMIT_CPU and RLIMIT_FSIZE are still applied.
    """
    import resource as resource_module
    from unittest.mock import call, patch

    from akgentic.tool.sandbox.local import _make_preexec

    fn = _make_preexec(cpu_s=30, mem_mb=512, fsize_mb=100)

    with (
        patch.object(resource_module, "setrlimit") as mock_setrlimit,
        patch("os.setpgrp") as mock_setpgrp,
        patch("akgentic.tool.sandbox.local.sys") as mock_sys,
    ):
        mock_sys.platform = "darwin"
        fn()

    mb = 1024**2
    expected_calls = [
        call(resource_module.RLIMIT_CPU, (30, 30)),
        call(resource_module.RLIMIT_FSIZE, (100 * mb, 100 * mb)),
    ]
    mock_setrlimit.assert_has_calls(expected_calls, any_order=False)
    # Verify RLIMIT_AS was NOT set
    for c in mock_setrlimit.call_args_list:
        assert c[0][0] != resource_module.RLIMIT_AS, "RLIMIT_AS should not be set on Darwin"
    mock_setpgrp.assert_called_once()


def test_make_preexec_sets_rlimit_as_on_linux() -> None:
    """AC2 (Story 8.5): On Linux, all three limits (RLIMIT_CPU, RLIMIT_AS,
    RLIMIT_FSIZE) are set as before.
    """
    import resource as resource_module
    from unittest.mock import call, patch

    from akgentic.tool.sandbox.local import _make_preexec

    fn = _make_preexec(cpu_s=30, mem_mb=512, fsize_mb=100)

    with (
        patch.object(resource_module, "setrlimit") as mock_setrlimit,
        patch("os.setpgrp") as mock_setpgrp,
        patch("akgentic.tool.sandbox.local.sys") as mock_sys,
    ):
        mock_sys.platform = "linux"
        fn()

    mb = 1024**2
    expected_calls = [
        call(resource_module.RLIMIT_CPU, (30, 30)),
        call(resource_module.RLIMIT_AS, (512 * mb, 512 * mb)),
        call(resource_module.RLIMIT_FSIZE, (100 * mb, 100 * mb)),
    ]
    mock_setrlimit.assert_has_calls(expected_calls, any_order=False)
    assert mock_setrlimit.call_count == 3
    mock_setpgrp.assert_called_once()


# ---------------------------------------------------------------------------
# Story 46.1 — argument tokenisation (AC1, AC2, AC5)
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_exec_keeps_a_quoted_argument_whole(mock_popen: MagicMock, tmp_path: Path) -> None:
    """AC1: ``echo "hello world"`` reaches the binary as two tokens, quotes consumed.

    Under ``cmd.split()`` this arrived as ``['echo', '"hello', 'world"']`` — no
    command could ever receive an argument containing a space.
    """
    backend = backend_at(tmp_path)
    popen_mock(mock_popen)

    backend.exec('echo "hello world"', "")

    assert mock_popen.call_args.args[0] == ["echo", "hello world"]


@patch(POPEN)
def test_exec_strips_backslash_escapes_posix_style(mock_popen: MagicMock, tmp_path: Path) -> None:
    """AC5: POSIX mode — ``echo a\\ b`` is two tokens, the escape consumed.

    Guards against a later ``posix=False``, which would keep the backslash
    literal and split ``a\\`` from ``b`` — the old behaviour wearing a new call.
    """
    backend = backend_at(tmp_path)
    popen_mock(mock_popen)

    backend.exec("echo a\\ b", "")

    assert mock_popen.call_args.args[0] == ["echo", "a b"]


def test_bash_dash_c_runs_the_whole_script_for_real(tmp_path: Path) -> None:
    """AC2: the documented escape hatch actually works — no mock.

    ``bash -c 'echo first && echo second'`` used to split into
    ``['bash', '-c', "'echo", 'first', '&&', ...]``, so bash received ``'echo``
    as its script and the rest as positional parameters. The allowlist check
    passed and the execution was nonsense. An argv assertion cannot show that;
    only running it can.
    """
    backend = backend_at(tmp_path)

    result = backend.exec("bash -c 'echo first && echo second'", "")

    assert result.exit_code == 0, result.stderr
    assert "first" in result.stdout
    assert "second" in result.stdout
