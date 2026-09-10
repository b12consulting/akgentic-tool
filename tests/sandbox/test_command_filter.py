"""The command filter, the allowlist, and ``ExecResult`` — through a backend.

These cases lived in ``tests/sandbox/test_sandbox_actor.py`` and were driven
through the actor's ``exec()``. The actor is gone; the filter is not — it lives
in ``sandbox/backend.py`` as ``validate_command`` and every backend's ``exec()``
calls it — so each obligation is asserted here through a real ``LocalBackend``
whose process is mocked, never through anything that no longer exists.

The four refusal shapes (an unbalanced quote, an empty command, a binary off
the list, a permitted binary reaching the process) are pinned against a real
process in ``tests/sandbox/test_backend_kill.py``; this file carries the rest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.sandbox.backend import (
    ALLOWED_COMMANDS,
    CommandNotAllowedError,
    CommandParseError,
    ExecResult,
    validate_command,
)
from akgentic.tool.sandbox.local import LocalBackend

POPEN = "akgentic.tool.sandbox.backend.subprocess.Popen"


def popen_mock(mock_popen: MagicMock, stdout: str = "", returncode: int = 0) -> MagicMock:
    """Shape *mock_popen* like a ``Popen``: ``communicate()`` pair plus ``returncode``."""
    proc = mock_popen.return_value
    proc.communicate.return_value = (stdout, "")
    proc.returncode = returncode
    proc.pid = 4242
    return proc


@pytest.fixture
def backend(tmp_path: Path) -> LocalBackend:
    """A ``LocalBackend`` pointed at *tmp_path*; the process is mocked per spec."""
    started = LocalBackend()
    started.workspace_path = tmp_path
    return started


# ---------------------------------------------------------------------------
# ExecResult
# ---------------------------------------------------------------------------


def test_exec_result_valid() -> None:
    """ExecResult accepts stdout, stderr, exit_code."""
    result = ExecResult(stdout="hello", stderr="", exit_code=0)
    assert result.stdout == "hello"
    assert result.stderr == ""
    assert result.exit_code == 0


def test_exec_result_requires_all_fields() -> None:
    """ExecResult raises ValidationError when required fields are missing."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExecResult(stdout="hello", stderr="")  # type: ignore[call-arg]


def test_exec_result_non_zero_exit_code() -> None:
    """ExecResult stores non-zero exit codes."""
    result = ExecResult(stdout="", stderr="error output", exit_code=1)
    assert result.exit_code == 1
    assert result.stderr == "error output"


# ---------------------------------------------------------------------------
# ALLOWED_COMMANDS
# ---------------------------------------------------------------------------


def test_allowed_commands_is_frozenset() -> None:
    """ALLOWED_COMMANDS is a frozenset."""
    assert isinstance(ALLOWED_COMMANDS, frozenset)


def test_allowed_commands_exact_set() -> None:
    """ALLOWED_COMMANDS contains exactly the binaries the sandbox offers.

    Pinned as an exact set on purpose: every entry widens what an agent can run
    inside the sandbox, so adding one is a decision that has to be made twice —
    once in the source and once here.
    """
    expected = frozenset(
        {
            # Python
            "python",
            "python3",
            "pytest",
            "ruff",
            "mypy",
            "uv",
            "pip",
            # Web
            "node",
            "npm",
            "npx",
            # bash
            "sh",
            "bash",
            "cat",
            "echo",
            "ls",
            "cp",
            "mv",
            "rm",
            "mkdir",
            "find",
            "grep",
            "sed",
            "awk",
            "jq",
            "wc",
            "xargs",
            "touch",
            "make",
            "git",
            "kill",
            # Network
            "curl",
            "wget",
        }
    )
    assert ALLOWED_COMMANDS == expected
    assert len(ALLOWED_COMMANDS) == 32


def test_git_is_allowed_and_the_journal_is_protected_by_the_mount() -> None:
    """``git`` is on the list, and taking it off would not have protected anything.

    It was briefly removed as defence in depth. That was never the boundary, and
    the test says why: ``bash`` and ``sh`` are on the list and only the first
    token is checked, so ``bash -c "git ..."`` walks straight past it. What
    actually protects the journal is that it lives at the sibling ``<root>.git``,
    outside every isolating backend's mount — see
    ``tests/sandbox/test_journal_placement.py``.
    """
    assert "git" in ALLOWED_COMMANDS
    assert "bash" in ALLOWED_COMMANDS


# ---------------------------------------------------------------------------
# The two exceptions
# ---------------------------------------------------------------------------


def test_command_parse_error_is_not_a_command_not_allowed_error() -> None:
    """The two are siblings, so an allowlist handler cannot swallow a parse error.

    ``CommandNotAllowedError``'s own message carries the allowed command list,
    and ``workspace_exec`` reports it verbatim as the run's failure. That is the
    wrong answer to a quoting mistake — it sends the model hunting for a binary
    it already has — so a parse error must never be caught as an allowlist one.
    """
    assert not issubclass(CommandParseError, CommandNotAllowedError)
    assert not issubclass(CommandNotAllowedError, CommandParseError)


def test_command_parse_error_is_exported_from_the_sandbox_package() -> None:
    """Importable from ``akgentic.tool.sandbox`` and listed in its ``__all__``."""
    import akgentic.tool.sandbox as sandbox_pkg

    assert sandbox_pkg.CommandParseError is CommandParseError
    assert "CommandParseError" in sandbox_pkg.__all__


# ---------------------------------------------------------------------------
# The filter, through a backend's exec()
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_a_disallowed_command_is_refused_naming_the_binary_and_never_spawns(
    mock_popen: MagicMock, backend: LocalBackend
) -> None:
    """A binary off the list surfaces through the wording exec has always used.

    ``malware`` is the exemplar rather than anything the sandbox plausibly
    wants: it has to stay off the list for this test to mean anything.
    """
    with pytest.raises(CommandNotAllowedError, match="malware"):
        backend.exec("malware --install", "")

    mock_popen.assert_not_called()


@patch(POPEN)
def test_allowlist_and_execution_derive_the_same_tokens(
    mock_popen: MagicMock, backend: LocalBackend
) -> None:
    """The validated binary is the shlex token, not a whitespace fragment.

    This is the spec that catches a check and an execution that tokenise
    differently: under ``cmd.split()`` the allowlist would validate ``"my`` while
    the backend ran something else. Both sides are one ``shlex.split`` on the
    same string, so they cannot disagree — and the error message proves which
    one ran.
    """
    with pytest.raises(CommandNotAllowedError) as excinfo:
        backend.exec('"my binary" --flag', "")

    mock_popen.assert_not_called()
    message = str(excinfo.value)
    assert "'my binary'" in message  # one token, quotes consumed
    assert '"my' not in message  # never the whitespace fragment


@patch(POPEN)
def test_empty_and_blank_commands_keep_the_existing_error(
    mock_popen: MagicMock, backend: LocalBackend
) -> None:
    """``shlex.split`` returns ``[]`` for both, so the empty branch is unchanged.

    A blank command must not become a *parse* failure: nothing about it is
    unparseable, there is simply no binary in it.
    """
    for blank in ("", "   "):
        with pytest.raises(CommandNotAllowedError, match="empty") as excinfo:
            backend.exec(blank, "")
        assert not isinstance(excinfo.value, CommandParseError)

    mock_popen.assert_not_called()


@patch(POPEN)
def test_exec_rm_rf_passes_allowlist(mock_popen: MagicMock, backend: LocalBackend) -> None:
    """``rm -rf /`` passes — only the ``rm`` binary is checked, not its arguments."""
    popen_mock(mock_popen)

    result = backend.exec("rm -rf /", "")

    assert isinstance(result, ExecResult)
    assert mock_popen.call_args.args[0] == ["rm", "-rf", "/"]


def test_validate_command_returns_the_tokens_the_backend_runs() -> None:
    """The tokens are returned, not discarded, so check and run are one string."""
    assert validate_command('echo "hello world"') == ["echo", "hello world"]


# ---------------------------------------------------------------------------
# The ask entry point forwards cwd and the budget unmodified
# ---------------------------------------------------------------------------


@patch(POPEN)
def test_exec_hands_cwd_and_the_budget_to_the_process(
    mock_popen: MagicMock, backend: LocalBackend, tmp_path: Path
) -> None:
    """``exec(cmd, cwd, timeout)`` reaches the process as the cwd and the budget.

    A budget that stops at the caller is decoration — a Python thread cannot be
    cancelled — so it is asserted where the wait actually happens:
    ``communicate(timeout=)``. ``None`` keeps the backend's own default.
    """
    popen_mock(mock_popen)

    backend.exec("python main.py", "sub", 7.5)
    assert mock_popen.call_args.kwargs["cwd"] == str(tmp_path / "sub")
    assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == 7.5

    backend.exec("python main.py", "", None)
    assert mock_popen.call_args.kwargs["cwd"] == str(tmp_path)
    assert mock_popen.return_value.communicate.call_args.kwargs["timeout"] == 30.0
