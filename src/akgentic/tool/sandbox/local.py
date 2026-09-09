"""LocalBackend — subprocess-based sandbox for local filesystem execution.

``LocalSandboxActor`` is kept beside it as a thin delegator: it owns the
actor lifecycle and the state it publishes, and nothing else.
"""

from __future__ import annotations

import logging
import os
import resource
import sys
from collections.abc import Callable
from pathlib import Path

from akgentic.tool.sandbox.actor import SandboxActor
from akgentic.tool.sandbox.backend import (
    DEFAULT_BACKEND_TIMEOUT_S,
    ExecResult,
    ProcessBackend,
    validate_command,
)

logger = logging.getLogger(__name__)

# Environment variables safe to pass through to sandboxed subprocesses.
# Keeps tools like git/xcode-select functional on macOS while stripping
# secrets, API keys, and shell customisation that could leak or interfere.
_SAFE_ENV_KEYS: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "DEVELOPER_DIR",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "GIT_EXEC_PATH",
        "GIT_TEMPLATE_DIR",
    }
)


_MACOS_DEVELOPER_DIRS: tuple[str, ...] = (
    "/Library/Developer/CommandLineTools",
    "/Applications/Xcode.app/Contents/Developer",
)


def _make_sandbox_env() -> dict[str, str]:
    """Build a minimal env dict from the host, keeping only safe keys.

    Falls back to a hardcoded PATH if the host PATH is missing.
    On macOS, sets ``DEVELOPER_DIR`` when absent — this avoids the
    ``xcode-select`` symlink lookup at ``/var/select/developer_dir`` which
    is blocked when the calling process is already sandboxed (e.g. Claude Code).
    """
    env: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    env.setdefault("PATH", "/usr/bin:/bin:/usr/local/bin")
    if sys.platform == "darwin" and "DEVELOPER_DIR" not in env:
        for candidate in _MACOS_DEVELOPER_DIRS:
            if Path(candidate).is_dir():
                env["DEVELOPER_DIR"] = candidate
                break
    return env


def _make_preexec(cpu_s: int = 30, mem_mb: int = 512, fsize_mb: int = 100) -> Callable[[], None]:
    """Return a preexec_fn callable that sets resource limits and new process group.

    Sets hard caps for:
    - ``RLIMIT_CPU``: CPU time in seconds
    - ``RLIMIT_AS``: Virtual address space in bytes (skipped on macOS/Darwin
      where it is not reliably enforceable)
    - ``RLIMIT_FSIZE``: Maximum file size in bytes

    Also calls ``os.setpgrp()`` to put the child process into a new process group,
    so that a timeout can kill the entire subtree.
    """

    def preexec() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024**2, mem_mb * 1024**2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb * 1024**2, fsize_mb * 1024**2))
        os.setpgrp()  # new process group → timeout kills entire subtree

    return preexec


class LocalBackend(ProcessBackend):
    """Subprocess-based sandbox for local filesystem execution.

    Creates and manages the workspace directory at
    ``<AKGENTIC_WORKSPACES_ROOT>/{workspace_path}/`` (default root:
    ``./workspaces``) — the path the card already resolved, joined and never
    re-derived, so the directory is by construction the one the card, the write
    gate and the journal are working on. No Docker daemon required.

    This backend does NOT provide filesystem isolation — an allowed command can
    still read files outside the workspace. It is a development convenience only,
    not a production security boundary.
    """

    def __init__(self) -> None:
        super().__init__()
        self.workspace_path: Path | None = None
        """The resolved host directory, set by :meth:`start`."""

    def start(self, workspace_path: str) -> None:
        """Join *workspace_path* to the workspaces root and create the directory."""
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        resolved = Path(base) / workspace_path
        resolved.mkdir(parents=True, exist_ok=True)
        self.workspace_path = resolved.resolve()
        logger.debug(
            "LocalBackend started: workspace=%s (no filesystem isolation)",
            self.workspace_path,
        )

    def exec(self, cmd: str, cwd: str = "", timeout: float | None = None) -> ExecResult:
        """Run *cmd* as a plain subprocess rooted at the workspace."""
        assert self.workspace_path is not None
        effective_cwd = self.workspace_path / cwd if cwd else self.workspace_path
        logger.debug("LocalBackend exec: cmd=%r cwd=%s", cmd, effective_cwd)
        try:
            return self._run(
                validate_command(cmd),
                cwd=str(effective_cwd),
                timeout=DEFAULT_BACKEND_TIMEOUT_S if timeout is None else timeout,
                preexec_fn=_make_preexec(),
                env=_make_sandbox_env(),
            )
        except FileNotFoundError:
            return ExecResult(
                stdout="",
                stderr=f"Working directory not found: {cwd or '.'}",
                exit_code=1,
            )

    def _release(self) -> None:
        """Nothing to release: the workspace directory outlives the backend."""
        logger.debug("LocalBackend stopped.")


class LocalSandboxActor(SandboxActor):
    """Subprocess-based sandbox actor for local filesystem execution.

    A thin delegator over :class:`LocalBackend`, which holds the implementation.

    This actor does NOT provide filesystem isolation — an allowed command can still
    read files outside the workspace. It is a development convenience only, not a
    production security boundary.
    """

    _backend: LocalBackend | None = None

    def _start_sandbox(self) -> None:
        self._backend = LocalBackend()
        self._backend.start(self.config.workspace_path)
        self.state.workspace_path = self._backend.workspace_path
        self.state.notify_state_change()

    def _stop_sandbox(self) -> None:
        if self._backend is not None:
            self._backend.stop()

    def _exec(self, cmd: str, cwd: str, timeout: float | None = None) -> ExecResult:
        assert self._backend is not None
        return self._backend.exec(cmd, cwd, timeout)
