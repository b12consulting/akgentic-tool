"""SeatbeltBackend — macOS Apple Seatbelt sandbox for policy-based filesystem isolation.

A plain strategy, built by ``resolve_mode`` and owned by ``#Workspace``'s
``ExecRunner``; no actor stands between the workspace and ``sandbox-exec``.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

from akgentic.tool.sandbox.backend import (
    DEFAULT_BACKEND_TIMEOUT_S,
    ExecResult,
    ProcessBackend,
    validate_command,
)

logger = logging.getLogger(__name__)

_SEATBELT_POLICY: str = """\
; akgentic-seatbelt.sb — write-restricted workspace sandbox
; Security model: allow all reads, restrict writes to workspace + tmpdir only.
; The command allowlist (ALLOWED_COMMANDS) is the primary security boundary.
(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow sysctl-read)

; allow all filesystem reads (tools need broad read access on macOS)
(allow file-read*)

; writes restricted to workspace, tmpdir, and /dev/null
(allow file-write* (literal "/dev/null"))
(allow file-write* (subpath "{workspace}"))
(allow file-write* (subpath "{tmpdir}"))

(allow network*)
(allow ipc-posix-shm)
(allow mach-lookup)
"""


class SeatbeltBackend(ProcessBackend):
    """macOS-only sandbox that uses Apple Seatbelt (``sandbox-exec``) for isolation.

    Requires ``sandbox-exec`` to be on PATH — it ships with macOS but is
    deprecated since macOS 10.15 Catalina and may be removed in a future
    macOS release. This backend is intended for macOS developer workstations
    only.

    Each ``exec()`` invocation writes a write-restricted SBPL policy to a
    temporary ``.sb`` file that allows all reads but restricts writes to the
    workspace directory and tmpdir. Network access is allowed. The temp file
    is deleted in a ``finally`` block after the subprocess completes.

    Unlike :class:`~akgentic.tool.sandbox.bwrap.BwrapBackend` and
    :class:`~akgentic.tool.sandbox.local.LocalBackend`, no ``preexec_fn``
    or env-stripping is applied: ``resource.setrlimit`` behaves differently on
    macOS and the Seatbelt policy handles the primary threat model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.workspace_path: Path | None = None
        """The resolved host directory, set by :meth:`start`."""

    def start(self, workspace_path: str) -> None:
        """Start the Seatbelt sandbox.

        Checks that ``sandbox-exec`` is available on PATH **and** that
        ``sandbox_apply`` actually works at runtime (macOS 15+ blocks it),
        then joins the card's already-resolved *workspace_path* to
        ``AKGENTIC_WORKSPACES_ROOT`` (defaulting to ``./workspaces``) and creates
        the directory if it does not yet exist — deriving nothing itself. Emits a
        ``DeprecationWarning`` noting that ``sandbox-exec`` is deprecated
        since macOS 10.15 Catalina.

        Raises:
            RuntimeError: If ``sandbox-exec`` is not found on PATH or if
                sandbox_apply is blocked by the OS (macOS 15+).
        """
        if shutil.which("sandbox-exec") is None:
            raise RuntimeError(
                "sandbox-exec not found on PATH. "
                "It ships with macOS but is absent on this system."
            )
        if platform.system() == "Darwin":
            probe = subprocess.run(
                ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
                capture_output=True,
                timeout=5,
            )
            if probe.returncode != 0:
                raise RuntimeError(
                    "sandbox-exec is on PATH but sandbox_apply is blocked "
                    f"(exit {probe.returncode}). Common causes: the calling process "
                    "is already sandboxed (e.g. running inside Claude Code) or a "
                    "future macOS removed sandbox-exec support. "
                    "Use mode='docker' or mode='local' instead."
                )
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        resolved = Path(base) / workspace_path
        resolved.mkdir(parents=True, exist_ok=True)
        self.workspace_path = resolved.resolve()
        logger.debug("SeatbeltBackend started: workspace=%s", self.workspace_path)
        warnings.warn(
            "sandbox-exec is deprecated since macOS 10.15 Catalina and may be removed "
            "in a future macOS release. SeatbeltBackend is for macOS developer "
            "workstations only.",
            DeprecationWarning,
            stacklevel=2,
        )

    def exec(self, cmd: str, cwd: str = "", timeout: float | None = None) -> ExecResult:
        """Execute a command inside an Apple Seatbelt policy sandbox.

        Writes the deny-by-default SBPL policy to a temporary ``.sb`` file
        with the workspace path substituted, then invokes ``sandbox-exec -f
        <policy_file>`` with the given command. The temp file is deleted in a
        ``finally`` block.

        No ``preexec_fn`` or env-stripping is applied — ``resource.setrlimit``
        behaviour differs on macOS and the SBPL policy covers the threat model.
        With no ``preexec_fn`` there is no process group of its own either, so a
        kill or a timeout signals the direct ``sandbox-exec`` child; on macOS
        ``sh -c`` execs a single command in place, so that child *is* the
        command.

        **The only writable subpath is the workspace root.** The journal at the
        sibling ``<root>.git`` is outside it, which is what keeps a sandboxed run
        from rewriting the history; widening this policy to a parent directory
        would undo that with no other visible effect.

        Args:
            cmd: Full command string to execute.
            cwd: Working directory for the sandboxed process. Falls back to the
                 workspace path when empty or not provided.
            timeout: Wall-clock budget in seconds, or ``None`` for the default.

        Returns:
            ExecResult with stdout, stderr, and exit_code from the process.
        """
        assert self.workspace_path is not None
        ws = self.workspace_path
        effective_cwd = str(ws / cwd) if cwd else str(ws)
        logger.debug("SeatbeltBackend exec: cmd=%r cwd=%s", cmd, effective_cwd)
        tmpdir = tempfile.gettempdir()
        policy = _SEATBELT_POLICY.replace("{workspace}", str(ws)).replace("{tmpdir}", tmpdir)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sb", delete=False
        ) as policy_file:
            policy_file.write(policy)
            policy_path = policy_file.name
        try:
            return self._run(
                ["sandbox-exec", "-f", policy_path] + validate_command(cmd),
                cwd=effective_cwd,
                timeout=DEFAULT_BACKEND_TIMEOUT_S if timeout is None else timeout,
            )
        except FileNotFoundError:
            return ExecResult(
                stdout="",
                stderr=f"Working directory not found: {cwd or '.'}",
                exit_code=1,
            )
        finally:
            os.unlink(policy_path)

    def _release(self) -> None:
        """Nothing to release: a ``sandbox-exec`` process lives only as long as its run."""
        logger.debug("SeatbeltBackend stopped.")
