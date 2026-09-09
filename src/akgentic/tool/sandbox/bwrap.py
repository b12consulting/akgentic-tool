"""BwrapBackend — Linux bubblewrap sandbox for filesystem-isolated command execution.

A plain strategy, built by ``resolve_mode`` and owned by ``#Workspace``'s
``ExecRunner``; no actor stands between the workspace and the namespace.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from akgentic.tool.sandbox.backend import (
    DEFAULT_BACKEND_TIMEOUT_S,
    ExecResult,
    ProcessBackend,
    validate_command,
)
from akgentic.tool.sandbox.local import _make_preexec

logger = logging.getLogger(__name__)


class BwrapBackend(ProcessBackend):
    """Linux-only sandbox that uses bubblewrap (bwrap) for filesystem isolation.

    Requires ``bwrap`` to be installed on PATH (``apt install bubblewrap`` or
    ``dnf install bubblewrap``). Each ``exec()`` invocation runs the command inside
    a fresh bubblewrap namespace where only the workspace directory is writable —
    ``/usr``, ``/lib*``, ``/tmp``, ``/dev``, and ``/proc`` are bound read-only or as
    virtual filesystems. Network access is disabled via ``--unshare-net``.

    Unlike :class:`~akgentic.tool.sandbox.local.LocalBackend`, this provides genuine
    filesystem isolation: paths outside the namespace (e.g., ``/etc``, ``/home``,
    parent directories of the workspace) are invisible to the sandboxed process.
    """

    def __init__(self, team_id: str = "") -> None:
        super().__init__(team_id)
        self.workspace_path: Path | None = None
        """The resolved host directory, set by :meth:`start`."""

    def start(self, workspace_path: str) -> None:
        """Start the bubblewrap sandbox.

        Checks that ``bwrap`` is available on PATH, then joins the card's
        already-resolved *workspace_path* to ``AKGENTIC_WORKSPACES_ROOT``
        (defaulting to ``./workspaces``) and creates the directory if it does not
        yet exist. Nothing is derived here: this backend has no notion of users,
        teams, sharing or metadata, which is what makes it unable to open a tree
        other than the one it was handed.

        Raises:
            RuntimeError: If ``bwrap`` is not found on PATH.
        """
        if shutil.which("bwrap") is None:
            raise RuntimeError(
                "bwrap not found on PATH. Install with:\n"
                "  apt install bubblewrap   (Debian/Ubuntu)\n"
                "  dnf install bubblewrap   (Fedora/RHEL)"
            )
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        resolved = Path(base) / workspace_path
        resolved.mkdir(parents=True, exist_ok=True)
        self.workspace_path = resolved.resolve()
        logger.debug("BwrapBackend started: workspace=%s", self.workspace_path)

    def exec(self, cmd: str, cwd: str = "", timeout: float | None = None) -> ExecResult:
        """Execute a command inside a bubblewrap namespace.

        Builds a ``bwrap`` command that mounts the workspace at ``/workspace``
        (read-write), bind-mounts ``/usr`` and ``/lib*`` read-only, provides
        virtual ``/tmp``, ``/dev``, and ``/proc``, and unshares the network and
        PID namespaces. The process is also subject to the same resource limits
        as ``LocalBackend`` (via ``_make_preexec()``) and runs with a
        minimal PATH-only environment.

        **Only the workspace root is bound.** The journal lives at the sibling
        ``<root>.git``, which is therefore not inside the namespace at all —
        that placement, not the command allowlist, is what keeps a sandboxed run
        from reaching the history. Binding a parent directory here for
        convenience would silently undo it.

        Args:
            cmd: Full command string to execute.
            cwd: Working directory inside the sandbox (relative to ``/workspace``).
                 Empty string means ``/workspace`` root.
            timeout: Wall-clock budget in seconds, or ``None`` for the default.

        Returns:
            ExecResult with stdout, stderr, and exit_code from the process.
        """
        assert self.workspace_path is not None
        effective_cwd = f"/workspace/{cwd}" if cwd else "/workspace"
        bwrap_cmd: list[str] = [
            "bwrap",
            "--bind", str(self.workspace_path), "/workspace",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind-try", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/lib32", "/lib32",
            "--tmpfs", "/tmp",
            "--dev", "/dev",
            "--proc", "/proc",
            "--unshare-net",
            "--unshare-pid",
            "--die-with-parent",
            "--new-session",
            "--chdir", effective_cwd,
        ] + validate_command(cmd)
        # ``process_group`` goes with ``_make_preexec``: the group it creates is
        # what a kill or a timeout signals. ``bwrap`` itself leads that group;
        # ``--new-session`` moves the sandboxed command into a session of its
        # own, and ``--die-with-parent`` is what ends it when bwrap dies.
        return self._run(
            bwrap_cmd,
            timeout=DEFAULT_BACKEND_TIMEOUT_S if timeout is None else timeout,
            preexec_fn=_make_preexec(),
            process_group=True,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
        )

    def _release(self) -> None:
        """Nothing to release: a bubblewrap namespace lives only as long as its run."""
        logger.debug("BwrapBackend stopped.")
