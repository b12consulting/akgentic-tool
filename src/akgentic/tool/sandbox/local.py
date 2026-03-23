"""LocalSandboxActor — subprocess-based sandbox for local filesystem execution."""

from __future__ import annotations

import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Callable

from akgentic.tool.sandbox.actor import ExecResult, SandboxActor


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


class LocalSandboxActor(SandboxActor):
    """Subprocess-based sandbox actor for local filesystem execution.

    Creates and manages a workspace directory under
    ``<AKGENTIC_WORKSPACES_ROOT>/{workspace_id or team_id}/`` (default root:
    ``./workspaces``). When ``SandboxConfig.workspace_id`` is set, that value is
    used as the directory name instead of ``team_id``, enabling directory sharing
    with ``WorkspaceTool(workspace_id=...)``. No Docker daemon required.

    This actor does NOT provide filesystem isolation — an allowed command can still
    read files outside the workspace. It is a development convenience only, not a
    production security boundary.
    """

    def _start_sandbox(self) -> None:
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        ws_name = self.config.workspace_id or self.config.team_id
        workspace_path = Path(base) / ws_name
        workspace_path.mkdir(parents=True, exist_ok=True)
        self.state.workspace_path = workspace_path.resolve()
        self.state.notify_state_change()

    def _stop_sandbox(self) -> None:
        pass  # no persistent resource to tear down

    def _exec(self, cmd: str, cwd: str) -> ExecResult:
        assert self.state.workspace_path is not None
        effective_cwd = self.state.workspace_path / cwd if cwd else self.state.workspace_path
        result = subprocess.run(
            cmd.split(),
            cwd=str(effective_cwd),
            capture_output=True,
            text=True,
            timeout=30,
            preexec_fn=_make_preexec(),
            env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
