"""LocalSandboxActor — subprocess-based sandbox for local filesystem execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

from akgentic.tool.sandbox.actor import ExecResult, SandboxActor


class LocalSandboxActor(SandboxActor):
    """Subprocess-based sandbox actor for local filesystem execution.

    Creates and manages a workspace directory under ./workspaces/{team_id}/.
    No Docker daemon required. Used when SANDBOX_MODE=local (default).
    """

    def _start_sandbox(self) -> None:
        workspace_path = Path(f"./workspaces/{self.config.team_id}")
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
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
