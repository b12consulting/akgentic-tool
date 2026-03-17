"""DockerSandboxActor — persistent Docker container per team."""

from __future__ import annotations

import shutil
import subprocess

from akgentic.tool.sandbox.actor import ExecResult, SandboxActor

SANDBOX_IMAGE: str = "akgentic-sandbox:latest"
DOCKER_EXEC_TIMEOUT: int = 60


class DockerSandboxActor(SandboxActor):
    """Persistent Docker container sandbox per team.

    Manages a single Docker container named sandbox-{team_id}.
    Container is started (or reused) on on_start(), stopped (not removed)
    on on_stop(). Used when SANDBOX_MODE=docker.
    """

    def _start_sandbox(self) -> None:
        container_name = f"sandbox-{self.config.team_id}"
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker CLI not found on PATH — cannot start DockerSandboxActor"
            )
        # Check if container already exists (any state)
        check = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={container_name}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
        )
        if container_name in check.stdout:
            subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "--network",
                    "none",
                    "-v",
                    f"workspaces/{self.config.team_id}:/workspace",
                    "-w",
                    "/workspace",
                    SANDBOX_IMAGE,
                    "sleep",
                    "infinity",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        self.state.container_name = container_name
        self.state.notify_state_change()

    def _stop_sandbox(self) -> None:
        assert self.state.container_name is not None
        subprocess.run(
            ["docker", "stop", self.state.container_name],
            capture_output=True,
            text=True,
        )
        # Do NOT run docker rm — container filesystem preserved between restarts

    def _exec(self, cmd: str, cwd: str) -> ExecResult:
        assert self.state.container_name is not None
        effective_workdir = f"/workspace/{cwd}" if cwd else "/workspace"
        docker_cmd = [
            "docker",
            "exec",
            "-w",
            effective_workdir,
            self.state.container_name,
        ] + cmd.split()
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=DOCKER_EXEC_TIMEOUT,
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
