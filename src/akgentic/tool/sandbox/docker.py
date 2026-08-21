"""DockerSandboxActor — persistent Docker container per team."""

from __future__ import annotations

import importlib.resources
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from akgentic.tool.sandbox.actor import DEFAULT_BACKEND_TIMEOUT_S, ExecResult, SandboxActor

logger = logging.getLogger(__name__)

SANDBOX_IMAGE: str = "akgentic-sandbox:latest"

DOCKER_EXEC_TIMEOUT: float = DEFAULT_BACKEND_TIMEOUT_S
"""Default budget for one ``docker exec``, when the caller names none.

This used to be 60 s — twice the orchestrator's 30 s stop backstop — which made
docker the one backend able to hold a team's teardown open past the point that
teardown gives up. A Python thread cannot be cancelled, so the difference is real
wall clock, not a formality. Docker is no longer the exception.
"""


class DockerSandboxActor(SandboxActor):
    """Persistent Docker container sandbox per team.

    Manages a single Docker container named ``sandbox-{team_id}``.
    Container is started (or reused) on on_start(), stopped (not removed)
    on on_stop(). The host-side volume mount path uses
    ``{AKGENTIC_WORKSPACES_ROOT}/{workspace_id or team_id}`` so that when
    ``SandboxConfig.workspace_id`` is set, the mounted directory matches the
    one used by ``WorkspaceTool(workspace_id=...)``. The container name always
    uses ``team_id`` — containers are per-team execution resources.
    """

    def _resolved_image(self) -> str:
        """Image name for docker run: AKGENTIC_SANDBOX_IMAGE override or the default."""
        return os.environ.get("AKGENTIC_SANDBOX_IMAGE", SANDBOX_IMAGE)

    def _ensure_image(self) -> None:
        """Build SANDBOX_IMAGE from the bundled Dockerfile if not present locally.

        Skipped entirely when AKGENTIC_SANDBOX_IMAGE is set — the caller owns the image.
        """
        if os.environ.get("AKGENTIC_SANDBOX_IMAGE"):
            return
        check = subprocess.run(
            ["docker", "images", "-q", SANDBOX_IMAGE], capture_output=True, text=True
        )
        if check.stdout.strip():
            return
        logger.info("Building %s from bundled Dockerfile (first use)...", SANDBOX_IMAGE)
        dockerfile_text = (
            importlib.resources.files("akgentic.tool.sandbox")
            .joinpath("sandbox.Dockerfile")
            .read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
            result = subprocess.run(["docker", "build", "-t", SANDBOX_IMAGE, tmpdir])
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to build {SANDBOX_IMAGE}. Check the docker build output above. "
                "Set AKGENTIC_SANDBOX_IMAGE to use a pre-built image instead."
            )
        logger.info("Built %s successfully.", SANDBOX_IMAGE)

    def _start_sandbox(self) -> None:
        container_name = f"sandbox-{self.config.team_id}"
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker CLI not found on PATH — cannot start DockerSandboxActor"
            )
        self._ensure_image()
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        ws_name = self.config.workspace_id or self.config.team_id
        volume = f"{(Path(base) / ws_name).resolve()}:/workspace"
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
        if container_name in check.stdout.splitlines():
            subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "-v",
                    volume,
                    "-w",
                    "/workspace",
                    self._resolved_image(),
                    "sleep",
                    "infinity",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"docker run failed (exit {result.returncode}): {result.stderr.strip()}"
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

    def _exec(self, cmd: str, cwd: str, timeout: float | None = None) -> ExecResult:
        """Execute a command in the team's container.

        Only ``<root>:/workspace`` is mounted (see ``_start_sandbox``), so the
        sibling journal at ``<root>.git`` is not visible inside the container.
        """
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
            timeout=DOCKER_EXEC_TIMEOUT if timeout is None else timeout,
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
