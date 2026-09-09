"""DockerBackend — persistent Docker container per team.

``DockerSandboxActor`` is kept beside it as a thin delegator: it owns the
actor lifecycle and the state it publishes, and nothing else.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from akgentic.tool.sandbox.actor import SandboxActor
from akgentic.tool.sandbox.backend import (
    DEFAULT_BACKEND_TIMEOUT_S,
    ExecResult,
    ProcessBackend,
    validate_command,
)

logger = logging.getLogger(__name__)

SANDBOX_IMAGE: str = "akgentic-sandbox:latest"

DOCKER_EXEC_TIMEOUT: float = DEFAULT_BACKEND_TIMEOUT_S
"""Default budget for one ``docker exec``, when the caller names none.

This used to be 60 s — twice the orchestrator's 30 s stop backstop — which made
docker the one backend able to hold a team's teardown open past the point that
teardown gives up. A Python thread cannot be cancelled, so the difference is real
wall clock, not a formality. Docker is no longer the exception.
"""


class DockerBackend(ProcessBackend):
    """Persistent Docker container sandbox per team.

    Manages a single Docker container named ``sandbox-{team_id}``.
    The container is started (or reused) on :meth:`start` and stopped — never
    removed — on :meth:`stop`. The host-side volume mount is
    ``{AKGENTIC_WORKSPACES_ROOT}/{workspace_path}`` — the path the card
    already resolved, joined and never re-derived, so the mounted directory is
    by construction the one the write gate and the journal are working on. The
    container name always uses ``team_id``: containers are per-team execution
    resources, and that is the only thing ``team_id`` decides here.

    Args:
        team_id: Names the container, and nothing else. Defaulted so that the
            backend is constructible with no arguments, which is what
            ``resolve_mode`` needs to build one from the registry.
    """

    def __init__(self, team_id: str = "") -> None:
        super().__init__()
        self.team_id = team_id
        self.container_name: str | None = None
        """The container this backend runs in, set by :meth:`start`."""

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

    def start(self, workspace_path: str) -> None:
        """Start or reuse the team's container, mounting *workspace_path* at ``/workspace``."""
        container_name = f"sandbox-{self.team_id}"
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker CLI not found on PATH — cannot start DockerSandboxActor"
            )
        self._ensure_image()
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        volume = f"{(Path(base) / workspace_path).resolve()}:/workspace"
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
        self.container_name = container_name

    def exec(self, cmd: str, cwd: str = "", timeout: float | None = None) -> ExecResult:
        """Execute a command in the team's container.

        Only ``<root>:/workspace`` is mounted (see :meth:`start`), so the
        sibling journal at ``<root>.git`` is not visible inside the container.
        """
        assert self.container_name is not None
        effective_workdir = f"/workspace/{cwd}" if cwd else "/workspace"
        docker_cmd = [
            "docker",
            "exec",
            "-w",
            effective_workdir,
            self.container_name,
        ] + validate_command(cmd)
        return self._run(
            docker_cmd,
            timeout=DOCKER_EXEC_TIMEOUT if timeout is None else timeout,
        )

    def kill(self) -> None:
        """End the local ``docker exec`` client — **not**, necessarily, what runs inside.

        The signal reaches the ``docker exec`` process on this host. The command
        it started inside the container is a child of the container's own init,
        not of this process, so it may keep running after this returns. That is
        the best available while the container is durable and shared across runs.

        :meth:`stop` is the authoritative end: it runs ``docker stop``, which
        does end everything inside. Use ``kill`` to abandon a run, ``stop`` to be
        certain it is over.
        """
        super().kill()

    def _release(self) -> None:
        """Stop the container, keeping its filesystem for the next run."""
        assert self.container_name is not None
        subprocess.run(
            ["docker", "stop", self.container_name],
            capture_output=True,
            text=True,
        )
        # Do NOT run docker rm — container filesystem preserved between restarts


class DockerSandboxActor(SandboxActor):
    """Persistent Docker container sandbox actor per team.

    A thin delegator over :class:`DockerBackend`, which holds the implementation.
    """

    _backend: DockerBackend | None = None

    def _start_sandbox(self) -> None:
        self._backend = DockerBackend(self.config.team_id)
        self._backend.start(self.config.workspace_path)
        self.state.container_name = self._backend.container_name
        self.state.notify_state_change()

    def _stop_sandbox(self) -> None:
        if self._backend is not None:
            self._backend.stop()

    def _exec(self, cmd: str, cwd: str, timeout: float | None = None) -> ExecResult:
        assert self._backend is not None
        return self._backend.exec(cmd, cwd, timeout)
