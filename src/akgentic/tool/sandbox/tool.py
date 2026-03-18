"""ExecTool — ToolCard proxy for SandboxActor execution backend."""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from pydantic import PrivateAttr

from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import TOOL_CALL, BaseToolParam, Channels, ToolCard, _resolve
from akgentic.tool.event import ActorToolObserver
from akgentic.tool.sandbox.actor import (
    ALLOWED_COMMANDS,
    SANDBOX_ACTOR_NAME,
    CommandNotAllowedError,
    SandboxActor,
    SandboxConfig,
)
from akgentic.tool.sandbox.docker import DockerSandboxActor
from akgentic.tool.sandbox.local import LocalSandboxActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# SANDBOX_ACTOR_CLASSES — mutable injection window for runtime registration
# ---------------------------------------------------------------------------

SANDBOX_ACTOR_CLASSES: dict[str, type[SandboxActor]] = {
    "local": LocalSandboxActor,
    "docker": DockerSandboxActor,
    # "e2b": E2BSandboxActor  ← injected by akgentic-infra at runtime
}


# ---------------------------------------------------------------------------
# Capability parameter model
# ---------------------------------------------------------------------------


class ExecCommand(BaseToolParam):
    """Execute a sandboxed shell command in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


# ---------------------------------------------------------------------------
# ExecTool ToolCard
# ---------------------------------------------------------------------------


class ExecTool(ToolCard):
    """ToolCard proxy that routes shell commands to the team's SandboxActor."""

    name: str = "Exec"
    description: str = "Execute sandboxed shell commands in the team workspace"
    exec_command: ExecCommand | bool = True
    mode: Literal["local", "docker"] = "local"
    workspace_id: str | None = None

    _sandbox_proxy: SandboxActor | None = PrivateAttr(default=None)

    def observer(self, observer: ActorToolObserver) -> "ExecTool":  # type: ignore[override]
        """Attach observer and set up the sandbox actor proxy.

        Resolves ``SANDBOX_ACTOR_CLASSES[self.mode]`` at call time (not import
        time) so that akgentic-infra can inject additional actor classes before
        any ExecTool is constructed (NFR-SB-7).  ``self.workspace_id`` is
        forwarded to ``SandboxConfig`` so the sandbox backend uses the same
        workspace directory as ``WorkspaceTool(workspace_id=...)``.

        Args:
            observer: Actor-aware observer providing orchestrator access.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If observer.orchestrator is None.
            KeyError: If ``self.mode`` names an unregistered backend.
        """
        self._observer = observer
        if observer.orchestrator is None:
            raise ValueError("ExecTool requires access to the orchestrator.")

        orchestrator_proxy = observer.proxy_ask(observer.orchestrator, Orchestrator)
        sandbox_addr = orchestrator_proxy.get_team_member(SANDBOX_ACTOR_NAME)

        if sandbox_addr is None:
            logger.info(f"ExecTool: create {SANDBOX_ACTOR_NAME}.")
            # KeyError on unknown mode — intentional (fail-fast, NFR-SB-7)
            actor_class = SANDBOX_ACTOR_CLASSES[self.mode]
            sandbox_addr = orchestrator_proxy.createActor(
                actor_class,
                config=SandboxConfig(
                    name=SANDBOX_ACTOR_NAME,
                    role="ToolActor",
                    team_id=str(observer._team_id),  # type: ignore[attr-defined]
                    workspace_id=self.workspace_id,
                    mode=self.mode,
                ),
            )

        self._sandbox_proxy = observer.proxy_ask(sandbox_addr, SandboxActor)
        return self

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return the exec_command tool callable when enabled."""
        tools: list[Callable[..., Any]] = []
        ec = _resolve(self.exec_command, ExecCommand)
        if ec is not None and TOOL_CALL in ec.expose:
            tools.append(self._exec_command_factory(ec))
        return tools

    def _exec_command_factory(self, params: ExecCommand) -> Callable[..., Any]:
        """Build the exec_command callable bound to the sandbox proxy."""
        assert self._sandbox_proxy is not None, "_sandbox_proxy must be set before get_tools()"
        sandbox_proxy = self._sandbox_proxy

        def exec_command(cmd: str, cwd: str = "") -> str:
            """Execute a sandboxed shell command in the team workspace.

            Args:
                cmd: Full command string. The binary (first token) must be in the allow-list.
                cwd: Subdirectory relative to workspace root. Defaults to workspace root.

            Returns:
                Combined stdout, stderr, and exit code summary as a string.
                On disallowed command: error string listing allowed commands.
            """
            try:
                result = sandbox_proxy.exec(cmd, cwd)
                return (
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                    f"\nexit_code: {result.exit_code}"
                )
            except CommandNotAllowedError as e:
                return f"CommandNotAllowedError: {e}. Allowed commands: {sorted(ALLOWED_COMMANDS)}"

        exec_command.__doc__ = params.format_docstring(exec_command.__doc__)
        return exec_command
