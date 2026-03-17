"""Sandbox submodule — sandboxed shell execution for SWE agents."""

from __future__ import annotations

from .actor import (
    ALLOWED_COMMANDS,
    SANDBOX_ACTOR_NAME,
    CommandNotAllowedError,
    ExecResult,
    SandboxActor,
    SandboxConfig,
    SandboxState,
)
from .docker import DockerSandboxActor
from .local import LocalSandboxActor
from .tool import ExecTool

__all__ = [
    "ALLOWED_COMMANDS",
    "CommandNotAllowedError",
    "DockerSandboxActor",
    "ExecResult",
    "ExecTool",
    "LocalSandboxActor",
    "SANDBOX_ACTOR_NAME",
    "SandboxActor",
    "SandboxConfig",
    "SandboxState",
]
