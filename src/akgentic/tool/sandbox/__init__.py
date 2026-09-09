"""Sandbox submodule — the exec backend ``workspace_exec`` runs on.

The four backends, the registry they sit in and the probe that picks one for
``mode="auto"`` are what lives here. The card that once wrapped them, ``ExecTool``,
is gone: sandboxed execution is a capability of ``WorkspaceTool`` —
``WorkspaceTool(workspace_exec=...)`` — and :func:`__getattr__` below says so to
anybody still importing the old name.
"""

from __future__ import annotations

from typing import Any

from .actor import (
    ALLOWED_COMMANDS,
    SANDBOX_ACTOR_NAME,
    CommandNotAllowedError,
    CommandParseError,
    ExecReport,
    ExecRequest,
    ExecResult,
    SandboxActor,
    SandboxConfig,
    SandboxState,
    sandbox_actor_name,
)
from .bwrap import BwrapSandboxActor
from .docker import DockerSandboxActor
from .local import LocalSandboxActor

# ``_resolve_auto_mode`` is re-exported so that ``workspace/execution.py`` reaches
# the probe through this package, exactly as it reaches the registry — a single
# import surface, and one attribute a test can replace.
from .registry import SANDBOX_ACTOR_CLASSES
from .registry import _resolve_auto_mode as _resolve_auto_mode
from .seatbelt import SeatbeltSandboxActor

__all__ = [
    "ALLOWED_COMMANDS",
    "BwrapSandboxActor",
    "CommandNotAllowedError",
    "CommandParseError",
    "DockerSandboxActor",
    "ExecReport",
    "ExecRequest",
    "ExecResult",
    "LocalSandboxActor",
    "SANDBOX_ACTOR_CLASSES",
    "SANDBOX_ACTOR_NAME",
    "SandboxActor",
    "SandboxConfig",
    "SandboxState",
    "SeatbeltSandboxActor",
    "sandbox_actor_name",
]

_EXEC_TOOL_REMOVED = (
    "ExecTool was removed from akgentic-tool: sandboxed execution is a capability of "
    "WorkspaceTool — use WorkspaceTool(workspace_exec=...) instead. It exposes the same "
    "execution as workspace_exec and workspace_exec_result, over the same sandbox "
    "backends, which still import from akgentic.tool.sandbox unchanged."
)


def __getattr__(name: str) -> Any:
    """Refuse ``ExecTool`` by name, and only it.

    A bare ``AttributeError: module has no attribute 'ExecTool'`` tells a reader
    nothing about where the capability went; an ``ImportError`` naming
    ``WorkspaceTool(workspace_exec=...)`` turns the break into an instruction.
    Every other missing name keeps the interpreter's ordinary ``AttributeError``
    — this is a pointer for one migration, not a catch-all. Runs only on a
    miss, so the import path's hot code is untouched.
    """
    if name == "ExecTool":
        raise ImportError(_EXEC_TOOL_REMOVED, name=__name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
