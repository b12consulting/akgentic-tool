"""Sandbox submodule — the exec backend ``workspace_exec`` runs on.

The sandbox is Docker, and it has no mode. :class:`DockerBackend` — a plain
strategy behind the :class:`SandboxBackend` Protocol — is installed at the
``SANDBOX_BACKEND`` slot, and ``#Workspace`` reads that slot through this
package when it builds its runner, then runs the backend on its own worker
thread. There is no host probe and no fallback: a host without Docker fails the
first command. A deployment that needs a different executor assigns its own
class to ``akgentic.tool.sandbox.SANDBOX_BACKEND`` in its wiring.

The card that once wrapped the backends, ``ExecTool``, is gone: sandboxed
execution is a capability of ``WorkspaceTool`` — ``WorkspaceTool(workspace_exec=...)``.
So are the host-process backends ``LocalBackend``, ``BwrapBackend`` and
``SeatbeltBackend``. :func:`__getattr__` below says so to anybody still
importing one of those names.
"""

from __future__ import annotations

from typing import Any

from .backend import (
    ALLOWED_COMMANDS,
    CommandNotAllowedError,
    CommandParseError,
    ExecReport,
    ExecResult,
    ProcessBackend,
    SandboxBackend,
    validate_command,
)
from .docker import DockerBackend
from .registry import SANDBOX_BACKEND

# The sandbox actor and its two persisted models, ``SandboxConfig`` and
# ``SandboxState``, used to be exported from here and are deleted, not
# tombstoned. Any persisted sandbox start event or checkpoint carrying their
# ``__model__`` tag is now unreadable — core's deserializer imports the tagged
# class before its guarded construction, so a stale tag fails the whole record
# rather than dropping a field. Accepted: the only affected records were the
# decision maker's own test teams, and agents, files and the UI are untouched.
# This exemption does not extend to the next removal — once anything is
# released and adopted, a persisted model is a migration.

__all__ = [
    "ALLOWED_COMMANDS",
    "CommandNotAllowedError",
    "CommandParseError",
    "DockerBackend",
    "ExecReport",
    "ExecResult",
    "ProcessBackend",
    "SANDBOX_BACKEND",
    "SandboxBackend",
    "validate_command",
]

_EXEC_TOOL_REMOVED = (
    "ExecTool was removed from akgentic-tool: sandboxed execution is a capability of "
    "WorkspaceTool — use WorkspaceTool(workspace_exec=...) instead. It exposes the same "
    "execution as workspace_exec and workspace_exec_result, over the same sandbox "
    "backends, which still import from akgentic.tool.sandbox unchanged."
)

_REMOVED_BACKENDS: frozenset[str] = frozenset({"LocalBackend", "BwrapBackend", "SeatbeltBackend"})
"""The host-process backends the sandbox no longer ships."""


def _backend_removed(name: str) -> str:
    """The refusal for one removed backend *name*: where the choice of executor went."""
    return (
        f"{name} was removed from akgentic-tool: the sandbox is Docker-only, and every "
        "exec-capable card runs its commands on DockerBackend, with no mode and no "
        "fallback. A deployment that needs a different executor assigns its own "
        "SandboxBackend class to akgentic.tool.sandbox.SANDBOX_BACKEND in its wiring."
    )


def __getattr__(name: str) -> Any:
    """Refuse ``ExecTool`` and the three removed backends by name, and only them.

    A bare ``AttributeError: module has no attribute 'ExecTool'`` tells a reader
    nothing about where the capability went; an ``ImportError`` naming
    ``WorkspaceTool(workspace_exec=...)`` turns the break into an instruction.
    ``LocalBackend``, ``BwrapBackend`` and ``SeatbeltBackend`` get the same
    treatment: the message says the sandbox is Docker-only and names the
    ``SANDBOX_BACKEND`` slot a deployment assigns another executor to. Every
    other missing name keeps the interpreter's ordinary ``AttributeError`` —
    these are pointers for removed names, not a catch-all. Runs only on a miss,
    so the import path's hot code is untouched.
    """
    if name == "ExecTool":
        raise ImportError(_EXEC_TOOL_REMOVED, name=__name__)
    if name in _REMOVED_BACKENDS:
        raise ImportError(_backend_removed(name), name=__name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
