"""The sandbox backend registry, and the probe that picks one for ``mode="auto"``.

``SANDBOX_ACTOR_CLASSES`` is the package's extension point: a mutable ``dict`` a
deployment assigns its own :class:`SandboxActor` subclass into, before any card
is constructed. ``workspace_exec`` resolves through it **at call time** — when a
card is wired (``resolve_mode`` in ``workspace/execution.py``) and again on every
run (``#Workspace``'s sandbox resolution) — so a class registered after this
module was imported is still found.

Both names are reached through ``akgentic.tool.sandbox``, never through this
module directly. The package is the documented import surface; this file is
where the names happen to live, and may move again behind it.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from typing import Literal

from akgentic.tool.sandbox.actor import SandboxActor
from akgentic.tool.sandbox.bwrap import BwrapSandboxActor
from akgentic.tool.sandbox.docker import DockerSandboxActor
from akgentic.tool.sandbox.local import LocalSandboxActor
from akgentic.tool.sandbox.seatbelt import SeatbeltSandboxActor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SANDBOX_ACTOR_CLASSES — mutable injection window for runtime registration
# ---------------------------------------------------------------------------

SANDBOX_ACTOR_CLASSES: dict[str, type[SandboxActor]] = {
    "local": LocalSandboxActor,
    "bwrap": BwrapSandboxActor,
    "seatbelt": SeatbeltSandboxActor,
    "docker": DockerSandboxActor,
    # "e2b": E2BSandboxActor  ← injected by akgentic-infra at runtime
}


# ---------------------------------------------------------------------------
# Auto-mode resolution
# ---------------------------------------------------------------------------


def _seatbelt_available() -> bool:
    """Return True if sandbox-exec is on PATH and actually works at runtime.

    macOS 15+ may block ``sandbox_apply`` even when ``sandbox-exec`` is present.
    A quick probe with ``(allow default)`` detects this at negligible cost.
    """
    if shutil.which("sandbox-exec") is None or platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _resolve_auto_mode() -> Literal["local", "bwrap", "seatbelt", "docker"]:
    """Probe the host and return the best available sandbox backend.

    Probe order:
    1. ``bwrap`` on PATH → ``"bwrap"`` (Linux bubblewrap)
    2. ``sandbox-exec`` on PATH + Darwin → ``"seatbelt"`` (macOS)
    3. ``docker`` on PATH → ``"docker"``
    4. fallback → ``"local"`` (no filesystem isolation)

    Returns:
        String key matching an entry in SANDBOX_ACTOR_CLASSES.
    """
    if shutil.which("bwrap") is not None:
        logger.debug("_resolve_auto_mode: selected bwrap")
        return "bwrap"
    if _seatbelt_available():
        logger.debug("_resolve_auto_mode: selected seatbelt")
        return "seatbelt"
    if shutil.which("docker") is not None:
        logger.debug("_resolve_auto_mode: selected docker")
        return "docker"
    logger.debug("_resolve_auto_mode: fallback to local (no isolation backend found)")
    return "local"
