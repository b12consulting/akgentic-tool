"""The sandbox backend registry and the ``auto`` probe, on their own.

These used to sit in ``tests/sandbox/test_exec_tool.py`` beside the card that
resolved through them. The card is gone (epic 41); the registry and the probe
are not deprecated and keep their coverage here, where a reader looking for the
backend's tests will look.

- ``SANDBOX_ACTOR_CLASSES`` holds the four shipped backends under their keys
  and is a plain mutable ``dict`` — the injection window a deployment writes
  its own backend into.
- ``_resolve_auto_mode`` probes ``bwrap`` → ``seatbelt`` → ``docker`` →
  ``local``, in that order, and the seatbelt probe is a runtime check rather
  than a PATH lookup.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from akgentic.tool.sandbox import (
    SANDBOX_ACTOR_CLASSES,
    BwrapSandboxActor,
    DockerSandboxActor,
    LocalSandboxActor,
    SeatbeltSandboxActor,
    _resolve_auto_mode,
)
from akgentic.tool.sandbox.registry import _seatbelt_available

# ---------------------------------------------------------------------------
# SANDBOX_ACTOR_CLASSES — the registry
# ---------------------------------------------------------------------------


def test_sandbox_actor_classes_has_local_key() -> None:
    """SANDBOX_ACTOR_CLASSES['local'] maps to LocalSandboxActor."""
    assert "local" in SANDBOX_ACTOR_CLASSES
    assert SANDBOX_ACTOR_CLASSES["local"] is LocalSandboxActor


def test_sandbox_actor_classes_has_docker_key() -> None:
    """SANDBOX_ACTOR_CLASSES['docker'] maps to DockerSandboxActor."""
    assert "docker" in SANDBOX_ACTOR_CLASSES
    assert SANDBOX_ACTOR_CLASSES["docker"] is DockerSandboxActor


def test_sandbox_actor_classes_has_bwrap_key() -> None:
    """Story 8.4: SANDBOX_ACTOR_CLASSES['bwrap'] maps to BwrapSandboxActor."""
    assert "bwrap" in SANDBOX_ACTOR_CLASSES
    assert SANDBOX_ACTOR_CLASSES["bwrap"] is BwrapSandboxActor


def test_sandbox_actor_classes_has_seatbelt_key() -> None:
    """Story 8.4: SANDBOX_ACTOR_CLASSES['seatbelt'] maps to SeatbeltSandboxActor."""
    assert "seatbelt" in SANDBOX_ACTOR_CLASSES
    assert SANDBOX_ACTOR_CLASSES["seatbelt"] is SeatbeltSandboxActor


def test_sandbox_actor_classes_is_mutable_dict() -> None:
    """SANDBOX_ACTOR_CLASSES is a regular dict (mutable — injection window)."""
    assert isinstance(SANDBOX_ACTOR_CLASSES, dict)


def test_the_registry_holds_exactly_the_four_shipped_backends() -> None:
    """Nothing is registered by import alone; a fifth key is a deployment's doing."""
    assert set(SANDBOX_ACTOR_CLASSES) == {"local", "bwrap", "seatbelt", "docker"}


# ---------------------------------------------------------------------------
# _resolve_auto_mode() — probe order
# ---------------------------------------------------------------------------


def test_resolve_auto_mode_returns_bwrap_when_bwrap_on_path() -> None:
    """_resolve_auto_mode() returns 'bwrap' when bwrap is on PATH."""
    with patch("akgentic.tool.sandbox.registry.shutil.which", return_value="/usr/bin/bwrap"):
        result = _resolve_auto_mode()
    assert result == "bwrap"


def test_resolve_auto_mode_returns_seatbelt_on_darwin_without_bwrap() -> None:
    """_resolve_auto_mode() returns 'seatbelt' on Darwin when sandbox-exec works."""

    def which_side_effect(cmd: str) -> str | None:
        return {
            "bwrap": None,
            "sandbox-exec": "/usr/bin/sandbox-exec",
            "docker": None,
        }.get(cmd)

    mock_probe = MagicMock(returncode=0)
    with (
        patch("akgentic.tool.sandbox.registry.shutil.which", side_effect=which_side_effect),
        patch("akgentic.tool.sandbox.registry.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.registry.subprocess.run", return_value=mock_probe),
    ):
        result = _resolve_auto_mode()
    assert result == "seatbelt"


def test_resolve_auto_mode_skips_seatbelt_when_probe_fails() -> None:
    """_resolve_auto_mode() falls through to docker/local when sandbox-exec probe fails."""

    def which_side_effect(cmd: str) -> str | None:
        return {
            "bwrap": None,
            "sandbox-exec": "/usr/bin/sandbox-exec",
            "docker": "/usr/bin/docker",
        }.get(cmd)

    mock_probe = MagicMock(returncode=71)  # Operation not permitted
    with (
        patch("akgentic.tool.sandbox.registry.shutil.which", side_effect=which_side_effect),
        patch("akgentic.tool.sandbox.registry.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.registry.subprocess.run", return_value=mock_probe),
    ):
        result = _resolve_auto_mode()
    assert result == "docker"


def test_resolve_auto_mode_returns_docker_when_docker_on_path() -> None:
    """_resolve_auto_mode() returns 'docker' when docker on PATH, no bwrap/seatbelt."""

    def which_side_effect(cmd: str) -> str | None:
        return {
            "bwrap": None,
            "sandbox-exec": None,
            "docker": "/usr/bin/docker",
        }.get(cmd)

    with (
        patch("akgentic.tool.sandbox.registry.shutil.which", side_effect=which_side_effect),
        patch("akgentic.tool.sandbox.registry._seatbelt_available", return_value=False),
    ):
        result = _resolve_auto_mode()
    assert result == "docker"


def test_resolve_auto_mode_returns_local_when_nothing_found() -> None:
    """_resolve_auto_mode() returns 'local' when no backends found."""
    with (
        patch("akgentic.tool.sandbox.registry.shutil.which", return_value=None),
        patch("akgentic.tool.sandbox.registry._seatbelt_available", return_value=False),
    ):
        result = _resolve_auto_mode()
    assert result == "local"


# ---------------------------------------------------------------------------
# _seatbelt_available() — a runtime probe, not a PATH check
# ---------------------------------------------------------------------------


def test_seatbelt_is_unavailable_off_darwin_even_with_the_binary_on_path() -> None:
    """The binary alone proves nothing — Seatbelt is a macOS facility."""
    with (
        patch("akgentic.tool.sandbox.registry.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.registry.platform.system", return_value="Linux"),
        patch("akgentic.tool.sandbox.registry.subprocess.run") as run,
    ):
        assert _seatbelt_available() is False
    run.assert_not_called()


def test_seatbelt_is_unavailable_when_the_probe_hangs() -> None:
    """A probe that times out is a backend that cannot be trusted to run anything."""
    with (
        patch("akgentic.tool.sandbox.registry.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.registry.platform.system", return_value="Darwin"),
        patch(
            "akgentic.tool.sandbox.registry.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="sandbox-exec", timeout=5),
        ),
    ):
        assert _seatbelt_available() is False


def test_seatbelt_is_unavailable_when_the_probe_cannot_start() -> None:
    """An ``OSError`` from the probe — a binary present but not executable — is a no."""
    with (
        patch("akgentic.tool.sandbox.registry.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.registry.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.registry.subprocess.run", side_effect=OSError("denied")),
    ):
        assert _seatbelt_available() is False
