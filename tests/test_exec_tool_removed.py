"""``ExecTool`` is gone, the break says where to go, and the backend did not move.

Epic 41 removed the card outright rather than carrying it as a deprecated shim.
Three properties hold the removal together, and each is pinned here on both
import paths the name ever had:

- ``from akgentic.tool import ExecTool`` and ``from akgentic.tool.sandbox import
  ExecTool`` both fail, and neither package's ``__all__`` lists the name.
- The failure is an ``ImportError`` naming ``WorkspaceTool(workspace_exec=...)``
  — an instruction, not a puzzle. Any *other* missing name keeps the
  interpreter's ordinary ``AttributeError``: the refusal is for one migration,
  not a catch-all.
- The sandbox **backend**'s import surface resolves: every name the README
  documents resolves from ``akgentic.tool.sandbox`` to the very object its
  module defines. It has since narrowed to Docker alone — the
  ``SANDBOX_BACKEND`` slot replaced the registry — and the three removed
  backends are refused with an explanation of their own, pinned below.
"""

from __future__ import annotations

import importlib
import re

import pytest

import akgentic.tool
import akgentic.tool.sandbox

REPLACEMENT = "WorkspaceTool(workspace_exec="
"""What the message must name — the call a reader should write instead."""

PACKAGES = ["akgentic.tool", "akgentic.tool.sandbox"]


def _from_import(module: str, name: str) -> object:
    """Run ``from <module> import <name>`` exactly as a consumer would write it."""
    namespace: dict[str, object] = {}
    exec(f"from {module} import {name}", namespace)
    return namespace[name]


class TestTheNameIsGone:
    @pytest.mark.parametrize("module", PACKAGES)
    def test_all_does_not_list_it(self, module: str) -> None:
        assert "ExecTool" not in importlib.import_module(module).__all__

    def test_no_module_under_the_package_defines_it(self) -> None:
        """The class is not merely un-exported; it does not exist anywhere."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("akgentic.tool.sandbox.tool")


class TestTheBreakNamesItsReplacement:
    @pytest.mark.parametrize("module", PACKAGES)
    def test_from_import_raises_an_import_error_naming_workspace_exec(self, module: str) -> None:
        with pytest.raises(ImportError) as caught:
            _from_import(module, "ExecTool")
        assert REPLACEMENT in str(caught.value)

    @pytest.mark.parametrize("module", PACKAGES)
    def test_attribute_access_raises_the_same_import_error(self, module: str) -> None:
        """``pkg.ExecTool`` takes the module ``__getattr__`` path directly, no conversion."""
        package = importlib.import_module(module)
        with pytest.raises(ImportError) as caught:
            _ = package.ExecTool
        assert REPLACEMENT in str(caught.value)

    @pytest.mark.parametrize("module", PACKAGES)
    def test_the_message_names_no_release(self, module: str) -> None:
        """No version number: the message must stay true across every later release."""
        package = importlib.import_module(module)
        with pytest.raises(ImportError) as caught:
            _ = package.ExecTool
        assert not re.search(r"\d+\.\d+", str(caught.value)), str(caught.value)

    def test_both_paths_raise_one_message(self) -> None:
        """The root delegates to the sandbox package, so the two cannot drift apart."""
        messages = []
        for module in PACKAGES:
            with pytest.raises(ImportError) as caught:
                _ = importlib.import_module(module).ExecTool
            messages.append(str(caught.value))
        assert messages[0] == messages[1]


class TestOtherMissingNamesAreUnaffected:
    @pytest.mark.parametrize("module", PACKAGES)
    def test_an_unrelated_missing_attribute_keeps_the_ordinary_error(self, module: str) -> None:
        package = importlib.import_module(module)
        with pytest.raises(AttributeError) as caught:
            _ = package.does_not_exist
        assert "does_not_exist" in str(caught.value)
        assert REPLACEMENT not in str(caught.value)

    @pytest.mark.parametrize("module", PACKAGES)
    def test_an_unrelated_from_import_is_the_interpreters_own_conversion(self, module: str) -> None:
        """CPython turns the ``AttributeError`` into a plain "cannot import name"."""
        with pytest.raises(ImportError) as caught:
            _from_import(module, "does_not_exist")
        assert "cannot import name" in str(caught.value)
        assert REPLACEMENT not in str(caught.value)


class TestTheBackendSurfaceDidNotMove:
    """Every documented name resolves from the package to the object its module defines."""

    @pytest.mark.parametrize(
        ("name", "home"),
        [
            ("SANDBOX_BACKEND", "akgentic.tool.sandbox.registry"),
            ("ALLOWED_COMMANDS", "akgentic.tool.sandbox.backend"),
            ("SandboxBackend", "akgentic.tool.sandbox.backend"),
            ("ProcessBackend", "akgentic.tool.sandbox.backend"),
            ("CommandNotAllowedError", "akgentic.tool.sandbox.backend"),
            ("CommandParseError", "akgentic.tool.sandbox.backend"),
            ("ExecResult", "akgentic.tool.sandbox.backend"),
            ("ExecReport", "akgentic.tool.sandbox.backend"),
            ("validate_command", "akgentic.tool.sandbox.backend"),
            ("DockerBackend", "akgentic.tool.sandbox.docker"),
        ],
    )
    def test_the_documented_name_resolves_to_its_defining_object(
        self, name: str, home: str
    ) -> None:
        resolved = _from_import("akgentic.tool.sandbox", name)
        assert resolved is getattr(importlib.import_module(home), name)
        assert name in akgentic.tool.sandbox.__all__

    def test_every_exported_name_resolves_and_the_actor_names_are_gone(self) -> None:
        """``__all__`` lists nothing that does not resolve, and none of the retired names.

        The first half is what makes the second mean something: a name could be
        absent from ``__all__`` and still importable, so the retired names are
        also asserted to raise on the ``from`` form.
        """
        for name in akgentic.tool.sandbox.__all__:
            assert _from_import("akgentic.tool.sandbox", name) is not None
        retired = {
            "SandboxActor",
            "SandboxConfig",
            "SandboxState",
            "SANDBOX_ACTOR_CLASSES",
            "SANDBOX_ACTOR_NAME",
            "sandbox_actor_name",
            "LocalSandboxActor",
            "BwrapSandboxActor",
            "SeatbeltSandboxActor",
            "DockerSandboxActor",
            "ExecRequest",
            "SANDBOX_BACKEND_CLASSES",
            "_resolve_auto_mode",
        }
        assert not retired & set(akgentic.tool.sandbox.__all__)
        assert not retired & set(akgentic.tool.__all__)
        for name in retired:
            with pytest.raises(ImportError):
                _from_import("akgentic.tool.sandbox", name)


REMOVED_BACKENDS = ["LocalBackend", "BwrapBackend", "SeatbeltBackend"]
"""The host-process backends the sandbox no longer ships."""


class TestTheRemovedBackendsAreRefusedWithAnExplanation:
    """The three removed backends fail with an ``ImportError`` naming the slot.

    A deployment that imported one of them reads why it is gone — the sandbox is
    Docker-only — and where the choice of executor went: the
    ``SANDBOX_BACKEND`` slot. Any other missing name keeps the interpreter's own
    error, ``SANDBOX_BACKEND_CLASSES`` included.
    """

    @pytest.mark.parametrize("name", REMOVED_BACKENDS)
    def test_the_from_import_raises_an_import_error_naming_docker_and_the_slot(
        self, name: str
    ) -> None:
        with pytest.raises(ImportError) as caught:
            _from_import("akgentic.tool.sandbox", name)
        assert "Docker" in str(caught.value)
        assert "SANDBOX_BACKEND" in str(caught.value)

    @pytest.mark.parametrize("name", REMOVED_BACKENDS)
    def test_attribute_access_raises_the_same_import_error(self, name: str) -> None:
        with pytest.raises(ImportError) as caught:
            getattr(akgentic.tool.sandbox, name)
        assert "Docker" in str(caught.value)
        assert "SANDBOX_BACKEND" in str(caught.value)
        assert name in str(caught.value)

    @pytest.mark.parametrize("name", REMOVED_BACKENDS)
    def test_the_message_names_no_release(self, name: str) -> None:
        with pytest.raises(ImportError) as caught:
            getattr(akgentic.tool.sandbox, name)
        assert not re.search(r"\d+\.\d+", str(caught.value)), str(caught.value)

    @pytest.mark.parametrize("name", REMOVED_BACKENDS)
    def test_all_does_not_list_it(self, name: str) -> None:
        assert name not in akgentic.tool.sandbox.__all__

    def test_the_exec_tool_message_is_unchanged(self) -> None:
        with pytest.raises(ImportError) as caught:
            _ = akgentic.tool.sandbox.ExecTool
        assert str(caught.value) == (
            "ExecTool was removed from akgentic-tool: sandboxed execution is a capability of "
            "WorkspaceTool — use WorkspaceTool(workspace_exec=...) instead. It exposes the same "
            "execution as workspace_exec and workspace_exec_result, over the same sandbox "
            "backends, which still import from akgentic.tool.sandbox unchanged."
        )

    def test_the_retired_registry_keeps_the_ordinary_error(self) -> None:
        with pytest.raises(AttributeError):
            _ = akgentic.tool.sandbox.SANDBOX_BACKEND_CLASSES
        with pytest.raises(ImportError) as caught:
            _from_import("akgentic.tool.sandbox", "SANDBOX_BACKEND_CLASSES")
        assert "cannot import name" in str(caught.value)
        assert "Docker" not in str(caught.value)
