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
- The sandbox **backend**'s import surface is byte-identical: every name the
  README documents still resolves from ``akgentic.tool.sandbox`` to the very
  object its module defines.
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
            ("SANDBOX_ACTOR_CLASSES", "akgentic.tool.sandbox.registry"),
            ("ALLOWED_COMMANDS", "akgentic.tool.sandbox.actor"),
            ("SandboxActor", "akgentic.tool.sandbox.actor"),
            ("CommandNotAllowedError", "akgentic.tool.sandbox.actor"),
            ("DockerSandboxActor", "akgentic.tool.sandbox.docker"),
            ("BwrapSandboxActor", "akgentic.tool.sandbox.bwrap"),
            ("SeatbeltSandboxActor", "akgentic.tool.sandbox.seatbelt"),
            ("LocalSandboxActor", "akgentic.tool.sandbox.local"),
        ],
    )
    def test_the_documented_name_resolves_to_its_defining_object(
        self, name: str, home: str
    ) -> None:
        resolved = _from_import("akgentic.tool.sandbox", name)
        assert resolved is getattr(importlib.import_module(home), name)
        assert name in akgentic.tool.sandbox.__all__

    def test_the_root_still_re_exports_the_two_platform_backends(self) -> None:
        assert akgentic.tool.BwrapSandboxActor is akgentic.tool.sandbox.BwrapSandboxActor
        assert akgentic.tool.SeatbeltSandboxActor is akgentic.tool.sandbox.SeatbeltSandboxActor
