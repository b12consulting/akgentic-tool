"""The sandbox cannot reach the workspace journal — placement, not the allowlist.

Story 29-5, AC10. ``git`` leaving ``ALLOWED_COMMANDS`` is defence in depth and
nothing more: only the *first token* of a command is checked and both ``bash``
and ``sh`` are on the list, so ``bash -c "git reset --hard"`` walks straight past
it. That is asserted here too, so nobody reads the removal as a boundary.

**The boundary is a filesystem fact.** The journal lives at the sibling
``<root>.git``, and every backend that constructs a mount names the workspace
root and nothing above or beside it. These tests put a journal directory next to
the tree and confirm it appears in no constructed argument and no rendered
policy — a regression guard, so that a later "just mount the parent"
convenience cannot destroy a team's history silently.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from akgentic.tool.sandbox.actor import ALLOWED_COMMANDS, SandboxConfig, SandboxState
from akgentic.tool.sandbox.bwrap import BwrapSandboxActor
from akgentic.tool.sandbox.docker import DockerSandboxActor
from akgentic.tool.sandbox.local import LocalSandboxActor
from akgentic.tool.sandbox.seatbelt import SeatbeltSandboxActor
from akgentic.tool.workspace.journal import git_dir_for


@pytest.fixture
def tree_with_journal(tmp_path: Path) -> Path:
    """A workspace root with a populated journal directory as its sibling."""
    root = (tmp_path / "team-1").resolve()
    root.mkdir()
    journal = git_dir_for(root)
    journal.mkdir()
    (journal / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    return root


def captured_argv(
    actor_class: type[Any], root: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Run one command through *actor_class* and return the argv it built."""
    actor = actor_class()
    actor.config = SandboxConfig(name="#SandboxActor", role="ToolActor", team_id="team-1")
    actor.state = SandboxState()
    actor.state.observer(actor)
    actor.state.workspace_path = root
    actor.state.container_name = "sandbox-team-1"

    seen: list[list[str]] = []

    def fake_run(argv: list[str], *args: Any, **kwargs: Any) -> Any:
        seen.append(list(argv))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    module = actor_class.__module__.rsplit(".", 1)[-1]
    monkeypatch.setattr(f"akgentic.tool.sandbox.{module}.subprocess.run", fake_run)
    actor._exec("echo hi", "", 1.0)
    assert len(seen) == 1
    return seen[0]


class TestTheJournalIsOutsideEveryMount:
    def test_bwrap_binds_the_root_and_nothing_beside_it(
        self, tree_with_journal: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        argv = captured_argv(BwrapSandboxActor, tree_with_journal, monkeypatch)

        assert argv[argv.index("--bind") + 1] == str(tree_with_journal)
        assert str(git_dir_for(tree_with_journal)) not in argv
        assert str(tree_with_journal.parent) not in argv

    def test_docker_mounts_the_root_and_nothing_beside_it(
        self, tree_with_journal: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The volume is built in _start_sandbox, so it is asserted there rather
        # than on the exec argv.
        actor = DockerSandboxActor()
        actor.config = SandboxConfig(
            name="#SandboxActor", role="ToolActor", team_id=tree_with_journal.name
        )
        actor.state = SandboxState()
        actor.state.observer(actor)
        monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tree_with_journal.parent))
        monkeypatch.setattr(
            "akgentic.tool.sandbox.docker.shutil.which", lambda _cmd: "/usr/bin/docker"
        )
        monkeypatch.setattr(DockerSandboxActor, "_ensure_image", lambda _self: None)

        seen: list[list[str]] = []

        def fake_run(argv: list[str], *args: Any, **kwargs: Any) -> Any:
            seen.append(list(argv))
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        monkeypatch.setattr("akgentic.tool.sandbox.docker.subprocess.run", fake_run)
        actor._start_sandbox()

        run_argv = seen[-1]
        volume = run_argv[run_argv.index("-v") + 1]
        assert volume == f"{tree_with_journal}:/workspace"
        assert str(git_dir_for(tree_with_journal)) not in volume

    def test_the_seatbelt_policy_makes_only_the_root_writable(
        self, tree_with_journal: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rendered: list[str] = []
        actor = SeatbeltSandboxActor()
        actor.config = SandboxConfig(name="#SandboxActor", role="ToolActor", team_id="team-1")
        actor.state = SandboxState()
        actor.state.observer(actor)
        actor.state.workspace_path = tree_with_journal

        import tempfile  # noqa: PLC0415

        real_named = tempfile.NamedTemporaryFile

        def capturing_named(*args: Any, **kwargs: Any) -> Any:
            handle = real_named(*args, **kwargs)
            real_write_method = handle.write

            def write(text: str) -> int:
                rendered.append(text)
                return real_write_method(text)

            handle.write = write  # type: ignore[method-assign]
            return handle

        monkeypatch.setattr(
            "akgentic.tool.sandbox.seatbelt.tempfile.NamedTemporaryFile", capturing_named
        )
        monkeypatch.setattr(
            "akgentic.tool.sandbox.seatbelt.subprocess.run",
            lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0),
        )
        actor._exec("echo hi", "", 1.0)

        policy = rendered[0]
        assert f'(allow file-write* (subpath "{tree_with_journal}"))' in policy
        journal = str(git_dir_for(tree_with_journal))
        write_rules = [line for line in policy.splitlines() if "file-write" in line]
        assert not any(journal in rule for rule in write_rules)

    def test_local_provides_no_isolation_and_this_test_says_so(
        self, tree_with_journal: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stated rather than pretended: LocalSandboxActor runs a plain
        # subprocess with a cwd, so the journal beside the tree is reachable
        # from it exactly as any other path on the host is. It is a development
        # convenience, not a boundary, and the class docstring says so.
        argv = captured_argv(LocalSandboxActor, tree_with_journal, monkeypatch)

        assert argv == ["echo", "hi"]


class TestTheAllowlistIsNotTheBoundary:
    def test_git_is_on_the_list_and_the_mount_is_what_protects_the_journal(self) -> None:
        # git was briefly taken off this list as defence in depth, and put back
        # when the allowlist was widened for real work. Nothing was lost, which
        # is the point of this class: the guarantee that holds is asserted by the
        # tests above — the journal lives at the sibling <root>.git, outside
        # every isolating backend's mount, so a `git reset --hard` from inside
        # the sandbox cannot reach it whether or not the binary is reachable.
        assert "git" in ALLOWED_COMMANDS

    def test_but_bash_walks_straight_past_it(self) -> None:
        # Only the first token is checked, and bash is on the list. Nothing in
        # this story may rely on the allowlist for safety.
        assert "bash" in ALLOWED_COMMANDS
        assert "sh" in ALLOWED_COMMANDS
