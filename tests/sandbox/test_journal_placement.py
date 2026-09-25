"""The sandbox cannot reach the workspace journal — placement, not the allowlist.

Story 29-5, AC10. ``git`` leaving ``ALLOWED_COMMANDS`` is defence in depth and
nothing more: only the *first token* of a command is checked and both ``bash``
and ``sh`` are on the list, so ``bash -c "git reset --hard"`` walks straight past
it. That is asserted here too, so nobody reads the removal as a boundary.

**The boundary is a filesystem fact.** The journal lives at the sibling
``<root>.git``, and the Docker container's mount names the workspace root and
nothing above or beside it. This test puts a journal directory next to the tree
and confirms it appears in no mount the backend constructs — a regression guard,
so that a later "just mount the parent" convenience cannot destroy a team's
history silently.

The placement rule is a property of the **mount**, so every spec here drives a
backend directly: the argv is built inside the strategy and there is no actor
in front of it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from akgentic.tool.sandbox.backend import ALLOWED_COMMANDS
from akgentic.tool.sandbox.docker import DockerBackend
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


class TestTheJournalIsOutsideEveryMount:
    def test_docker_mounts_the_root_and_nothing_beside_it(
        self, tree_with_journal: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The volume is built in start(), so it is asserted there rather than
        # on the exec argv.
        backend = DockerBackend()
        monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tree_with_journal.parent))
        monkeypatch.setattr(
            "akgentic.tool.sandbox.docker.shutil.which", lambda _cmd: "/usr/bin/docker"
        )
        monkeypatch.setattr(DockerBackend, "_ensure_image", lambda _self: None)

        seen: list[list[str]] = []

        def fake_run(argv: list[str], *args: Any, **kwargs: Any) -> Any:
            seen.append(list(argv))
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        monkeypatch.setattr("akgentic.tool.sandbox.docker.subprocess.run", fake_run)
        backend.start(tree_with_journal.name)

        run_argv = seen[-1]
        # EVERY bind mount, not the first one: the argv also carries the generated
        # /etc/passwd, so reading `index("-v")` would assert about whichever of the
        # two is written first. The invariant here is about the journal — no mount
        # may expose <root>.git or the parent that holds it — and it is only worth
        # anything if it is checked against all of them.
        volumes = [run_argv[index + 1] for index, token in enumerate(run_argv) if token == "-v"]
        workspace = [volume for volume in volumes if volume.endswith(":/workspace")]
        assert workspace == [f"{tree_with_journal}:/workspace"]
        for volume in volumes:
            assert str(git_dir_for(tree_with_journal)) not in volume
            assert str(tree_with_journal.parent) not in volume.split(":")[0] or volume in workspace


class TestTheAllowlistIsNotTheBoundary:
    def test_git_is_on_the_list_and_the_mount_is_what_protects_the_journal(self) -> None:
        # git was briefly taken off this list as defence in depth, and put back
        # when the allowlist was widened for real work. Nothing was lost, which
        # is the point of this class: the guarantee that holds is asserted by the
        # tests above — the journal lives at the sibling <root>.git, outside
        # the container's mount, so a `git reset --hard` from inside
        # the sandbox cannot reach it whether or not the binary is reachable.
        assert "git" in ALLOWED_COMMANDS

    def test_but_bash_walks_straight_past_it(self) -> None:
        # Only the first token is checked, and bash is on the list. Nothing in
        # this story may rely on the allowlist for safety.
        assert "bash" in ALLOWED_COMMANDS
        assert "sh" in ALLOWED_COMMANDS
