"""The human runs the workspace from slash commands.

``/workspace_mkdir``, ``/workspace_rmdir``, ``/workspace_delete``,
``/workspace_write``, ``/workspace_read``, ``/workspace_list``, ``/workspace_exec``
and ``/workspace_exec_result`` are served on the ``COMMAND`` channel and dispatched
by the agent's ``CommandRegistry``. The commands run on the agent's own card, so
the property that matters most is the negative one: **the human's change is never
the agent's view.** Each write-side command goes through the gate's human mode —
no view check, no observation recorded, the commit attributed to ``human``.
"""

from __future__ import annotations

import shlex
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

import akgentic.tool.sandbox
from akgentic.tool.core import COMMAND, TOOL_CALL, CommandRegistry, ToolFactory
from akgentic.tool.errors import RetriableError
from akgentic.tool.sandbox.backend import ALLOWED_COMMANDS
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.card import WorkspaceTool as CardWorkspaceTool
from akgentic.tool.workspace.lock import LockTicket, exec_busy, mutation_busy
from akgentic.tool.workspace.locks import LOCKS_DIR_NAME
from akgentic.tool.workspace.read.params import WorkspaceList, WorkspaceRead
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool
from akgentic.tool.workspace.workspace import meta_dir_for
from akgentic.tool.workspace.write.gate import lock_file_for
from akgentic.tool.workspace.write.params import (
    WorkspaceDelete,
    WorkspaceMkdir,
    WorkspaceRmdir,
    WorkspaceWrite,
)
from tests.sandbox._process_backend import LocalBackend
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    ExecHarness,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    SandboxScript,
    card_for,
    exec_card_for,
    journal_log,
    mutate,
    read,
    requires_git,
    tool_named,
)

WRITE_COMMANDS = {
    "workspace_write",
    "workspace_delete",
    "workspace_mkdir",
    "workspace_rmdir",
}
READ_COMMANDS = {"workspace_read", "workspace_list"}
EXEC_COMMANDS = {"workspace_exec", "workspace_exec_result"}


def registry(card: WorkspaceTool, **factory_kwargs: Any) -> CommandRegistry:
    """The agent's command registry over *card* alone, as ``BaseAgent`` builds it."""
    return ToolFactory(tool_cards=[card], **factory_kwargs).get_command_registry()


def run(card: WorkspaceTool, text: str) -> str | None:
    """Dispatch *text* exactly as ``BaseAgent._dispatch_command`` does."""
    return registry(card).dispatch(text)


def command_names(card: WorkspaceTool) -> set[str]:
    """The workspace commands *card* serves, by ``__name__``."""
    return {fn.__name__ for fn in card.get_commands().values()} - {
        "_expand_media_refs",
    }


def populate(tree: Path, files: dict[str, str]) -> None:
    for relative, body in files.items():
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def lock_files() -> set[str]:
    locks = meta_dir_for(WORKSPACE_PATH) / LOCKS_DIR_NAME
    return {entry.name for entry in locks.iterdir()} if locks.exists() else set()


class _RecordingCache:
    def __init__(self) -> None:
        self.marked: list[list[str]] = []

    def mark_paths_stale(self, paths: list[str]) -> None:
        self.marked.append(list(paths))


def hold_the_tree(card: WorkspaceTool) -> str:
    """Take the exec hold on *card*'s tree, as another agent's run would; return its id."""
    backend = card._lock_backend
    assert backend is not None
    grant = backend.acquire(
        card._workspace_path,
        LockTicket(agent_id="agent-9", agent_name="builder", cmd="make", budget_s=60.0),
    )
    assert grant.run_id
    return grant.run_id


def _wired(observer: FakeActorToolObserver, **fields: Any) -> WorkspaceTool:
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, **fields)
    card.observer(observer)
    return card


# ---------------------------------------------------------------------------
# B1 — the eight commands, their signatures and their descriptions
# ---------------------------------------------------------------------------


@pytest.fixture
def exec_card(
    orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
) -> Generator[WorkspaceTool, None, None]:
    card, observer = exec_card_for(orchestrator_proxy)
    yield card
    del observer


class TestTheCommandSurface:
    EXPECTED = {
        "workspace_mkdir": [("path", True)],
        "workspace_rmdir": [("path", True)],
        "workspace_delete": [("filepath", True)],
        "workspace_write": [("filepath", True), ("content", True)],
        "workspace_read": [("filepath", True)],
        "workspace_list": [("path", False), ("depth", False)],
        "workspace_exec": [("cmd", True)],
        "workspace_exec_result": [("run_id", True)],
    }

    def test_the_descriptors_name_each_command_and_its_arguments(
        self, exec_card: WorkspaceTool
    ) -> None:
        descriptors = {d.name: d for d in registry(exec_card).descriptors()}
        for name, args in self.EXPECTED.items():
            assert [(arg.name, arg.required) for arg in descriptors[name].args] == args, name

    def test_every_description_is_for_a_human_and_carries_an_example(
        self, exec_card: WorkspaceTool
    ) -> None:
        descriptors = {d.name: d for d in registry(exec_card).descriptors()}
        for name in self.EXPECTED:
            assert f"Example: /{name}" in descriptors[name].description, name

    def test_list_depth_defaults_to_the_configured_max_depth(
        self, observer: FakeActorToolObserver, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a/b.md": "b\n"})
        card = _wired(observer, workspace_list=WorkspaceList(max_depth=0))
        assert run(card, "/workspace_list docs") == tool_named(card, "workspace_list")("docs", 0)

    def test_the_commands_survive_the_retry_wrapper(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # ``BaseAgent`` builds its registry with a retry exception, which wraps
        # every command; the rest-of-line marker must still be found through it.
        wrapped = registry(wired_card, retry_exception=RuntimeError)
        assert wrapped.dispatch('/workspace_write q.md say "hi"') == "Written: q.md (8 bytes)"
        assert (workspace_tree / "q.md").read_text(encoding="utf-8") == 'say "hi"'


# ---------------------------------------------------------------------------
# B2, B3 — which commands a card serves
# ---------------------------------------------------------------------------


class TestWhichCommandsAreServed:
    def test_command_is_on_by_default_for_the_seven_params(self) -> None:
        for param in (
            WorkspaceMkdir,
            WorkspaceRmdir,
            WorkspaceDelete,
            WorkspaceWrite,
            WorkspaceRead,
            WorkspaceList,
            WorkspaceExec,
        ):
            assert param().expose == {TOOL_CALL, COMMAND}, param.__name__

    def test_a_default_card_serves_the_file_commands_and_no_exec(
        self, wired_card: WorkspaceTool
    ) -> None:
        assert command_names(wired_card) == WRITE_COMMANDS | READ_COMMANDS

    def test_an_exec_card_serves_the_exec_pair_too(self, exec_card: WorkspaceTool) -> None:
        assert command_names(exec_card) == WRITE_COMMANDS | READ_COMMANDS | EXEC_COMMANDS

    def test_a_persisted_param_on_tool_call_alone_serves_no_command(
        self, observer: FakeActorToolObserver, workspace_tree: Path
    ) -> None:
        stored = WorkspaceTool(workspace_id=WORKSPACE_NAME).model_dump()
        stored["workspace_write"] = {**WorkspaceWrite().model_dump(), "expose": ["tool_call"]}
        card = WorkspaceTool.model_validate(stored)
        card.observer(observer)
        assert "workspace_write" not in command_names(card)
        assert "workspace_write" in {tool.__name__ for tool in card.get_tools()}

    def test_command_alone_serves_the_command_and_no_tool(
        self, observer: FakeActorToolObserver, workspace_tree: Path
    ) -> None:
        card = _wired(observer, workspace_write=WorkspaceWrite(expose={COMMAND}))
        assert "workspace_write" in command_names(card)
        assert "workspace_write" not in {tool.__name__ for tool in card.get_tools()}

    def test_exec_on_command_alone_serves_the_pair_and_no_tool(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME, workspace_exec=WorkspaceExec(expose={COMMAND})
        )
        observer = FakeActorToolObserver(orchestrator_proxy)
        card.observer(observer)
        assert EXEC_COMMANDS <= command_names(card)
        assert not EXEC_COMMANDS & {tool.__name__ for tool in card.get_tools()}

    def test_read_only_serves_the_read_side_only(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, read_only=True, workspace_exec=True)
        observer = FakeActorToolObserver(orchestrator_proxy)
        card.observer(observer)
        assert command_names(card) == READ_COMMANDS

    def test_a_field_set_to_false_serves_neither_tool_nor_command(
        self, observer: FakeActorToolObserver, workspace_tree: Path
    ) -> None:
        card = _wired(observer, workspace_delete=False, workspace_read=False)
        assert "workspace_delete" not in command_names(card)
        assert "workspace_read" not in command_names(card)
        tools = {tool.__name__ for tool in card.get_tools()}
        assert "workspace_delete" not in tools
        assert "workspace_read" not in tools

    def test_two_workspace_tools_on_one_agent_collide(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # Pinned, not fixed: the registry keys commands by name, and two cards
        # serving one name is a wiring-time error. ``/expand_media_refs`` already
        # collided before these commands existed.
        first, _first_observer = card_for(orchestrator_proxy, "alice")
        second, _second_observer = card_for(orchestrator_proxy, "alice", workspace_id="other")
        with pytest.raises(ValueError, match="Command name collision"):
            ToolFactory(tool_cards=[first, second]).get_command_registry()


# ---------------------------------------------------------------------------
# B4, B5, B8 — /workspace_write takes the rest of the line verbatim
# ---------------------------------------------------------------------------


class TestWorkspaceWrite:
    def test_quotes_are_written_verbatim(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        result = run(wired_card, """/workspace_write notes.md it's a "quoted" word""")
        assert result == "Written: notes.md (20 bytes)"
        assert (workspace_tree / "notes.md").read_bytes() == b"""it's a "quoted" word"""

    def test_multi_line_content_keeps_its_newlines(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        run(wired_card, "/workspace_write notes.md\nfirst line\n\nthird line\n")
        assert (workspace_tree / "notes.md").read_text(encoding="utf-8") == (
            "first line\n\nthird line\n"
        )

    def test_any_extension_and_missing_parents(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        assert run(wired_card, "/workspace_write a/b/data.json {}") == (
            "Written: a/b/data.json (2 bytes)"
        )
        assert (workspace_tree / "a" / "b" / "data.json").read_text(encoding="utf-8") == "{}"

    def test_no_content_is_the_missing_argument_result(self, wired_card: WorkspaceTool) -> None:
        assert run(wired_card, "/workspace_write notes.md") == (
            "Command 'workspace_write' failed: missing required argument(s): content"
        )

    def test_a_broken_quote_in_the_path_is_a_result_not_a_crash(
        self, wired_card: WorkspaceTool
    ) -> None:
        assert run(wired_card, '/workspace_delete "a.md') == (
            "Command 'workspace_delete' failed: No closing quotation"
        )


# ---------------------------------------------------------------------------
# B6 — the human's change is not the agent's view
# ---------------------------------------------------------------------------


class TestTheHumansChangeIsNotTheAgentsView:
    def test_a_human_overwrite_refuses_the_agents_next_blind_write(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"a.md": "old\n"})
        read(wired_card, "a.md")
        run(wired_card, "/workspace_write a.md new")

        with pytest.raises(RetriableError, match="it changed since you read it"):
            mutate(wired_card, "workspace_write", "a.md", "agent's version\n")
        assert (workspace_tree / "a.md").read_text(encoding="utf-8") == "new"

    def test_a_human_create_is_not_a_read(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        run(wired_card, "/workspace_write b.md x")
        with pytest.raises(RetriableError, match="you have not read it"):
            mutate(wired_card, "workspace_edit", "b.md", "x", "y")

    @pytest.mark.parametrize(
        "text",
        [
            "/workspace_write a.md new",
            "/workspace_delete a.md",
            "/workspace_mkdir fresh",
            "/workspace_rmdir docs",
            "/workspace_read a.md",
            "/workspace_list",
        ],
    )
    def test_no_command_touches_the_observation_map(
        self, wired_card: WorkspaceTool, workspace_tree: Path, text: str
    ) -> None:
        populate(workspace_tree, {"a.md": "old\n", "docs/d.md": "d\n"})
        read(wired_card, "a.md")
        read(wired_card, "docs/d.md")
        before = dict(wired_card._observations)

        run(wired_card, text)

        assert dict(wired_card._observations) == before


# ---------------------------------------------------------------------------
# B7 — not gated on views, but the tree is still guarded
# ---------------------------------------------------------------------------


class TestTheTreeIsStillGuarded:
    @pytest.mark.parametrize(
        "text",
        [
            "/workspace_write a.md new",
            "/workspace_delete a.md",
            "/workspace_mkdir fresh",
            "/workspace_rmdir docs",
        ],
    )
    def test_a_held_tree_refuses_every_write_side_command(
        self, wired_card: WorkspaceTool, workspace_tree: Path, text: str
    ) -> None:
        populate(workspace_tree, {"a.md": "old\n", "docs/d.md": "d\n"})
        run_id = hold_the_tree(wired_card)

        assert run(wired_card, text) == mutation_busy(run_id, "builder")
        assert (workspace_tree / "a.md").read_text(encoding="utf-8") == "old\n"
        assert (workspace_tree / "docs" / "d.md").exists()
        assert not (workspace_tree / "fresh").exists()

    def test_write_and_delete_lock_their_path(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        meta = meta_dir_for(WORKSPACE_PATH)
        run(wired_card, "/workspace_write w.md x")
        assert lock_file_for(meta, "w.md").exists()
        populate(workspace_tree, {"d.md": "d\n"})
        run(wired_card, "/workspace_delete d.md")
        assert lock_file_for(meta, "d.md").exists()

    def test_mkdir_and_rmdir_lock_nothing(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/d.md": "d\n"})
        read(wired_card, "docs/d.md")
        before = lock_files()
        run(wired_card, "/workspace_mkdir fresh")
        run(wired_card, "/workspace_rmdir docs")
        assert lock_files() == before

    @requires_git
    @pytest.mark.parametrize(
        ("text", "subject"),
        [
            ("/workspace_write notes.md hello", "write: notes.md"),
            ("/workspace_delete seed.md", "delete: seed.md"),
            ("/workspace_rmdir docs", "rmdir: docs"),
        ],
    )
    def test_the_commit_is_the_humans(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        text: str,
        subject: str,
    ) -> None:
        populate(workspace_tree, {"seed.md": "s\n", "docs/d.md": "d\n"})
        alice, _observer = card_for(orchestrator_proxy, "alice", git_journal=True)
        mutate(alice, "workspace_mkdir", "warm")  # commits the seeded tree out of band

        run(alice, text)

        head = journal_log(workspace_tree)[-1]
        assert head.subject == subject
        assert head.author_name == "human"
        assert head.author_email.startswith("human@")

    def test_the_touched_files_are_marked_stale(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"d.md": "d\n", "docs/a.md": "a\n", "docs/b/c.md": "c\n"})
        cache = _RecordingCache()
        wired_card._document_cache = cache  # type: ignore[assignment]

        run(wired_card, "/workspace_write w.md x")
        run(wired_card, "/workspace_delete d.md")
        run(wired_card, "/workspace_rmdir docs")

        assert cache.marked == [["w.md"], ["d.md"], ["docs/a.md", "docs/b/c.md"]]


# ---------------------------------------------------------------------------
# B9, B10, B11 — delete, rmdir, mkdir
# ---------------------------------------------------------------------------


class TestDeleteRmdirMkdir:
    def test_delete_removes_one_file(self, wired_card: WorkspaceTool, workspace_tree: Path) -> None:
        populate(workspace_tree, {"a.md": "a\n"})
        assert run(wired_card, "/workspace_delete a.md") == "Deleted: a.md"
        assert not (workspace_tree / "a.md").exists()

    def test_delete_refuses_a_directory_and_a_missing_file(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n"})
        assert run(wired_card, "/workspace_delete docs") == (
            "Is a directory: docs — use /workspace_rmdir"
        )
        assert run(wired_card, "/workspace_delete nowhere.md") == "File not found: nowhere.md"
        assert (workspace_tree / "docs" / "a.md").exists()

    def test_rmdir_removes_the_tree_without_a_view_check(
        self,
        wired_card: WorkspaceTool,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n", "docs/b.md": "b\n"})
        read(wired_card, "docs/a.md")
        bob, _bob_observer = card_for(orchestrator_proxy, "bob")
        read(bob, "docs/a.md")
        mutate(bob, "workspace_write", "docs/a.md", "bob's\n")

        assert run(wired_card, "/workspace_rmdir docs") == "Removed: docs (2 files)"
        assert not (workspace_tree / "docs").exists()

    def test_rmdir_is_empty_only_where_recursion_is_not_allowed(
        self, observer: FakeActorToolObserver, workspace_tree: Path
    ) -> None:
        card = _wired(observer, workspace_rmdir=WorkspaceRmdir(allow_recursive=False))
        populate(workspace_tree, {"docs/a.md": "a\n"})
        (workspace_tree / "empty").mkdir()

        assert run(card, "/workspace_rmdir docs") == (
            "Directory not empty: docs (1 entries) — this workspace removes empty directories only"
        )
        assert run(card, "/workspace_rmdir empty") == "Removed: empty"
        assert (workspace_tree / "docs" / "a.md").exists()

    @pytest.mark.parametrize("target", [".", "a/..", "./"])
    def test_rmdir_refuses_the_root(
        self, wired_card: WorkspaceTool, workspace_tree: Path, target: str
    ) -> None:
        populate(workspace_tree, {"keep.md": "k\n"})
        assert "workspace root" in str(run(wired_card, f"/workspace_rmdir {target}"))
        assert (workspace_tree / "keep.md").exists()

    def test_rmdir_refuses_a_file_and_a_symlink(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"a.md": "a\n", "real/r.md": "r\n"})
        (workspace_tree / "link").symlink_to(workspace_tree / "real")
        assert run(wired_card, "/workspace_rmdir a.md") == (
            "Not a directory: a.md — use /workspace_delete"
        )
        assert run(wired_card, "/workspace_rmdir link") == (
            "link is a symlink — use /workspace_delete"
        )
        assert (workspace_tree / "real" / "r.md").exists()

    def test_mkdir_creates_like_apply_mkdir(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        assert run(wired_card, "/workspace_mkdir a/b/c") == "Created: a/b/c"
        assert (workspace_tree / "a" / "b" / "c").is_dir()
        assert run(wired_card, "/workspace_mkdir a/b/c") == "Created: a/b/c"


# ---------------------------------------------------------------------------
# B12, B13 — read and list
# ---------------------------------------------------------------------------


class TestReadAndList:
    def test_read_returns_the_text_and_records_nothing(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"a.md": "alpha\nbravo\n"})
        assert run(wired_card, "/workspace_read a.md") == "alpha\nbravo\n"
        assert wired_card.observation_for("a.md") is None

    def test_read_is_capped_at_the_default_limit(
        self, observer: FakeActorToolObserver, workspace_tree: Path
    ) -> None:
        card = _wired(observer, workspace_read=WorkspaceRead(default_limit=2))
        populate(workspace_tree, {"a.md": "one\ntwo\nthree\nfour\n"})
        assert run(card, "/workspace_read a.md") == "one\ntwo\n… (4 lines; showing 1–2)"

    def test_a_file_that_is_not_utf8_is_refused(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        (workspace_tree / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
        assert run(wired_card, "/workspace_read blob.bin") == "Not a text file: blob.bin"

    def test_a_document_never_reaches_the_document_reader(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def no_extraction(*_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("a human command invoked the document reader")

        monkeypatch.setattr("akgentic.tool.workspace.read._extract_document", no_extraction)
        (workspace_tree / "deck.pdf").write_bytes(b"%PDF-1.7\n\xe2\xe3\xcf\xd3\n")
        assert run(wired_card, "/workspace_read deck.pdf") == "Not a text file: deck.pdf"

    def test_read_of_a_missing_file(self, wired_card: WorkspaceTool) -> None:
        assert run(wired_card, "/workspace_read nowhere.md") == "File not found: nowhere.md"

    @pytest.mark.parametrize(("path", "depth"), [("", 1), ("docs", 1), ("docs", 2), ("", 0)])
    def test_list_renders_exactly_what_the_tool_renders(
        self, wired_card: WorkspaceTool, workspace_tree: Path, path: str, depth: int
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n", "docs/sub/b.md": "bb\n", "top.md": "t\n"})
        text = f"/workspace_list {path} {depth}" if path else f'/workspace_list "" {depth}'
        assert run(wired_card, text) == tool_named(wired_card, "workspace_list")(path, depth)

    def test_list_coerces_the_depth(self, wired_card: WorkspaceTool, workspace_tree: Path) -> None:
        populate(workspace_tree, {"docs/sub/b.md": "bb\n"})
        assert run(wired_card, "/workspace_list docs 2") == ".\n└── sub/\n    └── b.md (3 bytes)"

    def test_list_of_a_missing_directory_is_a_result(self, wired_card: WorkspaceTool) -> None:
        assert run(wired_card, "/workspace_list nowhere") == "Directory not found: nowhere"


# ---------------------------------------------------------------------------
# B15 — escape and containment
# ---------------------------------------------------------------------------


class TestEscape:
    @pytest.mark.parametrize(
        "text",
        [
            "/workspace_write ../escape.md x",
            "/workspace_delete ../escape.md",
            "/workspace_mkdir ../escape",
            "/workspace_rmdir ../neighbour",
            "/workspace_read ../escape.md",
            "/workspace_list ../neighbour",
        ],
    )
    def test_an_escape_is_a_refusal_and_touches_nothing(
        self, wired_card: WorkspaceTool, workspace_tree: Path, text: str
    ) -> None:
        outside = workspace_tree.parent
        (outside / "escape.md").write_text("outside\n", encoding="utf-8")
        (outside / "neighbour").mkdir()

        result = str(run(wired_card, text))

        assert "escape" in result.lower() or "outside" in result.lower()
        assert (outside / "escape.md").read_text(encoding="utf-8") == "outside\n"
        assert (outside / "neighbour").is_dir()
        assert not (outside / "escape").exists()


# ---------------------------------------------------------------------------
# B14 — /workspace_exec runs a bash command line as the human
# ---------------------------------------------------------------------------


@pytest.fixture
def exec_setup(
    orchestrator_proxy: FakeOrchestratorProxy,
    workspace_tree: Path,
    sandbox_script: SandboxScript,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[WorkspaceTool, WorkspaceActor, ExecHarness], None, None]:
    """An exec card with a journal, its actor, and the harness — a completing poll."""
    card, observer = exec_card_for(
        orchestrator_proxy, poll_attempts=50, poll_delay_seconds=0.01, git_journal=True
    )
    _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
    assert isinstance(actor, WorkspaceActor)
    harness = ExecHarness(actor, orchestrator_proxy)
    harness.install(monkeypatch)
    yield card, actor, harness
    harness.close()
    del observer


class TestWorkspaceExec:
    def test_bash_is_on_the_allow_list(self) -> None:
        # Without it ``/workspace_exec`` is silently useless: every line it runs
        # is ``bash -c <line>``.
        assert "bash" in ALLOWED_COMMANDS

    def test_the_raw_line_runs_as_bash_c(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, _actor, harness = exec_setup
        sandbox_script.gate.set()
        sandbox_script.stdout = "42"

        result = str(run(card, """/workspace_exec ls -la | wc -l && echo "done" > out.txt"""))
        harness.join()

        line = """ls -la | wc -l && echo "done" > out.txt"""
        assert sandbox_script.commands == [(shlex.join(["bash", "-c", line]), "")]
        assert "exit_code: 0 (OK)" in result
        assert "42" in result

    @requires_git
    def test_the_runs_changes_are_committed_as_the_human(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        card, _actor, harness = exec_setup
        sandbox_script.gate.set()
        sandbox_script.files = [("built.txt", "x\n")]

        run(card, "/workspace_exec make")
        harness.join()

        head = journal_log(workspace_tree)[-1]
        assert head.files == ["built.txt"]
        assert head.author_name == "human"

    @requires_git
    def test_the_agents_own_run_is_still_the_agents(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        card, _actor, harness = exec_setup
        sandbox_script.gate.set()
        sandbox_script.files = [("built.txt", "x\n")]

        tool_named(card, "workspace_exec")(cmd="make")
        harness.join()

        assert journal_log(workspace_tree)[-1].author_name == "alice"

    def test_a_run_that_outlives_the_wait_is_collected_by_its_command(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, _observer = exec_card_for(orchestrator_proxy, poll_attempts=1)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        sandbox_script.stdout = "finally"
        try:
            handoff = str(run(card, "/workspace_exec sleep 1"))
            run_id = harness.runs[0].run_id
            assert run_id in handoff
            sandbox_script.gate.set()
            harness.join()

            collected = str(run(card, f"/workspace_exec_result {run_id}"))
            assert "finally" in collected
            # One namespace: the agent's own collector reads the same run.
            assert "finally" in tool_named(card, "workspace_exec_result")(run_id)
        finally:
            harness.close()

    def test_a_second_command_while_the_tree_is_held_is_refused(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, _observer = exec_card_for(orchestrator_proxy, poll_attempts=1)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        try:
            run(card, "/workspace_exec sleep 1")
            assert sandbox_script.started.wait(timeout=HANDSHAKE_TIMEOUT_S)

            assert run(card, "/workspace_exec echo again") == exec_busy()
        finally:
            sandbox_script.gate.set()
            harness.join()
            harness.close()

    def test_a_run_records_no_observation(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, _actor, harness = exec_setup
        sandbox_script.gate.set()
        sandbox_script.files = [("built.txt", "x\n")]
        run(card, "/workspace_exec make")
        harness.join()
        assert card.observation_for("built.txt") is None
        with pytest.raises(RetriableError, match="you have not read it"):
            mutate(card, "workspace_write", "built.txt", "agent\n")


@pytest.fixture
def local_exec_card(
    orchestrator_proxy: FakeOrchestratorProxy,
    workspace_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[WorkspaceTool, ExecHarness], None, None]:
    """An exec card over the test ``LocalBackend`` — the line genuinely runs in bash.

    The slot is set before the card binds, so the actor builds a real
    ``LocalBackend``; the harness supplies the worker thread and the address
    the report comes back to.
    """
    monkeypatch.setattr(akgentic.tool.sandbox, "SANDBOX_BACKEND", LocalBackend)
    card, observer = exec_card_for(
        orchestrator_proxy, poll_attempts=400, poll_delay_seconds=0.01, timeout_s=5.0
    )
    _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
    assert isinstance(actor, WorkspaceActor)
    harness = ExecHarness(actor, orchestrator_proxy)
    harness.install(monkeypatch)
    yield card, harness
    harness.close()
    del observer


class TestWorkspaceExecOnARealShell:
    def test_a_pipe_reports_a_count(
        self, local_exec_card: tuple[WorkspaceTool, ExecHarness], workspace_tree: Path
    ) -> None:
        card, harness = local_exec_card
        populate(workspace_tree, {"a.txt": "a\n", "b.txt": "b\n", "c.txt": "c\n"})

        result = str(run(card, "/workspace_exec ls *.txt | wc -l"))
        harness.join()

        assert "exit_code: 0 (OK)" in result
        assert result.split("**stdout**\n", 1)[1].strip() == "3"


# ---------------------------------------------------------------------------
# Every human-mode mutation converges on ``_gated``, as the human
# ---------------------------------------------------------------------------


class TestHumanModeConverges:
    def test_every_human_method_reaches_gated_with_the_human_identity(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        populate(workspace_tree, {"a.md": "a\n", "docs/d.md": "d\n"})
        seen: list[tuple[str, str]] = []
        real = CardWorkspaceTool._gated

        def watched(
            card: WorkspaceTool, capability: str, paths: Any, run_: Any, **kwargs: Any
        ) -> Any:
            identity = kwargs.get("identity")
            seen.append((capability, identity.name if identity is not None else "agent"))
            return real(card, capability, paths, run_, **kwargs)

        monkeypatch.setattr(CardWorkspaceTool, "_gated", watched)
        humans = sorted(name for name in dir(WorkspaceTool) if name.startswith("human_"))
        assert humans == ["human_delete", "human_mkdir", "human_rmdir", "human_write"]

        wired_card.human_write("w.md", "x")
        wired_card.human_delete("a.md")
        wired_card.human_mkdir("fresh")
        wired_card.human_rmdir("docs", True)

        assert seen == [
            ("write", "human"),
            ("delete", "human"),
            ("mkdir", "human"),
            ("rmdir", "human"),
        ]
