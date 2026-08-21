"""``workspace_exec``: the capability, the lease, and the discovered write set.

Story 29-5. Every concurrency assertion here is an event handshake with an upper
bound that is a *failure budget*, never a delay: the fake backend blocks on an
event the test sets, so a run is held open for exactly as long as the test wants
and not one millisecond of wall clock more. Nothing in this file starts docker,
bubblewrap or ``sandbox-exec``, and nothing sleeps for seconds.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from akgentic.core.agent_config import BaseConfig

from akgentic.tool.core.deferred import DEFAULT_WORKER_TIMEOUT_S, WORKER_NAME_PREFIX
from akgentic.tool.errors import RetriableError
from akgentic.tool.sandbox.actor import (
    DEFAULT_BACKEND_TIMEOUT_S,
    SANDBOX_ACTOR_NAME,
    SandboxActor,
    SandboxConfig,
    SandboxState,
    sandbox_actor_name,
)
from akgentic.tool.sandbox.bwrap import BwrapSandboxActor
from akgentic.tool.sandbox.docker import DockerSandboxActor
from akgentic.tool.sandbox.local import LocalSandboxActor
from akgentic.tool.sandbox.seatbelt import SeatbeltSandboxActor
from akgentic.tool.sandbox.tool import ExecTool
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_TIMEOUT_S,
    MAX_TRACKED_RUNS,
    RUN_ID_CHARS,
    ExecConfig,
    ExecOutcome,
    ExecPayload,
    ExecState,
    ExecWorker,
    new_run_id,
    poll_attempts_within,
)
from akgentic.tool.workspace.journal import MAX_COMMIT_BODY_CHARS
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool

from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    ExecHarness,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    SandboxScript,
    exec_card_for,
    journal_body,
    journal_log,
    mutate,
    read,
    requires_git,
    tool_named,
    working_tree_is_clean,
)

AGENT = "agent-1"

REAL_BACKENDS: dict[str, type[SandboxActor]] = {
    "local": LocalSandboxActor,
    "bwrap": BwrapSandboxActor,
    "seatbelt": SeatbeltSandboxActor,
    "docker": DockerSandboxActor,
}
"""The four real backends, named directly rather than read from the registry.

``SANDBOX_ACTOR_CLASSES`` is the injection window this suite writes a fake into,
so a budget test that read the registry would be asserting about the fake.
"""

# ---------------------------------------------------------------------------
# Fixtures — an exec-capable card, its actor, and a worker that really threads
# ---------------------------------------------------------------------------


@pytest.fixture
def exec_setup(
    orchestrator_proxy: FakeOrchestratorProxy,
    workspace_tree: Path,
    sandbox_script: SandboxScript,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkspaceTool, WorkspaceActor, ExecHarness]:
    """An exec-capable card, the singleton behind it, and the worker harness."""
    card, _observer = exec_card_for(orchestrator_proxy)
    _, actor = orchestrator_proxy.children[workspace_actor_name(workspace_tree.name)]
    assert isinstance(actor, WorkspaceActor)
    harness = ExecHarness(actor, orchestrator_proxy)
    harness.install(monkeypatch)
    return card, actor, harness


def start_run(
    actor: WorkspaceActor, script: SandboxScript, cmd: str = "echo hi", agent: str = AGENT
) -> str:
    """Request a run and wait until it is genuinely inside the backend."""
    start = actor.request_exec(agent, cmd)
    assert start.run_id, start.refusal
    assert script.started.wait(timeout=HANDSHAKE_TIMEOUT_S), "the run never reached the backend"
    return start.run_id


def finish_run(script: SandboxScript, harness: ExecHarness) -> None:
    """Release the blocked run and wait for the worker to report."""
    script.gate.set()
    harness.join()


# ---------------------------------------------------------------------------
# AC1 — the capability, its default, and the read_only gate
# ---------------------------------------------------------------------------


class TestTheCapability:
    def test_it_is_off_by_default(self) -> None:
        # A security default, not a style choice: True would give every
        # WorkspaceTool in existence sandboxed shell execution through a
        # dependency bump.
        assert WorkspaceTool().workspace_exec is False

    def test_off_registers_neither_callable(self, wired_card: WorkspaceTool) -> None:
        names = {tool.__name__ for tool in wired_card.get_tools()}
        assert "workspace_exec" not in names
        assert "workspace_exec_result" not in names

    def test_on_registers_both_callables(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        card, _actor, _harness = exec_setup
        names = {tool.__name__ for tool in card.get_tools()}
        assert {"workspace_exec", "workspace_exec_result"} <= names

    def test_read_only_withholds_both(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        # Exec mutates the tree whatever the command is, so it belongs on the
        # write side of the gate — and both callables have to go together.
        card, _ = exec_card_for(orchestrator_proxy, read_only=True)
        names = {tool.__name__ for tool in card.get_tools()}
        assert "workspace_exec" not in names
        assert "workspace_exec_result" not in names

    def test_off_creates_no_sandbox_actor_and_probes_nothing(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The whole of what the default buys: no host probe at wiring time and
        # no #SandboxActor in a team that never asked for one.
        def explode() -> str:
            raise AssertionError("a card with exec off probed the host for a backend")

        monkeypatch.setattr("akgentic.tool.sandbox.tool._resolve_auto_mode", explode)
        card = WorkspaceTool(workspace_id=workspace_tree.name)
        card.observer(FakeActorToolObserver(orchestrator_proxy))

        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert not any(name.startswith(SANDBOX_ACTOR_NAME) for name in created)

    def test_read_only_creates_no_sandbox_actor_either(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        exec_card_for(orchestrator_proxy, read_only=True)
        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert not any(name.startswith(SANDBOX_ACTOR_NAME) for name in created)

    def test_on_creates_the_sandbox_actor(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        orchestrator_proxy: FakeOrchestratorProxy,
    ) -> None:
        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert sandbox_actor_name(WORKSPACE_NAME) in created

    def test_two_workspaces_in_one_team_get_two_sandbox_actors(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        # getChildrenOrCreate resolves on config.name alone, so a constant name
        # handed the second card the FIRST card's actor — whose directory is the
        # first card's tree. The second agent's commands then ran in tree `a`
        # while its own #Workspace-b gated, discovered and committed tree `b`:
        # `a` mutated entirely outside the gate, nothing raised, nothing logged.
        #
        # Asserted on where each backend actually ended up, not only on the
        # names: the name is the mechanism, the directory is the consequence.
        for name in ("alpha", "beta"):
            (workspaces_root / name).mkdir(parents=True, exist_ok=True)
        exec_card_for(orchestrator_proxy, name="a", workspace_id="alpha")
        exec_card_for(orchestrator_proxy, name="b", workspace_id="beta")

        alpha = orchestrator_proxy.children[sandbox_actor_name("alpha")][1]
        beta = orchestrator_proxy.children[sandbox_actor_name("beta")][1]

        assert alpha is not beta
        assert alpha.state.workspace_path == (workspaces_root / "alpha").resolve()
        assert beta.state.workspace_path == (workspaces_root / "beta").resolve()

    def test_two_cards_on_one_workspace_still_share_one_sandbox_actor(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        # The other half of "one actor per tree": naming by workspace must not
        # turn into naming by card. Two agents over one tree share one backend,
        # exactly as they share one #Workspace.
        exec_card_for(orchestrator_proxy, name="a")
        exec_card_for(orchestrator_proxy, name="b")

        created = [
            config.name
            for _cls, config in orchestrator_proxy.create_calls
            if config.name.startswith(SANDBOX_ACTOR_NAME)
        ]
        assert set(created) == {sandbox_actor_name(WORKSPACE_NAME)}

    def test_off_the_tool_channel_creates_no_sandbox_actor_and_probes_nothing(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The two halves of the capability have to agree on what "on" means. A
        # card that takes exec off the tool channel registers no callable, so it
        # must not resolve a backend, warn about the fallback, or bring up a
        # #SandboxActor either — on the docker backend that last one is a running
        # container, brought up to serve tools that do not exist.
        def explode() -> str:
            raise AssertionError("a card with exec off the tool channel probed the host")

        monkeypatch.setattr("akgentic.tool.sandbox.tool._resolve_auto_mode", explode)
        card = WorkspaceTool(
            workspace_id=workspace_tree.name,
            workspace_exec=WorkspaceExec(expose=set()),
        )
        card.observer(FakeActorToolObserver(orchestrator_proxy))

        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert not any(name.startswith(SANDBOX_ACTOR_NAME) for name in created)
        names = {tool.__name__ for tool in card.get_tools()}
        assert "workspace_exec" not in names

    def test_a_backend_announcement_that_fails_does_not_take_the_card_down(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The registration one line earlier already degrades rather than raises,
        # for the stand-in proxies and dead actors this window is full of. This
        # message has to degrade the same way: the cost is an exec request refused
        # for want of a backend, which is visible and recoverable, where a raise
        # at wiring time takes the whole agent with it.
        def refuse(_self: WorkspaceActor, _config: ExecConfig) -> None:
            raise RuntimeError("the actor died between the get-or-create and here")

        monkeypatch.setattr(WorkspaceActor, "configure_exec", refuse)
        card, _ = exec_card_for(orchestrator_proxy)

        assert "workspace_exec" in {tool.__name__ for tool in card.get_tools()}

    def test_an_unknown_mode_fails_at_wiring_time(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # A typo in a card is a configuration error, and configuration errors
        # belong at start-up rather than at the first command.
        card = WorkspaceTool(workspace_id=workspace_tree.name, workspace_exec=WorkspaceExec())
        object.__setattr__(card.workspace_exec, "mode", "e2b")

        with pytest.raises(KeyError):
            card.observer(FakeActorToolObserver(orchestrator_proxy))

    def test_the_card_round_trips_with_the_field_intact(self) -> None:
        card = WorkspaceTool(workspace_exec=WorkspaceExec(mode="docker", timeout_s=9.0))
        restored = WorkspaceTool.model_validate(card.model_dump())
        assert isinstance(restored.workspace_exec, WorkspaceExec)
        assert restored.workspace_exec.mode == "docker"
        assert restored.workspace_exec.timeout_s == 9.0

    def test_every_model_crossing_the_boundary_round_trips(self) -> None:
        # The behavioural half of "no arbitrary_types_allowed is introduced": a
        # non-serializable type leaking into a field shows up here as a model
        # that will not round-trip. The declaration itself is not observable —
        # pydantic materialises an inherited ``model_config`` onto every class.
        payload = ExecPayload(
            deferred_key="abc12345",
            cmd="pytest",
            cwd="src",
            mode="docker",
            team_id="t1",
            workspace_id="ws",
            timeout_s=12.0,
        )
        assert ExecPayload.model_validate(payload.model_dump()) == payload


# ---------------------------------------------------------------------------
# AC3 / AC4 / AC5 — the lease
# ---------------------------------------------------------------------------


class TestTheLease:
    def test_a_run_holds_it_and_a_mutation_is_refused(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        card, actor, harness = exec_setup
        (workspace_tree / "notes.md").write_text("original\n", encoding="utf-8")
        read(card, "notes.md")
        start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_write", "notes.md", "mine\n")

        # Nothing happened: the file is byte-for-byte what it was.
        assert (workspace_tree / "notes.md").read_text(encoding="utf-8") == "original\n"
        finish_run(sandbox_script, harness)

    def test_the_refusal_names_the_holder(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        orchestrator_proxy: FakeOrchestratorProxy,
    ) -> None:
        card, actor, harness = exec_setup
        actor.register_agent(AGENT, "builder")
        start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="builder"):
            mutate(card, "workspace_mkdir", "src")
        finish_run(sandbox_script, harness)

    @pytest.mark.parametrize(
        ("tool_name", "args"),
        [
            ("workspace_write", ("a.md", "x")),
            ("workspace_edit", ("a.md", "x", "y")),
            ("workspace_multi_edit", ([EditItem(path="a.md", old_string="x", new_string="y")],)),
            ("workspace_patch", ("--- a/a.md\n+++ b/a.md\n@@ -1 +1 @@\n-x\n+y\n",)),
            ("workspace_delete", ("a.md",)),
            ("workspace_mkdir", ("sub",)),
        ],
    )
    def test_every_mutation_is_refused(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        tool_name: str,
        args: tuple[Any, ...],
    ) -> None:
        card, actor, harness = exec_setup
        start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, tool_name, *args)
        finish_run(sandbox_script, harness)

    def test_a_second_exec_is_refused(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        start_run(actor, sandbox_script)

        second = actor.request_exec("agent-2", "echo again")
        assert not second.run_id
        assert "workspace busy" in second.refusal
        finish_run(sandbox_script, harness)

    def test_a_refusal_costs_no_file_read(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Asserted behaviourally rather than by counting: a path that does not
        # exist, and one whose read raises, are both refused with the busy
        # message. Either would produce a different answer if the busy check ran
        # after the live-hash read.
        card, actor, harness = exec_setup
        start_run(actor, sandbox_script)

        def explode(path: str) -> bytes:
            raise AssertionError("the gate read a file under a lease")

        monkeypatch.setattr(actor._workspace, "read", explode)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_write", "never-existed.md", "x")
        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_delete", "never-existed.md")
        finish_run(sandbox_script, harness)

    def test_a_refused_mutation_records_nothing(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        card, actor, harness = exec_setup
        (workspace_tree / "notes.md").write_text("original\n", encoding="utf-8")
        read(card, "notes.md")
        before = actor.observation_for(card._agent_id, "notes.md")
        start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_write", "notes.md", "mine\n")

        assert actor.observation_for(card._agent_id, "notes.md") == before
        assert actor._last_writers == {}
        finish_run(sandbox_script, harness)

    def test_a_mutation_succeeds_immediately_after_a_run_completes(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        assert "Created" in mutate(card, "workspace_mkdir", "src")

    def test_a_failed_run_releases_it(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        sandbox_script.raise_with = RuntimeError("the backend fell over")
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        assert "Created" in mutate(card, "workspace_mkdir", "src")

    def test_a_run_killed_by_its_budget_releases_it(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        sandbox_script.raise_with = subprocess.TimeoutExpired(cmd="sleep", timeout=1)
        run_id = start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        status = actor.exec_status(AGENT, run_id)
        assert status.state is ExecState.DONE
        assert status.outcome is not None and status.outcome.timed_out
        assert "Created" in mutate(card, "workspace_mkdir", "src")

    def test_a_worker_that_never_started_releases_it(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        harness.spawn_error = RuntimeError("no thread available")

        start = actor.request_exec(AGENT, "echo hi")
        assert start.run_id  # the id was issued; the spawn is what failed

        assert "Created" in mutate(card, "workspace_mkdir", "src")
        assert actor.exec_status(AGENT, start.run_id).state is ExecState.FAILED

    def test_a_late_report_does_not_clear_a_newer_lease(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        # The reclaim path is what makes this reachable: a run whose lease was
        # taken back on deadline can still report afterwards.
        _card, actor, _harness = exec_setup
        stale = new_run_id()
        actor._lease = None
        current = actor.request_exec("agent-2", "echo current")

        actor.deliver(stale, ExecOutcome(stdout="", stderr="", exit_code=0))

        assert actor._lease is not None
        assert actor._lease.run_id == current.run_id


class TestADisallowedCommand:
    """What an agent is actually told when the allowlist refuses its command.

    The allowlist check lives in ``SandboxActor.exec``, which from this story runs
    on the **worker's** thread — so ``CommandNotAllowedError`` never propagates
    out of ``request_exec`` or ``exec_status`` to a caller. It arrives as a
    reported failure instead, and this is where that is asserted end to end. The
    handler in ``ExecTool.exec_command`` is defence against a future path that
    raises synchronously, and the tests over it in ``tests/sandbox/`` say so.
    """

    def test_it_is_reported_as_a_failure_naming_the_binary_and_the_allowed_list(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        # The fake backend is a real SandboxActor subclass, so exec() runs the
        # real allowlist — the command never reaches _exec at all.
        start = actor.request_exec(AGENT, "git status")
        assert start.run_id, start.refusal
        harness.join()

        status = actor.exec_status(AGENT, start.run_id)
        assert status.state is ExecState.FAILED
        assert "git" in status.reason
        assert "pytest" in status.reason  # the allowed list travels with it
        assert not sandbox_script.commands  # it was refused before the backend

        answer = mutate(card, "workspace_exec_result", start.run_id)
        assert "git" in answer

    def test_it_does_not_leave_the_tree_leased(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The lease is taken before the worker is spawned, so a command the
        # backend refuses is one more exit that has to release it.
        card, actor, harness = exec_setup
        actor.request_exec(AGENT, "git status")
        harness.join()

        assert actor._lease is None
        assert "Created" in mutate(card, "workspace_mkdir", "src")


class TestReadsDuringARun:
    def test_every_read_still_works(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        card, actor, harness = exec_setup
        (workspace_tree / "notes.md").write_text("hello\n", encoding="utf-8")
        start_run(actor, sandbox_script)

        assert "hello" in read(card, "notes.md")
        assert "notes.md" in tool_named(card, "workspace_list")()
        assert "notes.md" in tool_named(card, "workspace_glob")("*.md")
        assert "hello" in tool_named(card, "workspace_grep")("hello")
        finish_run(sandbox_script, harness)

    def test_a_full_read_still_records_its_observation(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        card, actor, harness = exec_setup
        (workspace_tree / "notes.md").write_text("hello\n", encoding="utf-8")
        start_run(actor, sandbox_script)

        read(card, "notes.md")

        seen = actor.observation_for(card._agent_id, "notes.md")
        assert seen is not None and seen.full
        finish_run(sandbox_script, harness)


class TestTheDeadline:
    def test_an_expired_lease_is_reclaimed_by_the_next_mutation(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        # The one exit a worker cannot report: killed during teardown. Without a
        # reclaim it wedges every mutation for the life of the team.
        card, actor, _harness = exec_setup
        actor._lease = None
        start = actor.request_exec(AGENT, "echo hi")
        assert actor._lease is not None
        actor._lease = actor._lease.model_copy(update={"deadline": 0.0})

        assert "Created" in mutate(card, "workspace_mkdir", "src")
        assert actor._lease is None
        assert start.run_id

    def test_a_live_lease_is_not_reclaimed(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_mkdir", "src")
        finish_run(sandbox_script, harness)

    def test_the_deadline_covers_the_budget_plus_a_grace(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        _card, actor, _harness = exec_setup
        actor.request_exec(AGENT, "echo hi")

        assert actor._lease is not None
        held = actor._lease.deadline - actor._lease.started_at
        assert held > DEFAULT_EXEC_TIMEOUT_S


# ---------------------------------------------------------------------------
# AC7 — the handoff and the collection
# ---------------------------------------------------------------------------


class TestCollectingARun:
    def test_a_slow_run_hands_back_its_id_before_it_finishes(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, _actor, harness = exec_setup
        result = mutate(card, "workspace_exec", "pytest")

        assert "in progress" in result
        assert sandbox_script.started.wait(timeout=HANDSHAKE_TIMEOUT_S)
        run_id = harness.payloads[0].deferred_key
        assert run_id in result
        finish_run(sandbox_script, harness)

    def test_the_run_id_is_short(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script)
        assert len(run_id) == RUN_ID_CHARS
        finish_run(sandbox_script, harness)

    def test_a_running_run_reports_running_with_the_same_id(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script)

        assert actor.exec_status(AGENT, run_id).state is ExecState.RUNNING
        assert run_id in mutate(card, "workspace_exec_result", run_id)
        finish_run(sandbox_script, harness)

    def test_polling_after_completion_returns_the_outcome(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        sandbox_script.stdout = "===== 5 passed ====="
        run_id = start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        collected = mutate(card, "workspace_exec_result", run_id)
        assert "exit_code: 0 (OK)" in collected
        assert "5 passed" in collected

    def test_a_failed_run_is_collected_as_a_failure_not_as_running(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        sandbox_script.raise_with = RuntimeError("the backend fell over")
        run_id = start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        status = actor.exec_status(AGENT, run_id)
        assert status.state is ExecState.FAILED
        assert "fell over" in status.reason
        assert "failed" in mutate(card, "workspace_exec_result", run_id)

    def test_an_unknown_run_id_lists_this_agents_runs_rather_than_raising(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script, agent=card._agent_id)
        finish_run(sandbox_script, harness)

        answer = mutate(card, "workspace_exec_result", "deadbeef")
        assert "Unknown run id" in answer
        assert run_id in answer

    def test_recent_run_ids_are_capped(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        _card, actor, _harness = exec_setup
        for _ in range(MAX_TRACKED_RUNS + 5):
            actor._track_run(AGENT, new_run_id())

        assert len(actor._recent_runs[AGENT]) == MAX_TRACKED_RUNS

    def test_a_settled_run_is_never_reported_as_running_once_its_result_is_evicted(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The two maps have different capacities: _recent_runs holds 32 ids PER
        # AGENT, _slots holds 128 results IN TOTAL. Past five agents the tracking
        # outlives the results, and answering from the tracking map then reports
        # a run that finished long ago as still running — a dead end for a model,
        # because no later poll can ever settle it.
        _card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)
        assert actor.exec_status(AGENT, run_id).state is ExecState.DONE

        for index in range(actor.cache_capacity):
            actor.deliver(f"other{index}", ExecOutcome(stdout="", stderr="", exit_code=0))

        # The tracking still holds it; the result no longer does.
        assert run_id in actor._recent_runs[AGENT]
        assert actor.get(run_id) is None
        assert run_id not in actor._in_flight

        status = actor.exec_status(AGENT, run_id)
        assert status.state is ExecState.UNKNOWN
        assert run_id in status.recent_run_ids  # still correctable, never "running"

    def test_running_is_answered_from_the_in_flight_set_not_from_the_tracking_map(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # A run is running iff it is in flight. Asserted in both directions
        # against the tracking map, which is what the two capacities make unable
        # to answer it: tracked-and-in-flight is RUNNING, tracked-and-not is not.
        _card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script)

        assert run_id in actor._in_flight
        assert actor.exec_status(AGENT, run_id).state is ExecState.RUNNING

        never_ran = new_run_id()
        actor._track_run(AGENT, never_ran)
        assert actor.exec_status(AGENT, never_ran).state is ExecState.UNKNOWN

        finish_run(sandbox_script, harness)

    def test_two_polls_during_one_run_do_not_queue_behind_it(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The property the worker exists for: the actor's own methods never touch
        # the sandbox, so a poll answers while the run is still blocked.
        _card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script)

        answered = threading.Event()

        def poll_twice() -> None:
            actor.exec_status(AGENT, run_id)
            actor.exec_status(AGENT, run_id)
            answered.set()

        thread = threading.Thread(target=poll_twice, daemon=True)
        thread.start()
        assert answered.wait(timeout=HANDSHAKE_TIMEOUT_S), "a poll queued behind the run"
        thread.join(timeout=HANDSHAKE_TIMEOUT_S)
        finish_run(sandbox_script, harness)


# ---------------------------------------------------------------------------
# AC8 — the budgets
# ---------------------------------------------------------------------------


class TestTheBudgets:
    def test_the_run_budget_reaches_the_backend(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        assert sandbox_script.timeouts == [DEFAULT_EXEC_TIMEOUT_S]

    @pytest.mark.parametrize("mode", ["local", "bwrap", "seatbelt", "docker"])
    def test_every_backend_hands_its_budget_to_the_subprocess(
        self, mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Captured rather than measured: nothing slow is run and no backend
        # binary has to be present. A budget that stops at the proxy is
        # decoration, so this asserts it reaches subprocess.run in all four.
        actor = REAL_BACKENDS[mode]()
        actor.config = SandboxConfig(name="#SandboxActor", role="ToolActor", team_id="t1")
        actor.state = SandboxState()
        actor.state.observer(actor)
        actor.state.workspace_path = tmp_path
        actor.state.container_name = "sandbox-t1"

        captured: list[float | None] = []

        def fake_run(*args: Any, **kwargs: Any) -> Any:
            captured.append(kwargs.get("timeout"))
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        monkeypatch.setattr(f"akgentic.tool.sandbox.{mode}.subprocess.run", fake_run)
        actor._exec("echo hi", "", 3.25)

        assert captured == [3.25]

    @pytest.mark.parametrize("mode", ["local", "bwrap", "seatbelt", "docker"])
    def test_no_budget_falls_back_to_the_backends_own(
        self, mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # ``None`` keeps each backend's default, so no existing caller changed
        # behaviour when the parameter arrived.
        actor = REAL_BACKENDS[mode]()
        actor.config = SandboxConfig(name="#SandboxActor", role="ToolActor", team_id="t1")
        actor.state = SandboxState()
        actor.state.observer(actor)
        actor.state.workspace_path = tmp_path
        actor.state.container_name = "sandbox-t1"

        captured: list[float | None] = []

        def fake_run(*args: Any, **kwargs: Any) -> Any:
            captured.append(kwargs.get("timeout"))
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        monkeypatch.setattr(f"akgentic.tool.sandbox.{mode}.subprocess.run", fake_run)
        actor._exec("echo hi", "", None)

        assert captured == [DEFAULT_BACKEND_TIMEOUT_S]

    def test_a_card_budget_above_the_workers_is_clamped(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, _ = exec_card_for(orchestrator_proxy)
        _, actor = orchestrator_proxy.children[workspace_actor_name(workspace_tree.name)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        actor.configure_exec(
            ExecConfig(mode="local", team_id=workspace_tree.name, timeout_s=999.0)
        )

        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        assert sandbox_script.timeouts == [DEFAULT_WORKER_TIMEOUT_S]

    def test_the_poll_budget_defaults_below_the_run_budget(self) -> None:
        params = WorkspaceExec()
        assert params.poll_attempts * params.poll_delay_seconds < params.timeout_s

    def test_a_poll_longer_than_the_run_is_clamped(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A poll outlasting the run is a sleep with no possible answer: by then
        # the run has reported or its own budget has killed it. Asserted on the
        # budget the closure hands ``poll_deferred``, which is where it becomes
        # wall clock the agent's thread actually spends.
        card, _ = exec_card_for(
            orchestrator_proxy, poll_attempts=1000, poll_delay_seconds=1.0
        )
        seen: list[tuple[int, float]] = []

        def capture(fetch: Any, attempts: int, delay: float) -> None:
            seen.append((attempts, delay))
            return None

        monkeypatch.setattr("akgentic.tool.workspace.tool.poll_deferred", capture)
        _, actor = orchestrator_proxy.children[workspace_actor_name(workspace_tree.name)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        sandbox_script.gate.set()

        tool_named(card, "workspace_exec")(cmd="pytest")
        harness.join()

        attempts, delay = seen[0]
        assert attempts * delay <= min(DEFAULT_EXEC_TIMEOUT_S, DEFAULT_WORKER_TIMEOUT_S)
        assert attempts >= 1

    def test_a_poll_that_already_fits_is_left_alone(self) -> None:
        assert poll_attempts_within(12, 0.4, 15.0) == 12
        assert poll_attempts_within(0, 0.4, 15.0) == 0  # opting out of polling stands
        assert poll_attempts_within(1000, 1.0, 15.0) == 15
        assert poll_attempts_within(1000, 60.0, 15.0) == 1  # never below one look


# ---------------------------------------------------------------------------
# AC9 — the deferred-result rules, in full
# ---------------------------------------------------------------------------


class TestTheDeferredRules:
    def test_the_worker_is_hash_prefixed(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        assert harness.worker_names
        assert all(name.startswith(WORKER_NAME_PREFIX) for name in harness.worker_names)

    def test_one_run_spawns_exactly_one_worker(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        assert len(harness.worker_names) == 1

    def test_the_payload_is_serializable_and_names_no_actor(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        payload = harness.payloads[0]
        assert isinstance(payload, ExecPayload)
        assert ExecPayload.model_validate(payload.model_dump()) == payload

    def test_a_failure_is_cached_negatively_rather_than_respawning(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        sandbox_script.raise_with = RuntimeError("boom")
        run_id = start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        # A second request for the same key must not spawn another worker.
        actor.request(run_id, harness.payloads[0])
        assert len(harness.worker_names) == 1

    def test_the_result_cache_is_lru_capped(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        _card, actor, _harness = exec_setup
        for index in range(actor.cache_capacity + 5):
            actor.deliver(f"run{index}", ExecOutcome(stdout="", stderr="", exit_code=0))

        assert len(actor._slots) == actor.cache_capacity

    def test_the_actors_own_methods_perform_no_sandbox_call(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script)
        commands_so_far = len(sandbox_script.commands)

        actor.exec_status(AGENT, run_id)
        actor.apply_mkdir(AGENT, "src")
        actor.get(run_id)

        assert len(sandbox_script.commands) == commands_so_far
        finish_run(sandbox_script, harness)

    def test_produce_never_returns_none(self, tmp_path: Path) -> None:
        # The base reads None as a failure, and a command that produced no output
        # is not a failure.
        worker = ExecWorker()
        worker.config = BaseConfig(name="#defer-x", role="ToolActor")
        payload = ExecPayload(
            deferred_key="abc",
            cmd="echo",
            mode="local",
            team_id="t1",
            timeout_s=1.0,
        )

        class _Silent:
            def exec(self, cmd: str, cwd: str, timeout: float | None = None) -> Any:
                class _Result:
                    stdout = ""
                    stderr = ""
                    exit_code = 0

                return _Result()

        worker._sandbox = lambda _payload: _Silent()  # type: ignore[assignment,method-assign]
        outcome = worker.produce(payload)

        assert isinstance(outcome, ExecOutcome)

    def test_a_wrong_payload_is_a_type_error(self) -> None:
        from akgentic.tool.core.deferred import DeferredPayload  # noqa: PLC0415

        worker = ExecWorker()
        worker.config = BaseConfig(name="#defer-x", role="ToolActor")
        with pytest.raises(TypeError):
            worker.produce(DeferredPayload(deferred_key="abc"))


# ---------------------------------------------------------------------------
# AC6 / AC11 — discovery and the commit
# ---------------------------------------------------------------------------


@requires_git
class TestTheDiscoveredWriteSet:
    def test_a_nested_untracked_directory_is_discovered_file_by_file(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # The ``-uall`` property, and the one this story's mutation check
        # targets. Bare ``--porcelain`` collapses ``dist/`` to a **single**
        # entry, which is wrong for exactly the case exec exists for — exec
        # mostly CREATES files.
        #
        # The assertion has to reach the **discovery**, not only the commit's
        # contents: ``git add -A`` expands an untracked directory by itself, so a
        # test that checked the files alone stays green with the flag removed and
        # proves nothing. What actually changes is what the write set was
        # *reported* to be — three files, or one directory named as if it were
        # the thing written.
        _card, actor, harness = exec_setup
        sandbox_script.files = [
            ("dist/a.txt", "a\n"),
            ("dist/nested/b.txt", "b\n"),
            ("dist/nested/deeper/c.txt", "c\n"),
        ]
        start_run(actor, sandbox_script, cmd="make build")
        finish_run(sandbox_script, harness)

        head = journal_log(workspace_tree)[-1]
        assert head.subject == "exec: 3 files"
        assert "dist/a.txt" in head.files
        assert "dist/nested/b.txt" in head.files
        assert "dist/nested/deeper/c.txt" in head.files

    def test_the_discovery_names_files_never_a_directory(
        self, workspace_tree: Path, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        # The same property, asserted directly on the journal's own answer, so
        # that dropping ``-uall`` is caught even if the commit path changes.
        _card, actor, _harness = exec_setup
        for relative in ("dist/a.txt", "dist/nested/b.txt"):
            target = workspace_tree / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x\n", encoding="utf-8")

        assert sorted(actor._journal.changed_paths()) == ["dist/a.txt", "dist/nested/b.txt"]

    def test_one_run_is_one_commit_and_history_stays_linear(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        _card, actor, harness = exec_setup
        sandbox_script.files = [(f"out/f{index}.txt", f"{index}\n") for index in range(9)]
        before = journal_log(workspace_tree)
        start_run(actor, sandbox_script, cmd="make all")
        finish_run(sandbox_script, harness)

        after = journal_log(workspace_tree)
        assert len(after) == len(before) + 1
        assert len(after[-1].parents) == 1
        assert after[-1].parents == [before[-1].sha]
        assert len(after[-1].files) == 9

    def test_the_commit_is_authored_by_the_requester(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        _card, actor, harness = exec_setup
        actor.register_agent(AGENT, "builder")
        sandbox_script.files = [("out.txt", "x\n")]
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        head = journal_log(workspace_tree)[-1]
        assert head.author_name == "builder"
        assert AGENT in head.author_email

    def test_a_dirty_tree_is_committed_out_of_band_first(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        _card, actor, harness = exec_setup
        (workspace_tree / "uploaded.md").write_text("from the frontend\n", encoding="utf-8")
        sandbox_script.files = [("built.txt", "x\n")]
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        log = journal_log(workspace_tree)
        assert log[-2].author_name == "out-of-band"
        assert log[-2].files == ["uploaded.md"]
        assert log[-1].files == ["built.txt"]

    def test_a_run_that_changes_nothing_adds_no_commit(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        _card, actor, harness = exec_setup
        before = journal_log(workspace_tree)
        start_run(actor, sandbox_script, cmd="echo hi")
        finish_run(sandbox_script, harness)

        assert journal_log(workspace_tree) == before

    def test_debris_is_excluded_by_the_seeded_ignore_list(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        _card, actor, harness = exec_setup
        sandbox_script.files = [
            ("__pycache__/mod.cpython-312.pyc", "junk"),
            ("real.txt", "x\n"),
        ]
        start_run(actor, sandbox_script, cmd="pytest")
        finish_run(sandbox_script, harness)

        assert journal_log(workspace_tree)[-1].files == ["real.txt"]

    def test_the_command_goes_in_the_body_never_the_subject(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        _card, actor, harness = exec_setup
        sandbox_script.files = [("out.txt", "x\n")]
        start_run(actor, sandbox_script, cmd="pytest tests/ -v")
        finish_run(sandbox_script, harness)

        head = journal_log(workspace_tree)[-1]
        assert head.subject == "exec: out.txt"
        assert "pytest" not in head.subject

    def test_the_command_reaches_the_body_sanitised(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # It is the one place untrusted text reaches the journal, so it is
        # stripped of control characters and passed through a message file
        # rather than interpolated into an argument.
        _card, actor, harness = exec_setup
        sandbox_script.files = [("out.txt", "x\n")]
        start_run(actor, sandbox_script, cmd="pytest\ntests/\x00 -v")
        finish_run(sandbox_script, harness)

        head = journal_log(workspace_tree)[-1]
        body = journal_body(workspace_tree, head.sha)
        assert body == "pytest tests/ -v"
        assert "\n" not in head.subject

    def test_an_over_long_command_is_capped(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        _card, actor, harness = exec_setup
        sandbox_script.files = [("out.txt", "x\n")]
        start_run(actor, sandbox_script, cmd="echo " + "a" * 2000)
        finish_run(sandbox_script, harness)

        body = journal_body(workspace_tree, journal_log(workspace_tree)[-1].sha)
        assert len(body) <= MAX_COMMIT_BODY_CHARS + 2

    def test_a_refused_mutation_adds_no_commit(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # The last clause of "nothing happened". The busy check returns ahead of
        # the out-of-band commit as well as ahead of the gate, so a refusal costs
        # no git fork either — not only no file read.
        card, actor, harness = exec_setup
        before = journal_log(workspace_tree)
        start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_mkdir", "src")

        assert journal_log(workspace_tree) == before
        finish_run(sandbox_script, harness)

    def test_the_tree_is_clean_after_a_run(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        _card, actor, harness = exec_setup
        sandbox_script.files = [("dist/a.txt", "a\n")]
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        assert working_tree_is_clean(workspace_tree)


class TestTheJournalOff:
    def test_exec_runs_with_the_journal_off(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, _ = exec_card_for(orchestrator_proxy, workspace_git=False)
        _, actor = orchestrator_proxy.children[workspace_actor_name(workspace_tree.name)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        sandbox_script.files = [("out.txt", "x\n")]

        run_id = start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_mkdir", "src")

        finish_run(sandbox_script, harness)
        assert actor.exec_status(AGENT, run_id).state is ExecState.DONE
        assert (workspace_tree / "out.txt").exists()
        assert not (workspace_tree.parent / f"{workspace_tree.name}.git").exists()

    @requires_git
    @pytest.mark.parametrize(
        "failure",
        [
            OSError("git could not be spawned"),
            subprocess.TimeoutExpired(cmd="git", timeout=15),
        ],
        ids=["spawn-failure", "timeout"],
    )
    def test_a_journal_failure_leaves_the_result_and_the_lease_alone(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
        failure: BaseException,
    ) -> None:
        # A journal failure is logged, never raised: the run's bytes are already
        # on disk by the time a commit is attempted, so there is nothing a raise
        # could usefully undo — and a lease left held would wedge the team.
        card, actor, harness = exec_setup
        sandbox_script.files = [("out.txt", "x\n")]
        run_id = start_run(actor, sandbox_script)

        def explode(*args: Any, **kwargs: Any) -> None:
            raise failure

        monkeypatch.setattr("akgentic.tool.workspace.journal.subprocess.run", explode)
        finish_run(sandbox_script, harness)

        assert actor.exec_status(AGENT, run_id).state is ExecState.DONE
        assert "Created" in mutate(card, "workspace_mkdir", "src")

    def test_a_non_zero_git_exit_leaves_the_result_and_the_lease_alone(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, actor, harness = exec_setup
        sandbox_script.files = [("out.txt", "x\n")]
        run_id = start_run(actor, sandbox_script)

        def refuse(*args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(returncode=128, stdout="", stderr="fatal: not a repository")

        monkeypatch.setattr("akgentic.tool.workspace.journal.subprocess.run", refuse)
        finish_run(sandbox_script, harness)

        assert actor.exec_status(AGENT, run_id).state is ExecState.DONE
        assert "Created" in mutate(card, "workspace_mkdir", "src")


# ---------------------------------------------------------------------------
# AC2 — the deprecated shim behaves identically
# ---------------------------------------------------------------------------


@requires_git
class TestTheShimAndTheCapabilityAgree:
    def test_the_same_command_produces_the_same_observable_outcome(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The equivalence is asserted on what an agent and a repository can see:
        # the returned text, the files on disk, and the lease being taken and
        # released — not on which code path was taken to get there.
        card, _ = exec_card_for(orchestrator_proxy, poll_attempts=50, poll_delay_seconds=0.01)
        _, actor = orchestrator_proxy.children[workspace_actor_name(workspace_tree.name)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)

        shim = ExecTool(mode="local", workspace_id=workspace_tree.name)
        with pytest.warns(DeprecationWarning):
            shim.observer(FakeActorToolObserver(orchestrator_proxy, name="bob"))

        sandbox_script.files = [("built.txt", "x\n")]
        sandbox_script.stdout = "done"

        through_capability = self._run(tool_named(card, "workspace_exec"), sandbox_script, harness)
        (workspace_tree / "built.txt").unlink()
        actor._journal.commit_out_of_band()
        sandbox_script.started.clear()
        sandbox_script.gate.clear()
        through_shim = self._run(
            next(t for t in shim.get_tools() if t.__name__ == "exec_command"),
            sandbox_script,
            harness,
        )

        assert through_capability == through_shim
        assert (workspace_tree / "built.txt").read_text(encoding="utf-8") == "x\n"
        assert actor._lease is None

    @staticmethod
    def _run(callable_: Any, script: SandboxScript, harness: ExecHarness) -> str:
        """Drive one surface to completion and return what the agent was told."""
        script.gate.set()
        answer = str(callable_(cmd="make build"))
        harness.join()
        return answer

    def test_both_surfaces_commit_as_their_own_agent(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, observer = exec_card_for(
            orchestrator_proxy, name="alice", poll_attempts=50, poll_delay_seconds=0.01
        )
        _, actor = orchestrator_proxy.children[workspace_actor_name(workspace_tree.name)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        shim = ExecTool(mode="local", workspace_id=workspace_tree.name)
        with pytest.warns(DeprecationWarning):
            shim.observer(FakeActorToolObserver(orchestrator_proxy, name="bob"))

        sandbox_script.gate.set()
        sandbox_script.files = [("one.txt", "1\n")]
        tool_named(card, "workspace_exec")(cmd="make one")
        harness.join()
        sandbox_script.files = [("two.txt", "2\n")]
        next(t for t in shim.get_tools() if t.__name__ == "exec_command")(cmd="make two")
        harness.join()

        log = journal_log(workspace_tree)
        assert log[-2].author_name == "alice"
        assert log[-1].author_name == "bob"


# ---------------------------------------------------------------------------
# AC13 — the tool surface stays honest
# ---------------------------------------------------------------------------


class TestTheToolSurface:
    def test_the_two_signatures_are_what_they_claim(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        import inspect  # noqa: PLC0415

        card, _actor, _harness = exec_setup
        run = inspect.signature(tool_named(card, "workspace_exec"))
        collect = inspect.signature(tool_named(card, "workspace_exec_result"))

        assert list(run.parameters) == ["cmd", "cwd"]
        assert run.parameters["cwd"].default == ""
        assert list(collect.parameters) == ["run_id"]

    def test_no_mutation_signature_changed(self, wired_card: WorkspaceTool) -> None:
        import inspect  # noqa: PLC0415

        expected = {
            "workspace_write": ["path", "content"],
            "workspace_delete": ["path"],
            "workspace_edit": ["path", "old_string", "new_string", "replace_all"],
            "workspace_multi_edit": ["edits"],
            "workspace_patch": ["patch_text"],
            "workspace_mkdir": ["path"],
        }
        for name, parameters in expected.items():
            signature = inspect.signature(tool_named(wired_card, name))
            assert list(signature.parameters) == parameters

    def test_nothing_lets_a_model_name_a_mode_a_timeout_or_a_git_argument(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        import inspect  # noqa: PLC0415

        card, _actor, _harness = exec_setup
        for tool in card.get_tools():
            forbidden = {"mode", "timeout", "timeout_s", "force", "digest", "expected"}
            assert not (set(inspect.signature(tool).parameters) & forbidden)


# ---------------------------------------------------------------------------
# AC12 — ownership, and the two PermissionError sources
# ---------------------------------------------------------------------------


@requires_git
class TestOwnershipIsNeverAssumed:
    def test_a_file_a_run_created_is_governed_by_its_content_hash(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        card, actor, harness = exec_setup
        sandbox_script.files = [("built.txt", "from the run\n")]
        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        # The agent that did not read it is refused …
        other, _ = exec_card_for(orchestrator_proxy, name="bob")
        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(other, "workspace_write", "built.txt", "mine\n")

        # … and the one that reads it first may overwrite it.
        read(card, "built.txt")
        assert "Written" in mutate(card, "workspace_write", "built.txt", "mine\n")


class TestPermissionErrorsAreDistinguished:
    def test_a_path_escape_keeps_its_exact_wording(self, wired_card: WorkspaceTool) -> None:
        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "../escape.md", "x")

        assert str(refusal.value).startswith(
            "Path escapes workspace root — use a path relative to the workspace"
        )

    def test_an_os_denial_says_something_else(
        self, wired_card: WorkspaceTool, workspace_actor: WorkspaceActor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # What a root-owned file from a container produces: publication by rename
        # means the host process must be able to replace the inode.
        def denied(path: str, data: bytes) -> None:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(workspace_actor._workspace, "write", denied)

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "new.md", "x")

        message = str(refusal.value)
        assert "Path escapes workspace root" not in message
        assert "did not escape" in message
