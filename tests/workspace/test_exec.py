"""``workspace_exec``: the capability, the lease, and the discovered write set.

Story 29-5. Every concurrency assertion here is an event handshake with an upper
bound that is a *failure budget*, never a delay: the fake backend blocks on an
event the test sets, so a run is held open for exactly as long as the test wants
and not one millisecond of wall clock more. Nothing in this file starts docker,
bubblewrap or ``sandbox-exec``, and nothing sleeps for seconds.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pykka
import pytest
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from pydantic import ValidationError

from akgentic.tool.errors import RetriableError
from akgentic.tool.sandbox import SANDBOX_BACKEND_CLASSES
from akgentic.tool.sandbox.backend import (
    DEFAULT_BACKEND_TIMEOUT_S,
    ExecReport,
    ExecResult,
)
from akgentic.tool.sandbox.bwrap import BwrapBackend
from akgentic.tool.sandbox.docker import DockerBackend
from akgentic.tool.sandbox.local import LocalBackend
from akgentic.tool.sandbox.seatbelt import SeatbeltBackend
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_TIMEOUT_S,
    EXEC_REPORT_MARGIN_S,
    EXEC_SHUTDOWN_GRACE_S,
    LEASE_GRACE_S,
    MAX_EXEC_BUDGET_S,
    MAX_TRACKED_RUNS,
    RUN_ID_CHARS,
    TIMED_OUT_EXIT_CODE,
    ExecConfig,
    ExecOutcome,
    ExecStart,
    ExecState,
    ExecStatus,
    RunningExec,
    effective_budget,
    exec_busy,
    format_status,
    in_progress,
    lock_unavailable,
    new_run_id,
    poll_attempts_within,
    timed_out,
)
from akgentic.tool.workspace.journal import MAX_COMMIT_BODY_CHARS
from akgentic.tool.workspace.lock import (
    EXEC_LOCK_FILENAME,
    FileLockBackend,
    LockBackend,
    LockGrant,
    LockTicket,
)
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool
from akgentic.tool.workspace.workspace import meta_dir_for
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_PATH,
    ExecHarness,
    FakeActorToolObserver,
    FakeBackend,
    FakeOrchestratorProxy,
    SandboxScript,
    SilentAgent,
    attached,
    exec_card_for,
    journal_body,
    journal_log,
    mutate,
    read,
    requires_git,
    tool_named,
    working_tree_is_clean,
    workspace_path_for,
)

AGENT = "agent-1"

AGENT_B = "agent-2"
AGENT_C = "agent-3"
"""Two more requesters, for everything the single ``AGENT`` constant hid.

The whole suite used one agent id, which is precisely why a run belonging to
somebody else being collectable by anybody was never caught: with one id there is
no "somebody else". Every ownership and queue spec below uses at least two.
"""

REAL_STRATEGIES: dict[str, type[Any]] = {
    "local": LocalBackend,
    "bwrap": BwrapBackend,
    "seatbelt": SeatbeltBackend,
    "docker": DockerBackend,
}
"""The four real backends, named directly rather than read from the registry.

``SANDBOX_BACKEND_CLASSES`` is the injection window this suite writes a fake
into, so a budget test that read the registry would be asserting about the fake.
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
) -> Generator[tuple[WorkspaceTool, WorkspaceActor, ExecHarness], None, None]:
    """An exec-capable card, the singleton behind it, and the exec harness.

    A generator, unlike its predecessor, because the worker is now a real
    ``ThreadPoolExecutor`` thread rather than a daemon: one left running per
    spec would be a hundred idle threads by the end of the session.
    """
    card, _observer = exec_card_for(orchestrator_proxy)
    _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
    assert isinstance(actor, WorkspaceActor)
    harness = ExecHarness(actor, orchestrator_proxy)
    harness.install(monkeypatch)
    yield card, actor, harness
    harness.close()


def start_run(
    actor: WorkspaceActor, script: SandboxScript, cmd: str = "echo hi", agent: str = AGENT
) -> str:
    """Request a run and wait until it is genuinely inside the backend."""
    start = actor.request_exec(agent, cmd)
    assert start.run_id, start.refusal
    assert script.started.wait(timeout=HANDSHAKE_TIMEOUT_S), "the run never reached the backend"
    return start.run_id


def finish_run(script: SandboxScript, harness: ExecHarness) -> None:
    """Release the blocked run and wait for the sandbox to report."""
    script.gate.set()
    harness.join()


def exec_marker() -> Path:
    """The exec lock's marker for the suite's tree — a sibling, never inside it.

    Derived through ``meta_dir_for`` rather than spelled out, so a spec asserting
    on it cannot drift from where the backend actually writes. It reads the
    environment at call time, so callers take the ``workspace_tree`` fixture.
    """
    return meta_dir_for(WORKSPACE_PATH) / EXEC_LOCK_FILENAME


def _ticket(agent_id: str = AGENT_B, cmd: str = "echo hi") -> LockTicket:
    """A ticket another process's backend would present for the suite's tree."""
    return LockTicket(
        agent_id=agent_id, cmd=cmd, budget_s=effective_budget(DEFAULT_EXEC_TIMEOUT_S)
    )


class _RaisingLock:
    """A lock backend whose filesystem is broken — both calls raise.

    Stands in for an unwritable metadata parent or a full disk, which is the
    only way either call can fail: neither is a code path the shipped backend
    can be talked into on its own.
    """

    def acquire(self, tree_key: str, ticket: LockTicket) -> LockGrant:
        raise OSError("the metadata directory is not writable")

    def release(self, tree_key: str, run_id: str) -> None:
        raise OSError("the metadata directory is not writable")


class _ReleaseObserver:
    """Wraps a real backend and records the tree's state at each ``release``.

    The only way to assert the ORDER of the commit and the release rather than
    the end state: by the time ``_finish_run`` returns both have happened,
    whichever came first.
    """

    def __init__(self, inner: LockBackend, tree: Path) -> None:
        self._inner = inner
        self._tree = tree
        self.clean_at_release: list[bool] = []

    def acquire(self, tree_key: str, ticket: LockTicket) -> LockGrant:
        return self._inner.acquire(tree_key, ticket)

    def release(self, tree_key: str, run_id: str) -> None:
        self.clean_at_release.append(working_tree_is_clean(self._tree))
        self._inner.release(tree_key, run_id)


class _TeardownOrderLock:
    """Wraps a real backend and records what the sandbox had done at ``release``.

    The teardown counterpart of :class:`_ReleaseObserver`: by the time
    ``on_stop`` returns every step has run, so presence proves nothing about
    order. What is read here is the state the *child* is in at the moment the
    marker goes back.
    """

    def __init__(self, inner: LockBackend, script: SandboxScript) -> None:
        self._inner = inner
        self._script = script
        self.kills_at_release: list[int] = []
        self.stopped_at_release: list[bool] = []

    def acquire(self, tree_key: str, ticket: LockTicket) -> LockGrant:
        return self._inner.acquire(tree_key, ticket)

    def release(self, tree_key: str, run_id: str) -> None:
        self.kills_at_release.append(self._script.kills)
        self.stopped_at_release.append(("stop",) in self._script.events)
        self._inner.release(tree_key, run_id)


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
        # exactly one actor — the workspace's own, a team child again since
        # 52-5 — in a team that never asked for exec. Asserted as an equality
        # over the created list rather than as the absence of a name, so it
        # cannot pass over an empty list; the empty ``resource_calls`` beside it
        # is the negative that says this process forwards to no host at all.
        def explode() -> str:
            raise AssertionError("a card with exec off probed the host for a backend")

        monkeypatch.setattr("akgentic.tool.sandbox._resolve_auto_mode", explode)
        card = WorkspaceTool(workspace_id=workspace_tree.name)
        card.observer(FakeActorToolObserver(orchestrator_proxy))

        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert created == [workspace_actor_name(WORKSPACE_PATH)]
        assert orchestrator_proxy.resource_calls == []

    def test_read_only_creates_only_the_workspace_actor_too(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        exec_card_for(orchestrator_proxy, read_only=True)
        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert created == [workspace_actor_name(WORKSPACE_PATH)]
        assert orchestrator_proxy.resource_calls == []

    def test_on_builds_a_runner_and_still_creates_only_the_workspace_actor(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        orchestrator_proxy: FakeOrchestratorProxy,
    ) -> None:
        # What "on" buys is a backend on #Workspace itself, anchored to this
        # card's tree — and no second actor. Three actors per exec-enabled team
        # became two: the workspace and the agent. The created list is exactly
        # the workspace, and nothing was forwarded to a host, so a second actor
        # of any name reddens one or the other.
        _card, actor, _harness = exec_setup

        assert actor._runner is not None
        assert actor._runner.workspace_path == WORKSPACE_PATH
        assert isinstance(actor._runner.backend, FakeBackend)
        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert created == [workspace_actor_name(WORKSPACE_PATH)]
        assert orchestrator_proxy.resource_calls == []

    def test_two_workspaces_in_one_team_get_two_runners_on_their_own_trees(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        # The hazard the per-workspace naming removed is still the hazard: a
        # second card that ended up sharing the first card's backend would run
        # its commands in tree `a` while its own #Workspace-b gated, discovered
        # and committed tree `b` — `a` mutated entirely outside the gate, with
        # nothing raised and nothing logged.
        #
        # It is now structural rather than nominal: the backend hangs off the
        # workspace actor, and there is one of those per tree. Asserted on the
        # tree each runner is anchored to, which is the consequence, not on the
        # actor names, which were only ever the mechanism.
        for leaf in ("alpha", "beta"):
            (workspaces_root / workspace_path_for(leaf)).mkdir(parents=True, exist_ok=True)
        exec_card_for(orchestrator_proxy, name="a", workspace_id="alpha")
        exec_card_for(orchestrator_proxy, name="b", workspace_id="beta")

        alpha = orchestrator_proxy.children[
            workspace_actor_name(workspace_path_for("alpha"))
        ][1]
        beta = orchestrator_proxy.children[workspace_actor_name(workspace_path_for("beta"))][1]

        assert alpha is not beta
        assert alpha._runner is not None
        assert beta._runner is not None
        assert alpha._runner is not beta._runner
        assert alpha._runner.backend is not beta._runner.backend
        assert alpha._runner.workspace_path == workspace_path_for("alpha")
        assert beta._runner.workspace_path == workspace_path_for("beta")

    def test_two_cards_on_one_workspace_still_share_one_backend(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        # The other half of "one backend per tree": two agents over one tree
        # share one, exactly as they share one #Workspace. The second card
        # announces a config equal to the first's, and an equal config must not
        # replace the runner — a replacement would leak the first backend, and on
        # the docker backend that is a container with nobody left to stop it.
        # The fake hands every observer a fresh team id, and that no longer
        # matters: ExecConfig carries no team, so the two announcements are
        # equal whatever team each card belongs to.
        first_card, first_observer = exec_card_for(orchestrator_proxy, name="a")
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        first = actor._runner

        second_observer = FakeActorToolObserver(orchestrator_proxy, name="b")
        assert second_observer.team_id != first_observer.team_id
        second_card = WorkspaceTool(
            workspace_id=first_card.workspace_id,
            workspace_exec=WorkspaceExec(mode="local", poll_attempts=1),
        )
        second_card.observer(second_observer)

        assert first is not None
        assert actor._runner is first
        assert actor._runner.backend is first.backend
        assert sandbox_script.stops == 0

    def test_the_actor_builds_the_backend_from_the_registry_with_the_cards_config(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        # ``resolve_mode`` hands back a strategy instance, and this is the caller
        # that keeps it. The class has to come from ``SANDBOX_BACKEND_CLASSES``
        # at call time — that is the injection window a deployment writes into —
        # and every value the backend needs has to arrive on the card's own
        # ``ExecConfig``. A backend built from anything else could open a
        # directory other than the one this #Workspace gates.
        exec_card_for(orchestrator_proxy)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)

        assert actor._exec_config == ExecConfig(
            mode="local",
            workspace_path=WORKSPACE_PATH,
            timeout_s=DEFAULT_EXEC_TIMEOUT_S,
        )
        runner = actor._runner
        assert runner is not None
        assert type(runner.backend) is SANDBOX_BACKEND_CLASSES["local"]
        assert isinstance(runner.backend, FakeBackend)
        assert runner.workspace_path == WORKSPACE_PATH

    def test_the_actor_is_given_the_cards_own_lock_backend(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
    ) -> None:
        # Built card-side in ``observer()`` and announced over the tell proxy,
        # so the object the actor decides admission on is the one the card
        # resolved — not a second instance built somewhere else under a
        # different environment.
        card, _observer = exec_card_for(orchestrator_proxy)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)

        assert isinstance(card._lock_backend, FileLockBackend)
        assert actor._lock is card._lock_backend

    def test_an_unknown_lock_backend_fails_at_wiring_time(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A configuration error belongs at start-up, in front of the admin who
        # set the variable — exactly as an unknown sandbox mode already fails.
        # Deferred to the first command it would surface as an unexplained
        # refusal, to an agent, hours later.
        monkeypatch.setenv("AKGENTIC_LOCK_BACKEND", "nope")

        with pytest.raises(KeyError):
            WorkspaceTool(workspace_id=workspace_tree.name).observer(
                FakeActorToolObserver(orchestrator_proxy)
            )

    def test_a_workspace_with_no_lock_refuses_every_run(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # A lost announcement degrades to "unconfigured" rather than to a run
        # admitted under no hold — which is a run two workers could both admit.
        _card, actor, _harness = exec_setup
        actor._lock = None

        start = actor.request_exec(AGENT, "echo hi")

        assert not start.run_id
        assert "no execution backend configured" in start.refusal
        assert sandbox_script.commands == []

    def test_a_lock_that_raises_is_refused_rather_than_crashing_the_caller(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # ``request_exec`` is an ASK: a raise here crosses the actor boundary
        # and reaches the agent as a crash rather than as an answer it can act
        # on. An unwritable metadata parent must be a refusal.
        #
        # It must be the ENVIRONMENT refusal, not the busy one, and asserting
        # only the shared prefix would not tell them apart: the whole reason
        # ``lock_unavailable`` exists beside ``exec_busy`` is that busy means
        # "retry and it will work" while this means "retrying in a loop will
        # not". A path that answered ``exec_busy()`` here would send the agent
        # round that loop, so the distinguishing wording is what is asserted.
        _card, actor, _harness = exec_setup
        actor._lock = _RaisingLock()

        start = actor.request_exec(AGENT, "echo hi")

        assert not start.run_id
        assert start.refusal == lock_unavailable()
        assert start.refusal.startswith("workspace busy")
        assert "environment failure" in start.refusal
        assert start.refusal != exec_busy()
        assert sandbox_script.commands == []
        assert actor._running is None

    def test_off_the_tool_channel_creates_no_sandbox_actor_and_probes_nothing(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The two halves of the capability have to agree on what "on" means. A
        # card that takes exec off the tool channel registers no callable, so it
        # must not resolve a backend, warn about the fallback, or create any
        # actor beyond the workspace's own.
        def explode() -> str:
            raise AssertionError("a card with exec off the tool channel probed the host")

        monkeypatch.setattr("akgentic.tool.sandbox._resolve_auto_mode", explode)
        card = WorkspaceTool(
            workspace_id=workspace_tree.name,
            workspace_exec=WorkspaceExec(expose=set()),
        )
        card.observer(FakeActorToolObserver(orchestrator_proxy))

        created = [config.name for _cls, config in orchestrator_proxy.create_calls]
        assert created == [workspace_actor_name(WORKSPACE_PATH)]
        assert orchestrator_proxy.resource_calls == []
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
        report = ExecReport(
            run_id="abc12345",
            result=ExecResult(stdout="out", stderr="err", exit_code=0),
        )
        assert ExecReport.model_validate(report.model_dump()) == report

        running = RunningExec(
            run_id="abc12345", agent_id=AGENT, cmd="pytest", started_at=1.0
        )
        assert RunningExec.model_validate(running.model_dump()) == running


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
        builder = attached(actor, "builder")
        start_run(actor, sandbox_script, agent=builder)

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

    def test_a_second_exec_is_refused_and_nothing_is_parked(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The queue is gone (ADR-051 Decision 5): a second command is refused
        # and its caller retries. What the queue was protecting is kept in the
        # *wording* rather than in a deque — the refusal names no run and no
        # agent, so a model's parallel batch still cannot read a sibling call's
        # id out of it and collect that as its own answer.
        _card, actor, harness = exec_setup
        first = start_run(actor, sandbox_script)

        second = actor.request_exec(AGENT_B, "echo again")

        assert not second.run_id
        assert second.refusal
        assert first not in second.refusal
        assert AGENT not in second.refusal
        # Nothing was parked: no id was issued, so the caller has nothing to
        # collect and the worker was never handed a second command.
        assert actor._recent_runs.get(AGENT_B) is None
        assert len(harness.runs) == 1
        finish_run(sandbox_script, harness)

    def test_the_card_raises_retriable_rather_than_waiting_its_turn(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        orchestrator_proxy: FakeOrchestratorProxy,
    ) -> None:
        # What the model actually sees, and the whole of the behaviour change:
        # ``RetriableError`` reaches pydantic-ai as a ``ModelRetry``, so the
        # collision costs one round trip instead of a place in a queue.
        _card, actor, harness = exec_setup
        second_card, _observer = exec_card_for(orchestrator_proxy, name="bob")
        start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="workspace busy"):
            tool_named(second_card, "workspace_exec")("echo again")

        assert len(harness.runs) == 1
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
        before = card.observation_for("notes.md")
        start_run(actor, sandbox_script)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_write", "notes.md", "mine\n")

        assert card.observation_for("notes.md") == before
        # And it recorded no write set, so no commit was made for it either.
        assert card._touched == []
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

    def test_a_run_that_cannot_be_submitted_fails_at_once_and_releases_it(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The submit fails BEFORE the request is made, and the assertions come
        # straight after ``request_exec`` returns — no poll in between. That is
        # the whole of "the hold is taken before the submit": a failure that
        # found no run to release would strand the tree, and one reported
        # asynchronously would leave the run marked RUNNING until some later
        # look. ``submit`` on an executor that has already been shut down raises
        # ``RuntimeError``, so the case is reachable rather than invented.
        card, actor, harness = exec_setup
        harness.executor.submit_raises = RuntimeError("cannot schedule new futures")

        start = actor.request_exec(AGENT, "echo hi")

        assert start.run_id  # the id was issued; the submit is what failed
        assert actor._running is None
        status = actor.exec_status(AGENT, start.run_id)
        assert status.state is ExecState.FAILED
        assert "cannot schedule new futures" in status.reason
        assert not sandbox_script.commands  # nothing reached the backend
        assert "Created" in mutate(card, "workspace_mkdir", "src")

    def test_a_late_report_does_not_clear_a_newer_lease(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        # The wedge release is what makes this reachable: a run whose hold was
        # given up past its budget can still report afterwards.
        _card, actor, _harness = exec_setup
        stale = new_run_id()
        actor._running = None
        current = actor.request_exec("agent-2", "echo current")

        actor.deliver(stale, ExecOutcome(stdout="", stderr="", exit_code=0))

        assert actor._running is not None
        assert actor._running.run_id == current.run_id


class TestADisallowedCommand:
    """What an agent is actually told when the allowlist refuses its command.

    The check runs on the **worker** thread, in ``ExecRunner._exec`` and again in
    each backend's own ``exec`` — so ``CommandNotAllowedError`` never propagates
    out of ``request_exec`` or ``exec_status`` to a caller. It is caught by
    ``perform`` and arrives as a reported failure instead, and this is where that
    is asserted end to end.

    **The check in the runner is the one that precedes provisioning**, and it is
    there for a cost rather than for the answer: the backends' own checks are
    what enforce the list, but they run after the lazy ``start()``, so without it
    a command that will be refused first builds an image and creates a container.
    """

    def test_it_is_reported_as_a_failure_naming_the_binary_and_the_allowed_list(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        # ExecRunner validates before the backend is ever reached, so the
        # command never lands on the fake at all. `ssh` is the exemplar because
        # it has to stay off the list for this to mean anything; a binary the
        # sandbox might plausibly want would eventually be added and turn this
        # into a test of nothing.
        start = actor.request_exec(card._agent_id, "ssh nowhere")
        assert start.run_id, start.refusal
        harness.join()

        status = actor.exec_status(card._agent_id, start.run_id)
        assert status.state is ExecState.FAILED
        assert "ssh" in status.reason
        assert "pytest" in status.reason  # the allowed list travels with it
        assert not sandbox_script.commands  # it was refused before the backend

        answer = mutate(card, "workspace_exec_result", start.run_id)
        assert "ssh" in answer

    def test_it_is_refused_before_the_backend_is_ever_provisioned(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        """A command that will be refused must not pay for a cold start first.

        ``start()`` is lazy and it is the expensive call — on the docker backend
        it builds an image and creates a container. Refusing after it means an
        agent's typo can cost minutes and leave a container behind, so the
        allowlist runs ahead of it.

        ``starts`` is empty rather than ``commands``: the sibling spec above
        already asserts nothing reached ``exec``, and a check that only moved
        earlier *within* the backend would pass that one unchanged.
        """
        card, actor, harness = exec_setup
        assert not sandbox_script.starts  # nothing provisioned yet

        start = actor.request_exec(card._agent_id, "ssh nowhere")
        assert start.run_id, start.refusal
        harness.join()

        assert sandbox_script.starts == [], (
            "a refused command provisioned the backend before it was refused"
        )
        assert actor.exec_status(card._agent_id, start.run_id).state is ExecState.FAILED

    def test_an_allowed_command_still_provisions_on_its_first_run(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        """The other half, without which the spec above passes on a broken lazy start.

        A ``start()`` that had simply stopped being called would satisfy "a
        refused command provisions nothing" perfectly, and this is what makes
        that reading unreachable.
        """
        card, actor, harness = exec_setup
        sandbox_script.gate.set()  # this run completes rather than blocking

        start = actor.request_exec(card._agent_id, "echo hello")
        assert start.run_id, start.refusal
        harness.join()

        assert len(sandbox_script.starts) == 1
        assert actor.exec_status(card._agent_id, start.run_id).state is ExecState.DONE

    def test_it_does_not_leave_the_tree_leased(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The tree is taken before the request is sent, so a command the
        # allowlist refuses is one more exit that has to release it.
        card, actor, harness = exec_setup
        actor.request_exec(AGENT, "ssh nowhere")
        harness.join()

        assert actor._running is None
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

        seen = card.observation_for("notes.md")
        assert seen is not None and seen.full
        finish_run(sandbox_script, harness)


# ---------------------------------------------------------------------------
# 29-8 — the lease covers the run, not the wait for the backend
# ---------------------------------------------------------------------------


class _TellTarget(Akgent[BaseConfig, BaseState]):
    """An actor with one tell-shaped method, for observing what a send does.

    Deliberately trivial: the spike below is about *the send primitive*, not
    about anything the receiver computes, so the receiver records the fact of
    delivery and nothing else.
    """

    notes: ClassVar[list[str]] = []
    noted: ClassVar[threading.Event] = threading.Event()

    def note(self, text: str) -> None:
        type(self).notes.append(text)
        type(self).noted.set()


class TestTheTwoSpikes:
    """The framework question the report path rests on, pinned by a spec.

    Not a guard over this package's own behaviour — a record of what the
    *framework* does, and the design is only correct while it holds. There used
    to be two spikes here. The first — that ``getChildrenOrCreate`` returns while
    a sandbox actor's ``on_start`` is still provisioning — pinned a property the
    exec path no longer rests on: nothing resolves a sandbox actor any more, and
    provisioning is ``ExecRunner``'s lazy ``start()`` on the worker thread,
    guarded by ``TestTheLazyStart``. It went with the actor.
    """

    def test_address_tell_raises_on_a_dead_actor_where_a_held_proxy_tell_is_silent(
        self, threaded_orchestrator_proxy: FakeOrchestratorProxy
    ) -> None:
        # Spike (b), and it is what decides the send primitive.
        #
        # ActorAddress.tell checks liveness and raises on this thread, so
        # _start_run learns immediately that the sandbox is gone and can fail
        # the run — which releases the gate and drains the queue.
        #
        # A tell proxy does not. Its calls go through Pykka's CallableProxy,
        # which has ASK semantics: a dead actor's ActorDeadError is set on the
        # returned future, and ProxyWrapper.tell_wrapper drops that future. The
        # send then vanishes with the run still marked running, and only a much
        # later poll would notice.
        _TellTarget.notes = []
        _TellTarget.noted = threading.Event()
        address = threaded_orchestrator_proxy.getChildrenOrCreate(
            _TellTarget, config=BaseConfig(name="tell-target", role="tester")
        )
        sender = SilentAgent(config=BaseConfig(name="sender", role="tester"))

        # Built while the target is alive, and held across its death — which is
        # the window _start_run would actually have: it resolves an address,
        # then sends, and the sandbox may stop in between.
        held = sender.proxy_tell(address, _TellTarget)
        held.note("alive")
        assert _TellTarget.noted.wait(timeout=HANDSHAKE_TIMEOUT_S), "the proxy never delivered"

        threaded_orchestrator_proxy.stop_all()

        with pytest.raises(pykka.ActorDeadError):
            address.tell(object())

        # Raises nothing — and delivers nothing. Both halves matter: silence
        # alone would be tolerable if the message still arrived.
        held.note("dead")
        assert _TellTarget.notes == ["alive"]

        # A tell proxy built fresh against an already-dead address does raise,
        # at construction — so the silence needs the proxy to predate the death.
        # Recorded because it is the half that makes the hazard easy to miss.
        with pytest.raises(pykka.ActorDeadError):
            sender.proxy_tell(address, _TellTarget).note("fresh")


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
        run_id = harness.runs[0].run_id
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
        # Started as the CARD's agent, because collection is now ownership-scoped
        # — an id the asker does not own comes back as unknown, whatever state it
        # is really in.
        run_id = start_run(actor, sandbox_script, agent=card._agent_id)
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
        run_id = start_run(actor, sandbox_script, agent=card._agent_id)
        finish_run(sandbox_script, harness)

        status = actor.exec_status(card._agent_id, run_id)
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
        for index in range(MAX_TRACKED_RUNS + 5):
            actor._track_run(AGENT, new_run_id(), f"echo {index}")

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
        actor._track_run(AGENT, never_ran, "echo nothing")
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

    def test_the_actors_own_methods_perform_no_sandbox_call(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The property the whole design rests on: #Workspace's own methods stay
        # O(1) and never reach the backend, which is what keeps its mailbox
        # draining while a run is held. exec_status runs a release check that can
        # reach _start_next() and a real submit, so the counts are what say it
        # did not.
        _card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script)
        commands_so_far = len(sandbox_script.commands)
        submits_so_far = len(harness.runs)
        starts_so_far = len(sandbox_script.starts)

        actor.exec_status(AGENT, run_id)
        actor.get(run_id)

        assert len(sandbox_script.commands) == commands_so_far
        assert len(harness.runs) == submits_so_far
        # ``start()`` counts too: provisioning is the expensive half, and the
        # whole reason it is lazy and on the worker is that it must never happen
        # on this thread.
        assert len(sandbox_script.starts) == starts_so_far
        finish_run(sandbox_script, harness)

    def test_the_result_cache_is_lru_capped(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        # The deferred base's cache half is the half this actor uses, and an
        # uncapped map on a team singleton leaks for the life of the team.
        _card, actor, _harness = exec_setup
        for index in range(actor.cache_capacity + 5):
            actor.deliver(f"run{index}", ExecOutcome(stdout="", stderr="", exit_code=0))

        assert len(actor._slots) == actor.cache_capacity


# ---------------------------------------------------------------------------
# AC8 — the budgets
# ---------------------------------------------------------------------------


def started_strategy(mode: str, workspace_path: Path) -> Any:
    """The backend of *mode*, as ``start()`` would have left it.

    Assembled rather than started: no bwrap, no sandbox-exec and no docker daemon
    has to be present for a budget to be asserted.
    """
    backend = REAL_STRATEGIES[mode]()
    backend.workspace_path = workspace_path
    backend.container_name = "sandbox-t1"
    return backend


def capture_budget(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    """Record the budget each backend hands the process, and return the list.

    One patch target for all four: the budget is now an argument to
    ``communicate()`` inside ``ProcessBackend._run``, which is the single place a
    process is started from. Captured rather than measured — nothing slow runs.
    """
    captured: list[float | None] = []

    def fake_popen(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            communicate=lambda timeout=None: (captured.append(timeout), ("", ""))[1],
            returncode=0,
            kill=lambda: None,
        )

    monkeypatch.setattr("akgentic.tool.sandbox.backend.subprocess.Popen", fake_popen)
    return captured


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
        # binary has to be present. A budget that stops at the caller is
        # decoration, so this asserts it reaches the process in all four.
        backend = started_strategy(mode, tmp_path)

        captured = capture_budget(monkeypatch)
        backend.exec("echo hi", "", 3.25)

        assert captured == [3.25]

    @pytest.mark.parametrize("mode", ["local", "bwrap", "seatbelt", "docker"])
    def test_no_budget_falls_back_to_the_backends_own(
        self, mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # ``None`` keeps each backend's default, so no existing caller changed
        # behaviour when the parameter arrived.
        backend = started_strategy(mode, tmp_path)

        captured = capture_budget(monkeypatch)
        backend.exec("echo hi", "", None)

        assert captured == [DEFAULT_BACKEND_TIMEOUT_S]

    def test_a_card_budget_above_the_workers_is_clamped(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, _ = exec_card_for(orchestrator_proxy)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        actor.configure_exec(
            ExecConfig(mode="local", workspace_path=WORKSPACE_PATH, timeout_s=999.0)
        )

        start_run(actor, sandbox_script)
        finish_run(sandbox_script, harness)

        # The command gets the ceiling, never the card's 999 s — and EXACTLY
        # the ceiling, not a hair under it. Nothing is spent before the command
        # any more: there is no readiness phase to shave the difference off.
        assert len(sandbox_script.timeouts) == 1
        assert sandbox_script.timeouts[0] == MAX_EXEC_BUDGET_S

    def test_the_default_poll_covers_the_run_budget_and_the_report_margin(self) -> None:
        # AC1/AC3/AC9. The default is the sentinel, so the wait it resolves to is
        # the whole run plus the margin the sandbox needs to report it — and no
        # more, which is the half that keeps this a budget rather than a licence.
        #
        # The wait is (attempts - 1) delays, not attempts: poll_deferred sleeps
        # between looks and never after the last. Asserting the product instead
        # would overstate the real wait by one delay and hide a poll that stops
        # short of the budget it claims to cover.
        params = WorkspaceExec()
        assert params.poll_attempts == -1
        run_budget = effective_budget(params.timeout_s)
        attempts = poll_attempts_within(
            params.poll_attempts, params.poll_delay_seconds, run_budget
        )
        wait = (attempts - 1) * params.poll_delay_seconds
        assert wait > run_budget
        assert wait <= run_budget + EXEC_REPORT_MARGIN_S

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

        monkeypatch.setattr("akgentic.tool.workspace.card.execution.poll_deferred", capture)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        sandbox_script.gate.set()

        tool_named(card, "workspace_exec")(cmd="pytest")
        harness.join()

        attempts, delay = seen[0]
        assert attempts * delay <= effective_budget(DEFAULT_EXEC_TIMEOUT_S)
        assert attempts >= 1

    def test_a_poll_that_already_fits_is_left_alone(self) -> None:
        assert poll_attempts_within(12, 0.4, 15.0) == 12
        assert poll_attempts_within(0, 0.4, 15.0) == 0  # opting out of polling stands
        assert poll_attempts_within(1000, 1.0, 15.0) == 15
        assert poll_attempts_within(1000, 60.0, 15.0) == 1  # never below one look
        # An explicit positive count is clamped to the run budget ALONE — the
        # report margin belongs to the sentinel, which is the only setting asking
        # to wait for the whole thing.
        assert poll_attempts_within(1000, 1.0, 15.0) * 1.0 <= 15.0

    def test_the_sentinel_resolves_against_the_budget_plus_the_margin(self) -> None:
        # AC1/AC3. Arithmetic, not wall clock: the resolution is what the wiring
        # does once, so it is asserted where it happens rather than by sleeping.
        # ``(attempts - 1) * delay`` is the wall clock poll_deferred actually
        # spends — it sleeps between looks, never after the last one. The case
        # that makes this worth spelling out is delay == margin: the product
        # form passes there while the real wait ends exactly at the budget,
        # reporting a timeout for a run that was about to answer.
        for delay, budget in ((0.5, 15.0), (0.5, 20.0), (1.0, 15.0), (0.1, 2.0), (1.0, 3.0)):
            attempts = poll_attempts_within(-1, delay, budget)
            wait = (attempts - 1) * delay
            assert wait > budget, (delay, budget)
            assert wait <= budget + EXEC_REPORT_MARGIN_S, (delay, budget)

    def test_the_sentinel_never_resolves_below_one_look(self) -> None:
        assert poll_attempts_within(-1, 60.0, 1.0) == 1
        assert poll_attempts_within(-1, 0.0, 15.0) == 1  # no wall clock to divide

    def test_the_zero_opt_out_survives_the_sentinel(self) -> None:
        # AC2. Zero is not "a small sentinel": it is the only way to take a run
        # id without looking once, and the sentinel must not have absorbed it.
        assert poll_attempts_within(0, 0.5, 15.0) == 0
        assert WorkspaceExec(poll_attempts=0).poll_attempts == 0

    def test_below_the_sentinel_is_a_validation_error(self) -> None:
        # AC2. Rejected by the card, not silently read as the sentinel — two
        # spellings of one meaning is how the meaning drifts.
        with pytest.raises(ValidationError):
            WorkspaceExec(poll_attempts=-2)


# ---------------------------------------------------------------------------
# 29-9 — the default waits out the run, and says so honestly when it cannot
# ---------------------------------------------------------------------------


class TestWaitingOutTheRun:
    """The sentinel's behaviour, end to end through the card's own callable.

    Every test here drives completion by releasing the fake backend's gate, and
    every budget is small enough that the poll's *failure* budget is a second or
    two of wall clock that is never actually spent — ``poll_deferred`` returns on
    the first settled look.
    """

    def test_a_command_that_completes_returns_its_output_and_no_run_id(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # AC7: the common case. Under the sentinel the model is never handed a
        # run id to ACT on — which is the whole point, because a run id it cannot
        # act on is what produced the busy-loop this default removes. The id it
        # now reads is its own, on a provenance line, and nothing asks it to
        # collect anything later.
        card, _ = exec_card_for(
            orchestrator_proxy, poll_attempts=-1, poll_delay_seconds=0.01, timeout_s=1.0
        )
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        sandbox_script.stdout = "hello from the run"
        sandbox_script.gate.set()

        answer = tool_named(card, "workspace_exec")(cmd="echo hi")
        harness.join()

        run_id = harness.runs[0].run_id
        assert "exit_code: 0 (OK)" in answer
        assert "hello from the run" in answer
        assert answer.startswith(f"Run {run_id} - exit_code:")
        assert "workspace_exec_result" not in answer  # no handoff, only the result

    def test_the_two_exhaustion_messages_are_chosen_by_the_budget_in_force(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # AC4 and AC5, asserted together so the pair cannot drift apart. The two
        # cards differ in exactly one setting and the run they start is
        # identical, which is what makes "chosen by which budget was in force,
        # not by anything about the run" a property rather than a claim.
        #
        # The two are now exhausted by two different mechanisms, because they ARE
        # two different mechanisms: since the queue landed, the sentinel is
        # deadline-driven (it waits out its turn as well as its run) while a
        # positive count is still attempt-driven. So the sentinel gets a deadline
        # shrunk to 0.05 s by zeroing the report margin, and the bounded card
        # keeps its stubbed poll. Neither sleeps for a second; both exhaust for
        # real, rather than being handed the message directly.
        monkeypatch.setattr("akgentic.tool.workspace.execution.EXEC_REPORT_MARGIN_S", 0.0)
        monkeypatch.setattr(
            "akgentic.tool.workspace.card.execution.poll_deferred",
            lambda fetch, attempts, delay: None,
        )
        waiting_card, _ = exec_card_for(
            orchestrator_proxy,
            name="sentinel",
            poll_attempts=-1,
            poll_delay_seconds=0.01,
            timeout_s=0.05,
        )
        bounded_card, _ = exec_card_for(
            orchestrator_proxy, name="bounded", poll_attempts=2, poll_delay_seconds=0.01
        )
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        sandbox_script.gate.set()

        # The run never settles as far as either poller can see, so what ends the
        # sentinel's wait is its deadline and nothing else.
        original = actor.exec_status
        monkeypatch.setattr(
            actor,
            "exec_status",
            lambda agent_id, run_id: original(agent_id, run_id).model_copy(
                update={"state": ExecState.RUNNING, "outcome": None}
            ),
        )

        waited = tool_named(waiting_card, "workspace_exec")(cmd="echo hi")
        harness.join()
        bounded = tool_named(bounded_card, "workspace_exec")(cmd="echo hi")
        harness.join()

        waited_run, bounded_run = (request.run_id for request in harness.runs)
        # The sentinel's answer: the budget was spent, and it says which budget.
        assert waited == timed_out(waited_run, 0.05)
        assert waited_run in waited
        assert "0.05s" in waited
        assert "still holds the workspace" in waited
        assert "workspace_exec_result" in waited
        # The instruction that produced four polls in six seconds — an agent
        # inside a tool call has no next turn to be told to wait for.
        assert "next turn" not in waited
        # The bounded card asked for a run id and still gets the handoff.
        assert bounded == in_progress(bounded_run)
        assert "next turn" in bounded

    def test_a_command_killed_by_its_budget_comes_back_as_an_outcome(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # AC3, and the test that goes red if EXEC_REPORT_MARGIN_S is dropped.
        # The command is killed AT its budget and the readable exit_code 124
        # outcome lands a moment after — so a poll bounded at exactly the budget
        # misses it by that moment and reports a timeout instead. Here the run
        # budget and the poll delay are equal, which puts the whole difference
        # between the two designs in the margin: with it, 21 looks; without it,
        # exactly one.
        card, _ = exec_card_for(
            orchestrator_proxy, poll_attempts=-1, poll_delay_seconds=0.05, timeout_s=0.05
        )
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        sandbox_script.raise_with = subprocess.TimeoutExpired(cmd="echo hi", timeout=0.05)

        # The gate is released by the first look and not before, so the first
        # look is guaranteed to find the run unsettled — which is what makes the
        # margin, rather than a scheduling accident, the thing under test.
        original = actor.exec_status
        released = threading.Event()

        def release_after_the_first_look(agent_id: str, run_id: str) -> Any:
            status = original(agent_id, run_id)
            if not released.is_set():
                released.set()
                sandbox_script.gate.set()
            return status

        monkeypatch.setattr(actor, "exec_status", release_after_the_first_look)

        answer = tool_named(card, "workspace_exec")(cmd="echo hi")
        harness.join()

        assert released.is_set(), "the run was never looked at"
        assert f"exit_code: {TIMED_OUT_EXIT_CODE}" in answer
        assert "was killed" in answer
        assert "without reporting" not in answer  # not the timeout message

    def test_a_running_run_asked_about_directly_is_still_in_progress(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # AC6: workspace_exec_result is untouched. That path has no budget of its
        # own to have exhausted, so "still in progress" is the accurate answer.
        card, actor, harness = exec_setup
        run_id = start_run(actor, sandbox_script, agent=card._agent_id)

        answer = tool_named(card, "workspace_exec_result")(run_id=run_id)

        assert answer == in_progress(run_id)
        finish_run(sandbox_script, harness)

    def test_the_poll_budget_cannot_extend_the_run_budget(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # AC9: one run budget, and the poll is not it. A card asking to poll
        # for a thousand seconds hands the backend the same run budget as any
        # other — the poll buys more looking, never more running.
        #
        # This is the one card here whose poll delay is a whole second, so the
        # sleep is stubbed: a look that lands before the run settles would
        # otherwise cost real wall clock, and the assertions below are about the
        # budgets handed to the backend, not about how long anything waited.
        monkeypatch.setattr("akgentic.tool.core.deferred.time.sleep", lambda _: None)
        card, _ = exec_card_for(
            orchestrator_proxy, poll_attempts=1000, poll_delay_seconds=1.0, timeout_s=999.0
        )
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        sandbox_script.gate.set()

        tool_named(card, "workspace_exec")(cmd="echo hi")
        harness.join()

        assert harness.runs[0].timeout_s == MAX_EXEC_BUDGET_S
        command_budget = sandbox_script.timeouts[0]
        assert command_budget is not None
        assert command_budget <= MAX_EXEC_BUDGET_S


class TestTheStartAnswerIsExactlyOne:
    """``ExecStart`` enforces its own invariant rather than documenting it.

    Folded in from epic 29's deferred findings. Every caller branches on
    ``if not start.run_id``, so an answer carrying neither field would reach the
    agent as an empty refusal, and one carrying both would run a command the
    actor had already decided to refuse.
    """

    def test_a_run_id_alone_is_accepted(self) -> None:
        assert ExecStart(run_id="abc12345").run_id == "abc12345"

    def test_a_refusal_alone_is_accepted(self) -> None:
        assert ExecStart(refusal="busy").refusal == "busy"

    def test_neither_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExecStart()

    def test_both_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExecStart(run_id="abc12345", refusal="busy")

    def test_it_still_round_trips(self) -> None:
        start = ExecStart(run_id="abc12345")
        assert ExecStart.model_validate(start.model_dump()) == start


# ---------------------------------------------------------------------------
# AC6 / AC11 — discovery and the commit
# ---------------------------------------------------------------------------


@requires_git
class TestTheDiscoveredWriteSet:
    @pytest.fixture
    def exec_setup(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[WorkspaceTool, WorkspaceActor, ExecHarness]:
        """The module fixture with the journal on.

        The card's default is off, and this class is about what the journal
        discovered, so it opts in rather than depending on a default it is not
        asserting.
        """
        card, _observer = exec_card_for(orchestrator_proxy, git_journal=True)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        return card, actor, harness

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
        builder = attached(actor, "builder")
        sandbox_script.files = [("out.txt", "x\n")]
        start_run(actor, sandbox_script, agent=builder)
        finish_run(sandbox_script, harness)

        head = journal_log(workspace_tree)[-1]
        assert head.author_name == "builder"
        assert builder in head.author_email

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
        card, _ = exec_card_for(orchestrator_proxy, git_journal=False)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
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
# Two cards over one tree, and a backend registered from outside the package
# ---------------------------------------------------------------------------


@requires_git
class TestTwoCardsOverOneTree:
    def test_each_card_commits_as_its_own_agent(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Two agents sharing one workspace share one #Workspace and one sandbox,
        # yet every run is journalled under the agent that requested it — the
        # author is carried by the run, not by the actor.
        alice, _ = exec_card_for(
            orchestrator_proxy,
            name="alice",
            poll_attempts=50,
            poll_delay_seconds=0.01,
            git_journal=True,
        )
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        bob, _ = exec_card_for(
            orchestrator_proxy, name="bob", poll_attempts=50, poll_delay_seconds=0.01
        )

        sandbox_script.gate.set()
        sandbox_script.files = [("one.txt", "1\n")]
        tool_named(alice, "workspace_exec")(cmd="make one")
        harness.join()
        sandbox_script.files = [("two.txt", "2\n")]
        tool_named(bob, "workspace_exec")(cmd="make two")
        harness.join()

        log = journal_log(workspace_tree)
        assert log[-2].author_name == "alice"
        assert log[-1].author_name == "bob"


class InjectedBackend(FakeBackend):
    """A backend registered from outside the package, as a deployment would.

    A subclass rather than the fixture's fake itself, so the assertion below is
    on *this* class having been resolved — the fixture already sits at the
    ``local`` key, and a test that resolved it would prove nothing about the
    registration it made.
    """


class TestARegisteredBackendIsReached:
    """``SANDBOX_BACKEND_CLASSES`` is the live extension point, and this is its contract.

    A deployment assigns its own backend into the registry before any card is
    constructed. What matters is not the spelling of the import but that
    ``workspace_exec`` — the wiring *and* the run — resolves through the registry
    at call time and therefore reaches the injected class.

    **This spec moved from the actor registry to the backend registry**, because
    that is where the extension point moved — and the actor registry is now
    gone with the actor. A spec written against a registry nothing reads would
    install a class nothing reaches and pass without executing a line of it —
    the exact shape of vacuity this suite is written against. The backend
    registry's own shape is asserted in ``tests/sandbox/test_registry.py`` and
    its mapping in ``tests/sandbox/test_backend_kill.py``.
    """

    def test_a_backend_assigned_into_the_registry_runs_the_command(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Registered under a key a card may name, before the card exists —
        # exactly the sequence the sandbox README documents.
        monkeypatch.setitem(SANDBOX_BACKEND_CLASSES, "docker", InjectedBackend)
        card = WorkspaceTool(
            workspace_id=workspace_tree.name,
            workspace_exec=WorkspaceExec(mode="docker", poll_attempts=50, poll_delay_seconds=0.01),
        )
        card.observer(FakeActorToolObserver(orchestrator_proxy))
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)

        # The wiring resolved the injected class, not the shipped docker backend.
        assert actor._runner is not None
        assert type(actor._runner.backend) is InjectedBackend

        sandbox_script.gate.set()
        sandbox_script.stdout = "ran in the injected backend"
        answer = tool_named(card, "workspace_exec")(cmd="make build")
        harness.join()
        harness.close()

        assert "ran in the injected backend" in answer
        assert sandbox_script.commands == [("make build", "")]


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
        self, wired_card: WorkspaceTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # What a root-owned file from a container produces: publication by rename
        # means the host process must be able to replace the inode.
        def denied(path: str, data: bytes) -> None:
            raise PermissionError(13, "Permission denied")

        assert wired_card._workspace is not None
        monkeypatch.setattr(wired_card._workspace, "write", denied)

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "new.md", "x")

        message = str(refusal.value)
        assert "Path escapes workspace root" not in message
        assert "did not escape" in message


# ---------------------------------------------------------------------------
# 47-1 — admission's third answer: the queue, and who a run belongs to
#
# Every spec here asserts on MODEL FIELDS first — status.state, .run_id,
# .command, script.commands, harness.runs, actor._running,
# actor._in_flight, and the marker on disk — and only then
# on a rendered string. A guard whose whole assertion is ``"X" in result`` is
# satisfied by static text that was already there, which is how two inert guards
# shipped in the story before this one.
# ---------------------------------------------------------------------------


class TestTheTreeIsGivenBackAndTheRetryRuns:
    """G1 — a refused caller's retry runs, and answers with its OWN output.

    The queue's guards, re-pointed rather than deleted. What the queue was
    protecting is not the ordering — that is deliberately gone — but the two
    invariants underneath it: the work is never silently lost, and one run's
    answer is never another's. Both survive the queue, and both are asserted
    here on the retry rather than on a dequeue.
    """

    def test_a_refused_caller_that_retries_runs_with_its_own_output(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The original defect in one test. Two commands, two distinguishable
        # outputs: the second must come back with ITS OWN, because the incident
        # was an agent collecting a sibling call's result and recording it as
        # the answer to a question it never asked.
        _card, actor, harness = exec_setup
        sandbox_script.stdout_by_cmd = {"echo head": "HEAD OUTPUT", "echo tail": "TAIL OUTPUT"}
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)

        refused = actor.request_exec(AGENT_B, "echo tail")

        assert refused.refusal and not refused.run_id
        assert sandbox_script.commands == [("echo head", "")]  # nothing else started

        finish_run(sandbox_script, harness)
        tail = actor.request_exec(AGENT_B, "echo tail")  # the retry
        harness.join()

        assert tail.run_id and tail.run_id != head
        assert sandbox_script.commands == [("echo head", ""), ("echo tail", "")]
        collected = actor.exec_status(AGENT_B, tail.run_id)
        assert collected.state is ExecState.DONE
        assert collected.run_id == tail.run_id
        assert collected.command == "echo tail"
        assert collected.outcome is not None
        assert collected.outcome.stdout == "TAIL OUTPUT"
        # And the head's answer is untouched and different — the two never merge.
        head_status = actor.exec_status(AGENT, head)
        assert head_status.outcome is not None
        assert head_status.outcome.stdout == "HEAD OUTPUT"
        assert format_status(collected).startswith(f"Run {tail.run_id} - exit_code:")

    def test_a_head_that_fails_still_gives_the_tree_back(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # Both exits reach _finish_run — deliver AND fail — so a run that blew
        # up must release the marker exactly as one that succeeded does. A
        # release wired to the success path alone locks the tree for the whole
        # staleness window after every failure.
        _card, actor, harness = exec_setup
        sandbox_script.raise_with = RuntimeError("the backend fell over")
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)

        finish_run(sandbox_script, harness)

        assert actor.exec_status(AGENT, head).state is ExecState.FAILED
        assert not exec_marker().exists()
        assert actor.request_exec(AGENT_B, "echo tail").run_id
        harness.join()

    def test_nothing_is_recorded_for_a_refused_caller(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # A refusal issues no id, so there is nothing in flight, nothing
        # tracked, and nothing for the holder's record to have been overwritten
        # by. An admission path that recorded the refused caller anyway would
        # leave it an id it could never collect.
        _card, actor, harness = exec_setup
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)

        refused = actor.request_exec(AGENT_B, "echo tail")

        assert [request.run_id for request in harness.runs] == [head]
        assert not refused.run_id
        assert actor._recent_runs.get(AGENT_B) is None
        assert actor._running is not None
        assert actor._running.run_id == head  # still the head's, untouched
        assert sandbox_script.commands == [("echo head", "")]
        finish_run(sandbox_script, harness)


class TestARunBelongsToTheAgentThatStartedIt:
    """G3 — exec_status answers only the asking agent's runs."""

    def test_a_second_agent_cannot_collect_the_first_agents_finished_run(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        sandbox_script.stdout_by_cmd = {"echo mine": "MINE"}
        mine = start_run(actor, sandbox_script, cmd="echo mine", agent=AGENT)
        finish_run(sandbox_script, harness)
        theirs = actor.request_exec(AGENT_B, "echo theirs")
        harness.join()

        assert actor.exec_status(AGENT, mine).state is ExecState.DONE

        foreign = actor.exec_status(AGENT_B, mine)

        assert foreign.state is ExecState.UNKNOWN
        assert foreign.run_id == mine
        assert foreign.outcome is None
        assert foreign.command == ""
        # The ASKER's ids come back, never the owner's — that is what makes the
        # answer recoverable rather than a dead end.
        assert foreign.recent_run_ids == [theirs.run_id]
        assert mine not in foreign.recent_run_ids
        rendered = format_status(foreign)
        assert rendered.startswith(f"Unknown run id '{mine}'")
        assert "MINE" not in rendered

    def test_a_foreign_run_is_unknown_whatever_state_it_is_really_in(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The ownership gate is ahead of every other branch, so no branch below
        # it can answer a foreign run — not the result cache, not the error map,
        # not _in_flight. (The queued case went with the queue: a refused
        # caller never receives an id for anyone to ask about.)
        _card, actor, harness = exec_setup
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        assert actor.exec_status(AGENT, head).state is ExecState.RUNNING

        foreign = actor.exec_status(AGENT_C, head)

        assert foreign.state is ExecState.UNKNOWN
        assert foreign.recent_run_ids == []
        finish_run(sandbox_script, harness)

    def test_a_failed_run_is_not_collectable_by_another_agent_either(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, harness = exec_setup
        sandbox_script.raise_with = RuntimeError("the backend fell over")
        mine = start_run(actor, sandbox_script, agent=AGENT)
        finish_run(sandbox_script, harness)

        assert actor.exec_status(AGENT, mine).state is ExecState.FAILED

        foreign = actor.exec_status(AGENT_B, mine)

        assert foreign.state is ExecState.UNKNOWN
        assert foreign.reason == ""
        assert "fell over" not in format_status(foreign)


class TestADoneResultNamesItsRunAndItsCommand:
    """G4 — the provenance line, without which two answers are indistinguishable."""

    def test_the_status_carries_the_command_and_the_rendering_leads_with_both(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        card, actor, harness = exec_setup
        sandbox_script.stdout = "===== 5 passed ====="
        run_id = start_run(actor, sandbox_script, cmd="pytest -q", agent=card._agent_id)
        finish_run(sandbox_script, harness)

        status = actor.exec_status(card._agent_id, run_id)

        assert status.state is ExecState.DONE
        assert status.run_id == run_id
        assert status.command == "pytest -q"
        collected = mutate(card, "workspace_exec_result", run_id)
        assert collected.startswith(f"Run {run_id} - exit_code:")
        assert "exit_code: 0 (OK)" in collected
        assert "5 passed" in collected

    def test_two_runs_of_different_commands_are_told_apart_by_their_answers(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The incident restated as a property: before this, the two answers were
        # byte-identical whenever the outputs happened to match, so a collected
        # result could not be checked against the command that produced it.
        _card, actor, harness = exec_setup
        first = start_run(actor, sandbox_script, cmd="python --version", agent=AGENT)
        finish_run(sandbox_script, harness)
        sandbox_script.started.clear()
        sandbox_script.gate.clear()
        second = start_run(actor, sandbox_script, cmd="pytest --version", agent=AGENT)
        finish_run(sandbox_script, harness)

        first_status = actor.exec_status(AGENT, first)
        second_status = actor.exec_status(AGENT, second)

        assert first_status.command == "python --version"
        assert second_status.command == "pytest --version"
        assert format_status(first_status) != format_status(second_status)


LONG_COMMAND = "sh -c 'sleep 30'"
"""A real command that outlives the whole teardown budget, for the real backend.

``sh`` is on the allowlist and every ``sh`` this runs on execs ``sleep`` in place
for a single trailing command, so the direct child *is* the sleeping process and
``Popen.kill`` reaches it. A shell that forked instead would leave a grandchild
holding the pipes — which is the hazard :data:`EXEC_SHUTDOWN_GRACE_S` bounds, and
which the wedged-worker spec below reproduces deliberately rather than by luck.

**Its budget is deliberately not shortened.** Under a short budget the *budget*
would end the run and every assertion below would pass with the kill removed —
the exact narrowing these specs are written against.
"""


@pytest.fixture
def live_exec(
    orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
) -> Generator[tuple[WorkspaceTool, WorkspaceActor], None, None]:
    """An exec-capable card over a **real** ``LocalBackend`` and the real executor.

    No ``sandbox_script``, so nothing is injected into the registry and the
    command genuinely runs; no harness, so the actor keeps the
    ``ThreadPoolExecutor`` its own ``on_start`` built. This is the only setup in
    which "the child is dead, and it was killed" is a statement about a process.
    """
    card, _observer = exec_card_for(
        orchestrator_proxy, poll_attempts=0, timeout_s=MAX_EXEC_BUDGET_S
    )
    _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
    assert isinstance(actor, WorkspaceActor)
    yield card, actor
    actor._executor.shutdown(wait=False, cancel_futures=True)


def await_child(actor: WorkspaceActor) -> subprocess.Popen[str]:
    """Wait until the real backend is holding a process handle, and return it.

    Polled rather than waited on an event because the handle belongs to
    ``ProcessBackend`` and nothing about the production path announces it. The
    bound is a failure budget: a run that never reaches ``Popen`` is a failure,
    not something to wait longer for.
    """
    runner = actor._runner
    assert runner is not None
    backend = runner.backend
    assert isinstance(backend, LocalBackend)
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
    while time.monotonic() < deadline:
        proc = backend._running
        if proc is not None:
            return proc
        time.sleep(0.01)
    raise AssertionError("the run never reached a subprocess")


class TestTheExecutorAndItsWorker:
    """AC 1, 2, 9, 10 — one worker per workspace, and what runs on it."""

    def test_every_workspace_owns_one_single_worker_executor(
        self, wired_card: WorkspaceTool, orchestrator_proxy: FakeOrchestratorProxy
    ) -> None:
        # AC1. Created unconditionally, exec capability or not: a
        # ThreadPoolExecutor spawns no thread until the first submit, so a
        # workspace that never runs a command pays for the object and nothing
        # else — which is what makes the branch-free version correct rather than
        # merely tidy. More than one worker would break nothing and prove
        # nothing: the tree admits one run at a time, so a second could only idle.
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)

        assert isinstance(actor._executor, ThreadPoolExecutor)
        assert actor._executor._max_workers == 1
        assert actor._executor._threads == set()  # nothing spawned, nothing submitted
        assert actor._pending is None

    def test_the_backend_is_never_touched_on_the_actors_thread(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # AC2 and AC9's thread half, in one place because they are one property:
        # neither ``start`` nor ``exec`` may happen where the mailbox drains.
        # ``start`` is the expensive one — on the docker backend it is a
        # ``docker build`` — and it is exactly what would serialise behind every
        # read, mutation and poll in the team if it ran here.
        _card, actor, harness = exec_setup
        actor_thread = threading.get_ident()
        run_id = start_run(actor, sandbox_script)

        # Answered while the command is genuinely in flight, not afterwards.
        assert actor.exec_status(AGENT, run_id).state is ExecState.RUNNING
        finish_run(sandbox_script, harness)

        assert sandbox_script.threads, "the backend was never reached"
        assert all(ident != actor_thread for _what, ident in sandbox_script.threads)
        # One worker, so one thread for both calls.
        assert len({ident for _what, ident in sandbox_script.threads}) == 1

    def test_start_runs_once_before_the_first_command_and_not_again(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # AC9. Lazy — nothing is provisioned by binding a card — and once, so a
        # second run does not rebuild a container it already has.
        _card, actor, harness = exec_setup
        assert sandbox_script.starts == []  # binding provisioned nothing

        start_run(actor, sandbox_script, cmd="echo one")
        finish_run(sandbox_script, harness)
        sandbox_script.started.clear()
        sandbox_script.gate.clear()
        start_run(actor, sandbox_script, cmd="echo two")
        finish_run(sandbox_script, harness)

        assert sandbox_script.starts == [WORKSPACE_PATH]
        assert sandbox_script.commands == [("echo one", ""), ("echo two", "")]
        # Ordered, not merely counted: a start that followed the first exec
        # would still be one start.
        assert sandbox_script.events[0] == ("start", WORKSPACE_PATH)
        assert sandbox_script.events[1] == ("exec-enter", "echo one")

    def test_a_start_that_fails_is_the_runs_answer_and_the_next_run_retries(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # AC10. The flag is set only AFTER start() returns, so a daemon that was
        # down is retried rather than latched off until the team restarts. It
        # costs a failing probe per run in a broken deployment, which is the
        # honest half of the trade.
        _card, actor, harness = exec_setup
        sandbox_script.gate.set()
        sandbox_script.start_raises = RuntimeError("docker daemon is not running")

        first = actor.request_exec(AGENT, "echo one")
        assert first.run_id, first.refusal
        harness.join()

        status = actor.exec_status(AGENT, first.run_id)
        assert status.state is ExecState.FAILED
        assert "docker daemon is not running" in status.reason
        assert sandbox_script.commands == []  # nothing ran
        assert actor._running is None  # and the tree was handed back

        sandbox_script.start_raises = None
        second = actor.request_exec(AGENT, "echo two")
        assert second.run_id, second.refusal
        harness.join()

        assert len(sandbox_script.starts) == 2  # retried, not latched off
        assert actor.exec_status(AGENT, second.run_id).state is ExecState.DONE


class TestTheBackendFakeIsReallyReached:
    """The migration's own guard: a fixture installed but unreached is silent.

    ``#Workspace`` resolves no sandbox actor, so a fake left at the retired
    actor registry's ``local`` key was installed, restored, and never called —
    and every spec in this file went green while exercising the real
    ``LocalBackend`` or nothing at all. This is the sentinel that says otherwise.
    """

    def test_the_answer_could_only_have_come_from_the_fake(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The discriminator is the OUTPUT, not the absence of an error: the real
        # LocalBackend answers "hi" to ``echo hi`` and cannot answer this, and
        # exit code 7 is not something ``echo`` produces either.
        card, actor, harness = exec_setup
        sandbox_script.stdout = "<<<only-the-fake-backend-says-this>>>"
        sandbox_script.exit_code = 7
        run_id = start_run(actor, sandbox_script, cmd="echo hi", agent=card._agent_id)
        finish_run(sandbox_script, harness)

        answer = mutate(card, "workspace_exec_result", run_id)

        assert "<<<only-the-fake-backend-says-this>>>" in answer
        assert "exit_code: 7" in answer
        assert sandbox_script.commands == [("echo hi", "")]

    def test_the_runner_holds_the_injected_backend_and_not_the_shipped_one(
        self, exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness]
    ) -> None:
        # The other half: the object itself, resolved through the registry the
        # fixture writes into.
        _card, actor, _harness = exec_setup

        runner = actor._runner
        assert runner is not None
        assert isinstance(runner.backend, FakeBackend)
        assert not isinstance(runner.backend, LocalBackend)


class TestReplacingTheConfiguration:
    """AC 8 — an equal config changes nothing; a different one stops the old backend."""

    def test_an_equal_config_keeps_the_same_runner_and_stops_nothing(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The common case, and close to the only one. Rebuilding here would leak
        # the old backend — on the docker backend, a container with nobody left
        # to stop it.
        _card, actor, _harness = exec_setup
        config = actor._exec_config
        assert config is not None
        first = actor._runner

        actor.configure_exec(config.model_copy())

        assert actor._runner is first
        assert sandbox_script.stops == 0
        assert sandbox_script.events == []

    def test_the_config_carries_no_team_so_two_teams_announce_equal_configs(self) -> None:
        """A hosted tree is bound by several teams; nothing a backend does is one team's.

        Whole-set equality on the fields, so a team id added back — under any
        name — fails it, and so does a field silently dropped. The equality
        below is the consequence the set exists for: two cards from two teams
        with the same settings announce configs ``configure_exec`` cannot tell
        apart.
        """
        assert set(ExecConfig.model_fields) == {"mode", "workspace_path", "timeout_s"}
        first_team = ExecConfig(mode="local", workspace_path=WORKSPACE_PATH, timeout_s=5.0)
        second_team = ExecConfig(mode="local", workspace_path=WORKSPACE_PATH, timeout_s=5.0)
        assert first_team == second_team

    def test_a_config_record_still_carrying_a_team_loads_without_it(self) -> None:
        """A record written before the team id was deleted must load, and drop it.

        ``ExecConfig`` travels by tell and is not persisted today, so this pins
        the unknown-key rule rather than a stored stream: a later
        ``extra="forbid"`` would turn every caller still passing the old key
        into a crash at bind, and this goes red first.
        """
        stored = {
            "mode": "local",
            "team_id": "team-42",
            "workspace_path": WORKSPACE_PATH,
            "timeout_s": 5.0,
        }

        restored = ExecConfig.model_validate(stored)

        assert restored == ExecConfig(mode="local", workspace_path=WORKSPACE_PATH, timeout_s=5.0)
        assert "team_id" not in restored.model_dump()

    def test_a_second_exec_card_keeps_the_runner_with_a_run_in_flight(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A second exec card binding the same tree must not tear down a live run.

        **Re-pointed by decision.** This used to be two *teams* sharing one
        hosted actor; two teams get two actors again since 52-5 (*Ruling A*), so
        the two cards that can reach one runner are now two cards of one team.
        The hazard is unchanged and is the reason the spec exists:
        ``configure_exec`` is last-writer-wins, and a second announcement with
        an identical config must not rebuild the runner — rebuilding kills the
        command inside it and, on docker, removes the container under it.
        """
        harness: ExecHarness | None = None
        try:
            _first_card, first_observer = exec_card_for(orchestrator_proxy, name="a")
            _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
            assert isinstance(actor, WorkspaceActor)
            harness = ExecHarness(actor, orchestrator_proxy)
            harness.install(monkeypatch)
            runner = actor._runner
            assert runner is not None
            start_run(actor, sandbox_script, agent=str(first_observer.myAddress.agent_id))

            _second_card, second_observer = exec_card_for(orchestrator_proxy, name="b")

            assert second_observer.myAddress.agent_id != first_observer.myAddress.agent_id
            # One tree, one actor, two cards — and the second bind was a hit.
            assert len(orchestrator_proxy.children) == 1
            assert actor._runner is runner
            assert sandbox_script.stops == 0
            assert ("stop",) not in sandbox_script.events
            assert actor._running is not None  # still in flight, never killed
            assert sandbox_script.kills == 0
            finish_run(sandbox_script, harness)
        finally:
            if harness is not None:
                harness.close()

    def test_a_different_config_stops_the_old_runner_and_builds_a_new_one(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        _card, actor, _harness = exec_setup
        config = actor._exec_config
        assert config is not None
        first = actor._runner
        assert first is not None

        actor.configure_exec(config.model_copy(update={"timeout_s": config.timeout_s + 1.0}))

        assert actor._runner is not first
        assert actor._runner is not None
        assert actor._runner.backend is not first.backend
        assert sandbox_script.stops == 1  # the replaced backend was released
        assert isinstance(actor._runner.backend, FakeBackend)
        assert actor._exec_config is not None
        assert actor._exec_config.timeout_s == config.timeout_s + 1.0

    def test_a_backend_that_raises_on_release_does_not_take_the_binding_down(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Same requirement as teardown's, and the same reason: the degradation
        # from a swallowed failure is a refused run, which is visible and
        # recoverable, while a raise on a binding path takes the card down.
        _card, actor, _harness = exec_setup
        config = actor._exec_config
        assert config is not None
        first = actor._runner
        assert first is not None
        monkeypatch.setattr(
            first.backend, "stop", _raise_on_stop, raising=True
        )

        actor.configure_exec(config.model_copy(update={"timeout_s": config.timeout_s + 1.0}))

        assert actor._runner is not first  # the replacement still happened


def _raise_on_stop() -> None:
    """A backend release that fails, for the wrapped-stop path."""
    raise RuntimeError("the backend fell over on release")


class TestTeardownEndsTheRunInFlight:
    """AC 12, 13 — a real child, really killed, and killed before the drain.

    Every natural guard for an ordered teardown is vacuous — ``on_stop`` did not
    raise, the executor reports itself shut down, the queue is empty — so these
    observe the **effect** on a real process instead.

    **What each assertion actually carries, measured rather than assumed.**
    ``ProcessBackend.stop()`` calls ``kill()`` itself, so with a real backend the
    child is killed by teardown's *fourth* step even when the second is deleted:
    the exit-signal assertion therefore proves that no command outlives
    teardown, and it does **not** distinguish step 2 from step 4. That
    distinction is what the elapsed bound carries here, with two orders of
    magnitude of margin — the kill lands in milliseconds, while an ``on_stop``
    that reaches the drain with a live child burns the whole grace — and what
    ``TestTeardownIsOrderedAndBounded`` carries structurally, by requiring
    ``("kill",)`` in the record at all.

    Deleting either step is red in both places. Saying which assertion does
    which job is the point: a guard whose owner is guessed is a guard nobody
    has checked.
    """

    def test_the_child_is_killed_and_the_kill_precedes_the_drain(
        self, live_exec: tuple[WorkspaceTool, WorkspaceActor]
    ) -> None:
        _card, actor = live_exec
        start = actor.request_exec(AGENT, LONG_COMMAND)
        assert start.run_id, start.refusal
        proc = await_child(actor)

        began = time.monotonic()
        actor.on_stop()
        elapsed = time.monotonic() - began

        # AC12 — the command did not outlive teardown, and it was KILLED rather
        # than allowed to finish. ``poll() is not None`` alone would also pass
        # for a command that simply ended, which is why the signal is named:
        # nothing but SIGKILL produces -9 here.
        assert proc.wait(timeout=HANDSHAKE_TIMEOUT_S) == -signal.SIGKILL
        assert proc.returncode == -signal.SIGKILL
        # AC13 — killing FIRST is what makes the drain cheap: the signal lands in
        # milliseconds, so an ``on_stop`` in this order returns two orders of
        # magnitude inside the grace. Reaching the drain with a live child burns
        # the whole of it, whether that is because the kill was deleted or
        # because it was moved after.
        assert elapsed < EXEC_SHUTDOWN_GRACE_S / 2

    def test_the_command_really_would_have_outlived_the_teardown(
        self, live_exec: tuple[WorkspaceTool, WorkspaceActor]
    ) -> None:
        # The positive control for the spec above. Without it, "the child is
        # dead" is compatible with a command that was never going to be alive:
        # a budget too short, a binary that exits at once, a run that never
        # started. This shows the same command, on the same setup, still running
        # well past the grace when nothing tears it down.
        _card, actor = live_exec
        start = actor.request_exec(AGENT, LONG_COMMAND)
        assert start.run_id, start.refusal
        proc = await_child(actor)

        assert proc.poll() is None
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=EXEC_SHUTDOWN_GRACE_S + 0.5)
        assert proc.poll() is None

        proc.kill()


class TestTeardownIsOrderedAndBounded:
    """AC 14–17 — the four steps, their order, and what a failing one may not skip."""

    def test_a_clean_teardown_does_not_leave_the_tree_locked(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # The marker outlives this process, which is what makes this step owed
        # at all: a team that stops mid-run and leaves it behind locks the tree
        # for the next worker until the staleness window expires — and nothing
        # in that worker can tell the difference between "held" and "abandoned".
        #
        # **This is the REPORT path, not the teardown one**, and the distinction
        # is worth stating because the name alone hides it: a real ``kill()``
        # ends the blocked child, so the run reports inside the drain and
        # ``_finish_run`` is what gives the marker back. Teardown's own release
        # then finds nothing to do — which is why deleting it outright leaves
        # this spec green. The guard that holds teardown to AC 12 is the wedged
        # child below, where no report can arrive at all.
        _card, actor, harness = exec_setup
        start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        assert exec_marker().is_file()  # the control: it really was taken

        actor.on_stop()
        sandbox_script.gate.set()
        harness.join()

        assert not exec_marker().exists()
        # And the tree is genuinely free, not merely tidy: another process's
        # backend can take it at once.
        assert FileLockBackend().acquire(WORKSPACE_PATH, _ticket()).run_id

    def test_a_wedged_run_gives_the_tree_back_at_teardown_and_only_then(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # AC 12, guarded where the step is the ONLY thing that can satisfy it: a
        # child that ignores the kill never returns, so no report arrives, so
        # ``_finish_run`` never runs and the marker can only be given back by
        # teardown. Remove the step and this goes red; the happy-path spec above
        # does not.
        #
        # It also pins the ORDER, on the state at the moment of the release
        # rather than on the end state — the only way to see it, since by the
        # time ``on_stop`` returns every step has run either way. The marker is
        # what a SECOND PROCESS decides admission on, so handing it back while
        # this process's child may still be writing admits a run into a tree it
        # is not alone in, and that run's own discovery sweeps the dying child's
        # files into a commit attributed to whoever asked next — the
        # misattribution ``commit_out_of_band`` exists to prevent, across a
        # process boundary this time.
        _card, actor, harness = exec_setup
        sandbox_script.kill_releases = False  # the child that ignores the kill
        observer = _TeardownOrderLock(actor._lock, sandbox_script)
        actor._lock = observer
        start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        assert exec_marker().is_file()

        actor.on_stop()

        assert ("exec-return", "echo head") not in sandbox_script.events, (
            "the child returned, so this is no longer the wedged case"
        )
        assert observer.kills_at_release == [1], "the tree was given back before the kill"
        assert observer.stopped_at_release == [True], (
            "the tree was given back before the backend was stopped"
        )
        assert not exec_marker().exists()
        assert FileLockBackend().acquire(WORKSPACE_PATH, _ticket()).run_id
        sandbox_script.gate.set()
        harness.join()

    def test_a_release_that_raises_does_not_raise_past_on_stop(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # ``_release_lock`` swallows and ``_teardown_step`` wraps on top of it,
        # so a filesystem error on the way out costs a marker left for staleness
        # to reclaim — never a raise out of ``on_stop``, which would strand the
        # base class's own teardown behind it. The kill and the backend stop
        # precede the release and are asserted as the control: this spec is
        # about the swallow, not about which step runs first.
        _card, actor, harness = exec_setup
        start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        actor._lock = _RaisingLock()
        sandbox_script.gate.set()

        actor.on_stop()  # must not raise

        harness.join()
        assert sandbox_script.kills == 1
        assert ("stop",) in sandbox_script.events

    def test_the_kill_lands_on_a_live_run_and_the_release_follows_the_drain(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # AC16, asserted on POSITIONS in one shared record rather than on
        # presence. Presence alone cannot tell "released after the worker
        # drained" from "released around the same time", which is the whole
        # difference between tearing a container down around a command that is
        # still writing and tearing it down after.
        #
        # The tail is the one deliberate wall clock in this suite, and it is
        # here to make a cross-thread order observable rather than raced: with
        # step 4 moved ahead of step 3, ``stop`` lands inside this window.
        _card, actor, harness = exec_setup
        sandbox_script.exec_tail_s = 0.2
        start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)

        actor.on_stop()
        harness.join()

        events = sandbox_script.events
        assert ("kill",) in events
        assert ("stop",) in events
        assert ("exec-return", "echo head") in events
        assert events.index(("kill",)) < events.index(("exec-return", "echo head"))
        assert events.index(("stop",)) > events.index(("exec-return", "echo head"))

    def test_a_worker_that_ignores_the_kill_is_abandoned_rather_than_waited_out(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # AC14, and the whole reason the drain is bounded at all.
        # ``shutdown(wait=True)`` takes no timeout, and a worker wedged in
        # ``_run``'s second drain never returns — so teardown would hang for
        # ever, not merely slowly.
        #
        # The load-bearing assertion is the one about the record, not the one
        # about the clock: with an unbounded drain, ``exec-return`` is in the
        # record BEFORE on_stop can return, whatever the machine's timing.
        _card, actor, harness = exec_setup
        sandbox_script.kill_releases = False
        start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)

        began = time.monotonic()
        actor.on_stop()
        elapsed = time.monotonic() - began
        during_teardown = list(sandbox_script.events)

        assert ("exec-return", "echo head") not in during_teardown
        assert ("stop",) in during_teardown  # step 4 still ran
        assert elapsed < HANDSHAKE_TIMEOUT_S
        assert elapsed >= EXEC_SHUTDOWN_GRACE_S  # it did wait for the grace
        sandbox_script.gate.set()
        harness.join()

    def test_a_kill_that_raises_still_leaves_the_drain_and_the_release_done(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # AC17. Every step is wrapped SEPARATELY, which is stricter than the
        # single wrapper the queue clear used to carry — and this is the reason:
        # one wrapper around the group would let a raising kill leave the
        # executor undrained and the container up.
        _card, actor, harness = exec_setup
        sandbox_script.kill_raises = RuntimeError("the backend fell over on kill")
        start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        sandbox_script.gate.set()

        actor.on_stop()  # must not raise
        harness.join()

        assert sandbox_script.kills == 1
        assert ("stop",) in sandbox_script.events
        assert actor._executor._inner._shutdown  # type: ignore[attr-defined]

    def test_on_stop_on_a_workspace_that_never_ran_a_command_does_nothing(
        self, wired_card: WorkspaceTool, orchestrator_proxy: FakeOrchestratorProxy
    ) -> None:
        # AC18. **Inert on its own** and worth nothing without AC 12–17 beside
        # it: it would pass for an ``on_stop`` whose body was deleted. What it
        # covers is the branch, not the behaviour — no runner, no backend, and an
        # executor that never spawned a thread.
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        assert actor._runner is None

        actor.on_stop()  # must not raise

        assert actor._executor._shutdown


class TestTeardownChainsToTheBase:
    def test_on_stop_chains_to_the_base(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # An on_stop that forgets super() leaves the actor's own teardown undone,
        # which is silent and permanent.
        _card, actor, _harness = exec_setup
        chained: list[bool] = []
        # Find the class that actually supplies ``on_stop`` above the actor rather
        # than assuming a position: the workspace actor is assembled from mixins,
        # so ``__mro__[1]`` is whichever mixin happens to be listed first and is
        # not the base whose teardown this guards.
        base = next(
            klass
            for klass in type(actor).__mro__[1:]
            if "on_stop" in vars(klass)
        )
        monkeypatch.setattr(base, "on_stop", lambda _self: chained.append(True))

        actor.on_stop()

        assert chained == [True]


class TestTheStatusVocabulary:
    """What ``settled`` means, now that admission has only two answers."""

    def test_only_done_and_failed_are_settled(self) -> None:
        # ``poll_deferred`` stops at the first settled look, so a state wrongly
        # marked settled hands the agent a run id on the ordinary path where the
        # command finishes in milliseconds — the degraded answer, for the common
        # case. (The queued state went with the queue: a refused caller never
        # receives an id for a poll to ask about.)
        assert not ExecStatus(state=ExecState.RUNNING, run_id="abc12345").settled
        assert not ExecStatus(state=ExecState.UNKNOWN, run_id="abc12345").settled
        assert ExecStatus(state=ExecState.DONE, run_id="abc12345").settled
        assert ExecStatus(state=ExecState.FAILED, run_id="abc12345").settled


# ---------------------------------------------------------------------------
# 47-2 — the sandbox reports back, and the two releases that need no report
#
# Same rule as the G-classes above: every assertion here reads a MODEL FIELD
# first — status.state, status.reason, actor._running, harness.runs,
# script.commands, journal_log(...) — and only then, if at
# all, a rendered string.
# ---------------------------------------------------------------------------


class TestTheHandlerAlwaysReports:
    """N1-N4 — whatever the command does, one report comes back."""

    def test_a_backend_that_raises_is_reported_and_the_tree_is_given_back(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # N1. The handler's finally is the whole contract: a run whose report is
        # dropped holds the tree until staleness releases it — and now that the
        # hold is a file, "until staleness" means for the next process too. So a
        # backend that falls over has to arrive as an answer, and the marker has
        # to go with it.
        _card, actor, harness = exec_setup
        sandbox_script.raise_with = RuntimeError("the backend fell over")
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)

        finish_run(sandbox_script, harness)

        status = actor.exec_status(AGENT, head)
        assert status.state is ExecState.FAILED
        assert "the backend fell over" in status.reason
        assert actor._running is None
        assert not exec_marker().exists()
        tail = actor.request_exec(AGENT_B, "echo tail")
        assert tail.run_id
        harness.join()
        assert sandbox_script.commands == [("echo head", ""), ("echo tail", "")]

    def test_a_timeout_from_the_backend_is_an_outcome_not_a_failure(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # N2. "Too slow" is the ordinary case for a shell, so it is an answer
        # the agent reads — the exit code timeout(1) uses, and the budget named.
        # Mapped to fail() instead, it would be uncollectable as an outcome.
        _card, actor, harness = exec_setup
        sandbox_script.raise_with = subprocess.TimeoutExpired(cmd="make slow", timeout=1)
        run_id = start_run(actor, sandbox_script, cmd="make slow")
        finish_run(sandbox_script, harness)

        status = actor.exec_status(AGENT, run_id)
        assert status.state is ExecState.DONE
        assert status.outcome is not None
        assert status.outcome.timed_out
        assert status.outcome.exit_code == TIMED_OUT_EXIT_CODE
        assert f"{effective_budget(DEFAULT_EXEC_TIMEOUT_S):g}s" in status.outcome.stderr

    def test_an_unbalanced_quote_reaches_the_caller_as_a_reported_failure(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # N3, the second half — the disallowed binary is TestADisallowedCommand.
        # exec() raises CommandParseError before any backend is reached, on the
        # sandbox's thread, so nothing but the handler's except is between it and
        # the agent.
        _card, actor, harness = exec_setup
        start = actor.request_exec(AGENT, 'echo "unbalanced')
        assert start.run_id, start.refusal
        harness.join()

        status = actor.exec_status(AGENT, start.run_id)
        assert status.state is ExecState.FAILED
        assert "quotes" in status.reason
        assert not sandbox_script.commands
        assert actor._running is None

    def test_an_exception_with_no_message_names_its_type_instead_of_escaping(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # ``str(RuntimeError())`` is the empty string, and an empty ``error``
        # fails ExecReport's exactly-one validator — so building the failure
        # report raises INSIDE the except clause and is lost onto a future
        # nobody reads, leaving the run holding the tree until the grace
        # releases it. That is the exact failure ``or repr(exc)`` exists to
        # prevent, reached through the runner's own answer.
        #
        # ``perform`` is called directly here rather than through the executor:
        # an escape on the worker's thread is swallowed onto the future, which
        # is precisely why it must be asserted on a thread that fails.
        _card, actor, harness = exec_setup
        sandbox_script.raise_with = RuntimeError()
        sandbox_script.gate.set()
        start = actor.request_exec(AGENT, "echo hi")
        assert start.run_id, start.refusal
        harness.join()
        runner = actor._runner
        assert runner is not None
        submitted = harness.runs[0]

        runner.perform(  # must not raise
            run_id=submitted.run_id,
            cmd=submitted.cmd,
            cwd=submitted.cwd,
            timeout_s=submitted.timeout_s,
            reply_to=harness.workspace_address,
        )

        status = actor.exec_status(AGENT, start.run_id)
        assert status.state is ExecState.FAILED
        assert "RuntimeError" in status.reason  # the type, since there is no message
        assert actor._running is None

    def test_one_request_per_run_carries_the_clamped_budget_and_a_reply_address(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # N4. One submit, never two — a second would run the command twice
        # against one tree — and request() is never reached at all: the deferred
        # base's worker half is not used here, only its cache.
        _card, actor, harness = exec_setup

        def no_worker(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("#Workspace spawned a deferred worker")

        monkeypatch.setattr(actor, "request", no_worker)
        run_id = start_run(actor, sandbox_script, cmd="echo hi")
        finish_run(sandbox_script, harness)

        assert len(harness.runs) == 1
        submitted = harness.runs[0]
        assert submitted.run_id == run_id
        assert submitted.cmd == "echo hi"
        assert submitted.timeout_s == effective_budget(DEFAULT_EXEC_TIMEOUT_S)
        # The address is captured on the ACTOR's thread and handed over, which is
        # what stops the worker reaching into ``self`` for it. ``myAddress``
        # builds a fresh wrapper per read, so identity is the wrong test —
        # naming the same actor is the property.
        assert submitted.reply_to.name == actor.myAddress.name


@requires_git
class TestACommitPrecedesTheRelease:
    """N5 — run A's write set is committed before the tree is given back."""

    @pytest.fixture
    def journalled(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[WorkspaceActor, ExecHarness]:
        exec_card_for(orchestrator_proxy, git_journal=True)
        _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert isinstance(actor, WorkspaceActor)
        harness = ExecHarness(actor, orchestrator_proxy)
        harness.install(monkeypatch)
        return actor, harness

    def test_the_write_set_is_committed_before_the_lock_is_given_back(
        self,
        journalled: tuple[WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # The release is the LAST thing _finish_run does, after the commit, and
        # this asserts the ORDER rather than the end state — which is the only
        # way to see it, since after the call both have happened either way.
        #
        # It matters more with a file lock than it did with a queue: the next
        # acquirer may be in ANOTHER PROCESS, so a release before the commit
        # hands over a tree still showing a.txt as untracked, and that run's own
        # discovery sweeps it into a commit attributed to whoever asked next.
        actor, harness = journalled
        ann = attached(actor, "ann")
        sandbox_script.files_by_cmd = {"make a": [("a.txt", "A\n")]}
        observer = _ReleaseObserver(actor._lock, workspace_tree)
        actor._lock = observer
        start_run(actor, sandbox_script, cmd="make a", agent=ann)

        finish_run(sandbox_script, harness)

        assert observer.clean_at_release == [True], (
            "the tree was given back before the run's write set was committed"
        )
        assert journal_log(workspace_tree)[-1].files == ["a.txt"]

    def test_bs_files_are_absent_from_as_discovered_commit(
        self,
        journalled: tuple[WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # The attribution half, on the retry path the queue's dequeue used to
        # cover: B's command runs against a tree A already committed, so neither
        # run's discovery can sweep up the other's files.
        actor, harness = journalled
        ann = attached(actor, "ann")
        bert = attached(actor, "bert")
        sandbox_script.files_by_cmd = {
            "make a": [("a.txt", "A\n")],
            "make b": [("b.txt", "B\n")],
        }
        before = journal_log(workspace_tree)
        start_run(actor, sandbox_script, cmd="make a", agent=ann)
        assert actor.request_exec(bert, "make b").refusal  # refused while A holds it
        finish_run(sandbox_script, harness)

        assert actor.request_exec(bert, "make b").run_id  # the retry is granted
        harness.join()

        log = journal_log(workspace_tree)
        assert len(log) == len(before) + 2
        a_commit, b_commit = log[-2], log[-1]
        assert a_commit.author_name == "ann"
        assert a_commit.files == ["a.txt"]
        assert "b.txt" not in a_commit.files
        assert b_commit.author_name == "bert"
        assert b_commit.files == ["b.txt"]


class TestAReportThatCannotLand:
    """N10 — ``#Workspace`` is stopping, and the worker has nowhere to report."""

    def test_a_dead_reply_address_does_not_escape_the_worker(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The guard the body carried over from the sandbox actor, and its purpose
        # is stronger here: the address is now #Workspace ITSELF, which may
        # already be part-way through its own on_stop when the worker returns.
        # ``ActorAddress.tell`` raises synchronously on a dead address, so
        # without the guard the exception surfaces on a Future nobody reads and
        # the report is lost with no log line at all.
        #
        # ``perform`` is driven directly, on a thread that fails: run through the
        # executor, an escape would be swallowed onto the future and this spec
        # would pass with the guard removed.
        _card, actor, harness = exec_setup
        sandbox_script.gate.set()
        runner = actor._runner
        assert runner is not None
        harness.workspace_address.dead = True

        runner.perform(  # must not raise
            run_id="run-0001",
            cmd="echo hi",
            cwd="",
            timeout_s=1.0,
            reply_to=harness.workspace_address,
        )

        # It really did run — otherwise "nothing escaped" would be the answer to
        # a command that was never attempted.
        assert sandbox_script.commands == [("echo hi", "")]

    def test_the_dead_address_case_is_reachable(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The positive control for the spec above: the same stand-in really does
        # raise when it is dead, so "the worker swallowed it" is a statement
        # about the guard rather than about an address that never raises.
        _card, _actor, harness = exec_setup
        harness.workspace_address.dead = True

        with pytest.raises(pykka.ActorDeadError):
            harness.workspace_address.tell(ExecReport(run_id="run-0001", error="boom"))


class TestAWedgedChild:
    """N7, N8, N9b — a subprocess that ignores the kill, and the report that follows."""

    def test_past_the_budget_and_the_grace_the_gate_is_released(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # N7. Two clocks are moved, never the wall clock: the run's own
        # ``started_at``, which is what this actor's release predicate reads, and
        # the **marker's mtime**, which is what a mutation now reads. By then the
        # backend has killed the child, so this is not a race against a live
        # writer.
        #
        # **Re-pointed by decision.** A mutation used to reach the actor, so it
        # both triggered the release and proceeded. It is gated card-side now and
        # answers from the marker on disk, which is the only clock a second
        # worker could ever have shared — so a hold aged only in this process's
        # memory is not stale to it, and rightly so. The release is triggered by
        # the next exec request, which is the one path that still reaches the
        # predicate.
        card, actor, harness = exec_setup
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        assert actor._running is not None
        budget = effective_budget(DEFAULT_EXEC_TIMEOUT_S)
        aged = time.time() - (budget + LEASE_GRACE_S + 1.0)
        actor._running.started_at = time.monotonic() - (budget + LEASE_GRACE_S + 1.0)
        os.utime(exec_marker(), (aged, aged))

        assert "Created" in mutate(card, "workspace_mkdir", "src")

        # The mutation proceeded without touching the actor at all, so the
        # in-memory record is still there until something asks the actor.
        assert actor._running is not None
        assert actor.exec_status(AGENT, head).state is ExecState.RUNNING
        finish_run(sandbox_script, harness)

    def test_a_hold_aged_only_in_memory_still_refuses_a_mutation(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        """The other side of the same decision, and the reason it is right.

        ``_running.started_at`` is one process's memory. A second worker over
        the same mounted tree has its own, sees neither this one nor its ageing,
        and would keep writing under a hold this process had privately decided
        was dead. The marker's mtime is the only clock both of them can read, so
        it is the one the gate answers from.
        """
        card, actor, harness = exec_setup
        start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        budget = effective_budget(DEFAULT_EXEC_TIMEOUT_S)
        assert actor._running is not None
        actor._running.started_at = time.monotonic() - (budget + LEASE_GRACE_S + 1.0)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_mkdir", "src")

        finish_run(sandbox_script, harness)

    def test_the_release_gives_the_marker_back_as_well_as_the_record(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        workspace_tree: Path,
    ) -> None:
        # N9's wedge half, re-pointed at the file. Clearing ``_running`` alone
        # would let mutations resume while the tree stayed locked ON DISK until
        # the marker's own staleness — so the next command would be refused by a
        # run this actor has already declared dead. The marker's mtime and the
        # record's started_at agree by construction, and this is the one path
        # that could make them disagree.
        _card, actor, harness = exec_setup
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        assert exec_marker().is_file()
        assert actor._running is not None
        budget = effective_budget(DEFAULT_EXEC_TIMEOUT_S)
        actor._running.started_at = time.monotonic() - (budget + LEASE_GRACE_S + 1.0)

        # A mutation no longer reaches this actor at all, so the one path left
        # that runs the predicate is the next exec request — which is also the
        # caller that needs the tree back.
        retry_start = actor.request_exec(AGENT_B, "echo tail")

        assert retry_start.run_id
        assert not exec_marker().exists() or exec_marker().is_file()
        # Admitted at once, without waiting out the marker's own staleness
        # window: the release gave the marker back before the acquire.
        assert retry_start.run_id != head
        assert [request.run_id for request in harness.runs] == [head, retry_start.run_id]
        finish_run(sandbox_script, harness)

    def test_inside_the_grace_the_mutation_is_still_refused(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
    ) -> None:
        # The other side of the same boundary, and what makes the grace itself
        # load-bearing rather than decorative: past the budget alone the run
        # still holds the tree.
        card, actor, harness = exec_setup
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        assert actor._running is not None
        budget = effective_budget(DEFAULT_EXEC_TIMEOUT_S)
        actor._running.started_at = time.monotonic() - (budget + LEASE_GRACE_S / 2)

        with pytest.raises(RetriableError, match="workspace busy"):
            mutate(card, "workspace_mkdir", "src")

        assert actor._running is not None
        assert actor._running.run_id == head
        finish_run(sandbox_script, harness)

    def test_the_late_report_commits_nothing_and_clears_nothing(
        self,
        exec_setup: tuple[WorkspaceTool, WorkspaceActor, ExecHarness],
        sandbox_script: SandboxScript,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # N8. Its outcome is still cached, so its owner collects it — what the
        # late report must not do is touch the tree or the newer run's hold.
        _card, actor, harness = exec_setup
        head = start_run(actor, sandbox_script, cmd="echo head", agent=AGENT)
        assert actor._running is not None
        budget = effective_budget(DEFAULT_EXEC_TIMEOUT_S)
        actor._running.started_at = time.monotonic() - (budget + LEASE_GRACE_S + 1.0)
        actor._holding_run()  # the release the next mutation or poll would do
        assert actor._running is None
        tail = actor.request_exec(AGENT_B, "echo tail")  # takes the freed tree
        assert tail.run_id
        assert actor._running is not None
        assert actor._running.run_id == tail.run_id
        discovered: list[str] = []
        monkeypatch.setattr(
            actor._journal,
            "commit_discovered",
            lambda identity, capability, detail="": discovered.append(detail),
        )

        finish_run(sandbox_script, harness)

        # "echo head" is absent: the late report committed nothing as its agent.
        assert discovered == ["echo tail"]
        assert actor.exec_status(AGENT, head).state is ExecState.DONE
