"""Fixtures and test doubles for the ``#Workspace`` actor and the recording read path.

The doubles here reach the actor only through the public surface a card uses —
``getChildrenOrCreate`` on a fake orchestrator, then ``proxy_ask``. The fake
orchestrator holds the real actor instances, which is what lets the singleton
test prove that an observation recorded through one card is visible through
another. A handful of assertions do read a card's or an actor's private
attribute where there is no public equivalent — which tree an actor took, which
proxy a card bound — and they say so where they do it.

Shaped after ``tests/notification/conftest.py``; deliberately a copy rather than
an import, because a test package is not a library for other test packages.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_address_impl import ActorAddressImpl
from akgentic.core.agent import Akgent, AkgentType
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.tool.core import ToolState
from akgentic.tool.core.deferred import DeferredPayload
from akgentic.tool.sandbox.actor import ExecResult, SandboxActor
from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.execution import ExecWorker
from akgentic.tool.workspace.journal import git_dir_for
from akgentic.tool.workspace.models import MutationOutcome, Observation
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool

from tests.conftest import MockActorAddress

WORKSPACE_NAME = "test-workspace"
"""The ``workspace_id`` the wired cards below share."""

HANDSHAKE_TIMEOUT_S = 5.0
"""Upper bound on a thread handshake — never a delay, only a failure budget."""

GIT_ON_PATH = shutil.which("git") is not None
"""Whether this host can run the journal at all.

The journal is the suite's first dependency on an external binary, and it is kept
contained: this one probe, the skip marker below, and — for the *absence* path —
a patched resolver rather than a mutated ``PATH``. Mutating the session's ``PATH``
would leak into every other test that shells out.
"""

requires_git = pytest.mark.skipif(not GIT_ON_PATH, reason="git is not on PATH")


@dataclass
class Commit:
    """One commit as the tests read it — parsed fields, never a formatted log line.

    Attributes:
        sha: The full hash.
        author_name: ``%an`` — the agent's display name, or ``out-of-band``.
        author_email: ``%ae``.
        parents: ``%P`` split — linear history means at most one.
        subject: ``%s``.
        files: Paths this commit touched.
    """

    sha: str
    author_name: str
    author_email: str
    parents: list[str]
    subject: str
    files: list[str]


def _git(tree: Path, *args: str) -> str:
    """Run one read-only git command against *tree*'s sibling journal.

    Reads under the same neutralised configuration the journal writes under. A
    developer whose ``~/.gitconfig`` carries ``core.autocrlf`` would otherwise
    see ``git show`` hand back different bytes from the ones committed, and the
    content assertions would go red on their machine and nowhere else.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    result = subprocess.run(
        [
            "git",
            "--git-dir",
            str(git_dir_for(tree)),
            "--work-tree",
            str(tree),
            *args,
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout


def journal_log(tree: Path) -> list[Commit]:
    """Return *tree*'s journal, oldest commit first.

    Assertions are made against these parsed fields rather than against a
    formatted log string, so a change in git's default output cannot turn a test
    red without a behaviour changing.
    """
    raw = _git(tree, "log", "--reverse", "--format=%H%x1f%an%x1f%ae%x1f%P%x1f%s")
    commits: list[Commit] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        sha, author_name, author_email, parents, subject = line.split("\x1f")
        files = _git(tree, "show", "--name-only", "--format=", sha).split()
        commits.append(
            Commit(
                sha=sha,
                author_name=author_name,
                author_email=author_email,
                parents=parents.split(),
                subject=subject,
                files=files,
            )
        )
    return commits


def journal_body(tree: Path, sha: str) -> str:
    """Return one commit's message body — everything below the subject line."""
    full = _git(tree, "log", "-1", "--format=%B", sha)
    _subject, _, body = full.partition("\n")
    return body.strip()


def git_show(tree: Path, revision: str) -> str:
    """Return the content of one object in *tree*'s journal, e.g. ``<sha>:notes.md``."""
    return _git(tree, "show", revision)


def journal_branches(tree: Path) -> list[str]:
    """Return every branch in *tree*'s journal — linear history means exactly one."""
    return _git(tree, "for-each-ref", "--format=%(refname:short)", "refs/heads").split()


def working_tree_is_clean(tree: Path) -> bool:
    """Whether git sees nothing to commit — ``-uall`` so an untracked directory expands."""
    return not _git(tree, "status", "--porcelain", "-uall").strip()


class SilentAgent(Akgent[BaseConfig, BaseState]):
    """A do-nothing agent, used to mint a real (serializable) ``ActorAddress``."""


class FakeOrchestratorProxy:
    """Get-or-create singletons by config name, exactly as the orchestrator does.

    With *live* set, the actors it creates are genuinely started on their own
    thread and handed out behind a real ``ActorAddressImpl``. That is what lets a
    test reach one through a real ``ProxyWrapper`` and assert a property of the
    mailbox rather than of a stand-in.
    """

    def __init__(self, live: bool = False) -> None:
        # The second element is a live actor instance in the inert mode and a
        # Pykka proxy over one in the live mode — both answer the same calls.
        self.children: dict[str, tuple[ActorAddress, Any]] = {}
        self.create_calls: list[tuple[type[Akgent[Any, Any]], BaseConfig]] = []
        self.live = live
        self._refs: list[Any] = []

    def getChildrenOrCreate(  # noqa: N802 — mirrors the orchestrator's method name
        self, actor_class: type[Akgent[Any, Any]], config: BaseConfig
    ) -> ActorAddress:
        self.create_calls.append((actor_class, config))
        existing = self.children.get(config.name)
        if existing is not None:
            return existing[0]
        if self.live:
            ref = actor_class.start(config=config)
            self._refs.append(ref)
            address: ActorAddress = ActorAddressImpl(ref)
            self.children[config.name] = (address, ref.proxy())
            return address
        actor = actor_class(config=config)
        actor.on_start()
        address = MockActorAddress(config.name, config.role)
        self.children[config.name] = (address, actor)
        return address

    def actor_for(self, address: ActorAddress) -> Any:
        """Return the actor behind *address*, or ``None`` when it is unknown."""
        for known_address, actor in self.children.values():
            if known_address is address:
                return actor
        return None

    def stop_all(self) -> None:
        """Stop every created actor — a live one on its thread, an inert one in place."""
        for ref in self._refs:
            ref.stop()
        self._refs.clear()
        if not self.live:
            for _, actor in self.children.values():
                actor.on_stop()
        self.children.clear()


class FakeActorToolObserver:
    """``ActorToolObserver`` stand-in wired to a :class:`FakeOrchestratorProxy`.

    *workspace_proxy*, when given, is handed back by ``proxy_ask`` in place of the
    live actor, and *workspace_tell_proxy* likewise by ``proxy_tell``. That is how
    the counting, failing and busy stand-ins below reach the card without any of
    them having to impersonate an orchestrator — and how a test can tell the two
    proxies apart, which is the only way to assert that a read records through
    the *tell* one.

    With no stand-in given and a live orchestrator, both methods build real
    proxies over the actor's address, through the agent this observer holds.
    """

    def __init__(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        name: str = "alice",
        workspace_proxy: object | None = None,
        workspace_tell_proxy: object | None = None,
    ) -> None:
        self._agent = SilentAgent(config=BaseConfig(name=name, role="tester"))
        self._address: ActorAddress = ActorAddressImpl(self._agent.actor_ref)
        self._orchestrator: ActorAddress | None = MockActorAddress("orchestrator")
        self._orchestrator_proxy = orchestrator_proxy
        self._workspace_proxy = workspace_proxy
        self._workspace_tell_proxy = workspace_tell_proxy
        self._team_id = uuid.uuid4()
        self._state_carrier = SimpleNamespace(tool_state=ToolState())
        self.events: list[object] = []
        self.ask_targets: list[ActorAddress] = []
        self.tell_targets: list[ActorAddress] = []

    @property
    def myAddress(self) -> ActorAddress:  # noqa: N802
        return self._address

    @property
    def state(self) -> SimpleNamespace:
        return self._state_carrier

    @property
    def orchestrator(self) -> ActorAddress | None:
        return self._orchestrator

    @property
    def team_id(self) -> uuid.UUID:
        return self._team_id

    def notify_event(self, event: object) -> None:
        self.events.append(event)

    def proxy_ask(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
        timeout: int | None = None,
    ) -> Any:
        if actor is self._orchestrator:
            return self._orchestrator_proxy
        self.ask_targets.append(actor)
        if self._workspace_proxy is not None:
            return self._workspace_proxy
        if self._orchestrator_proxy.live:
            return self._agent.proxy_ask(actor, actor_type)
        return self._orchestrator_proxy.actor_for(actor)

    def proxy_tell(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
    ) -> Any:
        self.tell_targets.append(actor)
        if self._workspace_tell_proxy is not None:
            return self._workspace_tell_proxy
        if self._workspace_proxy is not None:
            return self._workspace_proxy
        if self._orchestrator_proxy.live:
            return self._agent.proxy_tell(actor, actor_type)
        return self._orchestrator_proxy.actor_for(actor)


class CountingProxy:
    """Counts recording calls and forwards them, and everything else, to a real actor.

    Exists for the one-call-per-invocation assertion: the property has to be
    *counted*, not inferred from the resulting map, which a per-line recorder
    would leave looking identical. Every other method — the six mutations
    included — passes straight through, so a card wired to one behaves normally.
    """

    def __init__(self, target: WorkspaceActor) -> None:
        self.target = target
        self.calls: list[tuple[str, str, Observation]] = []

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        self.calls.append((agent_id, path, observation))
        self.target.record_observation(agent_id, path, observation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)


class FailingProxy:
    """Raises on every recording call — a dead actor or an unreachable proxy."""

    def __init__(self) -> None:
        self.calls = 0

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        self.calls += 1
        raise RuntimeError("actor is dead")


class AskOnlyProxy:
    """An ask proxy that refuses to carry an observation.

    Handed to a card as its **ask** proxy alongside a working tell proxy: if any
    read path still records through the ask side, the read fails loudly instead
    of passing while quietly holding the wrong invariant.
    """

    def __init__(self, target: WorkspaceActor) -> None:
        self.target = target

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        raise AssertionError("a read recorded through the ask proxy — it must use proxy_tell")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)


class RecordingTellProxy:
    """A tell proxy that forwards observations and remembers them."""

    def __init__(self, target: WorkspaceActor) -> None:
        self.target = target
        self.calls: list[tuple[str, str, Observation]] = []

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        self.calls.append((agent_id, path, observation))
        self.target.record_observation(agent_id, path, observation)


class BusyProxy:
    """Serializes calls behind one lock, the way a mailbox does.

    :meth:`occupy` stands in for another agent's in-flight call: it holds the
    lock until :attr:`release` is set, so a concurrent ``record_observation``
    queues behind it exactly as it would behind a busy actor thread.

    :attr:`queued` is what makes the contention real rather than incidental.
    Without it a test can only release the occupier and hope the reader had
    already arrived; a scheduler that ran the occupier to completion first
    would leave the test green having exercised no contention at all. The
    recorder sets it *before* reaching for the lock, so a test that waits on it
    knows the read is committed to blocking.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.occupied = threading.Event()
        self.queued = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []

    def occupy(self) -> None:
        with self._lock:
            self.occupied.set()
            self.release.wait(timeout=HANDSHAKE_TIMEOUT_S)

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        self.queued.set()
        with self._lock:
            self.calls.append(path)


@pytest.fixture(autouse=True)
def _no_real_workspaces_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test in this package off the default ``./workspaces``.

    ``get_workspace`` falls back to ``./workspaces`` relative to the *current
    working directory* when the variable is unset, so a test that forgets the
    :func:`workspaces_root` fixture writes into the developer's own checkout —
    and, since 29-4, runs ``git init`` there. That looks like nothing at all
    until it does. Tests that want a named base still request
    :func:`workspaces_root`, whose ``setenv`` runs after this one and wins.
    """
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path / "unclaimed-workspaces"))


@pytest.fixture
def workspaces_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``get_workspace`` at a temporary base, for the card and the actor alike."""
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(root))
    return root


@pytest.fixture
def workspace_tree(workspaces_root: Path) -> Path:
    """The tree ``WORKSPACE_NAME`` resolves to."""
    tree = workspaces_root / WORKSPACE_NAME
    tree.mkdir(parents=True, exist_ok=True)
    return tree


@pytest.fixture
def orchestrator_proxy() -> Generator[FakeOrchestratorProxy, None, None]:
    """A fake orchestrator whose created actors are stopped after the test."""
    proxy = FakeOrchestratorProxy()
    yield proxy
    proxy.stop_all()


@pytest.fixture
def observer(orchestrator_proxy: FakeOrchestratorProxy) -> FakeActorToolObserver:
    """An observer for agent ``alice``. Held by the test — the card holds it weakly."""
    return FakeActorToolObserver(orchestrator_proxy)


@pytest.fixture
def wired_card(
    observer: FakeActorToolObserver,
    workspace_tree: Path,
) -> WorkspaceTool:
    """A ``WorkspaceTool`` wired to the fake observer, with a live actor behind it."""
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
    card.observer(observer)
    return card


@pytest.fixture
def workspace_actor(
    orchestrator_proxy: FakeOrchestratorProxy,
    wired_card: WorkspaceTool,
) -> WorkspaceActor:
    """The live singleton actor behind :func:`wired_card`."""
    _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_NAME)]
    assert isinstance(actor, WorkspaceActor)
    return actor


@pytest.fixture
def threaded_orchestrator_proxy() -> Generator[FakeOrchestratorProxy, None, None]:
    """A fake orchestrator that starts its actors on real threads."""
    proxy = FakeOrchestratorProxy(live=True)
    yield proxy
    proxy.stop_all()


def card_for(
    orchestrator_proxy: FakeOrchestratorProxy,
    name: str,
    workspace_id: str = WORKSPACE_NAME,
    git_journal: bool = False,
) -> tuple[WorkspaceTool, FakeActorToolObserver]:
    """Wire a second (or third) agent's card onto the same workspace.

    The observer comes back with the card because the card holds it weakly — a
    test that drops it would collect the agent mid-assertion.

    ``git_journal`` mirrors the card's own default, which is off. A test about
    the journal opts in explicitly, so the suite never depends on a default it
    is not asserting.
    """
    observer = FakeActorToolObserver(orchestrator_proxy, name=name)
    card = WorkspaceTool(workspace_id=workspace_id, git_journal=git_journal)
    card.observer(observer)
    return card, observer


def tool_named(card: WorkspaceTool, name: str) -> Any:
    """Return the card's LLM-facing callable named *name*."""
    for tool in card.get_tools():
        if tool.__name__ == name:
            return tool
    raise AssertionError(f"{name} is not exposed by this card")


def read(card: WorkspaceTool, path: str, **kwargs: Any) -> str:
    """Read *path* through *card*, exactly as its agent would."""
    return str(tool_named(card, "workspace_read")(path, **kwargs))


def mutate(card: WorkspaceTool, name: str, *args: Any, **kwargs: Any) -> str:
    """Call one of *card*'s mutation callables and return what the agent sees."""
    return str(tool_named(card, name)(*args, **kwargs))


def outcome_of(actor: WorkspaceActor, method: str, *args: Any) -> MutationOutcome:
    """Call one of the actor's ``apply_*`` methods directly, for status assertions."""
    result = getattr(actor, method)(*args)
    assert isinstance(result, MutationOutcome)
    return result


##
## Exec (29-5) — a fake backend at the ``local`` key, and a worker on a real
## thread.  No docker, no bwrap, no sandbox-exec, and no wall-clock sleeps: a run
## is held open by an event and released by the test, so every concurrency
## assertion is a handshake with a failure budget rather than a wait.
##


@dataclass
class SandboxScript:
    """What the fake backend does when a run reaches it, and what it saw.

    Attributes:
        started: Set the moment the backend is entered — a test waits on this to
            know a run is genuinely in flight before asserting anything about it.
        gate: The run blocks here until the test sets it. Set from the start when
            a test wants a run that simply completes.
        files: ``(relative path, content)`` written before the run returns —
            including nested paths, which is how the ``-uall`` property is
            exercised.
        stdout, stderr, exit_code: What the run reports.
        raise_with: Raised instead of returning, for the failure path.
        timeouts: Every budget the backend was handed, in order.
        commands: Every ``(cmd, cwd)`` it was handed, in order.
        ready_raise: Raised by ``ready()`` instead of answering it. A stand-in
            for the ``pykka.Timeout`` a real ask proxy raises when the backend
            has not finished starting: the harness hands the worker the actor
            itself rather than a proxy, so the refusal has to come from inside.
        ready_calls: How many times readiness was asked for.
    """

    started: threading.Event = field(default_factory=threading.Event)
    gate: threading.Event = field(default_factory=threading.Event)
    files: list[tuple[str, str]] = field(default_factory=list)
    stdout: str = "ok"
    stderr: str = ""
    exit_code: int = 0
    raise_with: BaseException | None = None
    timeouts: list[float | None] = field(default_factory=list)
    commands: list[tuple[str, str]] = field(default_factory=list)
    ready_raise: BaseException | None = None
    ready_calls: int = 0


class FakeSandboxActor(SandboxActor):
    """A backend that writes what a test asks for and blocks when a test asks it to.

    Injected into ``SANDBOX_ACTOR_CLASSES`` at the ``local`` key, which the
    module documents as a mutable injection window. It is a real
    :class:`SandboxActor` subclass, so it goes through the same
    ``getChildrenOrCreate`` the production path uses and honours the same
    allowlist — what it does not do is start a process.
    """

    script: ClassVar[SandboxScript] = SandboxScript()

    def _start_sandbox(self) -> None:
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        name = self.config.workspace_id or self.config.team_id
        root = Path(base) / name
        root.mkdir(parents=True, exist_ok=True)
        self.state.workspace_path = root.resolve()

    def _stop_sandbox(self) -> None:
        pass

    def ready(self) -> bool:
        script = type(self).script
        script.ready_calls += 1
        if script.ready_raise is not None:
            raise script.ready_raise
        return super().ready()

    def _exec(self, cmd: str, cwd: str, timeout: float | None = None) -> ExecResult:
        script = type(self).script
        script.commands.append((cmd, cwd))
        script.timeouts.append(timeout)
        script.started.set()
        assert script.gate.wait(timeout=HANDSHAKE_TIMEOUT_S), "the run was never released"
        assert self.state.workspace_path is not None
        for relative, body in script.files:
            target = self.state.workspace_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        if script.raise_with is not None:
            raise script.raise_with
        return ExecResult(stdout=script.stdout, stderr=script.stderr, exit_code=script.exit_code)


class DeadAddress(MockActorAddress):
    """An address that reports itself dead, so telemetry never leaves the worker.

    ``Akgent._notify_orchestrator`` reaches into ``ActorAddressImpl._actor_ref``
    for anything it believes is alive, which a stand-in does not have. Reporting
    dead is the honest answer here — there is no orchestrator behind this address
    — and it keeps the worker's own ``StartMessage`` and state notifications out
    of the way of what these tests are about.
    """

    def is_alive(self) -> bool:
        return False


class WorkerAddress(MockActorAddress):
    """A worker's address whose liveness follows the worker's own thread.

    The lease records this address and ``#Workspace`` asks it whether the run
    still has anybody to report it, so a stand-in that always claimed to be alive
    would make the reclaim path untestable and one that always claimed to be dead
    would make the refusal path untestable. Following the thread means a test
    holding a run open has a genuinely live worker, and a test that has joined
    the harness has a genuinely dead one.
    """

    def __init__(self, name: str, role: str) -> None:
        super().__init__(name, role)
        self.thread: threading.Thread | None = None

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


class ExecHarness:
    """Runs ``#Workspace``'s exec worker on a real thread, with no actor system.

    The actor itself stays inert — the tests call its methods directly, exactly
    as the other workspace suites do — but the worker genuinely runs elsewhere,
    which is the only way a lease can be observed *while it is held*. Follows the
    ``createActor`` / ``proxy_tell`` patching in ``tests/test_core_deferred.py``
    rather than starting a second actor system.

    The base's ``request`` is untouched, so its in-flight de-duplication and its
    spawn-failure path are the real ones.
    """

    def __init__(self, actor: WorkspaceActor, orchestrator_proxy: FakeOrchestratorProxy) -> None:
        self.actor = actor
        self.orchestrator_proxy = orchestrator_proxy
        self.threads: list[threading.Thread] = []
        self.worker_names: list[str] = []
        self.addresses: list[WorkerAddress] = []
        self.payloads: list[DeferredPayload] = []
        self.spawn_error: BaseException | None = None
        self.ask_timeouts: list[int | None] = []
        """Every timeout the worker's asks carried, in order.

        A worker holds the lease, the tree and its parent's teardown for as long
        as it blocks, so an ask it makes without a timeout is unbounded on all
        three. Recorded here so that property is asserted rather than assumed.
        """
        self._orchestrator = DeadAddress("orchestrator")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point the actor's spawn path at this harness."""
        monkeypatch.setattr(self.actor, "createActor", self._create_actor)
        monkeypatch.setattr(self.actor, "proxy_tell", self._proxy_tell)

    def _create_actor(
        self,
        actor_class: type[Akgent[Any, Any]],
        agent_id: uuid.UUID | None = None,
        config: BaseConfig | None = None,
    ) -> ActorAddress:
        if self.spawn_error is not None:
            raise self.spawn_error
        assert config is not None
        assert issubclass(actor_class, ExecWorker)
        self.worker_names.append(config.name)
        address = WorkerAddress(config.name, config.role)
        self.addresses.append(address)
        return address

    def _proxy_tell(self, address: ActorAddress, actor_type: Any = None) -> Any:
        return _WorkerLauncher(self)

    def _run_worker(self, payload: DeferredPayload) -> None:
        """Build a worker, wire its two proxies to this harness, and let it produce."""
        worker = ExecWorker()
        worker.config = BaseConfig(name=self.worker_names[-1], role="ToolActor")
        worker._parent = DeadAddress("#Workspace")
        worker._orchestrator = self._orchestrator
        worker.on_start()

        def proxy_ask(
            target: ActorAddress, actor_type: Any = None, timeout: int | None = None
        ) -> Any:
            self.ask_timeouts.append(timeout)
            if target is self._orchestrator:
                return self.orchestrator_proxy
            return self.orchestrator_proxy.actor_for(target)

        def proxy_tell(target: ActorAddress, actor_type: Any = None) -> Any:
            return self.actor  # deliver() / fail() land on the real actor

        worker.proxy_ask = proxy_ask  # type: ignore[method-assign]
        worker.proxy_tell = proxy_tell  # type: ignore[method-assign]
        worker.stop = lambda *args, **kwargs: None  # type: ignore[method-assign,assignment]
        worker.receiveMsg_DeferredPayload(payload)

    def join(self) -> None:
        """Wait for every spawned worker, bounded — a hang is a failure, not a wait."""
        for thread in self.threads:
            thread.join(timeout=HANDSHAKE_TIMEOUT_S)
            assert not thread.is_alive(), "an exec worker never finished"
        self.threads.clear()


class _WorkerLauncher:
    """The tell proxy ``request`` hands its payload to — starts the worker's thread."""

    def __init__(self, harness: ExecHarness) -> None:
        self.harness = harness

    def receiveMsg_DeferredPayload(self, payload: DeferredPayload) -> None:  # noqa: N802
        self.harness.payloads.append(payload)
        thread = threading.Thread(target=self.harness._run_worker, args=(payload,), daemon=True)
        self.harness.threads.append(thread)
        # Attached before the thread starts, and before ``request`` returns: the
        # lease reads this address the moment the spawn call comes back.
        self.harness.addresses[-1].thread = thread
        thread.start()


@pytest.fixture
def sandbox_script() -> Generator[SandboxScript, None, None]:
    """Install :class:`FakeSandboxActor` at the ``local`` key for one test."""
    script = SandboxScript()
    FakeSandboxActor.script = script
    previous = SANDBOX_ACTOR_CLASSES["local"]
    SANDBOX_ACTOR_CLASSES["local"] = FakeSandboxActor
    yield script
    SANDBOX_ACTOR_CLASSES["local"] = previous
    script.gate.set()  # never leave a worker blocked behind a failed assertion


def exec_card_for(
    orchestrator_proxy: FakeOrchestratorProxy,
    name: str = "alice",
    workspace_id: str = WORKSPACE_NAME,
    poll_attempts: int = 1,
    poll_delay_seconds: float = 0.0,
    **card_kwargs: Any,
) -> tuple[WorkspaceTool, FakeActorToolObserver]:
    """Wire an exec-capable card onto *workspace_id*, with a tight poll by default.

    ``poll_attempts=1`` is what keeps the suite free of real sleeps: a test that
    wants the ``in progress`` handoff gets it in one attempt, and a test that
    wants a completed run raises the count against a 10 ms delay instead.
    """
    observer = FakeActorToolObserver(orchestrator_proxy, name=name)
    card = WorkspaceTool(
        workspace_id=workspace_id,
        workspace_exec=WorkspaceExec(
            mode="local",
            poll_attempts=poll_attempts,
            poll_delay_seconds=poll_delay_seconds,
        ),
        **card_kwargs,
    )
    card.observer(observer)
    return card, observer
