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
import time
import uuid
from collections.abc import Generator
from concurrent import futures
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pykka import ActorDeadError

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_address_impl import ActorAddressImpl
from akgentic.core.agent import Akgent, AkgentType
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.tool.core import ToolState
from akgentic.tool.sandbox import SANDBOX_BACKEND_CLASSES
from akgentic.tool.sandbox.backend import ExecResult, validate_command
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.execution import DEFAULT_EXEC_TIMEOUT_S, RunningExec
from akgentic.tool.workspace.journal import git_dir_for
from akgentic.tool.workspace.models import MutationOutcome, Observation
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool

from tests.conftest import MockActorAddress

WORKSPACE_NAME = "test-workspace"
"""The ``workspace_id`` the wired cards below share."""

DEFAULT_TEST_PRINCIPAL = "u-alice"
"""The ``user_id`` the fake observer carries unless a test names another.

Every workspace now resolves under its owner, so the trees the suite writes to
live at ``<root>/u-alice/<leaf>``. :func:`workspace_root_for` builds that path so
no test has to spell the layout out twice.
"""


WORKSPACE_PATH = f"{DEFAULT_TEST_PRINCIPAL}/{WORKSPACE_NAME}"
"""What ``WORKSPACE_NAME`` **resolves** to — the two-segment path, not the leaf.

The distinction is the whole of ADR-048 in one line: ``WORKSPACE_NAME`` is what
a card declares, and this is the directory and the actor-name suffix it reaches.
Assertions about a tree or an actor name use this one; assertions about what an
author wrote use the other.
"""


def workspace_path_for(leaf: str, user_id: str = DEFAULT_TEST_PRINCIPAL) -> str:
    """The two-segment path *leaf* resolves to for *user_id*."""
    return f"{user_id}/{leaf}"


def workspace_root_for(base: Path, leaf: str, user_id: str = DEFAULT_TEST_PRINCIPAL) -> Path:
    """The on-disk root of the workspace *leaf* belonging to *user_id*."""
    return base / user_id / leaf

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
        self.metadata: Any = None
        """What :meth:`get_metadata` answers — the team's metadata, or ``None``.

        A card only asks when it declares ``workspace_metadata_keys``, so the
        default of ``None`` is also the assertion that a bare ``WorkspaceTool()``
        gained no bind-time round trip: :attr:`metadata_calls` stays at zero.
        """
        self.metadata_calls = 0

    def getChildrenOrCreate(  # noqa: N802 — mirrors the orchestrator's method name
        self, actor_class: type[Akgent[Any, Any]], config: BaseConfig
    ) -> ActorAddress:
        self.create_calls.append((actor_class, config))
        existing = self.children.get(config.name)
        # A child that is no longer alive is skipped and replaced, exactly as
        # the real orchestrator does — which is what makes "the next admission
        # recreates the sandbox" observable through ``create_calls``.
        if existing is not None and existing[0].is_alive():
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

    def get_metadata(self) -> Any:
        """Return the team's metadata, exactly as the orchestrator does."""
        self.metadata_calls += 1
        return self.metadata

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
        user_id: str | None = DEFAULT_TEST_PRINCIPAL,
    ) -> None:
        self._agent = SilentAgent(config=BaseConfig(name=name, role="tester"))
        self._address: ActorAddress = ActorAddressImpl(self._agent.actor_ref)
        self._orchestrator: ActorAddress | None = MockActorAddress("orchestrator")
        self._orchestrator_proxy = orchestrator_proxy
        self._workspace_proxy = workspace_proxy
        self._workspace_tell_proxy = workspace_tell_proxy
        self._team_id = uuid.uuid4()
        self.user_id = user_id
        """The owning principal — a plain attribute, so a test can hand two
        cards two different principals and watch them reach two trees. The
        Protocol declares a property; an attribute satisfies it structurally."""
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
    """The tree ``WORKSPACE_NAME`` resolves to — under its owner, not at the root."""
    tree = workspaces_root / WORKSPACE_PATH
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
    _, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
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
## Exec — a fake **backend** at the ``local`` key, and the actor's own worker
## thread.  No docker, no bwrap, no sandbox-exec, and no wall-clock sleeps: a run
## is held open by an event and released by the test, so every concurrency
## assertion is a handshake with a failure budget rather than a wait.
##
## **The injection window is** ``SANDBOX_BACKEND_CLASSES``, **not**
## ``SANDBOX_ACTOR_CLASSES``.  ``#Workspace`` builds its own backend in
## ``configure_exec`` and runs it on its own single-worker executor; it resolves
## no sandbox actor at all, so a fake installed at the actor key would be
## installed and never reached — every spec below would go green while testing
## nothing.  Injecting the *backend* keeps the whole production path live:
## ``configure_exec`` → ``resolve_mode`` → the registry → ``ExecRunner`` → the
## real executor → ``perform``.  Only the four lines that would touch a real
## process are the fake's.
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
        files_by_cmd: Per-command files, consulted before :attr:`files`. One
            fake backend serves every run in a test, so without this two runs
            write byte-identical trees and "run A's commit contains only run A's
            files" cannot be asserted at all. A command absent from the map falls
            back to :attr:`files`, so every existing test is unaffected — the
            same shape as :attr:`stdout_by_cmd`.
        stdout, stderr, exit_code: What the run reports.
        stdout_by_cmd: Per-command stdout, consulted before :attr:`stdout`. One
            fake backend serves every run in a test, so without this two runs
            report byte-identical output and "a run returns its OWN command's
            output" cannot be asserted at all. A command absent from the map
            falls back to :attr:`stdout`, so every existing test is unaffected.
        raise_with: Raised instead of returning, for the failure path.
        start_raises: Raised by :meth:`FakeBackend.start` instead of
            provisioning, for the cold-start failure path. Cleared by a test
            that wants the retry to succeed.
        timeouts: Every budget the backend was handed, in order.
        commands: Every ``(cmd, cwd)`` it was handed, in order.
        starts: Every ``workspace_path`` ``start()`` was called with, in order.
            Its **length** is the whole of the "started once, lazily" assertion.
        kills: How many times ``kill()`` was called.
        stops: How many times ``stop()`` was called.
        kill_raises: Raised by ``kill()`` instead of ending the run, for the
            "a failing step must not skip the ones after it" path.
        kill_releases: Whether ``kill()`` ends the blocked run, as a real
            backend's does. Turned **off** to reproduce the child that ignores
            the kill, which is what the bounded drain exists for.
        exec_tail_s: Wall clock ``exec`` spends *after* it is released, before
            it returns. Zero everywhere except the one spec that asserts a
            teardown ordering across two threads, where it is what makes the
            order observable rather than a race.
        threads: ``("start" | "exec", thread ident)`` for every call, in order.
            The whole of "no backend call happens on the actor's thread": the
            identity of the thread is the property, and a name would only be a
            proxy for it.
        events: One shared ordered record of everything the backend was asked to
            do — ``("start", path)``, ``("exec-enter", cmd)``,
            ``("exec-return", cmd)``, ``("kill",)``, ``("stop",)``. Teardown
            order is a property of *positions* in this list; the separate
            counters above answer "how many", which is a different question and
            cannot express an order at all.
    """

    started: threading.Event = field(default_factory=threading.Event)
    gate: threading.Event = field(default_factory=threading.Event)
    files: list[tuple[str, str]] = field(default_factory=list)
    files_by_cmd: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    stdout: str = "ok"
    stderr: str = ""
    exit_code: int = 0
    stdout_by_cmd: dict[str, str] = field(default_factory=dict)
    raise_with: BaseException | None = None
    start_raises: BaseException | None = None
    timeouts: list[float | None] = field(default_factory=list)
    commands: list[tuple[str, str]] = field(default_factory=list)
    starts: list[str] = field(default_factory=list)
    kills: int = 0
    stops: int = 0
    kill_raises: BaseException | None = None
    kill_releases: bool = True
    exec_tail_s: float = 0.0
    threads: list[tuple[str, int]] = field(default_factory=list)
    events: list[tuple[str, ...]] = field(default_factory=list)


class FakeBackend:
    """A backend that writes what a test asks for and blocks when a test asks it to.

    Installed at ``SANDBOX_BACKEND_CLASSES["local"]``, which the registry
    documents as a mutable injection window. It **exposes the four Protocol
    names** — which is all ``@runtime_checkable`` would check anyway — and
    honours the same allowlist the four shipped backends do, by calling
    ``validate_command`` in ``exec`` exactly as they each do. What it does not do
    is start a process.

    The script is a class attribute rather than a constructor argument because
    ``resolve_mode`` constructs the backend itself, from the registry, with only
    a ``team_id`` — which is the production path and the reason this fake is
    reached at all.
    """

    script: ClassVar[SandboxScript] = SandboxScript()

    def __init__(self, team_id: str = "") -> None:
        self.team_id = team_id
        self.workspace_path: Path | None = None

    def start(self, workspace_path: str) -> None:
        # Joins the path it was handed and derives nothing, exactly as the four
        # shipped backends do — a fake that still resolved would hide the very
        # thing the shipped ones stopped doing.
        script = type(self).script
        script.starts.append(workspace_path)
        script.threads.append(("start", threading.get_ident()))
        script.events.append(("start", workspace_path))
        if script.start_raises is not None:
            raise script.start_raises
        base = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
        root = Path(base) / workspace_path
        root.mkdir(parents=True, exist_ok=True)
        self.workspace_path = root.resolve()

    def exec(self, cmd: str, cwd: str = "", timeout: float | None = None) -> ExecResult:
        script = type(self).script
        # Ahead of every recording, so a refused command leaves no trace of
        # having reached the backend — the same order the four shipped backends
        # produce by validating while they build their argv.
        validate_command(cmd)
        script.commands.append((cmd, cwd))
        script.timeouts.append(timeout)
        script.threads.append(("exec", threading.get_ident()))
        script.events.append(("exec-enter", cmd))
        script.started.set()
        try:
            assert script.gate.wait(timeout=HANDSHAKE_TIMEOUT_S), "the run was never released"
            if script.exec_tail_s:
                time.sleep(script.exec_tail_s)
            assert self.workspace_path is not None
            for relative, body in script.files_by_cmd.get(cmd, script.files):
                target = self.workspace_path / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
            if script.raise_with is not None:
                raise script.raise_with
            stdout = script.stdout_by_cmd.get(cmd, script.stdout)
            return ExecResult(stdout=stdout, stderr=script.stderr, exit_code=script.exit_code)
        finally:
            # In a ``finally`` because "the worker left ``exec``" is the event
            # teardown ordering is asserted against, and a raise leaves it too.
            script.events.append(("exec-return", cmd))

    def kill(self) -> None:
        script = type(self).script
        script.kills += 1
        script.events.append(("kill",))
        if script.kill_raises is not None:
            raise script.kill_raises
        if script.kill_releases:
            # A real backend's kill ends the blocked command. Turning this off
            # is how the child that ignores the kill is reproduced.
            script.gate.set()

    def stop(self) -> None:
        script = type(self).script
        script.stops += 1
        script.events.append(("stop",))


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


class WorkspaceAddress(MockActorAddress):
    """``#Workspace``'s own address, as its worker sees it — the reply stand-in.

    The actor under test is inert: the suite calls its methods directly, so its
    real ``myAddress`` names an inbox nobody drains and a report told to it would
    simply vanish (a never-started Pykka actor even reports ``is_alive()`` as
    ``True``). So the harness rewrites ``reply_to`` to this, which calls the
    handler directly — on the worker's thread, exactly where production's mailbox
    hand-off lands the work.

    Do not "fix" this by starting the workspace actor for real: the point of the
    inert actor is that a test can read ``_running`` and ``_queue`` while a run
    is held open.
    """

    def __init__(self, name: str, role: str, actor: WorkspaceActor) -> None:
        super().__init__(name, role)
        self._actor = actor
        self.dead = False
        """When set, :meth:`tell` raises, exactly as a stopping actor's does."""

    def tell(self, message: Any) -> None:
        if self.dead:
            raise ActorDeadError(f"{self.name} not found")
        self._actor.receiveMsg_ExecReport(message)


@dataclass
class SubmittedRun:
    """One call the actor made to ``executor.submit``, as it made it.

    The five values plus the reply address are what ``_start_run`` hands the
    worker, so recording them here is recording the whole of what crosses the
    thread boundary — the successor to the ``ExecRequest`` the sandbox actor
    used to receive.
    """

    run_id: str
    cmd: str
    cwd: str
    timeout_s: float
    reply_to: ActorAddress


class RecordingExecutor:
    """Stands in for ``#Workspace``'s own executor, and delegates to a real one.

    Three jobs, and it is deliberately not a mock for any of them:

    - it **records** what the actor submitted, which is the only place the run
      id and the reply address are observable now that no request model crosses
      the boundary;
    - it **redirects** ``reply_to`` to the inert actor's stand-in, exactly as the
      old harness rewrote it on the ``ExecRequest``;
    - it **runs the real callable on a real single worker thread**, so the
      command genuinely runs elsewhere. That is the only way the tree's hold can
      be observed *while it is held*, and it is what makes "call ``perform``
      inline instead of submitting" an observable mutation rather than an
      invisible one.

    ``submit_raises`` is the reachable failure of the submit itself: production
    raises ``RuntimeError`` here when the executor has already been shut down.
    """

    def __init__(self, workspace_address: WorkspaceAddress, actor: WorkspaceActor) -> None:
        self._inner = ThreadPoolExecutor(max_workers=1)
        self._workspace_address = workspace_address
        self._actor = actor
        self.runs: list[SubmittedRun] = []
        self.futures: list[Future[None]] = []
        self.holds: list[RunningExec | None] = []
        """``actor._running`` as it stood at each submit, in order.

        Read between the hold being taken and the work being handed over, which
        is the only window in which the value is observable: a run released by an
        already-set gate can report and clear ``_running`` before the call that
        took it has even returned.
        """
        self.submit_raises: BaseException | None = None

    def submit(self, fn: Any, **kwargs: Any) -> Future[None]:
        self.holds.append(self._actor._running)
        self.runs.append(SubmittedRun(**kwargs))
        if self.submit_raises is not None:
            raise self.submit_raises
        future = self._inner.submit(fn, **{**kwargs, "reply_to": self._workspace_address})
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)


class ExecHarness:
    """Gives the inert actor a recording executor and an address to report to.

    The workspace actor stays inert — the tests call its methods directly,
    exactly as the other workspace suites do — but the command genuinely runs on
    a worker thread, which is the only way the tree's hold can be observed *while
    it is held*.

    Nothing about the actor's exec path is stubbed: it builds its backend through
    ``configure_exec`` and ``resolve_mode`` exactly as production does, and it
    submits the real ``ExecRunner.perform``. What the harness supplies is the two
    ends — an executor whose worker it can wait on, and an address to reply to.
    """

    def __init__(self, actor: WorkspaceActor, orchestrator_proxy: FakeOrchestratorProxy) -> None:
        self.actor = actor
        self.orchestrator_proxy = orchestrator_proxy
        self.workspace_address = WorkspaceAddress("#Workspace", "ToolActor", actor)
        self.executor = RecordingExecutor(self.workspace_address, actor)
        self._orchestrator = DeadAddress("orchestrator")

    @property
    def runs(self) -> list[SubmittedRun]:
        """Every run the actor submitted, in order — one per started run."""
        return self.executor.runs

    @property
    def holds(self) -> list[RunningExec | None]:
        """``actor._running`` as it stood at each submit, in order."""
        return self.executor.holds

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Put the recording executor on the actor, and give it an orchestrator."""
        monkeypatch.setattr(self.actor, "_orchestrator", self._orchestrator)
        monkeypatch.setattr(self.actor, "proxy_ask", self._proxy_ask)
        monkeypatch.setattr(self.actor, "_executor", self.executor)

    def _proxy_ask(
        self, target: ActorAddress, actor_type: Any = None, timeout: int | None = None
    ) -> Any:
        if target is self._orchestrator:
            return self.orchestrator_proxy
        return self.orchestrator_proxy.actor_for(target)

    def join(self) -> None:
        """Wait for every submitted run, bounded — a hang is a failure, not a wait."""
        for future in self.executor.futures:
            futures.wait([future], timeout=HANDSHAKE_TIMEOUT_S)
            assert future.done(), "a sandbox run never finished"
            future.result()  # a callable that raised would otherwise be silent
        self.executor.futures.clear()

    def close(self) -> None:
        """Release the worker thread. Never leaves one behind a failed assertion."""
        self.executor.shutdown(wait=False, cancel_futures=True)


@pytest.fixture
def sandbox_script() -> Generator[SandboxScript, None, None]:
    """Install :class:`FakeBackend` at the ``local`` key for one test."""
    script = SandboxScript()
    FakeBackend.script = script
    previous = SANDBOX_BACKEND_CLASSES["local"]
    SANDBOX_BACKEND_CLASSES["local"] = FakeBackend
    yield script
    SANDBOX_BACKEND_CLASSES["local"] = previous
    script.gate.set()  # never leave a worker blocked behind a failed assertion


def exec_card_for(
    orchestrator_proxy: FakeOrchestratorProxy,
    name: str = "alice",
    workspace_id: str = WORKSPACE_NAME,
    poll_attempts: int = 1,
    poll_delay_seconds: float = 0.0,
    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
    **card_kwargs: Any,
) -> tuple[WorkspaceTool, FakeActorToolObserver]:
    """Wire an exec-capable card onto *workspace_id*, with a tight poll by default.

    ``poll_attempts=1`` is what keeps the suite free of real sleeps: a test that
    wants the ``in progress`` handoff gets it in one attempt, and a test that
    wants a completed run raises the count against a 10 ms delay instead. It is
    deliberately **not** the card's own default (the wait-out-the-run sentinel),
    which resolves against the run budget and would give most of this suite a
    poll measured in seconds for no assertion's benefit; a test about the
    sentinel passes ``-1`` and a small ``timeout_s`` explicitly.
    """
    observer = FakeActorToolObserver(orchestrator_proxy, name=name)
    card = WorkspaceTool(
        workspace_id=workspace_id,
        workspace_exec=WorkspaceExec(
            mode="local",
            poll_attempts=poll_attempts,
            poll_delay_seconds=poll_delay_seconds,
            timeout_s=timeout_s,
        ),
        **card_kwargs,
    )
    card.observer(observer)
    return card, observer
