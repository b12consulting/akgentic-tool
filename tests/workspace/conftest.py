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

import threading
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_address_impl import ActorAddressImpl
from akgentic.core.agent import Akgent, AkgentType
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.models import MutationOutcome, Observation
from akgentic.tool.workspace.tool import WorkspaceTool

from tests.conftest import MockActorAddress

WORKSPACE_NAME = "test-workspace"
"""The ``workspace_id`` the wired cards below share."""

HANDSHAKE_TIMEOUT_S = 5.0
"""Upper bound on a thread handshake — never a delay, only a failure budget."""


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
        self.events: list[object] = []
        self.ask_targets: list[ActorAddress] = []
        self.tell_targets: list[ActorAddress] = []

    @property
    def myAddress(self) -> ActorAddress:  # noqa: N802
        return self._address

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
) -> tuple[WorkspaceTool, FakeActorToolObserver]:
    """Wire a second (or third) agent's card onto the same workspace.

    The observer comes back with the card because the card holds it weakly — a
    test that drops it would collect the agent mid-assertion.
    """
    observer = FakeActorToolObserver(orchestrator_proxy, name=name)
    card = WorkspaceTool(workspace_id=workspace_id)
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
