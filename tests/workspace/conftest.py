"""Fixtures and test doubles for the ``#Workspace`` actor and the recording read path.

The doubles here reach the actor only through the public surface a card uses —
``getChildrenOrCreate`` on a fake orchestrator, then ``proxy_ask`` — so nothing
in these tests depends on actor internals. The fake orchestrator holds the real
actor instances, which is what lets the singleton test prove that an observation
recorded through one card is visible through another.

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
from akgentic.tool.workspace.models import Observation
from akgentic.tool.workspace.tool import WorkspaceTool

from tests.conftest import MockActorAddress

WORKSPACE_NAME = "test-workspace"
"""The ``workspace_id`` the wired cards below share."""

HANDSHAKE_TIMEOUT_S = 5.0
"""Upper bound on a thread handshake — never a delay, only a failure budget."""


class SilentAgent(Akgent[BaseConfig, BaseState]):
    """A do-nothing agent, used to mint a real (serializable) ``ActorAddress``."""


class FakeOrchestratorProxy:
    """Get-or-create singletons by config name, exactly as the orchestrator does."""

    def __init__(self) -> None:
        self.children: dict[str, tuple[ActorAddress, Akgent[Any, Any]]] = {}
        self.create_calls: list[tuple[type[Akgent[Any, Any]], BaseConfig]] = []

    def getChildrenOrCreate(  # noqa: N802 — mirrors the orchestrator's method name
        self, actor_class: type[Akgent[Any, Any]], config: BaseConfig
    ) -> ActorAddress:
        self.create_calls.append((actor_class, config))
        existing = self.children.get(config.name)
        if existing is not None:
            return existing[0]
        actor = actor_class(config=config)
        actor.on_start()
        address = MockActorAddress(config.name, config.role)
        self.children[config.name] = (address, actor)
        return address

    def actor_for(self, address: ActorAddress) -> Akgent[Any, Any] | None:
        """Return the actor behind *address*, or ``None`` when it is unknown."""
        for known_address, actor in self.children.values():
            if known_address is address:
                return actor
        return None

    def stop_all(self) -> None:
        """Run ``on_stop`` on every created actor."""
        for _, actor in self.children.values():
            actor.on_stop()
        self.children.clear()


class FakeActorToolObserver:
    """``ActorToolObserver`` stand-in wired to a :class:`FakeOrchestratorProxy`.

    *workspace_proxy*, when given, is handed back by ``proxy_ask`` in place of the
    live actor. That is how the counting, failing and busy stand-ins below reach
    the card without any of them having to impersonate an orchestrator.
    """

    def __init__(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        name: str = "alice",
        workspace_proxy: object | None = None,
    ) -> None:
        self._agent = SilentAgent(config=BaseConfig(name=name, role="tester"))
        self._address: ActorAddress = ActorAddressImpl(self._agent.actor_ref)
        self._orchestrator: ActorAddress | None = MockActorAddress("orchestrator")
        self._orchestrator_proxy = orchestrator_proxy
        self._workspace_proxy = workspace_proxy
        self._team_id = uuid.uuid4()
        self.events: list[object] = []

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
        if self._workspace_proxy is not None:
            return self._workspace_proxy
        return self._orchestrator_proxy.actor_for(actor)


class CountingProxy:
    """Counts recording calls and forwards them to a real actor.

    Exists for the one-call-per-invocation assertion: the property has to be
    *counted*, not inferred from the resulting map, which a per-line recorder
    would leave looking identical.
    """

    def __init__(self, target: WorkspaceActor) -> None:
        self.target = target
        self.calls: list[tuple[str, str, Observation]] = []

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        self.calls.append((agent_id, path, observation))
        self.target.record_observation(agent_id, path, observation)

    def observation_for(self, agent_id: str, path: str) -> Observation | None:
        return self.target.observation_for(agent_id, path)


class FailingProxy:
    """Raises on every recording call — a dead actor or an unreachable proxy."""

    def __init__(self) -> None:
        self.calls = 0

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        self.calls += 1
        raise RuntimeError("actor is dead")


class BusyProxy:
    """Serializes calls behind one lock, the way a mailbox does.

    :meth:`occupy` stands in for another agent's in-flight call: it holds the
    lock until :attr:`release` is set, so a concurrent ``record_observation``
    queues behind it exactly as it would behind a busy actor thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.occupied = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []

    def occupy(self) -> None:
        with self._lock:
            self.occupied.set()
            self.release.wait(timeout=HANDSHAKE_TIMEOUT_S)

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
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


def tool_named(card: WorkspaceTool, name: str) -> Any:
    """Return the card's LLM-facing callable named *name*."""
    for tool in card.get_tools():
        if tool.__name__ == name:
            return tool
    raise AssertionError(f"{name} is not exposed by this card")
