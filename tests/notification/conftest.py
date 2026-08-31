"""Fixtures and test doubles for the notification tool.

The delivery class is configured by dotted path, so the classes below double as
fixtures *and* as import targets: tests name them as
``tests.notification.conftest.<name>``. They deliberately stand in for
``akgentic.agent.messages.AgentMessage`` — this package does not depend on
``akgentic-agent``, and its standalone CI does not install it.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_address_impl import ActorAddressImpl
from akgentic.core.agent import Akgent, AkgentType
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message
from pydantic import Field

from akgentic.tool.core import ToolState
from akgentic.tool.notification.actor import NOTIFICATION_ACTOR_NAME, NotificationActor
from akgentic.tool.notification.tool import NotificationTool
from tests.conftest import MockActorAddress

FAKE_MESSAGE_PATH = "tests.notification.conftest.FakeNotificationMessage"
"""Dotted path of the stand-in delivery class used throughout these tests."""


class FakeNotificationMessage(Message):
    """Stand-in for the configured delivery class: carries ``content`` and ``type``."""

    type: str = "request"
    content: str = ""


class NarrowTypeMessage(Message):
    """Declares both fields, but its ``type`` cannot hold ``"notification"``.

    The defect the resolver's fourth check exists for: declaring ``type`` is not
    the same as accepting the value delivery writes into it.
    """

    type: Literal["chat"] = "chat"
    content: str = ""


class NonEmptyContentMessage(Message):
    """Accepts ``"notification"``, but constrains ``content`` away from ``""``.

    Delivery never writes an empty content, so this class is perfectly usable and
    the resolver must not refuse it over a payload it would never build.
    """

    type: str = "notification"
    content: str = Field(min_length=1, default="x")


class NotAMessage:
    """Resolvable, but not a ``Message`` subclass."""


class SilentAgent(Akgent[BaseConfig, BaseState]):
    """A do-nothing agent, used to mint a real (serializable) ``ActorAddress``."""


@pytest.fixture
def owner_agent() -> SilentAgent:
    """An unstarted agent instance the test keeps alive, so its address stays live."""
    return SilentAgent(config=BaseConfig(name="alice", role="tester"))


@pytest.fixture
def owner_address(owner_agent: SilentAgent) -> ActorAddress:
    """A real ``ActorAddressImpl`` — the only address shape that round-trips."""
    return ActorAddressImpl(owner_agent.actor_ref)


class FakeOrchestratorProxy:
    """Get-or-create singletons by config name, exactly as the orchestrator does.

    Holds the real actor instances so a card's ask proxy resolves to a live
    ``NotificationActor`` — which is what lets the singleton test prove that
    state scheduled through one card is visible through another.
    """

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
        """Run ``on_stop`` on every created actor, joining their tick threads."""
        for _, actor in self.children.values():
            actor.on_stop()
        self.children.clear()


class FakeActorToolObserver:
    """``ActorToolObserver`` stand-in wired to a :class:`FakeOrchestratorProxy`."""

    def __init__(self, orchestrator_proxy: FakeOrchestratorProxy, name: str = "alice") -> None:
        # A real address, not a stub: the actor persists its state on every
        # mutation, and only a real one survives that serialization round trip.
        self._agent = SilentAgent(config=BaseConfig(name=name, role="tester"))
        self._address: ActorAddress = ActorAddressImpl(self._agent.actor_ref)
        self._orchestrator: ActorAddress | None = MockActorAddress("orchestrator")
        self._orchestrator_proxy = orchestrator_proxy
        self._team_id = uuid.uuid4()
        self._state_carrier = SimpleNamespace(tool_state=ToolState())
        self.events: list[object] = []

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
        return self._orchestrator_proxy.actor_for(actor)

    def proxy_tell(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
    ) -> Any:
        return self._orchestrator_proxy.actor_for(actor)


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
def wired_card(observer: FakeActorToolObserver) -> NotificationTool:
    """A ``NotificationTool`` wired to the fake observer, with a live actor behind it."""
    card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
    card.observer(observer)
    return card


@pytest.fixture
def notification_actor(
    orchestrator_proxy: FakeOrchestratorProxy,
    wired_card: NotificationTool,
) -> NotificationActor:
    """The live singleton actor behind :func:`wired_card`."""
    _, actor = orchestrator_proxy.children[NOTIFICATION_ACTOR_NAME]
    assert isinstance(actor, NotificationActor)
    return actor
