"""Shared pytest fixtures and helpers for akgentic-tool tests."""

from __future__ import annotations

import uuid

import pytest
from akgentic.core.actor_address import ActorAddress

from akgentic.tool.vector_store.protocol import WEAVIATE_API_KEY_ENV, WEAVIATE_URL_ENV


@pytest.fixture(autouse=True)
def _no_ambient_weaviate_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide a developer's exported cluster from every test in the package.

    ``CollectionConfig.backend`` resolves from ``AKGENTIC_WEAVIATE_URL`` at
    instantiation, so without this the suite means different things depending on
    whose shell it runs in: green on a CI runner that exports nothing, red for the
    developer running the local Weaviate the feature exists to support.

    Tests that want a cluster opt in with ``monkeypatch.setenv``.
    """
    monkeypatch.delenv(WEAVIATE_URL_ENV, raising=False)
    monkeypatch.delenv(WEAVIATE_API_KEY_ENV, raising=False)


class MockActorAddress(ActorAddress):
    """Minimal ActorAddress stub used across multiple test modules."""

    def __init__(self, name: str = "test-agent", role: str = "test-role") -> None:
        self._name = name
        self._role = role
        self._agent_id = uuid.uuid4()

    @property
    def agent_id(self) -> uuid.UUID:
        return self._agent_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> str:
        return self._role

    @property
    def team_id(self) -> uuid.UUID | None:
        return None

    @property
    def squad_id(self) -> uuid.UUID | None:
        return None

    def send(self, recipient: object, message: object) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    @property
    def is_user_proxy(self) -> bool:
        return False

    def serialize(self) -> dict:  # type: ignore[type-arg]
        return {"name": self._name, "role": self._role, "agent_id": str(self._agent_id)}

    def __repr__(self) -> str:
        return f"MockActorAddress(name={self._name})"


@pytest.fixture
def mock_actor_address() -> MockActorAddress:
    """Return a default MockActorAddress instance."""
    return MockActorAddress()
