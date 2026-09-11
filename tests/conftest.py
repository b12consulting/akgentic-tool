"""Shared pytest fixtures and helpers for akgentic-tool tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from akgentic.core.actor_address import ActorAddress

from akgentic.tool.vector_store.backends.qdrant import QDRANT_API_KEY_ENV, QDRANT_URL_ENV
from akgentic.tool.vector_store.backends.weaviate import WEAVIATE_API_KEY_ENV, WEAVIATE_URL_ENV
from akgentic.tool.workspace.workspace import SHARED_KINDS_ENV

#: The pid every ``popen_mock`` helper in this suite stamps on its fake process.
MOCK_CHILD_PID = 4242


@pytest.fixture(autouse=True)
def _no_group_kill_on_the_mock_pid() -> Iterator[None]:
    """Refuse a real ``os.killpg`` aimed at the fake process's pid.

    ``local`` and ``bwrap`` spawn under ``process_group=True``, so a spec that
    mocks ``Popen`` and then reaches the timeout or ``kill()`` path without
    patching ``os.killpg`` would send ``SIGKILL`` to whatever real process group
    happens to own pid 4242 on this host. Every such spec patches ``killpg``
    today; this turns a forgotten patch into a red test rather than a dead
    process. A spec that runs a real child goes through to the real call, and a
    spec's own ``patch`` of the same target still wins, since it nests inside.
    """
    real_killpg = os.killpg

    def guarded(pgid: int, sig: int) -> None:
        if pgid == MOCK_CHILD_PID:
            raise AssertionError(
                f"os.killpg reached with the mock pid {MOCK_CHILD_PID}: this spec mocks "
                "Popen on a group-kill path and must patch "
                "akgentic.tool.sandbox.backend.os.killpg"
            )
        real_killpg(pgid, sig)

    with patch("akgentic.tool.sandbox.backend.os.killpg", guarded):
        yield


@pytest.fixture(autouse=True)
def _no_ambient_weaviate_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide a developer's exported cluster from every test in the package.

    ``VectorStoreParam.backend`` resolves from the backends' environment
    variables at instantiation, so without this the suite means different things
    depending on whose shell it runs in: green on a CI runner that exports
    nothing, red for the developer running the local cluster the feature exists
    to support.

    Tests that want a cluster opt in with ``monkeypatch.setenv``.
    """
    monkeypatch.delenv(WEAVIATE_URL_ENV, raising=False)
    monkeypatch.delenv(WEAVIATE_API_KEY_ENV, raising=False)
    monkeypatch.delenv(QDRANT_URL_ENV, raising=False)
    monkeypatch.delenv(QDRANT_API_KEY_ENV, raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_shared_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide a developer's or runner's exported shared-kind permission from every test.

    ``WorkspaceTool.observer`` reads ``AKGENTIC_WORKSPACE_SHARED_KINDS`` at every
    bind, so without this the suite means different things depending on whose
    shell it runs in: an exported ``team,id,meta`` turns every "unset refuses"
    spec red and makes every "permitted binds" spec vacuous, since it would bind
    whether or not the spec granted anything.

    Here rather than in ``tests/workspace/conftest.py`` because a
    ``WorkspaceTool`` is bound outside that directory too. Tests that want a
    permission grant it with ``monkeypatch``, which runs after this one and wins.
    """
    monkeypatch.delenv(SHARED_KINDS_ENV, raising=False)


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
