"""Shared helpers for the mailbox test modules."""

from __future__ import annotations

import uuid

from akgentic.core import ActorAddressProxy
from akgentic.core.messages import UserMessage


def _address(name: str, role: str = "Agent") -> ActorAddressProxy:
    """Create a mock ActorAddress for testing."""
    return ActorAddressProxy(
        {
            "__actor_address__": True,
            "__actor_type__": "test.Agent",
            "agent_id": str(uuid.uuid4()),
            "name": name,
            "role": role,
            "team_id": str(uuid.uuid4()),
            "squad_id": str(uuid.uuid4()),
            "is_user_proxy": False,
        }
    )


def _user_message(sender: str, content: str) -> UserMessage:
    """Create a UserMessage carrying a mock sender address."""
    message = UserMessage(content=content)
    message.sender = _address(sender)
    return message
