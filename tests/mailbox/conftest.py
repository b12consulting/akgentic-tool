"""Shared helpers for the mailbox test modules."""

from __future__ import annotations

import uuid

from akgentic.core import ActorAddressProxy
from akgentic.core.messages import Message, UserMessage


class TypedMessage(Message):
    """A message carrying a declared ``type``, as ``AgentMessage`` does.

    The real one lives in ``akgentic-agent``, which this package may not import
    (module boundary rules), so the shape the reply-protocol renderer reads
    duck-typed is reproduced here.
    """

    type: str
    content: str = ""


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


def _typed_message(sender: str, message_type: str, content: str = "") -> TypedMessage:
    """Create a TypedMessage carrying a mock sender address."""
    message = TypedMessage(type=message_type, content=content)
    message.sender = _address(sender)
    return message
