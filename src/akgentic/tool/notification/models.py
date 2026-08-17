"""Data models for the notification tool, and the delivery-class resolver.

Everything that crosses an actor boundary here is a Pydantic model. ``Tick`` is a
plain :class:`SerializableBaseModel` rather than a ``Message`` on purpose: the
``Akgent`` telemetry sandwich fires only for ``Message`` instances, and a
``Message`` tick would put ``#NotificationTool`` in busy telemetry once a second.

``resolve_message_class`` lives here beside :class:`NotificationConfig` because it
validates that config value: the card checks it at ``observer()`` bind time, and
the actor resolves the same string when it starts (ADR-035 §2).
"""

from __future__ import annotations

from importlib import import_module

from pydantic import AwareDatetime, Field

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message
from akgentic.core.utils.serializer import SerializableBaseModel

DEFAULT_MESSAGE_CLASS = "akgentic.agent.messages.AgentMessage"
"""Dotted path of the message class delivered when a notification comes due.

A string, never an import: ``akgentic-tool`` may import ``akgentic-core`` only,
so the concrete class is named by the deployment and resolved at wiring time.
"""

REQUIRED_MESSAGE_FIELDS = ("content", "type")
"""Model fields the resolved delivery class must declare."""


class PendingNotification(SerializableBaseModel):
    """One scheduled notification, owned by the agent that asked for it.

    ``owner`` is an :class:`ActorAddress` — data, never a proxy — so a pending
    entry cannot pin its owning agent in memory.

    Attributes:
        notification_id: Monotonic per-actor identifier.
        owner: Address of the agent that scheduled it, and of the recipient.
        content: The message content to deliver.
        created_at: When the entry was scheduled (UTC).
        fire_at: Absolute due time (UTC), not a remaining delay — which is what
            makes a delay that expired while the team was down fire on resume.
    """

    notification_id: int
    owner: ActorAddress
    content: str
    created_at: AwareDatetime
    fire_at: AwareDatetime


class Tick(SerializableBaseModel):
    """Heartbeat payload told to the actor by its tick thread. Not a ``Message``."""


class NotificationConfig(BaseConfig):
    """Configuration of the ``#NotificationTool`` singleton.

    Attributes:
        message_class: Dotted path of the class delivered on fire, propagated
            from the card so the actor can resolve it after a team resume.
    """

    message_class: str = DEFAULT_MESSAGE_CLASS


class NotificationState(BaseState):
    """Persisted actor state: the pending entries and the id counter.

    ``pending`` self-drains — every entry is capped by ``max_delay_seconds`` and
    removed on fire or on cancel — so it needs no capacity bound.
    """

    pending: dict[int, PendingNotification] = Field(default_factory=dict)
    next_id: int = 1

    def serializable_copy(self) -> BaseState:
        """Snapshot the state without its observer, and without the inherited context.

        ``BaseState.serializable_copy`` hands the owning agent itself to the
        deserializer as the address-resolution context, but an ``Akgent`` does not
        implement ``resolve_address`` — so the inherited implementation raises for
        any state carrying an ``ActorAddress``, which this one does by design.

        Resolving with no context turns each address into an
        ``ActorAddressProxy``, which is exactly what ``snapshot_addresses``
        produces for an address crossing the same boundary inside a message. The
        live state keeps its live addresses; only the snapshot is proxied.
        """
        return self.__class__.model_validate(self.model_dump())


def resolve_message_class(dotted_path: str) -> type[Message]:
    """Import *dotted_path* and check it can carry a notification.

    Args:
        dotted_path: Dotted import path of the delivery message class.

    Returns:
        The resolved class.

    Raises:
        ValueError: When the path is not importable, does not name a ``Message``
            subclass, or names one that lacks a ``content`` or ``type`` field.
            All three are configuration defects and surface at wiring time,
            never when a notification comes due.
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"message_class {dotted_path!r} is not a dotted path to a class.")
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"message_class {dotted_path!r} is not importable: {exc}") from exc

    resolved = getattr(module, class_name, None)
    if resolved is None:
        raise ValueError(
            f"message_class {dotted_path!r} is not importable: "
            f"module {module_path!r} has no attribute {class_name!r}."
        )
    if not (isinstance(resolved, type) and issubclass(resolved, Message)):
        raise ValueError(
            f"message_class {dotted_path!r} resolves to {resolved!r}, "
            f"which is not a Message subclass."
        )
    missing = [name for name in REQUIRED_MESSAGE_FIELDS if name not in resolved.model_fields]
    if missing:
        raise ValueError(
            f"message_class {dotted_path!r} does not declare the required "
            f"field(s): {', '.join(missing)}."
        )
    return resolved
