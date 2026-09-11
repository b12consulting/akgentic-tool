"""``WorkspaceAttached``: the domain event a team's stream carries for each bind.

A leaf module: it imports the standard library and nothing else, so a client or a
replay can resolve the payload class without importing the card or the actor.
"""

import uuid
from dataclasses import dataclass

__all__ = ["WorkspaceAttached"]


@dataclass(frozen=True)
class WorkspaceAttached:
    """An agent bound its card to the ``#Workspace`` that owns *workspace_path*.

    Built and emitted by the binding card itself, through
    ``ToolObserver.notify_event``, which wraps it in core's ``EventMessage`` and
    puts it on the binding team's own stream **unread**. One event per successful
    bind, **including a get-or-create hit**: two agents of one team on one tree
    are two events and one actor. There is no detach event.

    **Keep it a module top-level class, in this module.** The serializer persists
    ``module.ClassName`` into every stored event, and replay resolves that string
    back to the class with no alias mechanism — moving or nesting it breaks every
    stream already written, the rule ``ClosedNotification`` states for its own
    payload. ``uuid`` is imported at module level for the same reason: the
    deserializer resolves these annotations against this module's globals to
    rebuild ``agent_id`` as a ``uuid.UUID``.

    Attributes:
        agent_id: The **binding agent**, never the envelope's sender. The
            envelope is sent by the orchestrator; this names the member whose
            card bound.
        workspace_path: The resolved two-segment path the card bound, e.g.
            ``"_meta/customer_id-ACME__case_id-42"``.
    """

    agent_id: uuid.UUID
    workspace_path: str
