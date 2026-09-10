"""``WorkspaceHost``: the process's one host of ``#Workspace-<path>`` actors.

Core's :class:`~akgentic.core.resource_host.ResourceHost` is the generic mechanism
— a registry keyed on ``config.name``, one store, the get-or-create on its own
mailbox. This module declares the workspace's **kind** of it, and nothing more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.resource_host import ResourceHost

if TYPE_CHECKING:
    from akgentic.core.actor_address import ActorAddress

__all__ = ["WorkspaceHost"]


class WorkspaceHost(ResourceHost):
    """The host every ``WorkspaceTool`` binds its tree through.

    **One per concrete class per process.** The orchestrator's forward looks the
    host up by exact class, never subclass-inclusive, so this host and core's
    base ``ResourceHost`` are two different hosts with two registries. That is
    why it is a subclass: a future memory tool declares its own host beside this
    one, and neither sees the other's registry, nor blocks on the other's store.

    **Created at wiring, right after the ``ActorSystem``, and never lazily.**
    Nothing in this package creates one. A process that wires only the base host
    fails the first workspace bind with core's "No WorkspaceHost is running"
    error, which is loud by design: the cure is to create this host there.

    **It carries nothing of its own**, and needs nothing. The registry, the store
    and the get-or-create are core's and are generic; what makes an actor a
    workspace is the class and config the card hands the forward, not the host.
    """


def workspace_host_address() -> ActorAddress | None:
    """The process's ``WorkspaceHost``, looked up at each use and never cached.

    A hosted workspace's only way home: it is nobody's child and has no
    orchestrator, so it finds the host that writes its deltas through core's
    registry, from its own thread. It names ``WorkspaceHost`` and never the base,
    because the lookup is by exact class — asking for ``ResourceHost`` in a
    process that runs both finds the wrong kind's registry, which drops the
    delta as an unknown scope.

    Never cached: the host is created at wiring and found, never held, and an
    empty answer is not an error — a test harness runs no host, and at process
    exit the host may already be gone.

    Deliberately absent from ``__all__``: the actor package is its only caller,
    and this module stays a leaf that imports only core, so both
    ``actor/__init__.py`` and ``actor/documents.py`` can reach it without a cycle.

    Returns:
        The host's address, or ``None`` when no ``WorkspaceHost`` is running.
    """
    hosts = ActorSystem.find_by_class(WorkspaceHost)
    return hosts[0] if hosts else None
