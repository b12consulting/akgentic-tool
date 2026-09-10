"""``WorkspaceHost``: the process's one host of ``#Workspace-<path>`` actors.

Core's :class:`~akgentic.core.resource_host.ResourceHost` is the generic mechanism
— a registry keyed on ``config.name``, one store, the get-or-create on its own
mailbox. This module declares the workspace's **kind** of it, and nothing more.
"""

from __future__ import annotations

from akgentic.core.resource_host import ResourceHost

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
