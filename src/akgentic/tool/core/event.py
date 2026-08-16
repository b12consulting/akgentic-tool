"""Event contracts shared by every tool domain.

Holds the tool-state envelope and the command-discovery models — the events any
tool may emit, named by ``core/commands.py`` and consumed across the package.

``ToolStateEvent.payload`` is typed structurally, as ``SerializableBaseModel``. That
is deliberate and load-bearing: a payload union naming concrete domain types would
make this global module depend on a domain package, which is the edge ``core/`` must
not have. Nothing about a payload's serialization depends on the annotation —
``SerializableBaseModel`` serializes from the runtime instance and tags it with a
``__model__`` marker, so a concrete payload round-trips as itself either way. The
knowledge graph's payload type, ``KnowledgeGraphStateEvent``, is re-exported from
``akgentic.tool.knowledge_graph.event``.
"""

from __future__ import annotations

from akgentic.core.actor_address import ActorAddress
from akgentic.core.messages import Message
from akgentic.core.utils.serializer import SerializableBaseModel


class ToolStateEvent(Message):
    """Generic tool-state event envelope (ADR-024, Story 17.1).

    Wraps a tool-specific delta payload so any stateful tool actor can broadcast
    typed state changes on the existing orchestrator event stream. Inherits
    ``team_id``, ``timestamp``, ``id``, ``sender``, and ``display_type`` from
    :class:`akgentic.core.messages.Message` without override.

    Attributes:
        tool_id: Tool-actor name emitting the event (e.g. ``"#KnowledgeGraphTool"``).
        seq: Per-tool monotonic sequence number (starts at 1, enforced in Story 17.2).
        payload: Tool-specific delta payload. Any ``SerializableBaseModel``; the
            concrete type is carried on the wire by its ``__model__`` marker.
    """

    tool_id: str
    seq: int
    payload: SerializableBaseModel


class CommandArg(SerializableBaseModel):
    """A single positional/keyword argument of a discoverable command (ADR-028 §Decision 3).

    Derived from a command callable's signature so consumers (the dispatch parser
    in Story 21.2 and the frontend help renderer) share one typed contract.

    Attributes:
        name: Argument name as it appears in the callable signature.
        type: JSON-schema type name (e.g. ``"string"``, ``"integer"``, ``"boolean"``).
        required: Whether the argument must be supplied (no default).
        description: Optional human-readable description; ``None`` when absent.
    """

    name: str
    type: str
    required: bool
    description: str | None = None


class CommandDescriptor(SerializableBaseModel):
    """A discoverable command exposed by a tool (ADR-028 §Decision 3).

    Describes one canonical command, its provenance, and its ordered argument
    list. The ``args`` order is load-bearing: it drives positional dispatch
    parsing (Story 21.2) and frontend help rendering.

    Attributes:
        name: Canonical command name (e.g. ``"hire_member"``).
        description: Command description, sourced from the callable docstring.
        args: Ordered list of :class:`CommandArg` entries (may be empty).
        tool_card: Provenance of the command (e.g. ``"TeamTool"``).
    """

    name: str
    description: str
    args: list[CommandArg]
    tool_card: str


class CommandsAnnouncedEvent(SerializableBaseModel):
    """Announcement of the command set an agent executes (ADR-028 §Decision 3).

    Emitted so downstream agents/frontends can render the available commands.
    Fully serializable: the ``__model__`` marker (from ``SerializableBaseModel``)
    lets consumers discriminate this inner event during deserialization.

    Attributes:
        agent: Address of the agent that executes these commands (core
            ``ActorAddress``; the tool layer MAY import core, NFR1).
        commands: The :class:`CommandDescriptor` set announced for ``agent``.
    """

    agent: ActorAddress
    commands: list[CommandDescriptor]
