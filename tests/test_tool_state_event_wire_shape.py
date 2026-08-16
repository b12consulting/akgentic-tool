"""Frozen wire shape of ``ToolStateEvent`` carrying a knowledge-graph payload.

``ToolStateEvent`` feeds the tool-state stream the web client consumes, so its
serialized form is an external contract. Story 27.6 retypes ``payload`` from a
``TypeAlias`` naming ``KnowledgeGraphStateEvent`` to the structural
``SerializableBaseModel``; the golden below was captured from the pre-split tree so
any drift caused by that retyping fails loudly.

The assertion is on the **serialized structure**, not on a round-trip: a round-trip
stays green even when the shape changes symmetrically on both sides. Every input is
pinned (explicit UUIDs, explicit timestamp) so the golden is stable.

One field legitimately changes and is therefore asserted separately rather than
folded into the golden: the envelope's own ``__model__`` marker is
``f"{cls.__module__}.{cls.__name__}"``, so moving ``ToolStateEvent`` into
``akgentic.tool.core.event`` moves the marker with it. The payload subtree — which
is what the retyping could actually have broken — stays byte-identical, and the
pre-split marker still deserializes through the façade. See the two tests below.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from akgentic.core.utils.deserializer import import_class

from akgentic.tool import KnowledgeGraphStateEvent, ToolStateEvent
from akgentic.tool.knowledge_graph.models import Entity, Relation

_EVENT_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
_TEAM_ID = uuid.UUID("00000000-0000-4000-8000-000000000002")
_ENTITY_ADDED_ID = uuid.UUID("00000000-0000-4000-8000-000000000003")
_ENTITY_MODIFIED_ID = uuid.UUID("00000000-0000-4000-8000-000000000004")
_ENTITY_REMOVED_ID = uuid.UUID("00000000-0000-4000-8000-000000000005")
_RELATION_ADDED_ID = uuid.UUID("00000000-0000-4000-8000-000000000006")
_RELATION_MODIFIED_ID = uuid.UUID("00000000-0000-4000-8000-000000000007")
_RELATION_REMOVED_ID = uuid.UUID("00000000-0000-4000-8000-000000000008")

_TIMESTAMP = datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=UTC)

# The pre-split module path, still the marker on any event persisted or in flight
# before this story shipped. The façade must keep it resolvable.
_LEGACY_ENVELOPE_MODEL = "akgentic.tool.event.ToolStateEvent"
_CURRENT_ENVELOPE_MODEL = "akgentic.tool.core.event.ToolStateEvent"


def _frozen_event() -> ToolStateEvent:
    """Build the envelope from fully pinned inputs — no default may vary."""
    payload = KnowledgeGraphStateEvent(
        entities_added=[
            Entity(
                id=_ENTITY_ADDED_ID,
                name="Alice",
                entity_type="Person",
                description="added",
                observations=["obs-a"],
                is_root=True,
            )
        ],
        entities_modified=[
            Entity(
                id=_ENTITY_MODIFIED_ID,
                name="Bob",
                entity_type="Person",
                description="modified",
            )
        ],
        entities_removed=[_ENTITY_REMOVED_ID],
        relations_added=[
            Relation(
                id=_RELATION_ADDED_ID,
                from_entity="Alice",
                to_entity="Bob",
                relation_type="KNOWS",
                description="added",
            )
        ],
        relations_modified=[
            Relation(
                id=_RELATION_MODIFIED_ID,
                from_entity="Bob",
                to_entity="Alice",
                relation_type="LIKES",
            )
        ],
        relations_removed=[_RELATION_REMOVED_ID],
    )
    return ToolStateEvent(
        id=_EVENT_ID,
        team_id=_TEAM_ID,
        timestamp=_TIMESTAMP,
        tool_id="#KnowledgeGraphTool",
        seq=7,
        payload=payload,
    )


def _serialized() -> dict[str, Any]:
    return _frozen_event().model_dump()


# Captured verbatim from the pre-split tree. Do NOT regenerate this literal to make a
# failing test pass: a diff here means the tool-state stream shape moved.
_GOLDEN_BODY: dict[str, Any] = {
    "id": "00000000-0000-4000-8000-000000000001",
    "parent_id": None,
    "team_id": "00000000-0000-4000-8000-000000000002",
    "timestamp": "2026-01-02T03:04:05.678000+00:00",
    "sender": None,
    "recipient": None,
    "display_type": "other",
    "tool_id": "#KnowledgeGraphTool",
    "seq": 7,
    "payload": {
        "entities_added": [
            {
                "id": "00000000-0000-4000-8000-000000000003",
                "name": "Alice",
                "entity_type": "Person",
                "description": "added",
                "observations": ["obs-a"],
                "is_root": True,
                "__model__": "akgentic.tool.knowledge_graph.models.Entity",
            }
        ],
        "entities_modified": [
            {
                "id": "00000000-0000-4000-8000-000000000004",
                "name": "Bob",
                "entity_type": "Person",
                "description": "modified",
                "observations": [],
                "is_root": False,
                "__model__": "akgentic.tool.knowledge_graph.models.Entity",
            }
        ],
        "entities_removed": ["00000000-0000-4000-8000-000000000005"],
        "relations_added": [
            {
                "id": "00000000-0000-4000-8000-000000000006",
                "from_entity": "Alice",
                "to_entity": "Bob",
                "relation_type": "KNOWS",
                "description": "added",
                "__model__": "akgentic.tool.knowledge_graph.models.Relation",
            }
        ],
        "relations_modified": [
            {
                "id": "00000000-0000-4000-8000-000000000007",
                "from_entity": "Bob",
                "to_entity": "Alice",
                "relation_type": "LIKES",
                "description": "",
                "__model__": "akgentic.tool.knowledge_graph.models.Relation",
            }
        ],
        "relations_removed": ["00000000-0000-4000-8000-000000000008"],
        "__model__": "akgentic.tool.knowledge_graph.models.KnowledgeGraphStateEvent",
    },
}


def test_tool_state_event_serialized_body_is_frozen() -> None:
    """Every field but the envelope marker equals the pre-split capture.

    The payload subtree is the part the ``SerializableBaseModel`` retyping could
    have broken; freezing it here proves the declared annotation never reaches the
    wire, ``__model__`` markers and field order included.
    """
    dumped = _serialized()
    assert {key: value for key, value in dumped.items() if key != "__model__"} == _GOLDEN_BODY


def test_payload_marker_survives_the_structural_retyping() -> None:
    """The payload still identifies as ``KnowledgeGraphStateEvent`` on the wire."""
    payload = _serialized()["payload"]
    assert payload["__model__"] == "akgentic.tool.knowledge_graph.models.KnowledgeGraphStateEvent"


def test_envelope_marker_tracks_the_class_module() -> None:
    """The envelope marker moved with the class — the one intended shape change.

    ``serialize_type`` builds the marker from ``cls.__module__``, so relocating
    ``ToolStateEvent`` necessarily relocates the marker. The class name — the part
    consumers discriminate on — is unchanged.
    """
    marker = _serialized()["__model__"]
    assert marker == _CURRENT_ENVELOPE_MODEL
    assert marker.rsplit(".", 1)[1] == _LEGACY_ENVELOPE_MODEL.rsplit(".", 1)[1]


def test_legacy_envelope_marker_still_deserializes() -> None:
    """An event persisted before the split still resolves to the moved class.

    ``import_class`` does ``import_module`` + ``getattr``, so the façade's PEP 562
    ``__getattr__`` answers the pre-split path and older payloads keep loading.
    """
    assert import_class(_LEGACY_ENVELOPE_MODEL) is ToolStateEvent
