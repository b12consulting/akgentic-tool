"""Knowledge-graph event surface.

Re-exports this domain's tool-state payload type, ``KnowledgeGraphStateEvent``, so a
consumer reaching for the knowledge graph's event contract finds it under ``event``
rather than having to know it is declared beside the graph models. The class itself
lives in ``models.py``, alongside the entity and relation types whose deltas it carries.

Re-exporting a symbol of this module's own domain is what the ``core/`` purity rule
permits; what it forbids is a domain module re-exporting a foreign one.
"""

from __future__ import annotations

from akgentic.tool.knowledge_graph.models import KnowledgeGraphStateEvent

__all__ = ["KnowledgeGraphStateEvent"]
