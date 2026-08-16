"""Knowledge-graph payload typing for the tool-state event stream.

``ToolStatePayload`` names the delta payload a ``ToolStateEvent`` carries for this
domain. It lives here, not beside the envelope: the envelope is package-global and
must name no domain type, while this alias is knowledge-graph business.

It is a real alias, not a string forward reference. The cycle the forward reference
was dodging — ``event.py`` needing ``models.py`` while ``models.py`` needed
``event.py`` — no longer exists now that ``ToolStateEvent.payload`` is typed
structurally, so the import runs in the ordinary direction.
"""

from __future__ import annotations

from typing import TypeAlias

from akgentic.tool.knowledge_graph.models import KnowledgeGraphStateEvent

#: Delta payload carried by a ``ToolStateEvent`` emitted by the knowledge-graph tool.
#
# Deliberately a ``TypeAlias`` rather than a PEP 695 ``type`` alias: ``type`` builds a
# ``TypeAliasType`` whose runtime value is a lazy wrapper, so ``ToolStatePayload`` would
# stop *being* the class. Consumers that resolve the alias at runtime need the class
# itself, which is what this name has always meant.
ToolStatePayload: TypeAlias = KnowledgeGraphStateEvent  # noqa: UP040

__all__ = ["KnowledgeGraphStateEvent", "ToolStatePayload"]
