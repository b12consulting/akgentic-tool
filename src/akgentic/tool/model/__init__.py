"""The model-switch capability's domain vocabulary and observer contract.

This package speaks in ``ModelRow`` — a serializable projection of one roster
entry — because ``akgentic-tool`` may import ``akgentic-core`` only, and the
roster's own configuration model belongs to ``akgentic-llm``.
"""

from akgentic.tool.model.observer import ModelSwitchToolObserver
from akgentic.tool.model.state import ModelRow

__all__ = [
    "ModelRow",
    "ModelSwitchToolObserver",
]
