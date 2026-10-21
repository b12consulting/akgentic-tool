"""The model-switch capability: its domain vocabulary, observer contract and card.

This package speaks in ``ModelRow`` — a serializable projection of one roster
entry — because ``akgentic-tool`` may import ``akgentic-core`` only, and the
roster's own configuration model belongs to ``akgentic-llm``.
"""

from akgentic.tool.model.observer import ModelSwitchToolObserver
from akgentic.tool.model.state import ActiveModelState, ModelCompaction, ModelRow
from akgentic.tool.model.tool import ActiveModel, ListModels, ModelTool, SwitchModel

__all__ = [
    "ActiveModel",
    "ActiveModelState",
    "ListModels",
    "ModelCompaction",
    "ModelRow",
    "ModelSwitchToolObserver",
    "ModelTool",
    "SwitchModel",
]
