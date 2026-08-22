"""Tool abstractions and factory for the akgentic tool package.

Defines the core contracts:
- ``BaseToolParam``: base for capability parameter models.
- ``ToolCard``: abstract base — tool configuration + callable factory in one class.
- ``ToolFactory``: resolves ``ToolCard`` instances into callable tools, prompts, and toolsets.
"""

from .card import ToolCard
from .channels import COMMAND, LLM_CONTEXT, SYSTEM_PROMPT, TOOL_CALL, Channels
from .commands import CommandRegistry
from .context_state import ContextState

# ``_resolve`` and ``_topological_sort`` stay on the façade: six modules under
# src/ and the test suite import them from ``akgentic.tool.core`` directly. The
# redundant ``as`` alias marks them as deliberate re-exports (mypy strict turns
# off implicit re-export) without promoting them into ``__all__``.
from .dependencies import _topological_sort as _topological_sort  # noqa: F401
from .factory import ToolFactory
from .params import BaseToolParam, normalize_system_prompt_to_llm_context
from .params import _resolve as _resolve  # noqa: F401

__all__ = [
    "COMMAND",
    "LLM_CONTEXT",
    "SYSTEM_PROMPT",
    "TOOL_CALL",
    "BaseToolParam",
    "Channels",
    "CommandRegistry",
    "ContextState",
    "ToolCard",
    "ToolFactory",
    "normalize_system_prompt_to_llm_context",
]
