"""The exec capability's parameter, beside the closures it configures.

Configuration only — a :class:`~akgentic.tool.core.BaseToolParam` subclass
carrying what the capability is configured *with*, never what its callable is
*called* with (ADR-020).

**Here rather than in ``card/params.py``, and the reason is an import-graph fact
rather than tidiness.** Importing anything under ``card/`` executes
``card/__init__.py``, and that module imports :class:`ExecFactories` from
``execution/card.py`` — so an exec module taking its own parameter from the card
would close a cycle Python refuses, as well as attributing the whole ``card``
package to this capability's import closure. ``card/params.py`` re-exports the
name instead, for the stored-record reason its own docstring gives.
"""

from __future__ import annotations

from pydantic import Field

from akgentic.tool.core import TOOL_CALL, BaseToolParam, Channels
from akgentic.tool.sandbox.backend import CardMode
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_POLL_ATTEMPTS,
    DEFAULT_EXEC_POLL_DELAY_S,
    DEFAULT_EXEC_TIMEOUT_S,
)


class WorkspaceExec(BaseToolParam):
    """Run a sandboxed shell command against the team workspace.

    Configuration only — nothing here duplicates an argument of the callables it
    enables. The two budgets it carries are two different things and are easy to
    conflate:

    - ``timeout_s`` bounds the **subprocess**, and reaches
      ``subprocess.run(timeout=...)`` in the backend. It is clamped to the
      :data:`~akgentic.tool.workspace.execution.MAX_EXEC_BUDGET_S`, which sits
      below the orchestrator's stop backstop.
    - ``poll_attempts`` × ``poll_delay_seconds`` bounds how long the **agent's
      own thread** waits inside the tool call. It cannot extend the first:
      raising it buys more looking, never more running.

    ``poll_attempts`` has three settings, and each is bounded by a different
    thing:

    - ``-1`` (the default) — **wait out the run.** Resolved at wiring time to
      the count whose wait is the longest still fitting the *effective run
      budget* (``effective_budget(timeout_s)``) plus
      :data:`~akgentic.tool.workspace.execution.EXEC_REPORT_MARGIN_S`, so the
      wait covers the sandbox's report and not merely the command. The common
      case then returns the command's own output and the model never sees a run
      id.
    - a **positive count** — a bounded look of ``count × poll_delay_seconds``,
      clamped to the effective run budget and **without** the margin. Exhausting
      it hands back a run id.
    - ``0`` — no polling at all: the run id comes back immediately.

    Anything below ``-1`` is a validation error rather than a second spelling of
    the sentinel.

    A run that outlives the wait is collected with ``workspace_exec_result``; it
    never outlives ``timeout_s``.
    """

    expose: set[Channels] = {TOOL_CALL}
    mode: CardMode = "auto"
    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S
    poll_attempts: int = Field(default=DEFAULT_EXEC_POLL_ATTEMPTS, ge=-1)
    poll_delay_seconds: float = DEFAULT_EXEC_POLL_DELAY_S
