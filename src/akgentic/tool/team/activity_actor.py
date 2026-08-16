"""Deferred summarization behind ``who_is_working`` (ADR-033 §Decision 3).

Three pieces, all consumers of ``akgentic.tool.core.deferred``:

* :class:`SummarizePayload` — one long task text plus everything needed to
  summarize it. The worker is spawned with a hardcoded ``BaseConfig``, so the
  model spec and the character budget have no channel other than the payload.
* :class:`SummarizerWorker` — builds a pydantic-ai ``Agent`` from that model
  spec string and performs exactly one call, on its own thread.
* :class:`TeamActivityActor` — the ``#TeamActivity`` singleton cache, keyed by
  the ``message_id`` of the task being summarized.

**No ``akgentic-llm`` import appears here, by design** (NFR1). ``akgentic-tool``
may reach for ``akgentic-core``, pydantic and pydantic-ai only, so ``ModelConfig``
does not exist at this layer; pydantic-ai's own model spec string
(``"openai:gpt-5.2-mini"``) needs no new dependency.

Known and accepted: these tokens are produced outside ``ReactAgent``, so they
reach neither cost accounting nor any usage limit. Summarization is opt-in,
per-``message_id`` and cached, which bounds the exposure to roughly one small
call per distinct long task per team.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.tool.core.deferred import (
    DeferredPayload,
    DeferredResultActor,
    DeferredWorker,
)

TEAM_ACTIVITY_ACTOR_NAME = "#TeamActivity"
TEAM_ACTIVITY_ACTOR_ROLE = "ToolActor"

SUMMARY_INSTRUCTION = (
    "Summarize what the following task asks for, in at most {max_chars} characters. "
    "Answer with the summary only: no preamble, no quotes, no trailing commentary."
)
"""Prompt wrapped around the task text.

Deliberately short and deterministic — one instruction plus the text — so the
call stays cheap and its output stays a single unadorned sentence.
"""


class SummarizePayload(DeferredPayload):
    """One task text to summarize, with its budget and its model.

    ``deferred_key`` narrows the base's ``Any`` to the ``message_id`` of the task,
    which is also the cache key. Never set it independently of the ``request()``
    key: ``request`` rebinds it, and a caller that let the two drift would clear
    a different in-flight mark and strand this one for the actor's lifetime.

    Attributes:
        deferred_key: ``message_id`` of the task being summarized.
        text: Full task text.
        model: pydantic-ai model spec string (e.g. ``"openai:gpt-5.2-mini"``).
        max_chars: Character budget for the produced summary.
    """

    deferred_key: uuid.UUID = Field(
        ...,
        description="message_id of the task being summarized; also the cache key.",
    )
    text: str = Field(..., description="Full task text to summarize.")
    model: str = Field(..., description="pydantic-ai model spec string.")
    max_chars: int = Field(..., description="Character budget for the summary.")


class SummarizerWorker(DeferredWorker):
    """Summarizes one task text with one model call, then stops.

    ``produce`` runs on the worker's own thread, which is why the pydantic-ai
    call is the synchronous ``run_sync`` form rather than an awaited one.
    """

    def produce(self, payload: DeferredPayload) -> Any:
        """Summarize ``payload.text`` within ``payload.max_chars``.

        :attr:`~akgentic.tool.core.deferred.DeferredWorker.timeout_s` is handed to
        the model call through ``ModelSettings``. A budget that does not reach the
        client is decoration: a Python thread cannot be cancelled, and this worker
        holds its parent's ``stop_children(blocking=True)`` open until it returns.

        Args:
            payload: A :class:`SummarizePayload`.

        Returns:
            The summary, clipped to ``payload.max_chars``.

        Raises:
            TypeError: If handed a payload that is not a :class:`SummarizePayload`.
        """
        if not isinstance(payload, SummarizePayload):
            raise TypeError(f"SummarizerWorker requires a SummarizePayload, got {type(payload)}")

        agent = Agent(payload.model)
        prompt = (
            f"{SUMMARY_INSTRUCTION.format(max_chars=payload.max_chars)}\n\n"
            f"Task:\n{payload.text}"
        )
        result = agent.run_sync(prompt, model_settings=ModelSettings(timeout=self.timeout_s))
        return str(result.output).strip()[: payload.max_chars]


class TeamActivityActor(DeferredResultActor[BaseConfig, BaseState, uuid.UUID, str]):
    """The ``#TeamActivity`` singleton: task summaries keyed by ``message_id``.

    Keeps the base's capacity and negative-TTL defaults. A summary is only ever
    requested for a task that is *currently open*, so the working set is bounded
    by the size of the team rather than by conversation length, and 128 slots is
    already generous for it.
    """

    def worker_class(self) -> type[DeferredWorker]:
        """Return :class:`SummarizerWorker`."""
        return SummarizerWorker
