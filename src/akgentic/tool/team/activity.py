"""``team_activity`` — who on the team is mid-handler, and on what (ADR-033 §Decision 3).

Team activity is a **capability on ``TeamTool``**, not a card of its own, and it
rests on two independent gates:

* ``TeamTool.get_team_activity`` decides whether ``team_activity`` is exposed at
  all. It defaults to ``True`` — the truncate-only report is pure telemetry and
  cheap enough to be on everywhere.
* :attr:`GetTeamActivity.summarizer` decides whether the ``#TeamActivity`` cache
  actor exists. The actor is there **only** to cache summaries, so with no
  summarizer there is nothing to cache: no actor, no worker, and no
  ``summarize_over`` parameter in the signature the model sees.

Nothing new is instrumented. ``Akgent.on_receive`` already wraps every handler in
a telemetry sandwich — ``ReceivedMessage`` before dispatch, ``ProcessedMessage``
after — and both carry the *receiving* agent's address as ``sender``. An agent
with a ``ReceivedMessage`` whose ``ProcessedMessage`` has not arrived is inside a
handler. The task text is one hop away: the matching ``SentMessage``, emitted by
the sender before the recipient could possibly receive it, carries the full
message.

Four properties of this derivation are load-bearing:

* **Grouping is by ``agent_id``, never by ``name``.** Firing a member frees its
  name for immediate reuse, so a name key merges two different actors into one row.
* **The stale cut-off is not a nicety.** Both telemetry messages are persisted and
  replayed on resume, so a team stopped mid-turn carries a ``ReceivedMessage``
  whose ``ProcessedMessage`` never existed and never will. Without the cut-off
  that member is reported as working for ever.
* **Only the filtered ``get_messages(message_type=...)`` form is used.** The
  unfiltered call returns the orchestrator's live list *by reference* and races
  its appends.
* **``summarize_over=None`` costs nothing.** Long tasks are truncated. Passing an
  integer is the opt-in — the threshold *is* the consent — and routes only longer
  tasks through the deferred cache, keyed by ``message_id`` so a follow-up call is
  free.

The last property is enforced structurally rather than by inspection:
:func:`apply_truncations` and :func:`apply_summaries` are separate functions, and
the truncate-only one has no cache proxy in scope at all.

**No ``akgentic-llm`` import appears here, by design** (NFR1). ``akgentic-tool``
may reach for ``akgentic-core``, pydantic and pydantic-ai only, so ``ModelConfig``
does not exist at this layer; pydantic-ai's own model spec string
(``"openai:gpt-5.2-mini"``) needs no new dependency.

Known and accepted: summarizer tokens are produced outside ``ReactAgent``, so they
reach neither cost accounting nor any usage limit. Summarization is opt-in,
per-``message_id`` and cached, which bounds the exposure to roughly one small
call per distinct long task per team.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message
from akgentic.core.messages.orchestrator import (
    ProcessedMessage,
    ReceivedMessage,
    SentMessage,
)
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.core import (
    COMMAND,
    TOOL_CALL,
    BaseToolParam,
    Channels,
)
from akgentic.tool.core.deferred import (
    DeferredPayload,
    DeferredResultActor,
    DeferredWorker,
    poll_deferred,
)

UNRESOLVED_TASK = "<task text unavailable>"
"""Reported when no ``SentMessage`` matches an open ``message_id``.

A member with an unresolvable task is still working, so it is still reported —
the text degrades, the row does not disappear.
"""

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

_ELLIPSIS = "…"


class AgentActivity(SerializableBaseModel):
    """One team member currently inside a message handler — the wire row.

    This model is read back by the calling model on every invocation, so every
    field is prompt cost. The derivation keys deliberately never appear here:
    ``agent_id`` (the grouping key) and ``message_id`` (the summary cache key)
    live on :class:`MemberRow`, and the busy duration is derivable from
    ``generated_at − started_at``.

    Attributes:
        name: Member name at the time the message was received.
        role: Member role.
        task: Task text — full, truncated, or summarized (see ``summarized``).
        summarized: True when ``task`` came back from the summarizer.
        started_at: When the handler started (the ``ReceivedMessage`` timestamp).
        suspect: True when more than one message is open for this member, which
            cannot happen on a live team and therefore signals replayed or
            malformed telemetry. The member is still reported.
    """

    name: str
    role: str
    task: str
    summarized: bool
    started_at: datetime
    suspect: bool = False


class TeamActivityReport(SerializableBaseModel):
    """Point-in-time answer to "who is working, and on what".

    Attributes:
        generated_at: When the report was derived.
        members: One row per busy member; empty when the team is idle.
        pending_summaries: Members whose task came back truncated because their
            summary was still in flight when the poll budget ran out.
    """

    generated_at: datetime
    members: list[AgentActivity]
    pending_summaries: int


class ActivitySummarizer(SerializableBaseModel):
    """Opt-in summarization of long task text.

    Configuring one is what brings the ``#TeamActivity`` cache actor into
    existence: the actor stores summaries and nothing else, so without a
    summarizer there is nothing for it to hold. Even with one configured, no
    model call happens until a caller passes ``summarize_over``.

    Attributes:
        model: pydantic-ai model spec string, never a ``ModelConfig`` — this
            package must not import ``akgentic-llm``.
        poll_attempts: Attempts spent waiting for in-flight summaries.
        poll_delay_seconds: Seconds slept between attempts.
    """

    model: str = Field(
        default="openai:gpt-5.2-mini",
        description="pydantic-ai model spec string used to summarize long tasks.",
    )
    poll_attempts: int = Field(
        default=5, description="Attempts spent waiting for in-flight summaries."
    )
    poll_delay_seconds: float = Field(
        default=0.4, description="Seconds slept between summary poll attempts."
    )


class GetTeamActivity(BaseToolParam):
    """Report which team members are currently working, and on what."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}
    summarizer: ActivitySummarizer | None = Field(
        default=None,
        description="Configure to summarize long tasks on demand. None: truncate, no actor.",
    )
    stale_after_seconds: float = Field(
        default=300.0,
        description="Open handlers older than this are treated as replayed history.",
    )
    max_task_chars: int = Field(
        default=200, description="Character budget for reported task text."
    )


class SummaryBudget(SerializableBaseModel):
    """Bind-time summarization configuration, read once per report.

    Attributes:
        model: pydantic-ai model spec string handed to the summarizer worker.
        max_chars: Character budget for both truncation and summaries.
        poll_attempts: Attempts spent waiting for in-flight summaries.
        poll_delay_seconds: Seconds slept between attempts.
    """

    model: str
    max_chars: int
    poll_attempts: int
    poll_delay_seconds: float

    @classmethod
    def from_params(cls, params: GetTeamActivity) -> SummaryBudget:
        """Build the budget from a param model that configures a summarizer.

        Args:
            params: Capability configuration whose ``summarizer`` is set.

        Returns:
            The bind-time budget.

        Raises:
            ValueError: If ``params.summarizer`` is ``None`` — the summarizing
                path must never be reachable without one.
        """
        if params.summarizer is None:
            raise ValueError("A summary budget requires a configured summarizer.")
        return cls(
            model=params.summarizer.model,
            max_chars=params.max_task_chars,
            poll_attempts=params.summarizer.poll_attempts,
            poll_delay_seconds=params.summarizer.poll_delay_seconds,
        )


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


@dataclass(slots=True)
class _OpenGroup:
    """Open handlers of one actor, plus the address they were reported under.

    Runtime bookkeeping local to one ``team_activity`` call — it never crosses an
    actor boundary, which is why it is a dataclass rather than a serializable
    model. Carrying the sender alongside the entries keeps the address non-optional
    downstream without re-deriving it from an arbitrary entry.
    """

    sender: ActorAddress
    entries: list[ReceivedMessage] = field(default_factory=list)


@dataclass(slots=True)
class MemberRow:
    """One busy member with its derivation keys, before any task text is applied.

    Runtime bookkeeping local to one ``team_activity`` call, like
    :class:`_OpenGroup` — it never crosses an actor boundary and never reaches
    the wire. It exists so the keys the derivation needs — ``agent_id`` for
    grouping, ``message_id`` for the summary cache — have a home that is not the
    report the model reads back (:class:`AgentActivity`).

    Attributes:
        name: Member name at the time the message was received.
        agent_id: Identity of the actor. The grouping key — names are reusable.
        role: Member role.
        message_id: The open message being handled. Also the summary cache key.
        started_at: When the handler started (the ``ReceivedMessage`` timestamp).
        suspect: True when more than one message is open for this member.
    """

    name: str
    agent_id: uuid.UUID
    role: str
    message_id: uuid.UUID
    started_at: datetime
    suspect: bool


@dataclass(slots=True)
class ActivitySnapshot:
    """Everything derived from telemetry, before any task text is applied.

    Runtime bookkeeping local to one ``team_activity`` call, like
    :class:`_OpenGroup`. It exists so the two report paths share one derivation:
    the truncate-only path and the summarizing path differ solely in what they do
    with ``texts``, never in how the rows are found.

    Attributes:
        generated_at: When the derivation ran; also the report timestamp.
        rows: One internal row per busy member, derivation keys included.
        texts: Resolved task text per open ``message_id``.
    """

    generated_at: datetime
    rows: list[MemberRow]
    texts: dict[uuid.UUID, str]


def _as_utc(value: datetime) -> datetime:
    """Read a timestamp as UTC, tolerating a naive one.

    Live telemetry is always tz-aware, but replayed history has crossed a
    serializer; a naive value must degrade rather than raise mid-comparison.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _is_excluded(sender: ActorAddress, caller_id: uuid.UUID) -> bool:
    """True for a sender that never belongs in the report.

    All three exclusions are structural, read off the address itself: the caller
    (by ``agent_id``, so a renamed actor cannot slip through), tool actors (the
    ``#`` prefix, which covers the deferred workers this very mechanism spawns),
    and the user proxy (by type — a human blocked on input is not working).
    """
    return (
        sender.agent_id == caller_id or sender.name.startswith("#") or sender.is_user_proxy
    )


def _truncate(text: str, max_chars: int) -> str:
    """Clip *text* to *max_chars*, marking the cut with an ellipsis."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + _ELLIPSIS


def _task_text(message: Message) -> str:
    """Best-effort task text for a sent message.

    ``SentMessage.message`` is a full ``Message`` and the concrete subclass varies:
    ``UserMessage`` / ``ResultMessage`` / ``NotificationMessage`` carry ``content``,
    others carry none. A message without usable text degrades to its class name
    rather than raising or dropping the member.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content
    return f"<{type(message).__name__}>"


def _collect_open_groups(
    orchestrator_proxy: Orchestrator,
    caller_id: uuid.UUID,
    now: datetime,
    stale_after_seconds: float,
) -> dict[uuid.UUID, _OpenGroup]:
    """Open handlers per actor, excluded and stale entries already removed.

    Two filtered history scans: ``ReceivedMessage`` minus the ``message_id``s that
    already have a ``ProcessedMessage``. Entries older than *stale_after_seconds*
    are dropped — on resume the orchestrator replays a pair that was never closed
    and never will be, and without this cut-off that member reads as busy for ever.
    """
    received = orchestrator_proxy.get_messages(message_type=ReceivedMessage)
    processed = orchestrator_proxy.get_messages(message_type=ProcessedMessage)
    processed_ids = {msg.message_id for msg in processed if isinstance(msg, ProcessedMessage)}
    cutoff = now - timedelta(seconds=stale_after_seconds)

    groups: dict[uuid.UUID, _OpenGroup] = {}
    for msg in received:
        sender = msg.sender
        if not isinstance(msg, ReceivedMessage) or sender is None:
            continue
        if msg.message_id in processed_ids or _is_excluded(sender, caller_id):
            continue
        if msg.timestamp is None or _as_utc(msg.timestamp) < cutoff:
            continue
        group = groups.setdefault(sender.agent_id, _OpenGroup(sender=sender))
        group.entries.append(msg)
    return groups


def _build_rows(groups: dict[uuid.UUID, _OpenGroup]) -> list[MemberRow]:
    """One row per group, reported against its OLDEST open entry.

    A group holding more than one open entry cannot arise on a live team — Pykka
    processes one message at a time per actor and the sandwich is strict, so the
    live difference is structurally 0 or 1. More than that is an anomaly, so the
    row names the entry the agent is most plausibly still inside and carries
    ``suspect=True``; it is never folded into a plain "working" and never dropped.

    Task text is not resolved here — :func:`_resolve_task_texts` does that.
    """
    rows: list[MemberRow] = []
    for agent_id, group in groups.items():
        oldest = min(group.entries, key=lambda msg: _as_utc(cast(datetime, msg.timestamp)))
        rows.append(
            MemberRow(
                name=group.sender.name,
                agent_id=agent_id,
                role=group.sender.role,
                message_id=oldest.message_id,
                started_at=_as_utc(cast(datetime, oldest.timestamp)),
                suspect=len(group.entries) > 1,
            )
        )
    return rows


def _resolve_task_texts(
    orchestrator_proxy: Orchestrator, rows: list[MemberRow]
) -> dict[uuid.UUID, str]:
    """Task text per open ``message_id``, walking ``SentMessage`` newest-first.

    A message is always sent before it can be received, so every open id sits
    near the end of the history; the walk stops the moment the last one is
    resolved rather than scanning the whole conversation. Indexed rather than
    ``reversed()`` so that "entries older than the last resolution are never
    inspected" is an observable property, not just an intended one.
    """
    wanted = {row.message_id for row in rows}
    if not wanted:
        return {}

    sent = orchestrator_proxy.get_messages(message_type=SentMessage)
    resolved: dict[uuid.UUID, str] = {}
    for index in range(len(sent) - 1, -1, -1):
        msg = sent[index]
        if not isinstance(msg, SentMessage) or msg.message.id not in wanted:
            continue
        resolved[msg.message.id] = _task_text(msg.message)
        wanted.discard(msg.message.id)
        if not wanted:
            break
    return resolved


def build_snapshot(
    orchestrator_proxy: Orchestrator,
    caller_id: uuid.UUID,
    stale_after_seconds: float,
) -> ActivitySnapshot:
    """Derive the busy members and their task texts from team telemetry.

    Shared by both report paths, and cost-free in the sense that matters: three
    filtered history scans on the orchestrator's own thread, no actor spawned and
    no model consulted.

    Args:
        orchestrator_proxy: Ask-proxy to the orchestrator.
        caller_id: ``agent_id`` of the asking agent, excluded from the report.
        stale_after_seconds: Replay cut-off for an unbalanced open handler.

    Returns:
        The rows, their resolved task texts, and the derivation timestamp.
    """
    now = datetime.now(UTC)
    groups = _collect_open_groups(orchestrator_proxy, caller_id, now, stale_after_seconds)
    rows = _build_rows(groups)
    return ActivitySnapshot(
        generated_at=now,
        rows=rows,
        texts=_resolve_task_texts(orchestrator_proxy, rows),
    )


def _to_activity(row: MemberRow, task: str, summarized: bool) -> AgentActivity:
    """Build the wire row from an internal one — the derivation keys stop here."""
    return AgentActivity(
        name=row.name,
        role=row.role,
        task=task,
        summarized=summarized,
        started_at=row.started_at,
        suspect=row.suspect,
    )


def apply_truncations(
    rows: list[MemberRow], texts: dict[uuid.UUID, str], max_chars: int
) -> list[AgentActivity]:
    """Attach truncated task text to every row.

    The whole summarizer-less path, and deliberately a function that takes **no
    cache proxy**: "no model call happens here" is then true by construction
    rather than by a runtime ``if`` that a later edit could invert.
    """
    out: list[AgentActivity] = []
    for row in rows:
        text = texts.get(row.message_id, UNRESOLVED_TASK)
        out.append(_to_activity(row, _truncate(text, max_chars), summarized=False))
    return out


def apply_summaries(
    rows: list[MemberRow],
    texts: dict[uuid.UUID, str],
    activity_proxy: TeamActivityActor,
    budget: SummaryBudget,
    summarize_over: int | None,
) -> tuple[list[AgentActivity], int]:
    """Attach the task text to every row, summarizing only where asked.

    ``summarize_over is None`` short-circuits before the cache is touched at all:
    no ``get``, no ``request``, no worker, no tokens. That short-circuit is the
    whole point of the default, not an optimization.

    A cached summary is clipped like any other text. The worker already clips to
    the budget of whoever requested it, but ``#TeamActivity`` is one singleton
    shared by every card on the team, so a card with a smaller ``max_task_chars``
    can read back a summary produced under a larger one.
    """
    tasks: dict[uuid.UUID, tuple[str, bool]] = {}
    pending: set[uuid.UUID] = set()
    for row in rows:
        text = texts.get(row.message_id, UNRESOLVED_TASK)
        if summarize_over is None or len(text) <= summarize_over:
            tasks[row.message_id] = (_truncate(text, budget.max_chars), False)
            continue
        cached = activity_proxy.get(row.message_id)
        if cached is not None:
            tasks[row.message_id] = (_truncate(cached, budget.max_chars), True)
            continue
        activity_proxy.request(
            row.message_id,
            SummarizePayload(
                deferred_key=row.message_id,
                text=text,
                model=budget.model,
                max_chars=budget.max_chars,
            ),
        )
        pending.add(row.message_id)
        tasks[row.message_id] = (_truncate(text, budget.max_chars), False)

    still_pending = (
        _harvest_summaries(tasks, pending, activity_proxy, budget) if pending else 0
    )
    return [_to_activity(row, *tasks[row.message_id]) for row in rows], still_pending


def _harvest_summaries(
    tasks: dict[uuid.UUID, tuple[str, bool]],
    pending: set[uuid.UUID],
    activity_proxy: TeamActivityActor,
    budget: SummaryBudget,
) -> int:
    """Poll ONCE for the whole pending set, then sweep for partial arrivals.

    ``poll_deferred`` returns the first non-``None`` result, so a ``fetch`` that
    answered with a partial dict would end the poll on the first arrival. It
    therefore answers only when every key has landed — and the sweep afterwards
    picks up whatever arrived while the budget was running out. Without the sweep
    ``pending_summaries`` over-counts members whose summary actually did land.

    Mutates ``tasks`` in place for every arrival and returns the count of keys
    still unresolved when the budget ran out.
    """
    keys = sorted(pending, key=str)

    def fetch_all() -> dict[uuid.UUID, str] | None:
        got = {key: value for key in keys if (value := activity_proxy.get(key)) is not None}
        return got if len(got) == len(keys) else None

    resolved = poll_deferred(
        fetch_all, attempts=budget.poll_attempts, delay=budget.poll_delay_seconds
    )
    if resolved is None:
        resolved = {key: value for key in keys if (value := activity_proxy.get(key)) is not None}

    for key, summary in resolved.items():
        tasks[key] = (_truncate(summary, budget.max_chars), True)
    return len(pending - resolved.keys())
