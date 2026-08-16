"""``who_is_working`` — who on the team is mid-handler, and on what (ADR-033 §Decision 3).

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
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from pydantic import Field, PrivateAttr

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent_config import BaseConfig
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
    ToolCard,
    _resolve,
)
from akgentic.tool.core.deferred import poll_deferred
from akgentic.tool.errors import RetriableError
from akgentic.tool.event import ActorToolObserver, ToolObserver
from akgentic.tool.team.activity_actor import (
    TEAM_ACTIVITY_ACTOR_NAME,
    TEAM_ACTIVITY_ACTOR_ROLE,
    SummarizePayload,
    TeamActivityActor,
)

logger = logging.getLogger(__name__)

UNRESOLVED_TASK = "<task text unavailable>"
"""Reported when no ``SentMessage`` matches an open ``message_id``.

A member with an unresolvable task is still working, so it is still reported —
the text degrades, the row does not disappear.
"""

_ELLIPSIS = "…"


class AgentActivity(SerializableBaseModel):
    """One team member currently inside a message handler.

    Attributes:
        name: Member name at the time the message was received.
        agent_id: Identity of the actor. The grouping key — names are reusable.
        role: Member role.
        message_id: The open message being handled. Also the summary cache key.
        task: Task text — full, truncated, or summarized (see ``summarized``).
        summarized: True when ``task`` came back from the summarizer.
        started_at: When the handler started (the ``ReceivedMessage`` timestamp).
        busy_for_seconds: Seconds between ``started_at`` and report generation.
        suspect: True when more than one message is open for this member, which
            cannot happen on a live team and therefore signals replayed or
            malformed telemetry. The member is still reported.
    """

    name: str
    agent_id: uuid.UUID
    role: str
    message_id: uuid.UUID
    task: str
    summarized: bool
    started_at: datetime
    busy_for_seconds: float
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


class GetTeamActivity(BaseToolParam):
    """Report which team members are currently working, and on what."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


@dataclass(slots=True)
class _OpenGroup:
    """Open handlers of one actor, plus the address they were reported under.

    Runtime bookkeeping local to one ``who_is_working`` call — it never crosses an
    actor boundary, which is why it is a dataclass rather than a serializable
    model. Carrying the sender alongside the entries keeps the address non-optional
    downstream without re-deriving it from an arbitrary entry.
    """

    sender: ActorAddress
    entries: list[ReceivedMessage] = field(default_factory=list)


class _SummaryBudget(SerializableBaseModel):
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


def _build_rows(groups: dict[uuid.UUID, _OpenGroup], now: datetime) -> list[AgentActivity]:
    """One row per group, reported against its OLDEST open entry.

    A group holding more than one open entry cannot arise on a live team — Pykka
    processes one message at a time per actor and the sandwich is strict, so the
    live difference is structurally 0 or 1. More than that is an anomaly, so the
    row names the entry the agent is most plausibly still inside and carries
    ``suspect=True``; it is never folded into a plain "working" and never dropped.

    ``task`` is a placeholder here — :func:`_resolve_task_texts` fills it in.
    """
    rows: list[AgentActivity] = []
    for agent_id, group in groups.items():
        oldest = min(group.entries, key=lambda msg: _as_utc(cast(datetime, msg.timestamp)))
        started_at = _as_utc(cast(datetime, oldest.timestamp))
        rows.append(
            AgentActivity(
                name=group.sender.name,
                agent_id=agent_id,
                role=group.sender.role,
                message_id=oldest.message_id,
                task=UNRESOLVED_TASK,
                summarized=False,
                started_at=started_at,
                busy_for_seconds=max((now - started_at).total_seconds(), 0.0),
                suspect=len(group.entries) > 1,
            )
        )
    return rows


def _resolve_task_texts(
    orchestrator_proxy: Orchestrator, rows: list[AgentActivity]
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


def _apply_summaries(
    rows: list[AgentActivity],
    texts: dict[uuid.UUID, str],
    activity_proxy: TeamActivityActor,
    budget: _SummaryBudget,
    summarize_over: int | None,
) -> tuple[list[AgentActivity], int]:
    """Attach the task text to every row, summarizing only where asked.

    ``summarize_over is None`` short-circuits before the cache is touched at all:
    no ``get``, no ``request``, no worker, no tokens. That short-circuit is the
    whole point of the default, not an optimization.
    """
    out: list[AgentActivity] = []
    pending: set[uuid.UUID] = set()
    for row in rows:
        text = texts.get(row.message_id, UNRESOLVED_TASK)
        if summarize_over is None or len(text) <= summarize_over:
            out.append(row.model_copy(update={"task": _truncate(text, budget.max_chars)}))
            continue
        cached = activity_proxy.get(row.message_id)
        if cached is not None:
            out.append(row.model_copy(update={"task": cached, "summarized": True}))
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
        out.append(row.model_copy(update={"task": _truncate(text, budget.max_chars)}))

    if not pending:
        return out, 0
    return _harvest_summaries(out, pending, activity_proxy, budget)


def _harvest_summaries(
    rows: list[AgentActivity],
    pending: set[uuid.UUID],
    activity_proxy: TeamActivityActor,
    budget: _SummaryBudget,
) -> tuple[list[AgentActivity], int]:
    """Poll ONCE for the whole pending set, then sweep for partial arrivals.

    ``poll_deferred`` returns the first non-``None`` result, so a ``fetch`` that
    answered with a partial dict would end the poll on the first arrival. It
    therefore answers only when every key has landed — and the sweep afterwards
    picks up whatever arrived while the budget was running out. Without the sweep
    ``pending_summaries`` over-counts members whose summary actually did land.
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

    harvested = [
        row.model_copy(update={"task": resolved[row.message_id], "summarized": True})
        if row.message_id in pending and row.message_id in resolved
        else row
        for row in rows
    ]
    still_pending = sum(
        1 for row in rows if row.message_id in pending and row.message_id not in resolved
    )
    return harvested, still_pending


class TeamActivityTool(ToolCard):
    """Reports which team members are mid-handler, and on what.

    A card of its own rather than a capability on ``TeamTool``: this one owns an
    actor, a cache and an optional model dependency, and a team that only wants
    hire/fire must not inherit those by accident.

    Attributes:
        get_team_activity: Enables ``who_is_working`` (default: True).
        summarizer_model: pydantic-ai model spec used when summarization is
            requested. Never consulted while ``summarize_over`` is None.
        summary_max_chars: Character budget for both truncated and summarized
            task text.
        stale_after_seconds: An open handler older than this is treated as
            replayed history and excluded.
        poll_attempts: Attempts spent waiting for in-flight summaries.
        poll_delay_seconds: Seconds slept between attempts.
    """

    get_team_activity: GetTeamActivity | bool = Field(
        default=True, description="Enable the who_is_working report (default: True)"
    )
    summarizer_model: str = Field(
        default="openai:gpt-5.2-mini",
        description="pydantic-ai model spec string used to summarize long tasks.",
    )
    summary_max_chars: int = Field(
        default=200, description="Character budget for reported task text."
    )
    stale_after_seconds: float = Field(
        default=300.0,
        description="Open handlers older than this are treated as replayed history.",
    )
    poll_attempts: int = Field(
        default=5, description="Attempts spent waiting for in-flight summaries."
    )
    poll_delay_seconds: float = Field(
        default=0.4, description="Seconds slept between summary poll attempts."
    )

    # Runtime handles: actor proxies are not serializable and never fields.
    _orchestrator_proxy: Orchestrator | None = PrivateAttr(default=None)
    _activity_proxy: TeamActivityActor | None = PrivateAttr(default=None)

    def observer(self, observer: ToolObserver) -> TeamActivityTool:
        """Attach the observer and bind the ``#TeamActivity`` singleton.

        Requires an ``ActorToolObserver``; the parameter keeps the base
        ``ToolObserver`` type so the override stays substitutable, and
        :meth:`_actor_observer` applies the narrower one.

        The cache actor is reached through an **ask** proxy and its TELL-shaped
        ``request`` is called on that same proxy. That is safe rather than a
        partial adoption of the mechanism: ``request`` adds to a set, spawns a
        worker and tells it the payload — all O(1) on the cache actor's thread, so
        the ask never waits on external work. What the mechanism forbids is a
        blocking ask on a method that performs the slow call.

        Raises:
            ValueError: If the observer exposes no orchestrator.
        """
        super().observer(observer)  # store the observer weakly via the base setter
        actor_observer = self._actor_observer()
        if actor_observer.orchestrator is None:
            raise ValueError("TeamActivityTool requires access to the orchestrator.")

        orchestrator_proxy = actor_observer.proxy_ask(actor_observer.orchestrator, Orchestrator)
        activity_addr = orchestrator_proxy.getChildrenOrCreate(
            TeamActivityActor,
            config=BaseConfig(
                name=TEAM_ACTIVITY_ACTOR_NAME,
                role=TEAM_ACTIVITY_ACTOR_ROLE,
            ),
        )
        self._orchestrator_proxy = orchestrator_proxy
        self._activity_proxy = actor_observer.proxy_ask(activity_addr, TeamActivityActor)
        return self

    def _actor_observer(self) -> ActorToolObserver:
        """Live observer typed as the actor protocol. Raises once the agent stops."""
        return cast(ActorToolObserver, self._observer)

    def _actor_observer_or_none(self) -> ActorToolObserver | None:
        """Live observer typed as the actor protocol; ``None`` once the agent stops."""
        return cast(ActorToolObserver | None, self._observer_or_none())

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return ``who_is_working`` when it is exposed on the tool-call channel."""
        gta = _resolve(self.get_team_activity, GetTeamActivity)
        if gta and TOOL_CALL in gta.expose:
            return [self._who_is_working_factory(gta)]
        return []

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return ``who_is_working`` when it is exposed on the command channel."""
        gta = _resolve(self.get_team_activity, GetTeamActivity)
        if gta and COMMAND in gta.expose:
            return {GetTeamActivity: self._who_is_working_factory(gta)}
        return {}

    def _who_is_working_factory(self, params: GetTeamActivity) -> Callable[..., Any]:
        """Build the ``who_is_working`` callable, reading configuration once."""
        orchestrator_proxy, activity_proxy = self._require_proxies()
        observer_or_none = self._actor_observer_or_none  # bound method -> weak edge to agent
        stale_after_seconds = self.stale_after_seconds
        budget = _SummaryBudget(
            model=self.summarizer_model,
            max_chars=self.summary_max_chars,
            poll_attempts=self.poll_attempts,
            poll_delay_seconds=self.poll_delay_seconds,
        )

        def who_is_working(summarize_over: int | None = None) -> TeamActivityReport:
            """Report which teammates are currently handling a message, and on what.

            Derived from team telemetry, so it costs no LLM call by default: task
            text longer than the report budget is simply truncated.

            Args:
                summarize_over: Omit (the default) for zero-cost truncation. Pass a
                    character count to summarize only tasks longer than it; the
                    summary is cached per task, so asking again is free. A summary
                    that has not arrived in time comes back truncated and is counted
                    in ``pending_summaries``.

            Returns:
                A ``TeamActivityReport``. Members that are idle, are you, are tool
                actors, or are the user proxy never appear.
            """
            observer = observer_or_none()
            if observer is None:
                raise RetriableError("Team is shutting down; cannot report team activity.")

            now = datetime.now(UTC)
            groups = _collect_open_groups(
                orchestrator_proxy, observer.myAddress.agent_id, now, stale_after_seconds
            )
            rows = _build_rows(groups, now)
            texts = _resolve_task_texts(orchestrator_proxy, rows)
            members, pending_summaries = _apply_summaries(
                rows, texts, activity_proxy, budget, summarize_over
            )
            return TeamActivityReport(
                generated_at=now, members=members, pending_summaries=pending_summaries
            )

        who_is_working.__doc__ = params.format_docstring(who_is_working.__doc__)
        return who_is_working

    def _require_proxies(self) -> tuple[Orchestrator, TeamActivityActor]:
        """Return the bound proxies, or fail loudly if :meth:`observer` never ran."""
        if self._orchestrator_proxy is None or self._activity_proxy is None:
            raise ValueError("TeamActivityTool.observer() must run before its callables are built.")
        return self._orchestrator_proxy, self._activity_proxy
