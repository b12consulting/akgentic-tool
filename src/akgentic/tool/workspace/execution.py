"""``workspace_exec``: the one writer whose write set cannot be declared.

Every other mutation in this package says what it is about to do, so the gate can
check a precondition against the file it names. A shell command cannot: its write
set is unknowable before it runs and only partly guessable after. So exec is
**fenced** rather than gated — an exclusive lease over the tree for the duration
of the run — and git is what tells us afterwards what it did (ADR-036 §5).

This module holds everything exec needs that is not the gate itself: the models
crossing the actor boundary, the budgets, and the one formatter ``workspace_exec``
renders through.

**The blocking call happens on ``#Workspace``'s own single worker thread**, and
:class:`ExecRunner` is what owns it. ``#Workspace`` submits one command at a time
and goes back to draining its mailbox; the worker tells the answer back to
``#Workspace`` itself. There is no second actor and no ``#defer-`` worker in
between — the worker only ever sat in an ``ask`` waiting for that same
subprocess, and the second actor cost a lifetime, a registry entry and a liveness
check to keep in agreement with this one.

**The module is named ``execution``, not ``exec``.** ``exec`` is a builtin, and a
module of that name shadows it at every import site in the package.

**This is where ``workspace/`` starts importing ``sandbox/``.** The two were
independent until the card surfaces merged, and the edge is now structural:
``#Workspace`` builds its backend here and reports through an ``ExecReport``
defined there. It is one-directional — ``workspace`` → ``sandbox``, never back —
and inside one package. Keep it that way: an import in the other direction makes
the pair a cycle, which is also why the report model lives on the sandbox side.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from enum import StrEnum
from uuid import uuid4

from pydantic import model_validator

from akgentic.core.actor_address import ActorAddress
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.sandbox.actor import (
    SANDBOX_ACTOR_ROLE,
    CardMode,
    SandboxConfig,
    SandboxMode,
    sandbox_actor_name,
)
from akgentic.tool.sandbox.backend import ExecReport, ExecResult, SandboxBackend

logger = logging.getLogger(__name__)

##
## ``SandboxMode`` and ``CardMode`` are defined in ``sandbox.actor`` and used
## here as they are: a resolved backend and a card's request are the same two
## vocabularies on both sides of the merge, and a second definition would be a
## second place to register a backend in.
##

DEFAULT_EXEC_TIMEOUT_S = 15.0
"""Wall-clock budget for the sandboxed command itself.

Below :data:`MAX_EXEC_BUDGET_S` (20 s), which is below the orchestrator's 30 s
stop backstop. The exec card's old default of 30 s sat *at* the backstop and
docker's sat above it, which is why exec could not simply keep it: a Python
thread cannot be cancelled, so a command still running past its budget holds its
parent's teardown open for the difference.
"""

MAX_EXEC_BUDGET_S = 20.0
"""The ceiling every run budget is clamped to, whatever a card asks for.

**What it bounds is a teardown, not a command.** The subprocess runs on
``#SandboxActor``'s own thread, and a Python thread cannot be cancelled — so the
orchestrator's ``stop_children(blocking=True)`` is held open for as long as the
command runs. The backstop on that stop is 30 s, so a run allowed past 20 s is a
team that cannot be shut down inside its own backstop.

The number is the one the retired ``#defer-`` worker used, and it is unchanged on
purpose: the constraint never belonged to the worker. A thread that cannot be
cancelled holds the blocking stop open whether it is a worker's thread or the
sandbox's, so the ceiling below the backstop is still owed and only its owner has
changed. Every budget arithmetic, README figure and existing spec therefore reads
the same.
"""


def effective_budget(timeout_s: float) -> float:
    """Return the budget a run will actually get — the card's ask, capped.

    The actor building the request and the capability resolving its poll both
    need this number, and each used to compute the ``min`` by hand. Two copies
    of a clamp is two places for a ceiling change to be applied once and missed
    once.

    Args:
        timeout_s: What the card asked for.

    Returns:
        ``min(timeout_s, MAX_EXEC_BUDGET_S)``.
    """
    return min(timeout_s, MAX_EXEC_BUDGET_S)


DEFAULT_EXEC_POLL_ATTEMPTS = -1
"""Sentinel: wait out the whole run rather than hand back a run id part-way.

Resolved once, at wiring time, by :func:`poll_attempts_within` into the attempt
count whose *wait* is the longest that still fits the run's effective budget plus
:data:`EXEC_REPORT_MARGIN_S`. Nothing downstream ever sees the ``-1``.

It is the **default**, and the instinct that a long synchronous tool call is
expensive is wrong here. An agent inside a tool call cannot do anything else; the
call is synchronous from the model's point of view, and it cannot yield and be
resumed. A short poll therefore does not save that latency — it converts it into
LLM round-trips against an answer that cannot change, and ends with the model
holding a run id and no way to be woken. Waiting costs one thread doing nothing;
not waiting cost four inferences in six seconds and a promise to a human that
nothing would ever deliver.

The tree is leased for the run's duration either way, so the *team* waits
identically in both settings. Only the requesting agent's turn count differs.

The other two settings still mean what they meant: a positive count is a bounded
poll clamped to the run budget, and ``0`` opts out of polling entirely and takes
the run id immediately.
"""

DEFAULT_EXEC_POLL_DELAY_S = 0.5
"""Seconds between two looks for a result, on the agent's own thread.

Only ever wall clock the agent was going to spend anyway (see above). It is the
granularity of the wait, not its length: the length comes from
:data:`DEFAULT_EXEC_POLL_ATTEMPTS` resolved against the run's budget.
"""

EXEC_REPORT_MARGIN_S = 1.0
"""How far past the run's budget the sentinel's poll keeps looking.

The run's budget bounds the **command**; the poll has to outlast it by the time
the sandbox takes to report, or the most ordinary slow case breaks. A command
killed at its 15 s budget produces a perfectly good ``exit_code: 124`` outcome a
moment later — an answer the agent can read — and a poll bounded at exactly 15 s
misses it by that moment and reports a timeout instead, turning a clean answer
into a confusing one.

**Only the sentinel gets it.** An explicit positive ``poll_attempts`` asked for a
bounded look, not for the whole run, and is clamped to the run budget alone.

A module constant rather than a card field: it describes how fast the sandbox
reports, which is a property of this runtime, not of what a user wants. Promote
it only if a deployment is ever observed where a report reliably takes longer.
"""

MAX_TRACKED_RUNS = 32
"""How many recent run ids are remembered per agent.

Bounded for the same reason every other map on ``#Workspace`` is: an uncapped map
on a team singleton leaks for the life of the team. It is what makes an unknown
run id *helpful* — a model that mistyped one reads the right one back — so the
cap only has to cover a conversation's worth of runs, not a team's.
"""

MAX_QUEUED_RUNS = 16
"""How many runs may wait for the tree behind the one holding it.

Bounded for the reason every other collection on ``#Workspace`` is: a team
singleton's deque with no ceiling grows with whatever a model emits, and a model
that has decided to emit commands in a loop will fill it faster than the head
drains it. The refusal over the cap is the only exec refusal left, and it names
nobody — which is what keeps it from reproducing the defect the queue removes.

Generous rather than tight: the workload that produced the queue is a parallel
batch of two or three probes, and a cap that a healthy batch could reach would
turn an ordinary response into a retry.
"""

LEASE_GRACE_S = 5.0
"""How long past its budget a run keeps the mutation gate with nothing reported.

One use, and it covers one case: a child that ignores the kill. Every ordinary
exit reports — a command that ran, a command the budget killed, a backend that
raised, an allowlist refusal — because the sandbox's handler reports in a
``finally``. What no report can cover is a subprocess still alive after its
budget, since the thread waiting on it is not free to say so.

So the gate is released **without** a report once the run is this far past its
budget, checked lazily by the next mutation: no timer, no extra thread. By then
the backend has killed the child, so the release is not a race against a live
writer — and the late report that may still arrive commits nothing and clears
nothing, because by then the tree may hold somebody else's work.

Measured from the moment the run actually started, which is the moment the work
was submitted: a submit onto an idle single-worker executor is O(1), so nothing
slow sits between admission and the command. A cold container backend spends its
provisioning inside the worker's own lazy ``start()``, which is inside the run
this clock is measuring — deliberately, because that provisioning is time the
command really does take.
"""

EXEC_SHUTDOWN_GRACE_S = 3.0
"""How long teardown waits for the killed run's worker to return.

**A bound is owed because ``ThreadPoolExecutor.shutdown`` takes no timeout**, so
``shutdown(wait=True)`` can wait for ever and the case is reachable:
``ProcessBackend._run``'s timeout path drains with a second ``communicate()``,
which reads both pipes to EOF, and a grandchild that inherited them and outlived
the killed child holds that call open. ``kill()`` signals the direct child only,
so it is no escape. The wait is therefore on the submitted ``Future``, and the
shutdown that follows it does not wait at all.

**Three seconds, and the arithmetic that picks it.** Teardown kills first, and a
healthy child dies in milliseconds, so this is generous for the case that is not
wedged. The worst case for the whole of exec teardown is this plus the backend's
own ``stop()``, and ``docker stop`` defaults to a 10 s SIGTERM grace before it
SIGKILLs — so ~13 s, comfortably under the orchestrator's 30 s stop backstop. If
this number changes, state that arithmetic again.

What it converts is the failure mode, not the hazard: an unbounded teardown hang
becomes three seconds of teardown latency. Signalling the process group would
close the root cause, but it would also change what a **timeout** does to a
subtree on every ordinary run, which is a decision that belongs to the ADR rather
than to a wiring story.
"""

SANDBOX_RESOLVE_TIMEOUT_S = 5
"""Seconds ``#Workspace`` will wait for the orchestrator to hand back the sandbox.

**No production caller is left.** ``#Workspace`` owns its backend directly and
resolves no second actor, so nothing reads this any more. It is a re-exported
public name and is kept until the sweep that retires the sandbox actor removes
the whole surface at once — an API removal is not something to fold into a
wiring change.

An ask made **on the team singleton's own thread**, which is the shape that must
never be untimed: everything else queued behind it — every read, every mutation,
every other agent's poll — waits for it. Five is generous rather than tight, since
what is being waited on is one O(1) orchestrator turn plus a thread start;
``getChildrenOrCreate`` returns before a cold backend's ``on_start`` has finished,
so a slow provision is not what this bounds.

On expiry the run **fails with a reason** rather than parking the singleton, which
is what keeps a wedged orchestrator from taking the workspace with it.

Whole seconds because ``proxy_ask`` takes an ``int``.
"""

RUN_ID_CHARS = 8
"""Length of a run id, in hex characters.

The id is a token an LLM has to copy back on a later turn, which is the same
hazard the design refuses to accept for a content digest — admitted here only
because the outcome has to be addressable at all. Short is the first of the three
mitigations; the other two are echoing it in the handoff message and making an
unknown id list the agent's recent ones instead of raising.
"""

TIMED_OUT_EXIT_CODE = 124
"""Exit code reported for a command its budget killed, following ``timeout(1)``."""

_UNCONFIGURED_MSG = (
    "This workspace has no execution backend configured — workspace_exec is not available here."
)


class ExecOutcome(SerializableBaseModel):
    """What a finished run produced.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Process exit code, or :data:`TIMED_OUT_EXIT_CODE`.
        timed_out: True when the run's budget killed it. "Too slow" is the
            ordinary case for a shell, so it arrives as an outcome and is
            collectible — never as a failure the agent cannot read.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class RunningExec(SerializableBaseModel):
    """The run that is holding the tree right now — the one mutation gate.

    ``#Workspace`` holds at most one of these at a time, in ``self._running``, and
    its presence *is* the answer to "is the tree taken". There is no second flag
    and no separate deadline: a report clears it, and a run wedged past
    ``budget + LEASE_GRACE_S`` is released by the shared predicate that both the
    mutation path and the admission path decide over.

    Attributes:
        run_id: The run holding it. A report is only allowed to clear the gate
            when its run id matches — a late report from a released run must not
            close out a newer one.
        agent_id: Who requested the run. Named in every refusal it causes, and
            who the discovered commit is attributed to.
        cmd: The command, kept for the discovered commit's body.
        started_at: Monotonic clock at the moment the work was submitted. The
            budget and the grace are both measured from here, and nothing
            re-bases it: a submit onto an idle single-worker executor is O(1), so
            there is nothing slow left between admission and the command for a
            fixed clock to mis-measure.
    """

    run_id: str
    agent_id: str
    cmd: str
    started_at: float


class ExecStart(SerializableBaseModel):
    """The answer to "may I run this, and under what id".

    Exactly one of the two fields is non-empty, and that is **enforced** rather
    than merely documented: every caller branches on ``if not start.run_id``, so
    an instance carrying neither would be reported to the agent as an empty
    refusal, and one carrying both would silently run a command the actor had
    already decided to refuse. A model rather than ``str | None`` because it
    crosses the actor boundary, and because a bare string would leave the caller
    guessing which of the two it holds.

    Attributes:
        run_id: The issued id, when the run was accepted.
        refusal: Why not, when it was not — the busy message, naming the holder.
    """

    run_id: str = ""
    refusal: str = ""

    @model_validator(mode="after")
    def _exactly_one(self) -> ExecStart:
        """Reject an answer that is both, or neither."""
        if bool(self.run_id) == bool(self.refusal):
            raise ValueError(
                "ExecStart carries exactly one of run_id or refusal — never both, never neither."
            )
        return self


class ExecState(StrEnum):
    """Where a run is, from the point of view of an agent asking about it.

    ``DONE`` and ``FAILED`` are both *settled*: a caller polling for a result
    stops on either. ``QUEUED``, ``RUNNING`` and ``UNKNOWN`` are not, and they
    are deliberately distinct — the cache's ``get`` returns ``None`` for an
    unknown key, an in-flight one and a negatively-cached one alike, so telling a
    model "still running" about an id it invented would be a lie it cannot
    recover from.

    ``QUEUED`` is admission's third answer: the tree is held by somebody else, so
    the command has not started and will. It must **not** be settled — a poller
    that stopped on it would hand back a run id the caller never needed, on the
    ordinary path where the head finishes in milliseconds.
    """

    DONE = "done"
    FAILED = "failed"
    QUEUED = "queued"
    RUNNING = "running"
    UNKNOWN = "unknown"


class QueuedExec(SerializableBaseModel):
    """One run waiting for the tree — inert bookkeeping, and nothing else.

    **Nothing is sent and no clock is started for a queued entry**, and the
    absence is the design rather than an omission: a request handed to the
    sandbox at enqueue would run out of order, and a clock started at enqueue
    would measure the wait instead of the run. Both happen at the dequeue, in
    ``_start_run``, which is why this model carries neither.

    Attributes:
        run_id: The id issued to the requester at enqueue time. Its owner holds a
            handle to its **own** work from the moment it asks, which is what
            stops any message naming somebody else's run.
        agent_id: Who asked, as a string. What the discovered commit is
            attributed to once the entry runs.
        cmd: The command string, exactly as the agent gave it.
        cwd: Working directory below the workspace root.
    """

    run_id: str
    agent_id: str
    cmd: str
    cwd: str = ""


class ExecStatus(SerializableBaseModel):
    """Where one run is, and whatever it has produced.

    Attributes:
        state: See :class:`ExecState`.
        run_id: The id that was asked about.
        outcome: The result, on :attr:`ExecState.DONE` only.
        reason: Why the run failed, on :attr:`ExecState.FAILED` only.
        command: The command this run was started with, on
            :attr:`ExecState.DONE` only — read from the asking agent's own
            tracking map, which is why it can only be filled for a run the asker
            owns. It is what makes a collected outcome nameable: without it the
            answer is byte-indistinguishable from any other run's.
        queue_position: The run's 1-based place in the FIFO, on
            :attr:`ExecState.QUEUED` only. ``1`` means "next to run when the head
            finishes"; the running head is not in the queue and is never
            position 0.
        recent_run_ids: This agent's recent runs, on :attr:`ExecState.UNKNOWN`
            only — what turns a mistyped id into a correctable one.
    """

    state: ExecState
    run_id: str
    outcome: ExecOutcome | None = None
    reason: str = ""
    command: str = ""
    queue_position: int = 0
    recent_run_ids: list[str] = []

    @property
    def settled(self) -> bool:
        """Whether there is nothing further to wait for."""
        return self.state in (ExecState.DONE, ExecState.FAILED)


class ExecConfig(SerializableBaseModel):
    """What an exec-capable card tells the actor once, at bind time.

    Deliberately not part of :class:`~akgentic.tool.workspace.models.WorkspaceConfig`.
    ``getChildrenOrCreate`` fixes that config at creation, and the card that
    creates the actor for a workspace is routinely a ``WorkspaceTool`` with no
    exec capability at all — the actor would then be permanently unable to run
    anything for the card that *does* have one.

    Attributes:
        mode: The resolved backend.
        team_id: The team, which names the container. Kept because containers
            are per-team execution resources, never because a directory is
            derived from it.
        workspace_path: The card's **already-resolved** two-segment path, the
            only thing a backend needs to open the right tree. It replaces the
            raw ``workspace_id`` this model used to forward: a backend that
            cannot re-derive the path cannot derive a different one, which is
            what removes the failure mode rather than making it less likely.
        timeout_s: The run budget this card asks for, before clamping.
    """

    mode: SandboxMode
    team_id: str
    workspace_path: str
    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S


def new_run_id() -> str:
    """Return a fresh run id — short, and never reused."""
    return uuid4().hex[:RUN_ID_CHARS]


def poll_attempts_within(attempts: int, delay: float, run_budget: float) -> int:
    """Turn a requested attempt count into a real one, bounded by the run itself.

    The one place a card's ``poll_attempts`` becomes the number
    ``poll_deferred`` is handed, which is what lets the sentinel be resolved
    exactly once, at wiring, with nothing downstream knowing it existed. Three
    requests, three meanings:

    - **negative** — :data:`DEFAULT_EXEC_POLL_ATTEMPTS`, "wait out the run":
      resolved so that the **last look** falls as late as it can inside
      ``run_budget`` **plus** :data:`EXEC_REPORT_MARGIN_S`, so the wait covers
      the sandbox's report and not merely the command (see that constant). The
      count is one higher than the wait divided by the delay, because
      ``poll_deferred`` sleeps between looks and not after the last.
    - **zero** — returned unchanged. A caller that polls zero times takes the run
      id immediately, and that opt-out is load-bearing: it is the only way to get
      a run id without waiting at all.
    - **positive** — a bounded look, clamped to ``run_budget`` and **without** the
      margin. A poll longer than the run parks the agent's thread past the point
      where there is anything left to wait for: by then the run has reported or
      its own budget has killed it, so every further attempt is a sleep with no
      possible answer.

    The bound is the **effective** run budget — the card's ``timeout_s`` after
    :data:`MAX_EXEC_BUDGET_S` — because that is what actually stops the run.
    Clamping against the requested value would leave a card asking for 999 s
    polling long past the 20 s the ceiling allows it.

    Args:
        attempts: What the card asked for; negative means the sentinel.
        delay: Seconds between attempts.
        run_budget: The effective wall-clock budget of the run being waited on.

    Returns:
        The count to poll with — never below 1 except for the zero opt-out,
        because a caller that looks zero times gets an exhaustion message without
        ever having looked, which is worse than looking once. A non-positive
        *delay* leaves the sentinel with no wall clock to divide, so it too
        resolves to a single look.
    """
    if attempts < 0:
        if delay <= 0:
            return 1
        # ``poll_deferred`` sleeps *between* looks and never after the last, so
        # N looks spend (N-1) delays of wall clock, not N. Resolving to the
        # count whose *product* fits would therefore stop looking one whole
        # delay early — at the shipped defaults that is 15.5 s against a 16 s
        # target, and at any delay above the margin the margin is gone entirely
        # and the poll ends at or before the run's own budget, which is the case
        # this margin exists to cover. The +1 puts the *last look* at
        # ``floor((run_budget + margin) / delay) * delay``: the largest multiple
        # of the delay that still fits, and never past it.
        return max(1, int((run_budget + EXEC_REPORT_MARGIN_S) // delay) + 1)
    if attempts == 0 or delay <= 0:
        return attempts
    if attempts * delay <= run_budget:
        return attempts
    return max(1, int(run_budget // delay))


def wait_out_the_turn(
    fetch: Callable[[], ExecStatus],
    run_id: str,
    run_budget: float,
    delay: float,
) -> str:
    """Poll until the run settles, waiting out the **queue in front of it** as well.

    The wait-out sentinel's loop, and the reason a batch of commands behaves as
    if it had been issued one at a time: each call returns *its own* output.
    Handing a queued caller a run id instead would be barely better than the
    refusal the queue replaced — the model would still be left managing results
    it never asked to manage.

    **Deadline-driven, not attempt-driven**, because the wait is no longer a
    property of the card alone: it depends on how many runs are ahead of this
    one, which is only knowable per look. A run at queue position ``p`` has at
    most ``p + 1`` run budgets left to wait — the ``p`` ahead of it, then its
    own — so that is what the deadline is re-armed to on every queued look.

    **The deadline re-arms on every queued look, and a separate ceiling fixed on
    entry is what bounds it.** The two do different jobs and neither is
    sufficient alone:

    - **Re-arming keeps a caller that is still advancing alive.** A run ahead
      costs its budget *plus* :data:`LEASE_GRACE_S` and the sandbox resolve, so
      a deadline
      fixed at the first look gives a caller at position 5 six budgets for work
      that honestly takes more — and abandons it **while it is still moving up
      the queue**, degrading to handoffs under exactly the load the queue exists
      for. Re-arming from the current position cannot do that: each look buys
      the time that position actually warrants.
    - **The ceiling bounds the case re-arming cannot.** A position that never
      decreases would otherwise re-start the clock for ever — reachable, because
      a head whose child ignores the kill keeps the gate until something arrives
      to release it. So ``(MAX_QUEUED_RUNS + 1) * run_budget +
      EXEC_REPORT_MARGIN_S`` is computed **once, on entry, and never re-armed**,
      and every deadline is clamped to it. That is what makes the worst case a
      property of the code rather than of the happy path.

    The ``+ 1`` is the caller's own run, and it is the same arithmetic as the
    1-based ``queue_position``: at the deepest legal position the caller waits
    for the :data:`MAX_QUEUED_RUNS` ahead of it **and then for itself**. A
    ceiling of exactly ``MAX_QUEUED_RUNS * run_budget`` would abandon a
    max-depth caller one budget before its own command finished.

    **The thread parked here is the caller's own tool thread, never the actor's.**
    ``exec_status`` is O(1) and the mailbox goes on draining, so nothing here
    blocks reads, mutations, or another agent's poll. It does **not** bound
    teardown: "only the head is on the sandbox" is a property of the actor side
    and says nothing about a caller parked in this loop.

    **Only the negative sentinel comes here.** A positive ``poll_attempts`` asked
    for an explicitly bounded look and keeps ``poll_deferred``; ``0`` opts out of
    polling entirely. ``poll_deferred`` is attempt-count-driven by construction
    and shared with other cards, so it is left alone rather than widened for this
    one caller.

    Args:
        fetch: Asks the actor where this run stands. Called once per look.
        run_id: The caller's own run — never another agent's.
        run_budget: The effective wall-clock budget of one run.
        delay: Seconds between looks. Non-positive means a single look.

    Returns:
        The rendered outcome once the run settles, or — when the deadline goes
        without one — the degraded handoff: :func:`queued` if it never started,
        :func:`timed_out` if it did and overran.
    """
    start = time.monotonic()
    ceiling = start + (MAX_QUEUED_RUNS + 1) * run_budget + EXEC_REPORT_MARGIN_S
    deadline = start + run_budget + EXEC_REPORT_MARGIN_S
    while True:
        status = fetch()
        if status.settled:
            return format_status(status)
        if status.state is ExecState.QUEUED:
            turn = time.monotonic() + (status.queue_position + 1) * run_budget
            deadline = min(turn + EXEC_REPORT_MARGIN_S, ceiling)
        if delay <= 0 or time.monotonic() > deadline:
            # A non-positive delay leaves no wall clock to divide, so it is one
            # look and out — the reading ``poll_attempts_within`` already gives it.
            if status.state is ExecState.QUEUED:
                return queued(run_id, status.queue_position)
            return timed_out(run_id, run_budget)
        time.sleep(delay)


def resolve_mode(mode: CardMode, *, team_id: str = "") -> tuple[SandboxMode, SandboxBackend]:
    """Turn a card's requested mode into a backend, warning where the host has none.

    Every wiring goes through this rather than probing for itself — a second
    copy of the probe is a second place for the warning to stop firing.

    **Both halves of the answer have a consumer, and they are different callers.**

    - ``_bind_sandbox`` calls it with the card's mode and uses only the
      **resolved mode**. It runs no commands, so it drops the instance. What it
      needs from here is the ``"auto"`` probe and its ``DeprecationWarning``,
      which must fire at wiring time, in front of the admin who configured the
      card.
    - ``#Workspace.configure_exec`` calls it with the already-resolved
      ``ExecConfig.mode`` and uses the **instance**, which becomes the backend
      its worker thread runs on. Because that mode is concrete, the probe
      short-circuits and no second warning fires for one card.

    Constructing a backend is inert — nothing is probed, created or started until
    ``start()`` — so building one at wiring time and discarding it costs nothing.

    Args:
        mode: What the card asked for, possibly ``"auto"``.
        team_id: The team the backend will run for. Only ``DockerBackend`` reads
            it, to name its container; the other three accept and ignore it, so
            that this function has one uniform constructor to call and never a
            type switch on the mode it has just resolved.

    Returns:
        The resolved mode and a fresh, unstarted backend for it.

    Raises:
        KeyError: If *mode* names no registered backend. Deliberately at wiring
            time rather than at the first command: a typo in a card is a
            configuration error, and configuration errors belong at start-up.
    """
    import warnings  # noqa: PLC0415 — only on the wiring path

    # Resolved at call time through the package, so a backend registered — or a
    # probe replaced — after this module was imported is what gets consulted.
    from akgentic.tool.sandbox import (  # noqa: PLC0415
        SANDBOX_BACKEND_CLASSES,
        _resolve_auto_mode,
    )

    resolved: SandboxMode = _resolve_auto_mode() if mode == "auto" else mode
    if mode == "auto" and resolved == "local":
        warnings.warn(
            "sandbox mode='auto': no isolation backend found (bwrap, sandbox-exec, "
            "docker). Falling back to LocalSandboxActor — no filesystem isolation.",
            DeprecationWarning,
            stacklevel=3,
        )
    return resolved, SANDBOX_BACKEND_CLASSES[resolved](team_id=team_id)


class ExecRunner:
    """One backend, one tree, and the body the worker thread runs.

    **Thread ownership is the whole reason this class exists**, and it is stated
    rather than enforced because there is nothing to enforce it with:

    - :meth:`perform` runs on ``#Workspace``'s single executor worker and touches
      nothing but this object. It is a plain method on a plain object rather than
      a closure over the actor, so the worker has no path to actor state at all —
      which is the exact race the mailbox exists to prevent, and one no test would
      catch until it had corrupted a run.
    - :meth:`kill` and :meth:`stop` are called from the **actor's** thread during
      teardown, and are the only two methods that may be.

    **No lock guards the backend, and none is owed.** Only the single worker ever
    calls ``start()`` or ``exec()``, so the one-worker executor is the lock, one
    level below where the mailbox is. ``kill()`` and ``stop()`` do cross threads,
    and :class:`~akgentic.tool.sandbox.backend.ProcessBackend` guards its own
    handle for exactly that reason.

    Attributes:
        backend: The strategy commands run on. Read by teardown and by specs;
            never swapped after construction.
        workspace_path: The already-resolved two-segment tree this runner's
            backend is started on. One runner is anchored to one tree for its
            whole life — a runner whose tree could change is a backend that
            could open a directory other than the one its ``#Workspace`` gates.
    """

    def __init__(self, backend: SandboxBackend, workspace_path: str) -> None:
        self.backend = backend
        self.workspace_path = workspace_path
        self._started = False

    def perform(
        self,
        *,
        run_id: str,
        cmd: str,
        cwd: str,
        timeout_s: float,
        reply_to: ActorAddress,
    ) -> None:
        """WORKER THREAD ONLY. Run one command and **always** report it.

        Every exit builds a report, because a run whose report is dropped holds
        the tree until the gate's grace releases it — and because the ``Future``
        this returns onto is never read, so an exception escaping here would
        surface nowhere at all.

        The three outcomes are three different things to the agent waiting: a
        command that ran is a result whatever it exited with; a command the
        budget killed is an answer that says so; anything else — the backend
        raised, the allowlist refused the binary, the quotes would not balance —
        is a failure with the reason in it.

        Args:
            run_id: The run being performed, echoed into the report.
            cmd: The command string, exactly as the agent gave it.
            cwd: Working directory below the workspace root.
            timeout_s: Wall-clock budget, already clamped by the caller.
            reply_to: ``#Workspace``'s own address, captured on the actor's
                thread at submit time. Never read from the actor here.
        """
        report: ExecReport | None = None
        try:
            result = self._exec(cmd, cwd, timeout_s)
            report = ExecReport(run_id=run_id, result=result)
        except subprocess.TimeoutExpired:
            report = ExecReport(run_id=run_id, timed_out=True)
        except Exception as exc:  # noqa: BLE001 — every failure is an answer, never a crash
            # ``or repr(exc)`` is load-bearing, not defensive. ``str(exc)`` is
            # the empty string for any exception raised with no message —
            # ``raise RuntimeError()`` — and an empty ``error`` fails
            # ``ExecReport``'s exactly-one validator, so the report that was
            # meant to carry the failure raises *inside this except clause* and
            # is lost onto a future nobody reads. ``repr`` always names the type.
            report = ExecReport(run_id=run_id, error=str(exc) or repr(exc))
        finally:
            if report is None:
                # Reachable only if building one of the reports above raises,
                # which the ``or repr(exc)`` overhead is there to stop — so this
                # is the branch for the exit nothing thought of. Kept rather
                # than argued: a run whose report is dropped holds the tree
                # until the gate's grace releases it.
                report = ExecReport(
                    run_id=run_id, error="The sandbox produced no report for this run."
                )
            try:
                reply_to.tell(report)
            except Exception:
                # The address this reports to is ``#Workspace`` itself, which may
                # already be part-way through its own ``on_stop``.
                # ``ActorAddress.tell`` raises synchronously on a dead address, and
                # a stopping workspace is nobody to report to — not a reason to
                # lose the report with no log line.
                logger.warning(
                    "Workspace %s could not report run %s to %s — swallowing",
                    self.workspace_path,
                    run_id,
                    reply_to.name,
                    exc_info=True,
                )

    def _exec(self, cmd: str, cwd: str, timeout_s: float) -> ExecResult:
        """WORKER THREAD ONLY. Start the backend if it is cold, then run *cmd*.

        **``start()`` is lazy, and that is a decision rather than an
        optimisation.** ``DockerBackend.start`` runs ``docker build``, which can
        take minutes; ``configure_exec`` runs on the team singleton's own thread,
        where every read, every mutation and every journal call in the team is
        serialised behind it. The container is created on the first command, not
        at bind time, and a workspace that never runs one provisions nothing.

        **The flag is set only after ``start()`` returns**, so a daemon that was
        down is retried by the next run. That costs a failing probe per run in a
        broken deployment, and is the honest trade against a latched failure
        that would need a restart to clear.
        """
        if not self._started:
            self.backend.start(self.workspace_path)
            self._started = True
        return self.backend.exec(cmd, cwd, timeout_s)

    def kill(self) -> None:
        """ACTOR THREAD ONLY, at teardown step 2. End the run in flight.

        Best-effort and idempotent, because the backend's is: with no run in
        flight there is nothing to signal, and a direct child that has already
        exited is what the caller wanted anyway.
        """
        self.backend.kill()

    def stop(self) -> None:
        """ACTOR THREAD ONLY, at teardown step 4. Release the backend.

        **This may run while the worker is still inside ``exec``**, and at
        teardown that is the right answer rather than a violation of the
        ordering the normal path protects. For docker, ``docker stop`` is the
        only thing that ends the process *inside* the container — ``kill()``
        reaches the local ``docker exec`` client and no further — so a bounded
        drain that gave up must still be followed by this. For the three local
        backends ``_release()`` is a no-op, so the case does not arise.
        """
        self.backend.stop()


def sandbox_config(config: ExecConfig) -> SandboxConfig:
    """Build the sandbox actor's configuration — in one place, for both callers.

    **No production caller is left.** Neither the card nor ``#Workspace`` creates
    a sandbox actor any more. It is a re-exported public name and is kept until
    the sweep that retires the actor removes the whole surface at once.

    ``getChildrenOrCreate`` keys on the actor **name**, so a config that differs
    in name creates a *second* actor per run instead of resolving the existing
    one; a config that differs in ``workspace_path`` would point the reused actor
    at the wrong directory. The card builds one at wiring time and ``#Workspace``
    builds one per run, and the two must be identical — so they are built here
    rather than twice by hand.

    **The name carries the workspace**, exactly as ``#Workspace-<workspace>``
    does. A constant name resolved two exec-capable cards on two workspaces onto
    the first actor, so one agent's commands ran in the other's tree while its
    own ``#Workspace`` gated an untouched one — see :func:`sandbox_actor_name`.

    **Nothing is derived here.** The path arrives already resolved from the card
    that built the ``ExecConfig``, and both the name and the directory are taken
    from that one value, so the two cannot disagree.

    Args:
        config: The card's resolved backend, team and workspace path.

    Returns:
        The configuration for ``#SandboxActor-<workspace>``.
    """
    return SandboxConfig(
        name=sandbox_actor_name(config.workspace_path),
        role=SANDBOX_ACTOR_ROLE,
        team_id=config.team_id,
        workspace_path=config.workspace_path,
        mode=config.mode,
    )


def format_outcome(outcome: ExecOutcome, run_id: str = "") -> str:
    """Render a finished run: one header line, then only the streams that spoke.

    One shape for every caller that renders an outcome, because two formats for
    one thing is how they drift.

    **An empty stream is omitted rather than labelled.** The previous shape
    printed ``stdout:`` and ``stderr:`` unconditionally, so a silent success
    spent four lines saying nothing twice, and a model reading a failure had to
    scan past an empty ``stdout:`` to reach the error. A heading that appears
    only when there is something under it means its presence is information.

    ``run_id`` joins the header rather than taking a line of its own: an answer
    that names no run is what let a collected result be mistaken for a sibling
    call's.

    Args:
        outcome: The finished run.
        run_id: The run this outcome belongs to. Omitted from the header when
            empty, which is the caller saying it has no run to name.

    Returns:
        The rendered answer, with ``stdout`` and ``stderr`` sections present
        only when the corresponding stream is non-empty.
    """
    status = "OK" if outcome.exit_code == 0 else "FAILED"
    head = f"Run {run_id} - " if run_id else ""
    parts = [f"{head}exit_code: {outcome.exit_code} ({status})"]
    if outcome.stdout.strip():
        parts.append(f"**stdout**\n{outcome.stdout}")
    if outcome.stderr.strip():
        parts.append(f"**stderr**\n{outcome.stderr}")
    return "\n\n".join(parts)


def format_status(status: ExecStatus) -> str:
    """Render any run state as the string an agent reads back.

    Every state produces a returned string, including the ones that are not a
    result. An unknown run id in particular does **not** raise: it lists the
    agent's recent runs, so a model that mistyped one reads the right one back.

    **The ``DONE`` branch names the run**, by handing its id to
    :func:`format_outcome` for the header line. An answer that names no run is
    what let a collected result be mistaken for a sibling call's.
    """
    if status.state is ExecState.DONE and status.outcome is not None:
        return format_outcome(status.outcome, run_id=status.run_id)
    if status.state is ExecState.FAILED:
        return f"Run {status.run_id} failed: {status.reason}"
    if status.state is ExecState.QUEUED:
        return queued(status.run_id, status.queue_position)
    if status.state is ExecState.RUNNING:
        return in_progress(status.run_id)
    known = ", ".join(status.recent_run_ids) or "none"
    return (
        f"Unknown run id '{status.run_id}'. Your recent runs: {known}. "
        "Pass one of those to workspace_exec_result."
    )


def in_progress(run_id: str) -> str:
    """The handoff message, with the run id echoed verbatim into the model's context.

    The answer to a card that asked for a **bounded** poll, and to any direct
    ``workspace_exec_result`` on a run still in flight. It is accurate in both:
    a card with a positive ``poll_attempts`` asked for a run id, and a run asked
    about directly has no budget of its own to have exhausted.
    """
    return (
        f"Run {run_id} is still in progress. It holds the workspace until it finishes. "
        f"Call workspace_exec_result('{run_id}') on your next turn to collect the output."
    )


def queued(run_id: str, position: int) -> str:
    """The handoff for a run that has not started, because the tree is held.

    Shaped like :func:`in_progress` deliberately: same id echoed verbatim, same
    pointer at ``workspace_exec_result``, so a model that already knows how to
    act on one acts on this without learning a second protocol. What it adds is
    the one fact the other cannot carry — the work has **not begun**, so nothing
    has been lost and nothing needs re-issuing.

    ``position`` is the **1-based place in the FIFO**: ``1`` means "next to run
    when the head finishes". The running head is not in the queue, so a position
    of ``0`` is never rendered for a queued run.

    Args:
        run_id: The queued run's id — the caller's own, in every branch.
        position: Its 1-based place in the queue.

    Returns:
        What the agent reads back.
    """
    return (
        f"Run {run_id} is queued at position {position}: another command holds the workspace, so "
        f"yours has not started yet. Nothing is lost — call workspace_exec_result('{run_id}') to "
        "collect the output once it runs."
    )


def queue_full() -> str:
    """Refusal for a run the queue has no room for — naming nobody.

    The one exec refusal left, and the only message on this path that hands back
    no run id at all. It names **no run and no agent** on purpose: a refusal that
    quoted the holder's id is exactly the defect the queue exists to remove, and
    a refusal is the one place an agent has no id of its own to be given instead.
    """
    return (
        f"Too many commands are already waiting for this workspace (the queue holds "
        f"{MAX_QUEUED_RUNS}). Nothing was started. Retry once some of them have finished, or "
        "collect the runs you already started first."
    )


def timed_out(run_id: str, budget_s: float) -> str:
    """The answer when a wait-out-the-run poll ran out — the run overran its budget.

    A different message from :func:`in_progress`, and deliberately: the sentinel
    already waited for everything there was to wait for, so a run still going is
    no longer "in progress" in any ordinary sense — it is past the budget that
    was supposed to stop it, and the workspace is still held by it.

    It says so, names the budget and the id, and offers ``workspace_exec_result``
    without instructing the model to call it "on your next turn". That
    instruction is what produced the busy-loop this default exists to remove: an
    agent inside a synchronous tool call has no next turn to wait for, so its
    only available move is to call the tool again immediately.
    """
    return (
        f"Run {run_id} passed its {budget_s:g}s budget without reporting, so there is no output "
        f"yet. It still holds the workspace. Nothing is lost — workspace_exec_result('{run_id}') "
        "collects the output whenever the run does report."
    )


def unconfigured() -> str:
    """Refusal for an exec request against a workspace with no backend bound."""
    return _UNCONFIGURED_MSG
