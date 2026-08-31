"""``workspace_exec``: the one writer whose write set cannot be declared.

Every other mutation in this package says what it is about to do, so the gate can
check a precondition against the file it names. A shell command cannot: its write
set is unknowable before it runs and only partly guessable after. So exec is
**fenced** rather than gated — an exclusive lease over the tree for the duration
of the run — and git is what tells us afterwards what it did (ADR-036 §5).

This module holds everything exec needs that is not the lease itself: the models
crossing the actor boundary, the budgets, the worker that performs the blocking
sandbox call off the actor's thread, and the one formatter both the
``workspace_exec`` capability and the deprecated ``ExecTool`` shim render through.

**The module is named ``execution``, not ``exec``.** ``exec`` is a builtin, and a
module of that name shadows it at every import site in the package.

**This is where ``workspace/`` starts importing ``sandbox/``.** The two have been
independent since the sandbox arrived; merging the card surface necessarily
creates the edge, because the worker needs ``SANDBOX_ACTOR_CLASSES`` and
``SandboxConfig``. It is one-directional — ``workspace`` → ``sandbox``, never
back — and inside one package. Keep it that way: an import in the other
direction makes the pair a cycle.
"""

from __future__ import annotations

import logging
import subprocess
import time
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import Field, PrivateAttr, model_validator

from akgentic.core.actor_address import ActorAddress
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.core.deferred import DeferredPayload, DeferredWorker
from akgentic.tool.sandbox.actor import (
    SANDBOX_ACTOR_ROLE,
    CardMode,
    SandboxActor,
    SandboxConfig,
    SandboxMode,
    sandbox_actor_name,
)

logger = logging.getLogger(__name__)

##
## ``SandboxMode`` and ``CardMode`` are defined in ``sandbox.actor`` and used
## here as they are: a resolved backend and a card's request are the same two
## vocabularies on both sides of the merge, and a second definition would be a
## second place to register a backend in.
##

DEFAULT_EXEC_TIMEOUT_S = 15.0
"""Wall-clock budget for the sandboxed command itself.

Below ``DEFAULT_WORKER_TIMEOUT_S`` (20 s), which is below the orchestrator's 30 s
stop backstop. The old ``ExecTool`` default of 30 s sat *at* the backstop and
docker's sat above it, which is why exec could not simply keep it: a worker
cannot cancel a thread, so a command still running past its worker's budget holds
its parent's teardown open for the difference.
"""

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
the worker takes to report, or the most ordinary slow case breaks. A command
killed at its 15 s budget produces a perfectly good ``exit_code: 124`` outcome a
moment later — an answer the agent can read — and a poll bounded at exactly 15 s
misses it by that moment and reports a timeout instead, turning a clean answer
into a confusing one.

**Only the sentinel gets it.** An explicit positive ``poll_attempts`` asked for a
bounded look, not for the whole run, and is clamped to the run budget alone.

A module constant rather than a card field: it describes how fast a worker
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

LEASE_GRACE_S = 5.0
"""How far past its budget a run's lease survives before a mutation reclaims it.

The lease is released on every exit a worker can report — success, failure,
timeout, even a spawn that never happened. The one gap is a worker killed during
teardown, which reports nothing, and without a reclaim that wedges every mutation
in the team for the rest of its life. A deadline checked lazily by the next
mutation closes it with no timer and no extra thread.

**The deadline is re-based when the run actually starts**, and the reclaim is
gated on the worker being gone. Both matter for the same reason: a deadline that
starts at *request* time is spent by anything slow ahead of the command — a
container image being built, above all — so without either guard a live run is
declared dead, the tree is handed to somebody else, and the run's report then
finds no lease of its own and drops its write set. What remains is the one case
the deadline was introduced for: a worker killed during teardown, which reports
nothing and is not alive to be waited on.
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

_NOT_READY_MSG = (
    "The execution backend was not ready within {seconds}s, so the command never ran and "
    "nothing was written. It is most likely still starting up — the first run on a container "
    "backend builds its image — so retry in a moment; a run that keeps failing this way means "
    "the backend cannot start on this host."
)


def ask_seconds(remaining: float) -> int:
    """Whole seconds an ask inside a worker may block for, never past its deadline.

    ``Akgent.proxy_ask`` takes an integer timeout, so a float budget has to land
    on one. It is **truncated**, not rounded: the deadline is the whole point of
    the budget, so an ask that outlived it would defeat the thing it is being
    given. A remainder below a second becomes ``0`` — which fails the ask
    immediately, and is the honest answer, since a backend does not start in the
    time left.

    Args:
        remaining: Seconds left of the worker's budget, possibly negative.

    Returns:
        The timeout to hand ``proxy_ask``, never below zero.
    """
    return max(0, int(remaining))


def backend_not_ready(seconds: int) -> str:
    """Why a run never started, and what the agent should do about it."""
    return _NOT_READY_MSG.format(seconds=seconds)


EXEC_REPLY_GRACE_S = 2
"""How much longer the ask waiting on a command lives than the command's budget.

A command killed by its budget is an **answer** — the backend raises, the worker
turns it into ``ExecOutcome(timed_out=True)``, and the agent reads what happened.
It stays one only while the ask waiting for that answer is still open, and the
ask has to cover the kill and the output collection that follow the budget
expiring. Cut to the truncated remainder instead, it expires *first* whenever the
card's budget reaches the worker's — so the one configuration that can genuinely
exhaust a run's budget was also the one that reported it as an opaque worker
failure rather than a timeout.

Whole seconds, because ``proxy_ask`` takes an int. Small, because the worker
still has to die well inside :data:`LEASE_GRACE_S` of its own deadline — which is
what keeps a liveness-gated reclaim from waiting on a worker that has outlived
the lease it holds.
"""


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


class ExecLease(SerializableBaseModel):
    """The exclusive hold one run has on the tree.

    Attributes:
        run_id: The run holding it. A report is only allowed to release the
            lease when its run id matches — a late report from a reclaimed run
            must not clear a newer lease.
        agent_id: Who requested the run. Named in every refusal it causes.
        cmd: The command, kept for the discovered commit's body.
        started_at: Monotonic clock at acquisition.
        budget: The run's effective wall-clock budget. Kept as a field rather
            than re-derived from the two clocks below, because the deadline moves
            when the run starts and a derivation would then have to un-do a grace
            it cannot see.
        deadline: When this lease may be reclaimed — the moment the run started
            (acquisition, until ``run_started`` says otherwise) plus
            :attr:`budget` plus :data:`LEASE_GRACE_S`. Checked lazily by the next
            mutation; nothing polls it.
    """

    run_id: str
    agent_id: str
    cmd: str
    started_at: float
    budget: float
    deadline: float

    _worker: ActorAddress | None = PrivateAttr(default=None)
    """The worker performing the run — runtime state, never serialized.

    A ``PrivateAttr`` rather than a field: an address is live actor state, and
    Golden Rule #1b keeps that out of a model's field set. Travelling *on the
    lease* is what matters, because a second attribute beside ``_lease`` would
    have to be cleared in every place the lease is, and the one that got missed
    would pin a dead worker's address onto a live run's lease.
    """

    def attach(self, worker: ActorAddress) -> None:
        """Record which worker is performing this run."""
        self._worker = worker

    @property
    def worker_alive(self) -> bool:
        """Whether the run still has a worker that could report it.

        A lease with **no** recorded worker counts as not alive, and that is the
        safe direction: a spawn that never happened leaves a lease nobody can
        release, and refusing to reclaim it would wedge every mutation in the
        team for the rest of its life.
        """
        worker = self._worker
        return worker is not None and worker.is_alive()


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
    stops on either. ``RUNNING`` and ``UNKNOWN`` are not, and they are
    deliberately distinct — the cache's ``get`` returns ``None`` for an unknown
    key, an in-flight one and a negatively-cached one alike, so telling a model
    "still running" about an id it invented would be a lie it cannot recover from.
    """

    DONE = "done"
    FAILED = "failed"
    RUNNING = "running"
    UNKNOWN = "unknown"


class ExecStatus(SerializableBaseModel):
    """Where one run is, and whatever it has produced.

    Attributes:
        state: See :class:`ExecState`.
        run_id: The id that was asked about.
        outcome: The result, on :attr:`ExecState.DONE` only.
        reason: Why the run failed, on :attr:`ExecState.FAILED` only.
        recent_run_ids: This agent's recent runs, on :attr:`ExecState.UNKNOWN`
            only — what turns a mistyped id into a correctable one.
    """

    state: ExecState
    run_id: str
    outcome: ExecOutcome | None = None
    reason: str = ""
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
        team_id: The team, which names the container.
        workspace_id: The card's ``workspace_id``, forwarded verbatim so the
            backend's directory resolution matches the card's.
        timeout_s: The run budget this card asks for, before clamping.
    """

    mode: SandboxMode
    team_id: str
    workspace_id: str | None = None
    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S


class ExecPayload(DeferredPayload):
    """One command handed across to a worker — plain data, and nothing else.

    No ``ActorAddress``, no proxy, no ``Filesystem``: the worker resolves the
    sandbox for itself from the ids below. ``deferred_key`` narrows the base's
    ``Any`` to the run id. Never set it independently of the ``request()`` key —
    ``request`` rebinds it, and a caller that let the two drift would clear a
    different in-flight mark and strand this one for the actor's lifetime.

    Attributes:
        deferred_key: The run id; also the cache key.
        cmd: The command string, exactly as the agent gave it.
        cwd: Working directory, relative to the workspace root.
        mode: The resolved backend.
        team_id: The team that owns the sandbox.
        workspace_id: The card's ``workspace_id``, or ``None``.
        timeout_s: The budget, already clamped to the worker's own.
    """

    deferred_key: str = Field(..., description="Run id; also the cache key.")
    cmd: str = Field(..., description="Command string as the agent supplied it.")
    cwd: str = Field(default="", description="Working directory below the workspace root.")
    mode: SandboxMode = Field(..., description="Resolved sandbox backend.")
    team_id: str = Field(..., description="Team that owns the sandbox actor.")
    workspace_id: str | None = Field(default=None, description="Card's workspace_id, verbatim.")
    timeout_s: float = Field(..., description="Wall-clock budget for the command.")


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
      the worker's report and not merely the command (see that constant). The
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
    the worker's own ceiling — because that is what actually stops the run.
    Clamping against the requested value would leave a card asking for 999 s
    polling long past the 20 s the worker allows it.

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


def resolve_mode(mode: CardMode) -> tuple[SandboxMode, type[SandboxActor]]:
    """Turn a card's requested mode into a backend, warning where the host has none.

    Both exec-capable cards go through this rather than each probing for
    themselves — a second copy of the probe is a second place for the warning to
    stop firing.

    Args:
        mode: What the card asked for, possibly ``"auto"``.

    Returns:
        The resolved mode and its actor class.

    Raises:
        KeyError: If *mode* names no registered backend. Deliberately at wiring
            time rather than at the first command: a typo in a card is a
            configuration error, and configuration errors belong at start-up.
    """
    import warnings  # noqa: PLC0415 — only on the wiring path

    from akgentic.tool.sandbox.tool import (  # noqa: PLC0415 — import cycle
        SANDBOX_ACTOR_CLASSES,
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
    return resolved, SANDBOX_ACTOR_CLASSES[resolved]


def sandbox_config(payload_or_config: ExecPayload | ExecConfig) -> SandboxConfig:
    """Build the sandbox actor's configuration — in one place, for both callers.

    ``getChildrenOrCreate`` keys on the actor **name**, so a config that differs
    in name creates a *second* actor per run instead of resolving the existing
    one; a config that differs in ``workspace_id`` would point the reused actor
    at the wrong directory. The card builds one at wiring time and the worker
    builds one per run, and the two must be identical — so they are built here
    rather than twice by hand.

    **The name carries the workspace**, exactly as ``#Workspace-<workspace>``
    does. A constant name resolved two exec-capable cards on two workspaces onto
    the first actor, so one agent's commands ran in the other's tree while its
    own ``#Workspace`` gated an untouched one — see :func:`sandbox_actor_name`.
    The workspace is resolved as ``workspace_id or team_id``, which is what
    ``Filesystem`` and every backend's ``_start_sandbox`` already do, so the name
    and the directory cannot disagree.

    Args:
        payload_or_config: Either side's carrier of the same four values.

    Returns:
        The configuration for ``#SandboxActor-<workspace>``.
    """
    workspace_name = payload_or_config.workspace_id or payload_or_config.team_id
    return SandboxConfig(
        name=sandbox_actor_name(workspace_name),
        role=SANDBOX_ACTOR_ROLE,
        team_id=payload_or_config.team_id,
        workspace_id=payload_or_config.workspace_id,
        mode=payload_or_config.mode,
    )


class ExecWorker(DeferredWorker):
    """Runs one sandboxed command on its own thread, reports it, and stops.

    The worker is not redundant with the lease, which already excludes a second
    run. If ``#Workspace`` made the blocking call itself its one thread would be
    occupied for the run's duration, and two things follow immediately: every
    ``workspace_exec_result`` poll would queue behind it — collapsing the bounded
    poll into one unbounded ask, the shape the deferred rules exist to forbid —
    and nothing could reclaim the lease of a sandbox that hung past its budget,
    because the thread that would do the reclaiming is the one that is stuck.

    It is also what keeps reads working during a run: the actor's mailbox goes on
    draining, and a read's observation ``tell`` lands on that same mailbox.
    """

    def produce(self, payload: DeferredPayload) -> Any:
        """Wait for the backend, then run the command — one budget, spent in order.

        The two waits are **separate and both bounded**, which is the whole point.
        A backend that has not finished starting answers nothing, and asking it to
        run a command is a wait of unknown length on a thread that holds the
        lease, the tree and its parent's teardown: the wait for the backend is not
        the run, and a budget that cannot tell them apart lets a cold start spend
        a run's whole lease before the command has started.

        So: fix a deadline, resolve the sandbox, ask :meth:`SandboxActor.ready`
        under what is left, tell the parent the run is starting, and only then
        hand the command whatever remains — waiting on it for that budget plus
        :data:`EXEC_REPLY_GRACE_S`, so a command the budget kills still comes back
        as the answer it is. **No ask here is made without a timeout** — that is
        what makes the worker mortal, which is in turn what makes a
        liveness-gated reclaim safe.

        A command that exits — well or badly — and a command its budget killed
        both come back as an :class:`ExecOutcome`, because both are answers an
        agent can read and act on. Anything else raises and is reported as a
        failure.

        ``None`` is never returned: the base treats it as a failure, and a
        command that legitimately produced no output is not a failure.

        Args:
            payload: An :class:`ExecPayload`.

        Returns:
            The :class:`ExecOutcome`.

        Raises:
            TypeError: If handed a payload that is not an :class:`ExecPayload`.
            RuntimeError: If the worker has no orchestrator to resolve the
                sandbox through, or if the backend never became ready.
        """
        if not isinstance(payload, ExecPayload):
            raise TypeError(f"ExecWorker requires an ExecPayload, got {type(payload)}")
        deadline = time.monotonic() + self.timeout_s
        address = self._sandbox_address(payload, deadline)
        self._await_ready(address, deadline)
        self._run_started(payload.deferred_key)
        remaining = deadline - time.monotonic()
        budget = max(0.0, min(payload.timeout_s, remaining))
        sandbox = self.proxy_ask(
            address, SandboxActor, timeout=ask_seconds(budget) + EXEC_REPLY_GRACE_S
        )
        try:
            result = sandbox.exec(payload.cmd, payload.cwd, timeout=budget)
        except subprocess.TimeoutExpired:
            return ExecOutcome(
                stdout="",
                stderr=f"Command exceeded its {budget:g}s budget and was killed.",
                exit_code=TIMED_OUT_EXIT_CODE,
                timed_out=True,
            )
        return ExecOutcome(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    def _sandbox_address(self, payload: ExecPayload, deadline: float) -> ActorAddress:
        """Get-or-create the team's ``#SandboxActor`` and return its address.

        Idempotent by construction (ADR-025): the card already created it at
        wiring time, so this resolves the existing one. The class comes from
        ``SANDBOX_ACTOR_CLASSES`` at call time, never at import time, so a
        backend injected by a deployment package is still found.

        The address rather than a proxy, because the two phases below need two
        proxies carrying two different timeouts.
        """
        from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES  # noqa: PLC0415 — cycle

        orchestrator = self.orchestrator
        if orchestrator is None:
            raise RuntimeError("An exec worker cannot resolve its sandbox without an orchestrator.")
        actor_class = SANDBOX_ACTOR_CLASSES[payload.mode]
        orchestrator_proxy = self.proxy_ask(
            orchestrator, Orchestrator, timeout=ask_seconds(deadline - time.monotonic())
        )
        address: ActorAddress = orchestrator_proxy.getChildrenOrCreate(
            actor_class, config=sandbox_config(payload)
        )
        return address

    def _await_ready(self, address: ActorAddress, deadline: float) -> None:
        """Block until the backend can serve messages, or give up saying so.

        ``ready()`` carries no information; being *answered* is the information.
        Pykka's mailbox is FIFO, so it cannot come back before ``on_start`` has
        returned — no flag, no polling, no state.

        Every failure of this one ask means the same thing to the agent waiting
        on it: the backend is not usable right now and the command did not run.
        A timeout is the expected one (a container image still being built); a
        dead actor is the other, and telling them apart would buy the model
        nothing it could act on differently.

        The original is **logged** as well as chained, because chaining alone
        loses it: ``receiveMsg_DeferredPayload`` reports failures as ``str(exc)``,
        which keeps the text above and drops the cause. Without this line a
        backend that raises rather than hangs leaves no trace anywhere of what it
        actually raised.

        Args:
            address: The sandbox actor.
            deadline: The worker's own deadline, monotonic.

        Raises:
            RuntimeError: When the backend did not answer in time — reported to
                the agent verbatim by ``receiveMsg_DeferredPayload``.
        """
        seconds = ask_seconds(deadline - time.monotonic())
        try:
            self.proxy_ask(address, SandboxActor, timeout=seconds).ready()
        except Exception as exc:
            logger.warning(
                "[%s] The execution backend did not answer readiness within %ss: %r",
                self.config.name,
                seconds,
                exc,
            )
            raise RuntimeError(backend_not_ready(seconds)) from exc

    def _run_started(self, run_id: str) -> None:
        """Tell ``#Workspace`` that *run_id*'s command is starting now.

        A **tell**, deliberately: the worker needs no answer, and an ask would be
        one more unbounded wait on the very thread this budget exists to bound.
        A parentless worker — the shape a few unit tests build — simply skips it.
        """
        parent = self._parent
        if parent is None:
            return
        from akgentic.tool.workspace.actor import WorkspaceActor  # noqa: PLC0415 — cycle

        self.proxy_tell(parent, WorkspaceActor).run_started(run_id)


def format_outcome(outcome: ExecOutcome) -> str:
    """Render a finished run the way ``exec_command`` has always rendered one.

    Deliberately the existing shape rather than a second one: a model already
    reads this, and two formats for one thing is how the pair drifts.
    """
    status = "OK" if outcome.exit_code == 0 else "FAILED"
    return (
        f"exit_code: {outcome.exit_code} ({status})"
        f"\nstdout:\n{outcome.stdout}"
        f"\nstderr (note: many tools write progress to stderr"
        f" even on success):\n{outcome.stderr}"
    )


def format_status(status: ExecStatus) -> str:
    """Render any run state as the string an agent reads back.

    Every state produces a returned string, including the ones that are not a
    result. An unknown run id in particular does **not** raise: it lists the
    agent's recent runs, so a model that mistyped one reads the right one back.
    """
    if status.state is ExecState.DONE and status.outcome is not None:
        return format_outcome(status.outcome)
    if status.state is ExecState.FAILED:
        return f"Run {status.run_id} failed: {status.reason}"
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
