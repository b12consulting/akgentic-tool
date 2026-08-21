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
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import Field

from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.core.deferred import DeferredPayload, DeferredWorker
from akgentic.tool.sandbox.actor import (
    SANDBOX_ACTOR_NAME,
    SANDBOX_ACTOR_ROLE,
    CardMode,
    SandboxActor,
    SandboxConfig,
    SandboxMode,
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

DEFAULT_EXEC_POLL_ATTEMPTS = 12
DEFAULT_EXEC_POLL_DELAY_S = 0.4
"""Caller-side poll: ~5 s of waiting inside the tool call before handing back a run id.

Deliberately longer than ``poll_deferred``'s own default of ~2 s, and the reason
is worth stating because the instinct is to minimise it. At a 2 s poll *every*
command slower than two seconds — every test run, every build — becomes a
two-turn interaction, which is most real uses of a shell. An agent inside a tool
call cannot do anything else anyway; the call is synchronous from the model's
point of view. So a longer poll costs that agent latency it was going to spend
regardless, and costs the team nothing, because the tree is leased either way.
The invariant is "bounded, with a degraded answer always available" — never "as
short as possible".
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

The accepted cost, stated plainly: a run that somehow outlives its deadline may
interleave with a mutation, and both changes end up in that run's discovered
commit. Strictly better than a permanently wedged team.
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


class ExecLease(SerializableBaseModel):
    """The exclusive hold one run has on the tree.

    Attributes:
        run_id: The run holding it. A report is only allowed to release the
            lease when its run id matches — a late report from a reclaimed run
            must not clear a newer lease.
        agent_id: Who requested the run. Named in every refusal it causes.
        cmd: The command, kept for the discovered commit's body.
        started_at: Monotonic clock at acquisition.
        deadline: ``started_at`` + the effective budget + :data:`LEASE_GRACE_S`.
            Checked lazily by the next mutation; nothing polls it.
    """

    run_id: str
    agent_id: str
    cmd: str
    started_at: float
    deadline: float


class ExecStart(SerializableBaseModel):
    """The answer to "may I run this, and under what id".

    Exactly one of the two fields is non-empty. A model rather than
    ``str | None`` because it crosses the actor boundary, and because a bare
    string would leave the caller guessing which of the two it holds.

    Attributes:
        run_id: The issued id, when the run was accepted.
        refusal: Why not, when it was not — the busy message, naming the holder.
    """

    run_id: str = ""
    refusal: str = ""


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
    """Return an attempt count whose poll cannot outlast the run it waits for.

    A poll longer than the run budget parks the agent's thread inside the tool
    call past the point where there is anything left to wait for: by then the run
    has either reported or been killed by its own budget, so every further
    attempt is a sleep with no possible answer. It is the one budget on this card
    that can be misconfigured into costing the team real time and buying nobody
    anything, so it is clamped rather than accepted.

    The bound is the **effective** run budget — the card's ``timeout_s`` after
    the worker's own ceiling — because that is what actually stops the run.
    Clamping against the requested value would leave a card asking for 999 s
    polling long past the 20 s the worker allows it.

    Args:
        attempts: What the card asked for.
        delay: Seconds between attempts.
        run_budget: The effective wall-clock budget of the run being waited on.

    Returns:
        *attempts* unchanged when the poll already fits, otherwise the largest
        count that does — never below 1, because a caller that polls zero times
        gets the handoff message without ever looking, which is worse than
        looking once.
    """
    if attempts <= 0 or delay <= 0:
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
    in name creates a *second* ``#SandboxActor`` per run instead of resolving the
    team's one; a config that differs in ``workspace_id`` would point the reused
    actor's container at the wrong directory. The card builds one at wiring time
    and the worker builds one per run, and the two must be identical — so they
    are built here rather than twice by hand.

    Args:
        payload_or_config: Either side's carrier of the same four values.

    Returns:
        The configuration for ``#SandboxActor``.
    """
    return SandboxConfig(
        name=SANDBOX_ACTOR_NAME,
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
        """Resolve the backend, run the command under the budget, report the outcome.

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
                sandbox through.
        """
        if not isinstance(payload, ExecPayload):
            raise TypeError(f"ExecWorker requires an ExecPayload, got {type(payload)}")
        sandbox = self._sandbox(payload)
        budget = min(payload.timeout_s, self.timeout_s)
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

    def _sandbox(self, payload: ExecPayload) -> SandboxActor:
        """Get-or-create the team's ``#SandboxActor`` and return an ask proxy to it.

        Idempotent by construction (ADR-025): the card already created it at
        wiring time, so this resolves the existing one. The class comes from
        ``SANDBOX_ACTOR_CLASSES`` at call time, never at import time, so a
        backend injected by a deployment package is still found.
        """
        from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES  # noqa: PLC0415 — cycle

        orchestrator = self.orchestrator
        if orchestrator is None:
            raise RuntimeError("An exec worker cannot resolve its sandbox without an orchestrator.")
        actor_class = SANDBOX_ACTOR_CLASSES[payload.mode]
        orchestrator_proxy = self.proxy_ask(orchestrator, Orchestrator)
        address = orchestrator_proxy.getChildrenOrCreate(
            actor_class, config=sandbox_config(payload)
        )
        return self.proxy_ask(address, SandboxActor)


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
    """The handoff message, with the run id echoed verbatim into the model's context."""
    return (
        f"Run {run_id} is still in progress. It holds the workspace until it finishes. "
        f"Call workspace_exec_result('{run_id}') on your next turn to collect the output."
    )


def unconfigured() -> str:
    """Refusal for an exec request against a workspace with no backend bound."""
    return _UNCONFIGURED_MSG
