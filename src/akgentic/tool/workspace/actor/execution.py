"""The exec lease, the run bookkeeping, and the deferred surface (ADR-045 §1).

**Exec is fenced, not gated, and that is the whole difference.** Every other
writer says what it is about to do, so the gate can check a precondition against
the file it names. A shell command cannot, so ``workspace_exec`` takes an
exclusive lease over the tree instead, and its write set is *discovered*
afterwards from ``git status --porcelain -uall`` (ADR-036 §5).

The deferred-result mechanism (ADR-033) is engaged in full: the blocking sandbox
call happens in a ``#defer-`` worker, never on the actor's thread. Everything
this module does on the ask path is O(1) plus the journal's bounded git calls.

**Two modules are called ``execution`` and they are not the same one.**
:mod:`akgentic.tool.workspace.execution` holds the exec models, ``ExecWorker``
and the sandbox edge; this module holds the mixin that drives them. The
dependency runs one way — this module imports **from** that one — and every
import here is absolute so the two never blur.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from akgentic.core.actor_address import ActorAddress
from akgentic.tool.core.deferred import DEFAULT_WORKER_TIMEOUT_S, DeferredResultActor
from akgentic.tool.workspace.execution import (
    LEASE_GRACE_S,
    MAX_TRACKED_RUNS,
    ExecConfig,
    ExecLease,
    ExecOutcome,
    ExecPayload,
    ExecStart,
    ExecState,
    ExecStatus,
    new_run_id,
    unconfigured,
)
from akgentic.tool.workspace.journal import GitJournal, Identity
from akgentic.tool.workspace.models import WorkspaceConfig, WorkspaceState

logger = logging.getLogger(__name__)

EXEC_CAPABILITY = "exec"
"""First field of a discovered commit's subject, beside ``write``, ``edit``, ``patch``."""

_BUSY_PREFIX = "workspace busy"
"""Opening words of every refusal a lease causes.

Fixed wording because it is what an agent recognises across all seven refused
operations — the six mutations and a second ``workspace_exec``.
"""

if TYPE_CHECKING:
    # ``deliver`` and ``fail`` call ``super()``, which resolves against this
    # mixin's own bases — ``(object,)`` at runtime. mypy needs the real base
    # there or it reports ``"deliver" undefined in superclass``. The MRO stays
    # consistent in both worlds: at runtime the mixin contributes only
    # ``object``, and the actor names the base itself.
    _ExecBase = DeferredResultActor[WorkspaceConfig, WorkspaceState, str, ExecOutcome]
else:
    _ExecBase = object


class ExecMixin(_ExecBase):
    """The lease, the run bookkeeping, and the deferred surface."""

    _exec_config: ExecConfig | None
    _lease: ExecLease | None
    _reclaimed: OrderedDict[str, ExecLease]
    _run_errors: OrderedDict[str, str]
    _recent_runs: dict[str, OrderedDict[str, str]]
    _journal: GitJournal

    if TYPE_CHECKING:
        # Supplied by ``ObservationMixin``; the MRO binds them at runtime.
        def _identity(self, agent_id: str) -> Identity: ...

        def _name_of(self, agent_id: str) -> str: ...

    ##
    ## Exec — the lease, the run, and the discovered commit
    ##
    def configure_exec(self, config: ExecConfig) -> None:
        """Record which backend to run commands on — **tell** path, once per card.

        The actor cannot take this from :class:`WorkspaceConfig`, because
        ``getChildrenOrCreate`` fixes that at creation and the card that creates
        the actor for a workspace is routinely one with no exec capability at
        all. So an exec-capable card announces itself here instead, at bind time,
        exactly as ``register_agent`` does.

        Last writer wins, and that is correct: two exec-capable cards over one
        tree must agree on the backend anyway, since they share one
        ``#SandboxActor-<workspace>`` — the sandbox actor is named per workspace,
        for the same reason this one is.

        Args:
            config: The resolved backend and the ids to build payloads from.
        """
        self._exec_config = config

    def request_exec(self, agent_id: str, cmd: str, cwd: str = "") -> ExecStart:
        """Take the lease and spawn the worker, or refuse — in one mailbox turn.

        Everything here is O(1) plus the journal's bounded git calls. **No
        sandbox call happens on this thread**: that is what keeps the actor's
        mailbox draining, which is what makes reads work during a run and a
        refused mutation cost one turn instead of the run's whole duration.

        Args:
            agent_id: Identity of the requesting agent, as a string.
            cmd: The command string.
            cwd: Working directory below the workspace root.

        Returns:
            The issued run id, or the refusal that stopped it.
        """
        busy = self._busy_refusal()
        if busy is not None:
            return ExecStart(refusal=busy)
        config = self._exec_config
        if config is None:
            return ExecStart(refusal=unconfigured())
        # The tree's existing dirt belongs to nobody, and must not end up inside
        # this run's discovered commit — which is exactly what would happen,
        # since that commit takes whatever the tree shows afterwards.
        self._journal.commit_out_of_band()
        run_id = new_run_id()
        budget = min(config.timeout_s, DEFAULT_WORKER_TIMEOUT_S)
        now = time.monotonic()
        self._lease = ExecLease(
            run_id=run_id,
            agent_id=agent_id,
            cmd=cmd,
            started_at=now,
            budget=budget,
            deadline=now + budget + LEASE_GRACE_S,
        )
        self._track_run(agent_id, run_id)
        # The lease is taken BEFORE request(), because a spawn failure reports
        # through fail() synchronously and must find a lease to release.
        worker = self.request(
            run_id,
            ExecPayload(
                deferred_key=run_id,
                cmd=cmd,
                cwd=cwd,
                mode=config.mode,
                team_id=config.team_id,
                workspace_id=config.workspace_id,
                timeout_s=budget,
            ),
        )
        self._attach_worker(run_id, worker)
        return ExecStart(run_id=run_id)

    def _attach_worker(self, run_id: str, worker: ActorAddress | None) -> None:
        """Record on the lease which worker is performing *run_id*.

        Guarded on the lease still being this run's, because ``request`` can
        release it before returning: a spawn that fails reports through ``fail``
        synchronously, which reaches :meth:`_finish_run` and clears the lease
        (29-5's ordering note). Attaching blindly would pin an address onto
        whatever lease came next.

        Args:
            run_id: The run the address belongs to.
            worker: The spawned worker, or ``None`` when nothing was spawned.
        """
        lease = self._lease
        if worker is None or lease is None or lease.run_id != run_id:
            return
        lease.attach(worker)

    def run_started(self, run_id: str) -> None:
        """TELL, from the worker. Re-base *run_id*'s lease onto the moment it began.

        The deadline is set when a run is *accepted*, and until this arrives that
        is all it can mean. Anything slow ahead of the command therefore spends
        it — a first run on a container backend builds the image, which is a
        minute against a 20 s lease — and the tree is then handed to another
        agent while a live run is still about to write into it. Re-basing here
        makes the deadline mean what its name says: the run's budget, measured
        from the run.

        A ``run_started`` for a run that no longer holds the lease is ignored
        rather than raised on. It is a tell from a worker whose lease was already
        reclaimed, and there is nothing left for it to extend.

        Args:
            run_id: The run whose command is starting now.
        """
        lease = self._lease
        if lease is None or lease.run_id != run_id:
            return
        self._lease = lease.model_copy(
            update={"deadline": time.monotonic() + lease.budget + LEASE_GRACE_S}
        )

    def exec_status(self, agent_id: str, run_id: str) -> ExecStatus:
        """Report where *run_id* stands, for *agent_id*.

        The base's ``get`` cannot answer this alone: it returns ``None`` for an
        unknown key, an in-flight one and a negatively-cached one alike, and
        telling a model "still running" about an id it invented is a dead end it
        cannot recover from. So a failure is read from this actor's own small
        error map.

        **A run is running iff it is in flight**, and that is the definition
        rather than a shortcut — which is why the base's ``_in_flight`` is read
        here directly. The tracking map cannot answer it: ``_recent_runs`` holds
        32 ids *per agent* while ``_slots`` holds 128 results *in total*, so past
        five agents the tracking outlives the results and a settled run whose
        outcome has been evicted would report as still running for ever.
        ``_in_flight`` is this actor's own attribute, read on its own thread, and
        it is cleared in the same mailbox turn that stores the outcome — so there
        is no window in which a run is neither in flight nor answerable.

        A settled run whose result has since been evicted therefore answers
        ``UNKNOWN`` with this agent's recent ids, which is recoverable; the
        alternative was a fourth state meaning "finished, result no longer held",
        which invents semantics for the agent to reason about.

        Args:
            agent_id: Identity of the asking agent, as a string.
            run_id: The run to report on.

        Returns:
            Done with the outcome, failed with the reason, running, or unknown
            with this agent's recent run ids.
        """
        outcome = self.get(run_id)
        if outcome is not None:
            return ExecStatus(state=ExecState.DONE, run_id=run_id, outcome=outcome)
        error = self._run_errors.get(run_id)
        if error is not None:
            return ExecStatus(state=ExecState.FAILED, run_id=run_id, reason=error)
        if run_id in self._in_flight:
            return ExecStatus(state=ExecState.RUNNING, run_id=run_id)
        return ExecStatus(
            state=ExecState.UNKNOWN,
            run_id=run_id,
            recent_run_ids=list(self._recent_runs.get(agent_id, {})),
        )

    def deliver(self, key: str, value: ExecOutcome) -> None:
        """TELL, from the worker. Cache the outcome, then close the run out."""
        super().deliver(key, value)
        self._finish_run(key)

    def fail(self, key: str, error: str) -> None:
        """TELL, from the worker. Record the failure, then close the run out.

        The base caches negatively with a TTL, which is what stops a broken
        backend from being respawned once per poll. The reason is kept here as
        well because the base deliberately does not expose it — ``get`` answers
        ``None`` for a failure exactly as it does for an unknown key, and a run
        that failed must never be reported as still running.
        """
        super().fail(key, error)
        self._run_errors[key] = error
        self._run_errors.move_to_end(key)
        while len(self._run_errors) > self.cache_capacity:
            self._run_errors.popitem(last=False)
        self._finish_run(key)

    def _finish_run(self, run_id: str) -> None:
        """Commit what the run produced and release its lease.

        **Only a report from the lease's own run releases it.** A late report
        from a run whose lease was already reclaimed must not clear a newer
        agent's lease — and must not commit as its own agent either, since by
        then the tree may hold somebody else's accepted mutations. What it must
        not do is what it used to: return in silence, leaving the files it
        produced in the tree for a later agent's discovery to sweep into *that*
        agent's commit.
        """
        lease = self._lease
        if lease is None or lease.run_id != run_id:
            self._orphaned_report(run_id)
            return
        self._lease = None
        self._journal.commit_discovered(
            self._identity(lease.agent_id), EXEC_CAPABILITY, detail=lease.cmd
        )

    def _orphaned_report(self, run_id: str) -> None:
        """Record a run that reported after its lease was taken back.

        Loud, because it is a real anomaly: a run outlived its budget, the tree
        was handed to somebody else, and files arrived from a command nobody is
        waiting for any more. And committed **out of band** rather than
        discovered, because attributing them to the late run would put whatever
        else is in the tree under its author — the mirror image of the silent
        drop, written into the journal where it looks authoritative.

        A report with nothing reclaimed under its id is a second report for a run
        already closed out, which left nothing behind and needs no record.

        Args:
            run_id: The run that reported late.
        """
        lease = self._reclaimed.pop(run_id, None)
        if lease is None:
            return
        logger.warning(
            "Workspace %s: run %s reported after its lease was reclaimed (agent %s, command %r). "
            "Whatever it wrote is committed as out-of-band — belonging to nobody — because the "
            "tree may since have been changed by another agent.",
            self.config.workspace_name,
            run_id,
            self._name_of(lease.agent_id),
            lease.cmd,
        )
        self._journal.commit_out_of_band()

    def _busy_refusal(self) -> str | None:
        """Refuse under a live lease, reclaim a dead one, or allow.

        Fail fast, never stall. Ten seconds of silence inside a tool call is
        indistinguishable from a hang and gives the model nothing to react to; an
        immediate refusal naming the holder lets it read a file, answer the user,
        or ask the holder. That is only affordable because the actor's thread is
        free — the blocking call is in a worker.

        **Past the deadline is not the same as gone.** A worker that is still
        alive is still going to write into this tree, so a past-deadline lease
        whose worker is alive is refused rather than reclaimed — with wording
        that says which of the two it is. That is only safe because the worker is
        mortal: its own budget bounds every ask it makes, so a lease cannot be
        held open for ever by a wait nobody can see.

        Returns:
            The refusal text, or ``None`` when the tree is free.
        """
        lease = self._lease
        if lease is None:
            return None
        if time.monotonic() <= lease.deadline:
            return (
                f"{_BUSY_PREFIX} — exec run {lease.run_id} is in progress "
                f"(agent '{self._name_of(lease.agent_id)}'). Reads still work; retry the change "
                f"once the run has finished."
            )
        if lease.worker_alive:
            return (
                f"{_BUSY_PREFIX} — exec run {lease.run_id} has passed its budget and is still "
                f"being waited on (agent '{self._name_of(lease.agent_id)}'). Reads still work; "
                f"retry the change once the run has finished."
            )
        logger.warning(
            "Workspace %s: reclaiming the lease of run %s (agent %s) — it passed its deadline "
            "and its worker is no longer alive, so nothing will ever report it. Anything it is "
            "still writing will land in a later commit.",
            self.config.workspace_name,
            lease.run_id,
            self._name_of(lease.agent_id),
        )
        self._reclaim(lease)
        return None

    def _reclaim(self, lease: ExecLease) -> None:
        """Take the tree back from *lease*, remembering the run it belonged to.

        The run is kept — capped, like every other map on a team singleton —
        because a reclaimed run may still report, and :meth:`_orphaned_report`
        needs its agent and its command to say whose command left the files it is
        about to commit as belonging to nobody.
        """
        self._lease = None
        self._reclaimed[lease.run_id] = lease
        self._reclaimed.move_to_end(lease.run_id)
        while len(self._reclaimed) > MAX_TRACKED_RUNS:
            self._reclaimed.popitem(last=False)

    def _track_run(self, agent_id: str, run_id: str) -> None:
        """Remember *run_id* as one of *agent_id*'s recent runs, LRU-capped.

        Capped for the reason every map on a team singleton is: an uncapped one
        leaks for the life of the team. Losing the oldest id is safe — a finished
        run is still answered from the result cache, and the list exists only to
        make a mistyped id correctable.
        """
        runs = self._recent_runs.setdefault(agent_id, OrderedDict())
        runs[run_id] = run_id
        runs.move_to_end(run_id)
        while len(runs) > MAX_TRACKED_RUNS:
            runs.popitem(last=False)
