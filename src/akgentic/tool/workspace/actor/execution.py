"""The exec lease, the run bookkeeping, and the deferred surface (ADR-045 §1).

**Exec is fenced, not gated, and that is the whole difference.** Every other
writer says what it is about to do, so the gate can check a precondition against
the file it names. A shell command cannot, so ``workspace_exec`` takes an
exclusive hold over the tree instead, and its write set is *discovered*
afterwards from ``git status --porcelain -uall`` (ADR-036 §5).

**The hold itself is no longer in this module.** It is an ``O_EXCL`` marker file
under the tree's metadata sibling, taken through a
:class:`~akgentic.tool.workspace.lock.LockBackend` (ADR-051 Decision 5), because
an actor's hold is exclusive only inside one process and two workers over one
mounted tree get two actors. What stays here is what the actor still owns: the
in-memory record of the running run, the bookkeeping, the worker thread, the
discovered commit, and the *prompt* release a report makes possible.

The blocking call happens on ``#Workspace``'s own single worker thread, never on
the actor's thread and no longer on a second actor's. Everything this module does
on the ask path is O(1) plus the journal's bounded git calls.

**Two modules are called ``execution`` and they are not the same one.**
:mod:`akgentic.tool.workspace.execution` holds the exec models, ``ExecRunner``
and the sandbox edge; this module holds the mixin that drives them. The
dependency runs one way — this module imports **from** that one — and every
import here is absolute so the two never blur.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from concurrent import futures
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from akgentic.tool.core.deferred import DeferredResultActor
from akgentic.tool.sandbox.backend import ExecReport
from akgentic.tool.workspace.execution import (
    _BUSY_PREFIX,
    DEFAULT_EXEC_TIMEOUT_S,
    EXEC_SHUTDOWN_GRACE_S,
    LEASE_GRACE_S,
    MAX_TRACKED_RUNS,
    TIMED_OUT_EXIT_CODE,
    ExecConfig,
    ExecOutcome,
    ExecRunner,
    ExecStart,
    ExecState,
    ExecStatus,
    RunningExec,
    effective_budget,
    lock_unavailable,
    resolve_mode,
    unconfigured,
)
from akgentic.tool.workspace.journal import GitJournal, Identity
from akgentic.tool.workspace.lock import LockBackend, LockTicket
from akgentic.tool.workspace.models import WorkspaceConfig, WorkspaceState

logger = logging.getLogger(__name__)

EXEC_CAPABILITY = "exec"
"""First field of a discovered commit's subject, beside ``write``, ``edit``, ``patch``."""

if TYPE_CHECKING:
    from collections.abc import Callable

    # ``deliver`` and ``fail`` call ``super()``, which resolves against this
    # mixin's own bases — ``(object,)`` at runtime. mypy needs the real base
    # there or it reports ``"deliver" undefined in superclass``. The MRO stays
    # consistent in both worlds: at runtime the mixin contributes only
    # ``object``, and the actor names the base itself.
    _ExecBase = DeferredResultActor[WorkspaceConfig, WorkspaceState, str, ExecOutcome]
else:
    _ExecBase = object


class ExecMixin(_ExecBase):
    """The hold on the tree, the run bookkeeping, the worker thread and its teardown."""

    _exec_config: ExecConfig | None
    _runner: ExecRunner | None
    _lock: LockBackend | None
    _executor: ThreadPoolExecutor
    _pending: Future[None] | None
    _running: RunningExec | None
    _run_errors: OrderedDict[str, str]
    _recent_runs: dict[str, OrderedDict[str, str]]
    _discarded_run: str | None
    _journal: GitJournal

    if TYPE_CHECKING:
        # Supplied by ``ObservationMixin``; the MRO binds them at runtime.
        def _identity(self, agent_id: str) -> Identity: ...

        def _name_of(self, agent_id: str) -> str: ...

    ##
    ## Exec — the hold on the tree, the running run, and the discovered commit
    ##
    def configure_lock(self, backend: LockBackend) -> None:
        """Receive the hold this tree is serialised by — **tell** path, at bind time.

        Built card-side, in ``observer()``, and announced here, because the card
        is where every other self-resolved runtime object is built (ADR-051
        Decision 8) and the actor is where admission, the discovered commit and
        the release still are. Splitting those would fork the release across two
        owners: the card only learns a run finished by polling, so a caller that
        took a run id instead of waiting would leave the tree held until
        staleness on the **normal** path.

        Last writer wins, like :meth:`configure_exec`, and nothing is released
        or rebuilt on a replacement: a backend holds no per-tree state, so two
        cards announcing two instances of the same class are interchangeable. A
        deployment that switched backends mid-run would strand one marker, which
        staleness reclaims.

        Args:
            backend: What ``acquire`` and ``release`` are called on.
        """
        self._lock = backend

    def configure_exec(self, config: ExecConfig) -> None:
        """Build the backend commands will run on — **tell** path, once per card.

        The actor cannot take this from :class:`WorkspaceConfig`, because the
        first bind fixes that for every card on the tree and the card that binds
        a tree first is routinely one with no exec capability at all. So an
        exec-capable card announces itself here instead, at bind time, right
        after its ``attach``.

        **This is the one place a backend is built**, and it is here because
        :class:`ExecConfig` is the one place ``mode``, ``workspace_path`` and
        ``timeout_s`` all arrive together. Nothing is
        probed, created or started by the construction: the container is
        provisioned by the worker thread on the first command.

        Last writer wins, and that is correct: two exec-capable cards over one
        tree must agree on the backend anyway. What is new is that overwriting
        now **leaks** — a backend with no owner, and for docker a container with
        nobody left to stop it. So an *equal* config changes nothing at all, which
        is the common case and close to the only one; a *different* one stops the
        old runner and builds a new one.

        **Equal across teams, by construction.** A hosted tree is bound by agents
        of several teams, and :class:`ExecConfig` carries no team, so a second
        team's card with the same settings announces an equal config and keeps
        the running runner — its run in flight included.

        **A replacement mid-run is deliberately not guarded**, for the reason
        :meth:`_run_budget` gives about the same situation: two exec-capable
        cards over one tree that disagree on the backend are a misconfiguration,
        not a race, and the honest failure is the run reporting an error rather
        than a lock nobody can see.

        **The old runner is released before the new one exists**, which leaves
        one window worth naming: a registered backend whose *constructor* raises
        strands this actor holding the runner it has just stopped, under the
        config it has just kept — and the equality check above then short-circuits
        every later announcement, so it holds it for good. It is left this way
        rather than reordered because the alternative leaks the very backend the
        release exists to reclaim, and because the degradation is loud: every run
        reports the stopped backend's error, which is an answer its caller reads.

        Args:
            config: The resolved backend, the tree, and the run budget.
        """
        if self._exec_config == config and self._runner is not None:
            return
        self._stop_runner()
        _mode, backend = resolve_mode(config.mode)
        self._runner = ExecRunner(backend, config.workspace_path)
        self._exec_config = config

    def _stop_runner(self) -> None:
        """Release the current runner's backend, if there is one, swallowing failures.

        Two callers with the same requirement: :meth:`configure_exec` replacing a
        runner, and teardown's fourth step. Neither may raise — one is on a
        binding path where the degradation is a refused run, and the other is
        inside ``on_stop``, where leaving a Pykka actor part-way stopped is worse
        than any error this could report.
        """
        runner = self._runner
        if runner is None:
            return
        try:
            runner.stop()
        except Exception:
            logger.warning(
                "Workspace %s: releasing the exec backend raised — swallowing",
                self.config.workspace_path,
                exc_info=True,
            )

    def request_exec(self, agent_id: str, cmd: str, cwd: str = "") -> ExecStart:
        """Take the tree and start the run, or refuse — in one mailbox turn.

        Admission has two answers now: the run id of the caller's **own** run, or
        a refusal that names nobody. Nothing this path produces can name a run
        belonging to another agent — the defect a model's parallel batch of
        ``workspace_exec`` calls used to reproduce every time, by reading a
        sibling call's id out of a refusal and collecting it.

        **The decision belongs to the lock, not to this actor**, and that is the
        whole of the change: an actor's hold is exclusive inside one process, and
        two workers over one mounted tree get two actors. The refused caller
        retries; there is no queue to wait in, and among several waiters the
        first to retry wins rather than the first to ask.

        **The run id comes from the grant**, so the id the agent holds and the id
        in the marker on disk are one value by construction rather than by
        agreement.

        The leading :meth:`_holding_run` call is housekeeping, not the decision:
        it releases a run of *this* actor's that is wedged past its budget, both
        in memory and on disk, so the acquire that follows is granted without
        waiting for the marker's own staleness. This actor is passive — nothing
        releases on a timer — so every path that can reach a wedged run makes
        this call.

        Everything here is O(1) plus the journal's bounded git calls and the
        lock's two file operations. **No sandbox call happens on this thread**:
        that is what keeps the actor's mailbox draining, which is what makes
        reads work during a run and a refused mutation cost one turn instead of
        the run's whole duration.

        Args:
            agent_id: Identity of the requesting agent, as a string.
            cmd: The command string.
            cwd: Working directory below the workspace root.

        Returns:
            The issued run id, or the refusal — which names no run and no agent.
        """
        config = self._exec_config
        lock = self._lock
        if config is None or lock is None:
            return ExecStart(refusal=unconfigured())
        self._holding_run()
        try:
            grant = lock.acquire(
                config.workspace_path,
                LockTicket(agent_id=agent_id, cmd=cmd, budget_s=self._run_budget()),
            )
        except Exception as exc:  # noqa: BLE001 — an ask path answers, it never crashes
            # An unwritable metadata parent, a full disk. This is an ``ask``: a
            # raise here crosses the actor boundary and reaches the agent as a
            # crash rather than as an answer it can act on.
            logger.warning(
                "Workspace %s: taking the exec lock raised — refusing the run: %r",
                self.config.workspace_path,
                exc,
                exc_info=True,
            )
            return ExecStart(refusal=lock_unavailable())
        if not grant.run_id:
            return ExecStart(refusal=grant.refusal)
        self._track_run(agent_id, grant.run_id, cmd)
        self._start_run(grant.run_id, agent_id, cmd, cwd, config)
        return ExecStart(run_id=grant.run_id)

    def _start_run(
        self, run_id: str, agent_id: str, cmd: str, cwd: str, config: ExecConfig
    ) -> None:
        """Record the granted run and submit it to the worker — the one accept path.

        Reached from :meth:`request_exec` and from nowhere else, now that the
        queue's dequeue is gone. It is still its own method because the
        out-of-band commit, the in-memory record and the submit have to happen
        together and in this order.

        **The out-of-band commit happens first.** The tree's existing dirt
        belongs to nobody, and a discovered commit takes whatever the tree shows
        afterwards — so anything already lying there would be attributed to an
        agent that never wrote it. That matters more with a file lock than it did
        with a queue: the previous holder may have been **another process**, so
        this run can be the first thing in this process to look at the tree.

        **The record is taken before the submit**, and that ordering is
        load-bearing: a submit that raises reports through :meth:`fail`
        *synchronously*, on this thread, and that has to find a run to close out
        or the tree is never given back. A ``submit`` onto an executor that has
        already been shut down raises ``RuntimeError``, so the case is reachable
        rather than theoretical.

        **This method submits and returns**, and nothing about the command runs
        here: no ``start``, no ``exec``, no ``subprocess``. That is what keeps the
        mailbox draining, which is what makes reads work during a run and a
        refused mutation cost one turn instead of the run's whole duration.

        **``reply_to`` is captured here, on the actor's thread**, and handed to
        the worker as an argument. The worker must never read ``self.myAddress``
        — or anything else on this actor — for itself.

        Args:
            run_id: The id the grant issued — the same one the marker holds.
            agent_id: Who asked, as a string.
            cmd: The command string, exactly as the agent gave it.
            cwd: Working directory below the workspace root.
            config: The budget to start it under.
        """
        self._journal.commit_out_of_band()
        self._running = RunningExec(
            run_id=run_id,
            agent_id=agent_id,
            cmd=cmd,
            started_at=time.monotonic(),
        )
        # What ``request()`` used to do, and ``exec_status`` still answers
        # RUNNING from it. The base's deliver/fail clear it, unchanged.
        self._in_flight.add(run_id)
        try:
            runner = self._runner
            if runner is None:
                raise RuntimeError("#Workspace has no execution backend to run this command on.")
            self._pending = self._executor.submit(
                runner.perform,
                run_id=run_id,
                cmd=cmd,
                cwd=cwd,
                timeout_s=effective_budget(config.timeout_s),
                reply_to=self.myAddress,
            )
        except Exception as exc:  # noqa: BLE001 — every failure here is the run's answer
            logger.warning(
                "Workspace %s: run %s never reached the worker: %r",
                self.config.workspace_path,
                run_id,
                exc,
                exc_info=True,
            )
            self.fail(run_id, f"The command was never handed to the sandbox: {exc}")

    def receiveMsg_ExecReport(self, report: ExecReport) -> None:
        """TELL, from this actor's own worker thread. Turn one report into an answer.

        The only thing that closes a run out on the ordinary path, and it has
        exactly three branches because the worker reports exactly three things.
        Two of them are *answers* the agent can read and one is a failure:

        - a **result** — whatever it exited with. A non-zero exit code is an
          answer, not a failure; a compiler that found errors has worked.
        - **timed out** — the budget killed the command. Also an answer, and
          rendered as one, with the exit code ``timeout(1)`` uses.
        - an **error** — the backend raised, the allowlist refused the binary,
          the quotes would not balance. Nothing ran, and the reason says why.

        Dispatched by name from ``Akgent.on_receive``, which is what a
        non-``Message`` payload delivered by ``ActorAddress.tell`` gets. It
        arrives through the mailbox exactly as it did from the sandbox actor, so
        this runs on the actor's own thread and the worker's does not touch a
        thing here. ``deliver`` and ``fail`` do the rest, so a report for a run
        that no longer holds the tree is handled in exactly one place
        (:meth:`_finish_run`) rather than here.

        A fourth case comes first and is not one of the three: the report of a
        run whose agent the liveness sweep dropped while it ran
        (:meth:`_discard_report`).

        Args:
            report: What the worker produced for one run.
        """
        if report.run_id == self._discarded_run:
            self._discard_report(report.run_id)
            return
        if report.error:
            self.fail(report.run_id, report.error)
            return
        if report.timed_out:
            budget = self._run_budget()
            self.deliver(
                report.run_id,
                ExecOutcome(
                    stdout="",
                    stderr=f"Command exceeded its {budget:g}s budget and was killed.",
                    exit_code=TIMED_OUT_EXIT_CODE,
                    timed_out=True,
                ),
            )
            return
        assert report.result is not None  # noqa: S101 — the model's validator guarantees it
        self.deliver(
            report.run_id,
            ExecOutcome(
                stdout=report.result.stdout,
                stderr=report.result.stderr,
                exit_code=report.result.exit_code,
            ),
        )

    def _discard_report(self, run_id: str) -> None:
        """Close out a swept agent's run without caching what it answered.

        **What "discarded" means: the report reaches the actor, and nothing
        about it is cached.** Neither ``deliver`` nor ``fail`` runs, so there is
        no ``_slots`` entry — ``get(run_id)`` is ``None`` and no LRU slot is
        spent — and no ``_run_errors`` entry; ``_in_flight`` is cleared here, as
        those two would have. Nobody could collect the outcome anyway: the sweep
        pruned the agent's ``_recent_runs``, so its polls answer ``UNKNOWN``, and
        a slot nobody can reach is a slot taken from a run somebody can.

        **What is not discarded is the tree's own record.** The run did write, so
        :meth:`_finish_run` still commits the discovered write set under the
        run's identity — by id, since the name was pruned with the holder — and
        still gives the tree back. Dropping that commit would sweep
        the run's files into the next agent's discovery, the misattribution the
        out-of-band commit exists to prevent.

        **The mark, not a membership test.** The sweep sets ``_discarded_run``
        for the run holding the tree when its holder was dropped; testing
        ``agent_id not in self._holders`` here instead would discard every run
        started under an id that never attached. One scalar, overwritten by a
        later sweep that marks a newer run: an earlier marked run reporting after
        that is then delivered into a slot nobody collects — the cost a late
        report of a lease-released run already has.

        Args:
            run_id: The marked run, now reporting.
        """
        self._discarded_run = None
        self._in_flight.discard(run_id)
        logger.info(
            "Workspace %s: run %s reported after its agent was swept — outcome discarded",
            self.config.workspace_path,
            run_id,
        )
        self._finish_run(run_id)

    def _drop_runs_of(self, agent_ids: list[str]) -> None:
        """Prune the runs of agents the liveness sweep dropped; start none of theirs.

        For each agent: its ``_recent_runs`` go, and with them every run id they
        held out of ``_run_errors``, which is keyed by run id and reachable only
        through them. Its **running** run is not killed: it is marked, completes
        on its budget, and :meth:`_discard_report` closes it out when it reports
        — which is also what gives the tree back, so a swept agent's run releases
        the hold exactly as a live agent's does. Nothing here kills anything;
        teardown owns the one kill path.

        **Its ``_slots`` results are left to the LRU.** The deferred base has no
        delete beyond expiry, and the entries are uncollectable once
        ``_recent_runs`` is gone. That is bounded — ``MAX_TRACKED_RUNS`` per agent
        against ``cache_capacity`` in total — and widening a base three packages
        share to reclaim an already-capped cost is not worth it.

        Args:
            agent_ids: The agents the sweep just dropped. Empty is a no-op.
        """
        if not agent_ids:
            return
        dropped = set(agent_ids)
        for agent_id in dropped:
            for run_id in self._recent_runs.pop(agent_id, {}):
                self._run_errors.pop(run_id, None)
        running = self._running
        if running is not None and running.agent_id in dropped:
            self._discarded_run = running.run_id

    def _run_budget(self) -> float:
        """The effective budget a run gets, from the bound card's configuration.

        One derivation, read by the timeout message and by the release predicate,
        so the number an agent is told its command exceeded is the number the
        gate measured it against. ``configure_exec`` is last-writer-wins, and two
        exec-capable cards over one tree must agree on the backend anyway, so a
        configuration replaced mid-run is not a case this guards.
        """
        config = self._exec_config
        return effective_budget(config.timeout_s if config is not None else DEFAULT_EXEC_TIMEOUT_S)

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

        **The ownership gate is first, ahead of every other branch**, and that
        ordering is the whole of it: a run absent from the asker's own tracking
        map is answered ``UNKNOWN`` whether it is running, done or failed for
        somebody else, because a later branch reached first would answer a
        foreign run with a foreign result. It is defence in depth — no refusal
        publishes another agent's id any more — and no new state is introduced,
        for the reason this method's own docstring gives about evicted results.

        The tracking map is capped at ``MAX_TRACKED_RUNS`` per agent, so making
        collection depend on it means an agent past its 33rd run can no longer
        collect its own oldest. That is accepted: a 33-runs-ago id is not
        something a model holds, and the answer is a recoverable ``UNKNOWN``
        rather than an error. A second, uncapped map keyed by run would leak for
        the life of the team.

        Args:
            agent_id: Identity of the asking agent, as a string.
            run_id: The run to report on.

        Returns:
            Done with the outcome and the command, failed with the reason,
            running, or unknown with this agent's recent run ids.
        """
        # Housekeeping first, ahead of even the ownership gate, and it is what
        # makes the release path reachable at all: this actor is passive and
        # nothing releases on a timer, so a run that will never report would
        # otherwise hold the mutation gate until some unrelated later message
        # arrived. It no longer drains anything — there is nothing to drain —
        # but it is still what releases a wedged run so mutations resume.
        self._holding_run()
        runs: OrderedDict[str, str] = self._recent_runs.get(agent_id, OrderedDict())
        if run_id not in runs:
            return ExecStatus(state=ExecState.UNKNOWN, run_id=run_id, recent_run_ids=list(runs))
        outcome = self.get(run_id)
        if outcome is not None:
            return ExecStatus(
                state=ExecState.DONE, run_id=run_id, outcome=outcome, command=runs[run_id]
            )
        error = self._run_errors.get(run_id)
        if error is not None:
            return ExecStatus(state=ExecState.FAILED, run_id=run_id, reason=error)
        if run_id in self._in_flight:
            return ExecStatus(state=ExecState.RUNNING, run_id=run_id)
        return ExecStatus(state=ExecState.UNKNOWN, run_id=run_id, recent_run_ids=list(runs))

    def deliver(self, key: str, value: ExecOutcome) -> None:
        """TELL, from the sandbox, through :meth:`receiveMsg_ExecReport`. Close the run out."""
        super().deliver(key, value)
        self._finish_run(key)

    def fail(self, key: str, error: str) -> None:
        """TELL, from the sandbox or from a send that never happened. Close the run out.

        The base caches negatively with a TTL, which is what stops a broken
        backend from being retried once per poll. The reason is kept here as
        well because the base deliberately does not expose it — ``get`` answers
        ``None`` for a failure exactly as it does for an unknown key, and a run
        that failed must never be reported as still running.
        """
        self._record_failure(key, error)
        self._finish_run(key)

    def _record_failure(self, key: str, error: str) -> None:
        """Record *key*'s failure without touching the tree or the gate.

        Split out of :meth:`fail` because the two callers want different halves.
        A reported failure is a run closing out, so it commits what it wrote and
        hands on the tree. A run abandoned because its sandbox died is not: there
        is no report, nothing is known about what it left behind, and committing
        it as that agent's work would attribute a half-written tree to somebody
        who never saw it finish.
        """
        super().fail(key, error)
        self._run_errors[key] = error
        self._run_errors.move_to_end(key)
        while len(self._run_errors) > self.cache_capacity:
            self._run_errors.popitem(last=False)

    def _finish_run(self, run_id: str) -> None:
        """Commit what the run produced and hand the tree on.

        **Only a report from the run that is holding the tree closes it out.** A
        report from a run whose hold was already released must not clear a newer
        agent's — and must not commit as its own agent either, since by then the
        tree may hold somebody else's accepted mutations. Its outcome is still
        cached by the caller above, so its owner can still collect it; what it
        does not do is touch the tree.

        Whatever such a run left behind is committed out of band by the next
        mutation or the next run's :meth:`_start_run`, so nothing is swept into
        another agent's discovery.

        **The release happens last, after the commit**, and with a file lock that
        ordering is sharper than it was with a queue. A run admitted before this
        run's write set was committed would have its own
        ``git status --porcelain -uall`` discovery sweep up this run's files —
        the misattribution the out-of-band commit exists to prevent, one step
        further along — and the next acquirer may now be in **another process**,
        so an early release hands an uncommitted tree to a discovery this process
        cannot see coming.
        """
        running = self._running
        if running is None or running.run_id != run_id:
            logger.warning(
                "Workspace %s: run %s reported after the workspace was handed on. Its outcome "
                "is still collectable by its owner; anything it wrote belongs to nobody and is "
                "committed out of band by the next mutation or run.",
                self.config.workspace_path,
                run_id,
            )
            return
        self._running = None
        self._journal.commit_discovered(
            self._identity(running.agent_id), EXEC_CAPABILITY, detail=running.cmd
        )
        self._release_lock(run_id)

    def _release_lock(self, run_id: str) -> None:
        """Give the tree back for *run_id*, swallowing whatever the backend raises.

        **The single point every release converges on** — :meth:`_finish_run`,
        :meth:`_holding_run` and teardown all come here, and none of them may
        inline ``self._lock.release(...)`` instead. A fourth release path is a
        fourth place to forget the log line or the swallow.

        Nothing here may raise: one caller is a run's own answer, one is a
        predicate a mutation decides over, and one is inside ``on_stop``. A
        filesystem error on the way out is a marker left for staleness to
        reclaim, which is a bounded cost; a raise on any of those three paths is
        not.

        Args:
            run_id: The run giving the tree back. A backend that finds a
                different run in the marker leaves it alone — this is a request,
                not an eviction.
        """
        lock = self._lock
        if lock is None:
            return
        try:
            lock.release(self.config.workspace_path, run_id)
        except Exception:
            logger.warning(
                "Workspace %s: releasing the exec lock for run %s raised — swallowing. The "
                "marker is left for staleness to reclaim.",
                self.config.workspace_path,
                run_id,
                exc_info=True,
            )

    def _holding_run(self) -> RunningExec | None:
        """Return the run genuinely holding the tree, releasing one that is wedged.

        **The one place that decides whether a mutation may proceed**, and it is
        a predicate rather than a message because its callers need the decision
        and only one of them needs words. Exec no longer decides here — the lock
        does — but exec still *calls* it, for the release below.

        **The release covers one case: a child that ignores the kill.** Every
        other exit reports, because the worker reports in a ``finally`` — a
        command that ran, one the budget killed, a backend that raised, an
        allowlist refusal, and now also a callable that raised on its way in. A
        subprocess still alive past its budget is the one thing no report can
        cover, since the thread waiting on it is not free to say so. By
        ``budget + LEASE_GRACE_S`` the backend has killed the child, so this is
        not a race against a live writer.

        **The release covers the marker as well as the record, and both halves
        are owed.** Clearing ``_running`` alone would let mutations resume while
        the tree stayed locked on disk until the marker's own staleness — so the
        two clocks, which agree by construction, would stop agreeing on exactly
        the path that made one of them fire. Releasing here is what lets the next
        ``request_exec`` on this actor be granted at once.

        **No liveness check anywhere any more.** There is no second actor to
        die: a callable that raises is caught in :meth:`ExecRunner.perform` and
        reported, a submit that raises is caught in :meth:`_start_run` and
        reported, and a callable that never returns is exactly the wedge this
        timestamp releases.

        Returns:
            The run holding the tree, or ``None`` when the tree is free — either
            because nothing was running or because a wedged run was just
            released.
        """
        running = self._running
        if running is None:
            return None
        if time.monotonic() <= running.started_at + self._run_budget() + LEASE_GRACE_S:
            return running
        logger.warning(
            "Workspace %s: releasing the workspace from run %s (agent %s, command %r) — it is "
            "past its budget and the grace with nothing reported, so its command is not going "
            "to answer. Anything it is still writing will land in a later commit.",
            self.config.workspace_path,
            running.run_id,
            self._name_of(running.agent_id),
            running.cmd,
        )
        self._running = None
        self._release_lock(running.run_id)
        return None

    ##
    ## Teardown — four steps, in one order, each independently wrapped
    ##
    def _teardown_exec(self) -> None:
        """Take the worker down in the one order that leaves nothing running.

        Called by ``WorkspaceActor.on_stop`` before it chains to the base, and
        split out of it so ``on_stop`` stays short and exec teardown stays beside
        the exec code.

        The four steps, and why they are in this order:

        1. **give the tree back.** A team that stops mid-run must not leave its
           marker behind: the next acquirer may be in another process, and
           without this it would wait out the staleness window for a run that
           ended the moment this actor did. It is first because it is the step
           that outlives this process — the three below only tidy up inside it.
        2. **kill the running subprocess.** This is what makes the drain that
           follows cheap: after the kill a healthy child dies in milliseconds. It
           is also the only step that stops a command from outliving the team's
           teardown, so it must be *attempted*, never assumed.
        3. **drain the executor**, bounded (see :meth:`_drain_executor`).
        4. **release the backend.** Last, because for docker it is the only thing
           that ends the process *inside* the container, so it must still happen
           when step 3 gave up — see :meth:`ExecRunner.stop`.

        **Every step is wrapped separately**, and it matters most for the new
        first one: a release that raised would otherwise skip the kill, the drain
        and the backend release, leaving a live child, an undrained executor and
        a container up.

        **A workspace with no exec capability never built a runner**, and no
        branch of this is a special case for it: steps 2 and 4 are skipped, step
        1 finds no run to release, and step 3 shuts down an executor that never
        spawned a thread.
        """
        runner = self._runner
        self._teardown_step("releasing the exec lock", self._release_running_lock)
        if runner is not None:
            self._teardown_step("killing the running command", runner.kill)
        self._teardown_step("draining the exec worker", self._drain_executor)
        self._teardown_step("releasing the exec backend", self._stop_runner)

    def _release_running_lock(self) -> None:
        """Give the tree back for whatever run this actor is holding, if any.

        Teardown's own entry into :meth:`_release_lock`, and it goes through the
        predicate rather than the attribute for nothing at all — ``_holding_run``
        would *itself* release a wedged run and answer ``None``, which is the one
        case teardown most needs released. So the attribute is read directly and
        the release is unconditional on the run's age: at teardown a run that is
        healthy and a run that is wedged both end here.
        """
        running = self._running
        if running is None:
            return
        self._running = None
        self._release_lock(running.run_id)

    def _teardown_step(self, what: str, step: Callable[[], None]) -> None:
        """Run one teardown step, swallowing whatever it raises.

        Nothing here may raise past ``super().on_stop()``: leaving a Pykka actor
        part-way stopped is worse than any error a step could report. Wrapping
        each step separately rather than the group is what keeps a failing one
        from skipping the ones after it.
        """
        try:
            step()
        except Exception:
            logger.warning(
                "Workspace %s: %s raised during on_stop — swallowing",
                self.config.workspace_path,
                what,
                exc_info=True,
            )

    def _drain_executor(self) -> None:
        """Wait a bounded time for the killed run, then shut the executor down.

        **``shutdown(wait=True)`` would be unbounded and the case is reachable**,
        so the wait is on the ``Future`` :meth:`_start_run` kept instead —
        ``ThreadPoolExecutor.shutdown`` takes no timeout, and a worker wedged in
        ``ProcessBackend._run``'s second drain never returns. Only the head is
        ever submitted, so there is at most one future to wait on.

        Past the grace the executor is shut down **without** waiting and with the
        queued futures cancelled, so a worker that ignored the kill costs
        :data:`EXEC_SHUTDOWN_GRACE_S` of teardown latency rather than holding
        teardown open for ever.

        **What the bound does not reach is interpreter exit**, and the boundary
        is worth knowing before somebody reads this as a fix. A pool's worker
        threads are non-daemon and stay registered for the whole process;
        ``shutdown(wait=False)`` abandons the wait here but deregisters nothing,
        and the interpreter joins every one of them on its way out. So a worker
        genuinely wedged in that second drain still stops **this actor** in
        bounded time and still blocks the **process** from exiting. The ordinary
        wedge — a shell's forked command holding the pipes — is closed by the
        backend's group kill on ``local`` and ``bwrap``; what can still wedge
        the worker is a process that left the group on its own, or docker's
        in-container process, which only ``stop()``'s removal ends.
        """
        pending = self._pending
        if pending is not None:
            futures.wait([pending], timeout=EXEC_SHUTDOWN_GRACE_S)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _busy_refusal(self) -> str | None:
        """Refuse a **mutation** while a run holds the tree, or allow it.

        Fail fast, never stall. Ten seconds of silence inside a tool call is
        indistinguishable from a hang and gives the model nothing to react to; an
        immediate refusal naming the holder lets it read a file, answer the user,
        or ask the holder. That is only affordable because the actor's thread is
        free — the blocking call is on the sandbox's.

        **This is the mutation message, and it is not the exec one.** Both
        refusals now exist and they are deliberately different: this one names
        the holder's run id and agent, and the exec refusal
        (:func:`~akgentic.tool.workspace.execution.exec_busy`) names nobody.

        Naming the id here is safe, and always was: it is uncollectable by
        anyone but its owner (see :meth:`exec_status`), so it informs a human
        reading the transcript without handing the model something to
        mis-collect. What ADR-047 removed was an *exec* refusal publishing a
        sibling call's id, which the model then collected as its own answer —
        and a refused exec caller, unlike a refused mutation, has no id of its
        own to be given instead. The exec refusal keeps naming nobody for that
        reason, and could not name the holder anyway: it is rendered under a
        ``LockBackend``, which has no access to ``_name_of``.

        **One wording, over the same predicate the exec path decides on.** There
        is no second message for a run past its budget: past the budget and the
        grace the run no longer holds the tree at all — :meth:`_holding_run`
        releases it — so the state that message described is not one a mutation
        can arrive in any more.

        Returns:
            The refusal text, or ``None`` when the tree is free.
        """
        running = self._holding_run()
        if running is None:
            return None
        return (
            f"{_BUSY_PREFIX} — exec run {running.run_id} is in progress "
            f"(agent '{self._name_of(running.agent_id)}'). Reads still work; retry the change "
            f"once the run has finished."
        )

    def _track_run(self, agent_id: str, run_id: str, cmd: str) -> None:
        """Remember *run_id* as one of *agent_id*'s recent runs, with its command.

        The map does two jobs, which is why the value slot stopped repeating the
        key: it is the **ownership record** :meth:`exec_status` gates on, and the
        source of the command a ``DONE`` result names itself with.

        Called the moment a grant issues an id, because a run is owned from the
        moment its id exists, and an id issued but untracked would be one its own
        requester could not collect.

        Capped for the reason every map on a tree singleton is: an uncapped one
        leaks for the life of the tree. Losing the oldest entry now costs the
        ability to collect that run as well as the ability to correct a mistyped
        id, and that is accepted — the answer is a recoverable ``UNKNOWN``.

        Args:
            agent_id: Identity of the requesting agent, as a string.
            run_id: The issued id.
            cmd: The command string, exactly as the agent gave it.
        """
        runs = self._recent_runs.setdefault(agent_id, OrderedDict())
        runs[run_id] = cmd
        runs.move_to_end(run_id)
        while len(runs) > MAX_TRACKED_RUNS:
            runs.popitem(last=False)
