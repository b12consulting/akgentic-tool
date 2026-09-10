"""The exec lease, the run bookkeeping, and the deferred surface (ADR-045 §1).

**Exec is fenced, not gated, and that is the whole difference.** Every other
writer says what it is about to do, so the gate can check a precondition against
the file it names. A shell command cannot, so ``workspace_exec`` takes an
exclusive lease over the tree instead, and its write set is *discovered*
afterwards from ``git status --porcelain -uall`` (ADR-036 §5).

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
from collections import OrderedDict, deque
from concurrent import futures
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from akgentic.tool.core.deferred import DeferredResultActor
from akgentic.tool.sandbox.backend import ExecReport
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_TIMEOUT_S,
    EXEC_SHUTDOWN_GRACE_S,
    LEASE_GRACE_S,
    MAX_QUEUED_RUNS,
    MAX_TRACKED_RUNS,
    TIMED_OUT_EXIT_CODE,
    ExecConfig,
    ExecOutcome,
    ExecRunner,
    ExecStart,
    ExecState,
    ExecStatus,
    QueuedExec,
    RunningExec,
    effective_budget,
    new_run_id,
    queue_full,
    resolve_mode,
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
    """The lease, the run bookkeeping, the worker thread and its ordered teardown."""

    _exec_config: ExecConfig | None
    _runner: ExecRunner | None
    _executor: ThreadPoolExecutor
    _pending: Future[None] | None
    _running: RunningExec | None
    _queue: deque[QueuedExec]
    _run_errors: OrderedDict[str, str]
    _recent_runs: dict[str, OrderedDict[str, str]]
    _discarded_run: str | None
    _journal: GitJournal

    if TYPE_CHECKING:
        # Supplied by ``ObservationMixin``; the MRO binds them at runtime.
        def _identity(self, agent_id: str) -> Identity: ...

        def _name_of(self, agent_id: str) -> str: ...

    ##
    ## Exec — the queue, the running run, and the discovered commit
    ##
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
        """Start the run, or queue it — in one mailbox turn, under the caller's own id.

        Admission has three answers and **the run id is issued in two of them**,
        which is the load-bearing detail. A caller leaves here holding a handle
        to its *own* work whether the tree was free or not, so no message this
        path produces can name a run belonging to another agent — the defect a
        model's parallel batch of ``workspace_exec`` calls used to reproduce
        every time, by reading a sibling call's id out of a refusal and
        collecting it.

        Everything here is O(1) plus the journal's bounded git calls. **No
        sandbox call happens on this thread**: that is what keeps the actor's
        mailbox draining, which is what makes reads work during a run and a
        refused mutation cost one turn instead of the run's whole duration.

        The decision comes from :meth:`_holding_run` rather than from
        ``self._running``, so a run wedged past its budget releases the tree here
        exactly as it does on the mutation path. Reading the attribute directly
        would park every subsequent run behind a run nobody will ever release,
        for the life of the team.

        Args:
            agent_id: Identity of the requesting agent, as a string.
            cmd: The command string.
            cwd: Working directory below the workspace root.

        Returns:
            The issued run id — for an accepted run and a queued one alike — or
            the one refusal left, which names nobody.
        """
        config = self._exec_config
        if config is None:
            return ExecStart(refusal=unconfigured())
        entry = QueuedExec(run_id=new_run_id(), agent_id=agent_id, cmd=cmd, cwd=cwd)
        if self._holding_run() is not None:
            if len(self._queue) >= MAX_QUEUED_RUNS:
                return ExecStart(refusal=queue_full())
            self._queue.append(entry)
            self._track_run(agent_id, entry.run_id, cmd)
            return ExecStart(run_id=entry.run_id)
        self._track_run(agent_id, entry.run_id, cmd)
        self._start_run(entry, config)
        return ExecStart(run_id=entry.run_id)

    def _start_run(self, entry: QueuedExec, config: ExecConfig) -> None:
        """Take the tree for *entry* and submit it to the worker — the one accept path.

        Reached from :meth:`request_exec` over a free tree and from
        :meth:`_start_next` when the head releases it, and there is deliberately
        only one of it: the out-of-band commit, the hold and the submit have to
        happen together and in this order, and a second copy is where the two
        paths would drift.

        **The out-of-band commit matters on the dequeue path too**, which is the
        non-obvious half. A dequeue that follows a release reaches a tree nobody
        committed, so anything already lying in it would be swept into this run's
        discovered commit and attributed to an agent that never wrote it.

        **The hold is taken before the submit**, and that ordering is
        load-bearing: a submit that raises reports through :meth:`fail`
        *synchronously*, on this thread, and that has to find a run to release or
        the queue behind it never drains. A ``submit`` onto an executor that has
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
            entry: The run to start.
            config: The budget to start it under.
        """
        # The tree's existing dirt belongs to nobody, and must not end up inside
        # this run's discovered commit — which is exactly what would happen,
        # since that commit takes whatever the tree shows afterwards.
        self._journal.commit_out_of_band()
        self._running = RunningExec(
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            cmd=entry.cmd,
            started_at=time.monotonic(),
        )
        # What ``request()`` used to do, and ``exec_status`` still answers
        # RUNNING from it. The base's deliver/fail clear it, unchanged.
        self._in_flight.add(entry.run_id)
        try:
            runner = self._runner
            if runner is None:
                raise RuntimeError("#Workspace has no execution backend to run this command on.")
            self._pending = self._executor.submit(
                runner.perform,
                run_id=entry.run_id,
                cmd=entry.cmd,
                cwd=entry.cwd,
                timeout_s=effective_budget(config.timeout_s),
                reply_to=self.myAddress,
            )
        except Exception as exc:  # noqa: BLE001 — every failure here is the run's answer
            logger.warning(
                "Workspace %s: run %s never reached the worker: %r",
                self.config.workspace_path,
                entry.run_id,
                exc,
                exc_info=True,
            )
            self.fail(entry.run_id, f"The command was never handed to the sandbox: {exc}")

    def _start_next(self) -> None:
        """Hand the freed tree to the queue head, if anything is waiting.

        Called from every place the tree is released and from nowhere else:
        :meth:`_finish_run` on a report (both exits — ``deliver`` and ``fail``
        alike, so a run that failed drains the queue exactly as a run that
        succeeded does), and the one release path that needs no report,
        :meth:`_holding_run` for a wedged child.

        A workspace whose exec configuration was never announced cannot start
        anything; the entries stay queued rather than being silently dropped,
        and :meth:`on_stop` clears them. In practice the state is unreachable —
        nothing can be queued before an exec-capable card has bound.
        """
        config = self._exec_config
        if not self._queue or config is None:
            return
        self._start_run(self._queue.popleft(), config)

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
        still hands the tree to the queue head. Dropping that commit would sweep
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
        through them. Its queued runs leave the queue — a shell nobody is
        waiting for must not start. Its **running** run is not killed: it is
        marked, completes on its budget, and :meth:`_discard_report` closes it
        out when it reports. Nothing here kills anything; teardown owns the one
        kill path.

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
        # A filter over the existing entries, in place: no ``QueuedExec`` is
        # rebuilt, so there is no field list here to fall out of date.
        kept = [entry for entry in self._queue if entry.agent_id not in dropped]
        self._queue.clear()
        self._queue.extend(kept)
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
        map is answered ``UNKNOWN`` whether it is queued, running, done or failed
        for somebody else, because a later branch reached first would answer a
        foreign run with a foreign result. It is defence in depth — with the
        queue in place nothing publishes another agent's id any more — and no
        new state is introduced, for the reason this method's own docstring
        gives about evicted results.

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
            queued with its place in the FIFO, running, or unknown with this
            agent's recent run ids.
        """
        # Housekeeping first, ahead of even the ownership gate, and it is what
        # makes the release path reachable at all: this actor is passive,
        # nothing releases on a timer, and a head that will never report with
        # entries behind it would otherwise wait for some unrelated later
        # message. The message that is *guaranteed* to arrive is this one —
        # every queued caller polls its own run by construction.
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
        position = self._queue_position(run_id)
        if position:
            return ExecStatus(state=ExecState.QUEUED, run_id=run_id, queue_position=position)
        if run_id in self._in_flight:
            return ExecStatus(state=ExecState.RUNNING, run_id=run_id)
        return ExecStatus(state=ExecState.UNKNOWN, run_id=run_id, recent_run_ids=list(runs))

    def _queue_position(self, run_id: str) -> int:
        """Return *run_id*'s 1-based place in the FIFO, or ``0`` if it is not in it.

        1-based rather than 0-based so the number reads as an answer: position
        ``1`` is "next to run when the head finishes". The head itself holds the
        tree and is not in the queue, which is what leaves ``0`` free to mean
        "not queued at all".
        """
        for index, entry in enumerate(self._queue, start=1):
            if entry.run_id == run_id:
                return index
        return 0

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

        **The dequeue happens last, after the commit.** A queued run started
        before this run's write set was committed would have its own discovery
        sweep up the head's files, which is the same misattribution the
        out-of-band commit exists to prevent — one step further along.
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
        self._start_next()

    def _holding_run(self) -> RunningExec | None:
        """Return the run genuinely holding the tree, releasing one that is wedged.

        **The one place that decides whether the tree is taken**, and it is a
        predicate rather than a message because both callers need the decision
        and only one of them needs words: a mutation renders it as a refusal, an
        exec request queues behind it. Splitting them is what keeps the two from
        diverging — an exec path that tested ``self._running is not None`` would
        never release, so one wedged run would park every subsequent one in the
        queue for the life of the team.

        **The release covers one case: a child that ignores the kill.** Every
        other exit reports, because the worker reports in a ``finally`` — a
        command that ran, one the budget killed, a backend that raised, an
        allowlist refusal, and now also a callable that raised on its way in. A
        subprocess still alive past its budget is the one thing no report can
        cover, since the thread waiting on it is not free to say so. By
        ``budget + LEASE_GRACE_S`` the backend has killed the child, so this is
        not a race against a live writer.

        **A release drains the queue, and that is what stops a newcomer
        overtaking.** Handing back the tree without starting the head would leave
        the queued entries with nothing scheduled to run them — they would wait
        for some unrelated later request — and worse, that later request would
        find the tree free and start *itself*, jumping ahead of everything
        already waiting. FIFO would break on exactly the path the release
        created. So the head is started here and this method answers with the run
        **it** now holds; only a genuinely empty queue answers ``None``.

        **No liveness check anywhere any more.** There is no second actor to
        die: a callable that raises is caught in :meth:`ExecRunner.perform` and
        reported, a submit that raises is caught in :meth:`_start_run` and
        reported, and a callable that never returns is exactly the wedge this
        timestamp releases.

        Returns:
            The run holding the tree — the original, or the queue head's after a
            release drained into it — or ``None`` when the tree is free and
            nothing was waiting for it.
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
        self._start_next()
        return self._running

    ##
    ## Teardown — four steps, in one order, each independently wrapped
    ##
    def _teardown_exec(self) -> None:
        """Take the worker down in the one order that leaves nothing running.

        Called by ``WorkspaceActor.on_stop`` before it chains to the base, and
        split out of it so ``on_stop`` stays short and exec teardown stays beside
        the exec code.

        The four steps, and why they are in this order:

        1. **cancel the queued runs.** They never started, were never submitted
           anywhere and produced nothing, so dropping them is free.
        2. **kill the running subprocess.** This is what makes the drain that
           follows cheap: after the kill a healthy child dies in milliseconds. It
           is also the only step that stops a command from outliving the team's
           teardown, so it must be *attempted*, never assumed.
        3. **drain the executor**, bounded (see :meth:`_drain_executor`).
        4. **release the backend.** Last, because for docker it is the only thing
           that ends the process *inside* the container, so it must still happen
           when step 3 gave up — see :meth:`ExecRunner.stop`.

        **Every step is wrapped separately**, which is stricter than the single
        wrapper the queue clear used to have and is the reason for the change: a
        ``kill()`` that raised would otherwise skip the drain and the release,
        leaving the executor undrained and the container up.

        **A workspace with no exec capability never built a runner**, and no
        branch of this is a special case for it: steps 2 and 4 are skipped, step
        1 clears an empty deque, and step 3 shuts down an executor that never
        spawned a thread.
        """
        runner = self._runner
        self._teardown_step("clearing the exec queue", self._queue.clear)
        if runner is not None:
            self._teardown_step("killing the running command", runner.kill)
        self._teardown_step("draining the exec worker", self._drain_executor)
        self._teardown_step("releasing the exec backend", self._stop_runner)

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

        **This is a mutation message and no longer an exec one**, and the
        asymmetry is deliberate: mutations are *gated*, so their refusal is a
        precondition failure that is cheap to re-issue, while exec is *fenced*
        and its work would be thrown away. So exec queues and mutations still
        refuse — which is also why naming the holder's run id here is safe.
        That id is uncollectable by anyone but its owner (see
        :meth:`exec_status`), so it informs a human reading the transcript
        without handing the model something to mis-collect.

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

        Called in both admission branches — accepted and queued — because a run
        is owned from the moment its id is issued, and an id issued but untracked
        would be one its own requester could not collect.

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
