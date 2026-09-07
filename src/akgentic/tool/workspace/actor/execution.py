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
from collections import OrderedDict, deque
from typing import TYPE_CHECKING

from akgentic.core.actor_address import ActorAddress
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core.deferred import DeferredResultActor
from akgentic.tool.sandbox.actor import ExecReport, ExecRequest
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_TIMEOUT_S,
    LEASE_GRACE_S,
    MAX_QUEUED_RUNS,
    MAX_TRACKED_RUNS,
    SANDBOX_RESOLVE_TIMEOUT_S,
    TIMED_OUT_EXIT_CODE,
    ExecConfig,
    ExecOutcome,
    ExecStart,
    ExecState,
    ExecStatus,
    QueuedExec,
    RunningExec,
    effective_budget,
    new_run_id,
    queue_full,
    sandbox_config,
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
    _running: RunningExec | None
    _queue: deque[QueuedExec]
    _run_errors: OrderedDict[str, str]
    _recent_runs: dict[str, OrderedDict[str, str]]
    _journal: GitJournal

    if TYPE_CHECKING:
        # Supplied by ``ObservationMixin``; the MRO binds them at runtime.
        def _identity(self, agent_id: str) -> Identity: ...

        def _name_of(self, agent_id: str) -> str: ...

    ##
    ## Exec — the queue, the running run, and the discovered commit
    ##
    def configure_exec(self, config: ExecConfig) -> None:
        """Record which backend to run commands on — **tell** path, once per card.

        The actor cannot take this from :class:`WorkspaceConfig`, because
        ``getChildrenOrCreate`` fixes that at creation and the card that creates
        the actor for a workspace is routinely one with no exec capability at
        all. So an exec-capable card announces itself here instead, at bind time,
        exactly as :meth:`register_agent` does.

        Last writer wins, and that is correct: two exec-capable cards over one
        tree must agree on the backend anyway, since they share one
        ``#SandboxActor-<workspace>`` — the sandbox actor is named per workspace,
        for the same reason this one is.

        Args:
            config: The resolved backend and the ids to build payloads from.
        """
        self._exec_config = config

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
        """Take the tree for *entry* and send it to the sandbox — the one accept path.

        Reached from :meth:`request_exec` over a free tree and from
        :meth:`_start_next` when the head releases it, and there is deliberately
        only one of it: the out-of-band commit, the hold and the send have to
        happen together and in this order, and a second copy is where the two
        paths would drift.

        **The out-of-band commit matters on the dequeue path too**, which is the
        non-obvious half. A dequeue that follows a release reaches a tree nobody
        committed, so anything already lying in it would be swept into this run's
        discovered commit and attributed to an agent that never wrote it.

        **The hold is taken before the send**, and that ordering is load-bearing:
        a resolve or a send that raises reports through :meth:`fail`
        *synchronously*, on this thread, and that has to find a run to release or
        the queue behind it never drains.

        **The send is** ``ActorAddress.tell``, **never a tell proxy.** A tell
        proxy's calls carry ask semantics underneath, so a dead sandbox's
        ``ActorDeadError`` is set on a future the proxy drops — the send vanishes
        with the run still marked running, and only a much later poll would
        notice. The address checks liveness and raises here, where the failure is
        an answer.

        Args:
            entry: The run to start.
            config: The backend to start it on.
        """
        # The tree's existing dirt belongs to nobody, and must not end up inside
        # this run's discovered commit — which is exactly what would happen,
        # since that commit takes whatever the tree shows afterwards.
        self._journal.commit_out_of_band()
        running = RunningExec(
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            cmd=entry.cmd,
            started_at=time.monotonic(),
        )
        self._running = running
        # What ``request()`` used to do, and ``exec_status`` still answers
        # RUNNING from it. The base's deliver/fail clear it, unchanged.
        self._in_flight.add(entry.run_id)
        try:
            address = self._resolve_sandbox(config)
            running.attach(address)
            address.tell(
                ExecRequest(
                    run_id=entry.run_id,
                    cmd=entry.cmd,
                    cwd=entry.cwd,
                    timeout_s=effective_budget(config.timeout_s),
                    reply_to=self.myAddress,
                )
            )
        except Exception as exc:  # noqa: BLE001 — every failure here is the run's answer
            logger.warning(
                "Workspace %s: run %s never reached the sandbox: %r",
                self.config.workspace_name,
                entry.run_id,
                exc,
                exc_info=True,
            )
            self.fail(entry.run_id, f"The command was never handed to the sandbox: {exc}")

    def _resolve_sandbox(self, config: ExecConfig) -> ActorAddress:
        """Get-or-create ``#SandboxActor-<workspace>`` and return its address.

        Idempotent by construction (ADR-025): the card already created it at
        wiring time, so this ordinarily resolves the existing one. It is also
        what recreates it after a crash — the orchestrator skips a child that is
        no longer alive, so the next run gets a fresh sandbox without anything
        here having to notice.

        The class comes from ``SANDBOX_ACTOR_CLASSES`` at call time, never at
        import time, so a backend injected by a deployment package is still
        found.

        **The ask carries a timeout**, because it is made on the team singleton's
        own thread and everything else queued behind it waits for it. It returns
        before a cold backend's ``on_start`` has finished provisioning — Pykka
        starts the actor's thread and returns — so what is being waited on is one
        orchestrator turn, not a container build.

        Args:
            config: The card's resolved backend and ids.

        Returns:
            The sandbox actor's address.

        Raises:
            RuntimeError: If this actor has no orchestrator to resolve through.
        """
        from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES  # noqa: PLC0415 — cycle

        orchestrator = self.orchestrator
        if orchestrator is None:
            raise RuntimeError("#Workspace cannot resolve its sandbox without an orchestrator.")
        orchestrator_proxy = self.proxy_ask(
            orchestrator, Orchestrator, timeout=SANDBOX_RESOLVE_TIMEOUT_S
        )
        address: ActorAddress = orchestrator_proxy.getChildrenOrCreate(
            SANDBOX_ACTOR_CLASSES[config.mode], config=sandbox_config(config)
        )
        return address

    def _start_next(self) -> None:
        """Hand the freed tree to the queue head, if anything is waiting.

        Called from every place the tree is released and from nowhere else:
        :meth:`_finish_run` on a report (both exits — ``deliver`` and ``fail``
        alike, so a run that failed drains the queue exactly as a run that
        succeeded does), and the two release paths that need no report,
        :meth:`_holding_run` for a wedged child and
        :meth:`_release_a_dead_sandbox` for a sandbox that stopped.

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
        """TELL, from the sandbox. Turn one report into the run's settled answer.

        The only thing that closes a run out on the ordinary path, and it has
        exactly three branches because the sandbox reports exactly three things.
        Two of them are *answers* the agent can read and one is a failure:

        - a **result** — whatever it exited with. A non-zero exit code is an
          answer, not a failure; a compiler that found errors has worked.
        - **timed out** — the budget killed the command. Also an answer, and
          rendered as one, with the exit code ``timeout(1)`` uses.
        - an **error** — the backend raised, the allowlist refused the binary,
          the quotes would not balance. Nothing ran, and the reason says why.

        Dispatched by name from ``Akgent.on_receive``, which is what a
        non-``Message`` payload delivered by ``ActorAddress.tell`` gets.
        ``deliver`` and ``fail`` do the rest, so a report for a run that no
        longer holds the tree is handled in exactly one place
        (:meth:`_finish_run`) rather than here.

        Args:
            report: What the sandbox produced for one run.
        """
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
        # makes both release paths reachable at all: this actor is passive,
        # nothing releases on a timer, and a head that will never report with
        # entries behind it would otherwise wait for some unrelated later
        # message. The message that is *guaranteed* to arrive is this one —
        # every queued caller polls its own run by construction.
        #
        # The liveness check lives HERE and only here. The mutation path needs
        # no address: a dead sandbox's run is released by the wedge timestamp
        # within budget + LEASE_GRACE_S anyway, and the queued callers' polls
        # catch it sooner than any mutation would. It costs a flag read, and
        # the start it can trigger is the same one ``request_exec`` performs on
        # the same thread.
        self._release_a_dead_sandbox()
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
                self.config.workspace_name,
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
        other exit reports, because the sandbox's handler reports in a
        ``finally`` — a command that ran, one the budget killed, a backend that
        raised, an allowlist refusal. A subprocess still alive past its budget is
        the one thing no report can cover, since the thread waiting on it is not
        free to say so. By ``budget + LEASE_GRACE_S`` the backend has killed the
        child, so this is not a race against a live writer.

        **A release drains the queue, and that is what stops a newcomer
        overtaking.** Handing back the tree without starting the head would leave
        the queued entries with nothing scheduled to run them — they would wait
        for some unrelated later request — and worse, that later request would
        find the tree free and start *itself*, jumping ahead of everything
        already waiting. FIFO would break on exactly the path the release
        created. So the head is started here and this method answers with the run
        **it** now holds; only a genuinely empty queue answers ``None``.

        **No liveness check here.** A sandbox that died is handled once, in
        :meth:`exec_status`, on the message that is guaranteed to arrive.

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
            self.config.workspace_name,
            running.run_id,
            self._name_of(running.agent_id),
            running.cmd,
        )
        self._running = None
        self._start_next()
        return self._running

    def _release_a_dead_sandbox(self) -> None:
        """Fail the running run when the sandbox performing it has stopped.

        A sandbox that dies mid-run takes the answer with it: no report is
        coming, and no send primitive can see it happen — the request was
        delivered to an actor that was alive at the time. So the run is recorded
        as failed with a reason naming the sandbox, which is an answer its owner
        can read, rather than being left to time out into silence.

        **Nothing is committed as the agent.** The run may have written half of
        something before the sandbox went, and attributing a half-written tree to
        the agent would put work in the journal under an author who never saw it
        finish. Whatever is there is committed out of band by the next
        :meth:`_start_run` or the next mutation, belonging to nobody.

        The next admission resolves a **new** sandbox: the orchestrator skips a
        child that is no longer alive, so ``getChildrenOrCreate`` creates one.
        """
        running = self._running
        if running is None:
            return
        sandbox = running.sandbox
        if sandbox is None or sandbox.is_alive():
            return
        reason = (
            f"The execution sandbox '{sandbox.name}' stopped while run {running.run_id} was "
            "running, so the command's outcome is lost. Nothing further is waiting on it — "
            "retry the command."
        )
        logger.warning(
            "Workspace %s: %s (agent %s, command %r)",
            self.config.workspace_name,
            reason,
            self._name_of(running.agent_id),
            running.cmd,
        )
        self._running = None
        self._record_failure(running.run_id, reason)
        self._start_next()

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

        Capped for the reason every map on a team singleton is: an uncapped one
        leaks for the life of the team. Losing the oldest entry now costs the
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
