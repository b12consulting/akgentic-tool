"""Exec factories for :class:`WorkspaceTool` — ``workspace_exec`` and its collector.

Move-only with respect to the sandbox: nothing behind the exec surface changes
here. The lease, the ``ExecWorker`` and the backend arrangement stay exactly
where ADR-017 and ADR-036 §5 put them.

**One directional sibling edge:** this module imports :func:`_bound` from
``card/write.py``. The helper is used by both the mutation factories and the
exec factories, and it belongs with the mutations; duplicating it would give the
gate two spellings, and hoisting it into ``card/__init__.py`` would make a mixin
module import the module that imports it — a cycle.

:class:`ExecFactories` is a **mixin**: it declares no Pydantic field, and the
two names it consumes off ``self`` are declared under ``if TYPE_CHECKING:`` so
they reach mypy without ever reaching Pydantic's field collection (ADR-045 §1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from akgentic.tool.core.deferred import DEFAULT_WORKER_TIMEOUT_S, poll_deferred
from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import WorkspaceActor
from akgentic.tool.workspace.card.params import WorkspaceExec
from akgentic.tool.workspace.card.write import _bound
from akgentic.tool.workspace.execution import (
    ExecStatus,
    format_status,
    in_progress,
    poll_attempts_within,
    timed_out,
)


def _settled_status(status: ExecStatus) -> ExecStatus | None:
    """Answer the poll only once there is something final to say.

    ``poll_deferred`` stops at the first non-``None``, so a fetch that answered
    with a *running* status would end the poll on its first attempt and hand the
    agent a run id it did not need. A failure, by contrast, is final — it is
    collected as a failure with its reason, never reported as still running.
    """
    return status if status.settled else None


class ExecFactories:
    """The exec factory bodies of :class:`WorkspaceTool`.

    Declares **no Pydantic field**: the annotations below are inside
    ``if TYPE_CHECKING:``, so they are never executed and never reach
    ``__annotations__``, which is where Pydantic v2 collects fields from across
    the MRO. mypy reads them normally.
    """

    if TYPE_CHECKING:
        _agent_id: str
        _workspace_proxy: WorkspaceActor | None

    def _exec_factory(self, params: WorkspaceExec) -> Callable[..., Any]:
        """Create the ``workspace_exec`` tool callable.

        Args:
            params: Exec capability configuration.

        Returns:
            Callable that runs a sandboxed command, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id
        delay = params.poll_delay_seconds
        # The effective run budget — the card's ask after the worker's ceiling —
        # is what actually stops the run, so it is what both the sentinel and the
        # clamp are measured against. A card asking for 999 s never gets more
        # than the worker allows it either way.
        run_budget = min(params.timeout_s, DEFAULT_WORKER_TIMEOUT_S)
        # Resolved once, here, so nothing downstream knows the sentinel existed.
        attempts = poll_attempts_within(params.poll_attempts, delay, run_budget)
        # Which budget was in force decides which exhaustion message is honest,
        # and it is a property of the card, not of the run — so it is computed
        # here rather than re-derived on every call.
        waits_out_the_run = params.poll_attempts < 0

        def workspace_exec(cmd: str, cwd: str = "") -> str:
            """Run a shell command in the team workspace, in a sandbox.

            This waits for the command and gives you its output. The workspace is
            held exclusively for the duration of the run: your teammates can still
            read files, but every change they attempt is refused until it
            finishes. Everything the command touched — files you never named
            included — is recorded as one change attributed to you.

            A run that outlives the wait is the exception, and then you get a run
            id instead of output; workspace_exec_result collects that run's output
            once it lands.

            Args:
                cmd: Full command string. The binary (first token) must be in
                    the allow-list.
                cwd: Subdirectory relative to workspace root. Defaults to root.

            Returns:
                Combined stdout, stderr and exit code — or, for a run that
                outlived the wait, a message naming the run id.

            Raises:
                RetriableError: If another agent's run holds the workspace.
            """
            start = _bound(proxy).request_exec(agent_id, cmd, cwd)
            if not start.run_id:
                raise RetriableError(start.refusal)
            run_id = start.run_id
            settled = poll_deferred(
                lambda: _settled_status(_bound(proxy).exec_status(agent_id, run_id)),
                attempts=attempts,
                delay=delay,
            )
            if settled is not None:
                return format_status(settled)
            if waits_out_the_run:
                return timed_out(run_id, run_budget)
            return in_progress(run_id)

        workspace_exec.__doc__ = params.format_docstring(workspace_exec.__doc__)
        return workspace_exec

    def _exec_result_factory(self, params: WorkspaceExec) -> Callable[..., Any]:
        """Create the ``workspace_exec_result`` tool callable.

        Args:
            params: Exec capability configuration.

        Returns:
            Callable that collects a finished run's output, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id

        def workspace_exec_result(run_id: str) -> str:
            """Collect the output of a command started by workspace_exec.

            Args:
                run_id: The id workspace_exec handed back.

            Returns:
                The command's output if it has finished, a note that it is still
                running, why it failed, or — for an id nothing was issued under —
                your recent run ids so you can retry with the right one.
            """
            return format_status(_bound(proxy).exec_status(agent_id, run_id))

        workspace_exec_result.__doc__ = params.format_docstring(workspace_exec_result.__doc__)
        return workspace_exec_result
