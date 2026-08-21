"""``#Workspace-<workspace_name>``: the team singleton that owns one workspace tree.

In this story the actor is **wired and observing, not yet gating**: it is told
what every agent read, and it decides nothing. Mutations still go straight to
each agent's own :class:`~akgentic.tool.workspace.workspace.Filesystem`. The
first decision it makes — the live-hash compare-and-swap of ADR-036 §3 — arrives
in the next story, and it makes it through the tree handle built here.

**The name carries the workspace, and that is load-bearing.**
``getChildrenOrCreate`` keys on the actor *name*, so a fixed ``#Workspace`` would
collapse two cards carrying different ``workspace_id`` values onto one actor
owning one of the two trees — silently, in a team where
``WorkspaceTool(workspace_id="shared")`` is a documented configuration. The
unicity domain of an actor must equal the resource it owns, and the resource is a
tree. The ``#`` prefix stays: it is the orchestrator's two-phase-stop invariant.

Nothing here blocks. Both ask-reachable methods are O(1) dict operations, so the
deferred-result mechanism (ADR-033) stays disengaged in full.
"""

from __future__ import annotations

import contextlib
import logging
from collections import OrderedDict
from pathlib import Path

from akgentic.core.agent import Akgent
from akgentic.tool.workspace.models import Observation, WorkspaceConfig, WorkspaceState
from akgentic.tool.workspace.workspace import Filesystem, get_workspace, is_staging_name

logger = logging.getLogger(__name__)

WORKSPACE_ACTOR_NAME = "#Workspace"
"""Base actor name. The live name appends the workspace — see :func:`workspace_actor_name`.

The ``#`` prefix is the orchestrator's teardown invariant: it is what classifies
the actor as a tool actor during the two-phase stop.
"""

WORKSPACE_ACTOR_ROLE = "ToolActor"


def workspace_actor_name(workspace_name: str) -> str:
    """Return the singleton actor name owning *workspace_name*.

    Args:
        workspace_name: The resolved workspace — a card's ``workspace_id``, or
            the team id when it has none.

    Returns:
        ``#Workspace-<workspace_name>``.
    """
    return f"{WORKSPACE_ACTOR_NAME}-{workspace_name}"


class WorkspaceActor(Akgent[WorkspaceConfig, WorkspaceState]):
    """Team singleton owning one workspace tree and the observations made of it.

    The observation map is a plain instance attribute rather than a state field:
    recording is not persisted state (see :class:`WorkspaceState`). It is keyed
    ``agent_id -> path -> Observation`` and each inner map is an
    :class:`~collections.OrderedDict`, which is the whole of the LRU — recording
    moves an entry to the end, eviction pops from the front.
    """

    def on_start(self) -> None:
        """Initialise state, take the tree handle, and sweep orphaned staging files."""
        super().on_start()
        self.state = WorkspaceState()
        self.state.observer(self)
        self._observations: dict[str, OrderedDict[str, Observation]] = {}
        self._workspace: Filesystem = get_workspace(self.config.workspace_name)
        self._sweep_staging_files()

    ##
    ## Ask-path methods — O(1) dict operations only
    ##
    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        """Record that *agent_id* observed *path* as described by *observation*.

        Tell-shaped, but reached through the **ask** proxy — ``ActorToolObserver``
        exposes ``proxy_ask`` and no tell proxy, exactly as
        ``DeferredResultActor.request`` is reached.

        Re-recording a known path refreshes its recency instead of adding an
        entry. Over the cap, the least recently observed path is evicted.

        Only the **path** dimension is capped. The agent dimension is deliberately
        unbounded: a team's roster is small and bounded by the team itself, and
        evicting an absent agent's observations would silently re-open the
        "never read it" hole for an agent that comes back. Eviction on the path
        dimension is safe for the opposite reason — a lost observation makes the
        gate *refuse* a write, which is a correctness-preserving degradation.

        Args:
            agent_id: Identity of the reading agent, as a string.
            path: Workspace-relative path that was read.
            observation: Digest of the file's bytes, and whether it was whole.
        """
        seen = self._observations.setdefault(agent_id, OrderedDict())
        seen[path] = observation
        seen.move_to_end(path)
        while len(seen) > self.config.max_observations_per_agent:
            seen.popitem(last=False)

    def observation_for(self, agent_id: str, path: str) -> Observation | None:
        """Return what *agent_id* last observed of *path*, or ``None``.

        A lookup does not refresh recency: the gate consults this on every
        mutation, and letting a write extend a path's lifetime would evict the
        paths an agent is actively reading in favour of the ones it writes.

        Args:
            agent_id: Identity of the agent, as a string.
            path: Workspace-relative path.

        Returns:
            The recorded observation, or ``None`` when there is none — which the
            gate reads as "you have not read this".
        """
        seen = self._observations.get(agent_id)
        return None if seen is None else seen.get(path)

    ##
    ## Startup housekeeping
    ##
    def _sweep_staging_files(self) -> None:
        """Delete staging files an interrupted write left behind, anywhere in the tree.

        ``Filesystem.write`` publishes by rename from ``.<name>.<32 hex>.tmp`` in
        the target's own directory. A process killed between the two steps leaves
        one behind for good, and nothing else ever removes it.

        The sweep runs **at actor start only** — never on a timer, never per
        mutation — and matches the full staging shape, so a user's own
        ``.notes.tmp`` survives. Every failure is suppressed: a directory this
        process cannot clean must not stop the team's workspace from starting.
        """
        root = self._workspace._root
        staged: list[Path] = []
        with contextlib.suppress(OSError):
            staged = [
                entry
                for entry in root.rglob("*")
                if is_staging_name(entry.name) and entry.is_file()
            ]
        removed = 0
        for entry in staged:
            with contextlib.suppress(OSError):
                entry.unlink()
                removed += 1
        if removed:
            logger.info(
                "Workspace %s: swept %d orphaned staging file(s) at start",
                self.config.workspace_name,
                removed,
            )
