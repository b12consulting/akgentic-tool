"""Who read what, who wrote what, and what an agent id is called (ADR-045 §1).

The two maps this mixin owns are the actor's memory of the team, and neither is
persisted state: recording is not state (see
:class:`~akgentic.tool.workspace.models.WorkspaceState`). The observation map is
keyed ``agent_id -> path -> Observation`` and each inner map is an
:class:`~collections.OrderedDict`, which is the whole of the LRU — recording
moves an entry to the end, eviction pops from the front. The last-writer map is
keyed by path across all agents and is capped independently.

Every map here is initialised in
:meth:`~akgentic.tool.workspace.actor.WorkspaceActor.on_start` and nowhere else;
the annotations below declare what this mixin consumes, they do not own it.
"""

from __future__ import annotations

from collections import OrderedDict

from akgentic.tool.workspace.journal import Identity
from akgentic.tool.workspace.models import (
    LastWrite,
    Observation,
    WorkspaceConfig,
    content_sha,
)


class ObservationMixin:
    """The observation map, the last-writer map, and agent-name registration."""

    _agent_names: OrderedDict[str, str]
    _observations: dict[str, OrderedDict[str, Observation]]
    _last_writers: OrderedDict[str, LastWrite]
    _touched: list[str]
    config: WorkspaceConfig

    ##
    ## Identity — reached through the card's **tell** proxy, once, at bind time
    ##
    def register_agent(self, agent_id: str, name: str) -> None:
        """Record the human-readable name behind *agent_id*.

        Fire-and-forget, sent once per card at bind time — O(1), never on the
        mutation path. The actor holds ``agent_id`` because that is what the
        card can capture without an edge back to the agent (ADR-030), but an
        ``agent_id`` is a **UUID**: a journal authored by UUID satisfies the
        letter of "the git log is the who-changed-what record" and defeats its
        purpose, and a refusal reading *"last written by agent '3f2a…'"* tells a
        model nothing it can act on.

        Capped like the last-writer map, and for the same reason: an uncapped map
        on a team singleton leaks for the life of the team. Losing a name is a
        safe degradation — the id is used instead.

        Args:
            agent_id: Identity of the agent, as a string.
            name: Its configured, human-readable name.
        """
        self._agent_names[agent_id] = name
        self._agent_names.move_to_end(agent_id)
        while len(self._agent_names) > self.config.max_tracked_writers:
            self._agent_names.popitem(last=False)

    def _name_of(self, agent_id: str) -> str:
        """Return *agent_id*'s registered name, falling back to the id itself."""
        return self._agent_names.get(agent_id) or agent_id

    def _identity(self, agent_id: str) -> Identity:
        """Compose the git identity for *agent_id*: name to read, id to distinguish."""
        return Identity(self._name_of(agent_id), agent_id)

    ##
    ## The observation map — reached through the card's **tell** proxy
    ##
    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        """Record that *agent_id* observed *path* as described by *observation*.

        Fire-and-forget: the card sends this through ``proxy_tell``, because the
        reader needs nothing back. An ask would make every reader wait on a
        mailbox that, from this story on, hashes files — and an ask carries no
        timeout, so a merely slow actor would stall reads rather than refuse
        writes.

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

    def _writer_of(self, path: str, live: bytes | None) -> str | None:
        """Name the agent whose accepted write produced *live*, if any.

        Attribution is granted only while the live bytes still hash to what that
        agent wrote. Once anything else has touched the path the last accepted
        writer is no longer the author of what is on disk, and naming them would
        pin an out-of-band change on whichever agent happened to write last.
        """
        if live is None:
            return None
        entry = self._last_writers.get(path)
        if entry is None or entry.sha != content_sha(live):
            return None
        return entry.agent_id

    ##
    ## Bookkeeping on the accept path
    ##
    def _accept(self, agent_id: str, path: str, data: bytes) -> None:
        """Record that *agent_id* wrote *data* to *path*, in the same mailbox turn.

        The writer's own observation is refreshed because an agent that has just
        written a file has by definition observed it in full — without this, its
        *next* write to the same path would be refused with a diff against its
        own content. Doing it here rather than in the closure keeps it inside the
        one turn; a second round trip would be a second window.

        This is also where the mutation's write set is collected, for the commit
        :meth:`~akgentic.tool.workspace.actor.gate.GateMixin._journalled` makes
        once the body returns. The list is plain actor state rather than a return
        value because the alternative is widening six signatures, and the actor
        is single-threaded — the mailbox is the lock.
        """
        self._touched.append(path)
        sha = content_sha(data)
        self.record_observation(agent_id, path, Observation(sha=sha, full=True))
        self._last_writers[path] = LastWrite(agent_id=agent_id, sha=sha)
        self._last_writers.move_to_end(path)
        while len(self._last_writers) > self.config.max_tracked_writers:
            self._last_writers.popitem(last=False)

    def _forget(self, agent_id: str, path: str) -> None:
        """Drop what an accepted delete invalidated.

        Only the deleting agent's observation goes: another agent still holding
        one is meant to be refused, because from its point of view the file
        vanished under it.

        A delete is a change to the path like any other, so it joins the write
        set — ``git add -A -- <path>`` stages a removal as readily as a write.
        """
        self._touched.append(path)
        seen = self._observations.get(agent_id)
        if seen is not None:
            seen.pop(path, None)
        self._last_writers.pop(path, None)
