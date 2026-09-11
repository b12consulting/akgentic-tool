"""Story 52-6: the tree is taken down by its team and by nothing else, and ``attach`` half-lives.

**The successor to ``test_liveness_sweep.py``, which is deleted.** That file
guarded a hosted actor's own lifetime — the sweep's tick, the reap grace, the
self-stop and the holder map they counted. All four are gone: they existed only
because a hosted actor sat outside every team's two-phase teardown, and this
actor is an ordinary team child. Deleting specs whose subject a decision retired
is correct; what is **not** correct is letting an invariant leave with them, so
the two things that file guarded and that survive are re-pointed here:

- the bind still registers the binding agent's **name**, which is the only source
  for the git journal's author and the exec busy refusal's holder (AC 5);
- a bound card whose tree is then left idle is still there, and still works, long
  past the interval the sweep used to tick at (AC 6).

**Each half is inert alone, which is why both are here.** The structural half
would pass against a sweep that had merely stopped firing; the behavioural half
would pass against a sweep armed with a huge interval. Together they say the
mechanism is gone rather than quiet.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from akgentic.core.agent_state import BaseState

from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.execution import mutation_busy
from akgentic.tool.workspace.models import WorkspaceConfig
from tests.conftest import MockActorAddress
from tests.workspace.conftest import (
    WORKSPACE_PATH,
    FakeOrchestratorProxy,
    card_for,
    tool_named,
)

IDLE_WAIT_S = 0.2
"""How long the behavioural half leaves a bound tree completely idle.

**It is deliberately not the sweep's former interval, and naming it so would
overclaim.** The deleted sweep defaulted to 30 s and read that from
``WorkspaceConfig``, so 51-3 could set it short and then genuinely outlive
several ticks. There is no field to set any more, so no wall-clock a spec can
afford proves a 30 s timer is absent — which is exactly why the structural half
carries that weight: ``test_starting_one_spawns_no_timer_thread`` measures
``threading.enumerate()`` across the start, so a timer at *any* interval, under
any name, armed from any helper, is caught. What this wait adds is the other
direction — that an idle tree is not degraded by being idle — and 0.2 s is
enough for that.
"""


def _actor_of(orchestrator_proxy: FakeOrchestratorProxy) -> WorkspaceActor:
    """The workspace actor the suite's card bound, read out of the team's children."""
    _address, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
    assert isinstance(actor, WorkspaceActor)
    return actor


class TestTheBindRegistersTheAgentsName:
    """AC 5. ``attach``'s holder half died; its name half is load-bearing and stayed.

    **Mutation**: drop ``workspace.attach(...)`` from
    ``WorkspaceTool._bind_workspace_actor``. Every spec here goes red, and so do
    ``test_exec.py``'s busy-refusal family and ``test_journal.py``'s attribution
    specs — which is the point: the name is not decoration, it is what a model
    reads out of a refusal and what a human reads out of ``git log``.
    """

    def test_the_bind_records_the_name_under_this_agents_id(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card, observer = card_for(orchestrator_proxy, "alice")
        actor = _actor_of(orchestrator_proxy)
        agent_id = str(observer.myAddress.agent_id)

        assert actor._name_of(agent_id) == str(observer.myAddress.name)
        assert card._agent_id == agent_id

    def test_an_unregistered_agent_falls_back_to_its_id_rather_than_raising(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Losing a name degrades the two messages; it never breaks either."""
        card_for(orchestrator_proxy, "alice")
        actor = _actor_of(orchestrator_proxy)

        assert actor._name_of("never-attached") == "never-attached"

    def test_the_name_map_still_keeps_its_cap(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """LRU by ``move_to_end``: the least recently recorded name is the one dropped."""
        card_for(orchestrator_proxy, "alice")
        actor = _actor_of(orchestrator_proxy)
        actor.config = actor.config.model_copy(update={"max_tracked_writers": 2})
        ann, bert, carl = (MockActorAddress(name) for name in ("ann", "bert", "carl"))

        actor.attach(ann, "ann")
        actor.attach(bert, "bert")
        actor.attach(ann, "ann")  # refreshes ann's recency, so bert is now the oldest
        actor.attach(carl, "carl")

        assert actor._name_of(str(ann.agent_id)) == "ann"
        assert actor._name_of(str(carl.agent_id)) == "carl"
        assert actor._name_of(str(bert.agent_id)) == str(bert.agent_id)  # evicted: id fallback

    def test_the_busy_refusal_reads_that_name_end_to_end(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The whole chain in one spec, in a process with no host of any kind.

        ``attach`` → ``_agent_names`` → ``_name_of`` → the lock marker's
        ``agent_name`` → :func:`mutation_busy`. Asserted against the rendered
        text rather than against the map, because the text is the product: a
        refusal reading *"agent '3f2a…'"* is something a model can read and
        nothing it can act on, and that is the regression a bare-id degradation
        would be.
        """
        _card, observer = card_for(orchestrator_proxy, "alice")
        actor = _actor_of(orchestrator_proxy)
        agent_id = str(observer.myAddress.agent_id)
        run_id = "r0000001"
        assert actor._name_of(agent_id) == "alice"

        refusal = mutation_busy(run_id, actor._name_of(agent_id))

        assert "agent 'alice'" in refusal
        assert agent_id not in refusal


class TestNothingSweepsAndNothingReaps:
    """AC 6, structural half: the mechanism is absent, not merely quiet."""

    @pytest.mark.parametrize(
        "name",
        [
            "receiveMsg_SweepTick",
            "_arm_sweep",
            "_close_sweep",
            "_reap",
            "_grace_expired",
            "_drop_dead_holders",
            "_drop_runs_of",
        ],
    )
    def test_the_actor_answers_none_of_the_lifetime_names(self, name: str) -> None:
        """A proxy forwards whatever name it is given, so an absence is the invariant."""
        assert not hasattr(WorkspaceActor, name)

    def test_the_two_timing_fields_left_the_config(self) -> None:
        """A stored config still carrying them loads unchanged; nothing reads them."""
        assert "sweep_interval_s" not in WorkspaceConfig.model_fields
        assert "reap_grace_s" not in WorkspaceConfig.model_fields
        stale = WorkspaceConfig.model_validate(
            {
                "name": workspace_actor_name(WORKSPACE_PATH),
                "role": "ToolActor",
                "workspace_path": WORKSPACE_PATH,
                "sweep_interval_s": 30.0,
                "reap_grace_s": 120.0,
            }
        )
        assert stale.workspace_path == WORKSPACE_PATH

    def test_the_state_is_a_bare_base_state_with_no_fields(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """AC 7, from the outside: the deferred machinery reads ``self.state`` and finds one."""
        card_for(orchestrator_proxy, "alice")
        actor = _actor_of(orchestrator_proxy)

        assert type(actor.state) is BaseState
        # ``model_fields``, not ``model_dump``: the serializer stamps a
        # ``__model__`` key into every dump, so an empty dump is not a thing a
        # ``SerializableBaseModel`` can produce and the field set is the claim.
        assert BaseState.model_fields == {}

    def test_starting_one_spawns_no_timer_thread(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The sweep armed a daemon ``threading.Timer`` in ``on_start``; nothing does now.

        Measured across the start rather than by name, so a timer renamed or
        armed from a helper is caught just the same. The executor spawns no
        thread until the first ``submit``, which is what makes zero the right
        number here.
        """
        before = set(threading.enumerate())

        card_for(orchestrator_proxy, "alice")

        assert set(threading.enumerate()) - before == set()


class TestABoundCardKeepsItsTreeAliveIndefinitely:
    """AC 6, behavioural half. Re-pointed from ``TestABoundCardKeepsItsTreeAlive``.

    51-3's version had to hold a *holder* to stay alive past a grace. There is no
    grace to outlive, so what this asserts is what a user would notice: a card
    bound to a tree that is then left completely idle still serves a mutation,
    with every surface a reap would have taken down still in place. **It does not
    claim to outlast the old 30 s tick** — see :data:`IDLE_WAIT_S`; the absence of
    any timer is the structural half's job, and this is the half that says the
    live path still works.
    """

    def test_an_idle_tree_still_serves_a_mutation(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card, _observer = card_for(orchestrator_proxy, "alice")
        actor = _actor_of(orchestrator_proxy)

        time.sleep(IDLE_WAIT_S)

        # Intact rather than merely un-stopped: the surfaces a reap would have
        # taken down are all still there, and a write still lands on disk.
        assert actor._executor is not None
        assert actor._workspace._root == workspace_tree.resolve()
        assert card._workspace_proxy is not None
        assert tool_named(card, "workspace_write")("late.md", "still here\n") == (
            "Written: late.md"
        )
        assert (workspace_tree / "late.md").read_text(encoding="utf-8") == "still here\n"

    def test_the_team_teardown_is_what_stops_it(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The positive control: without it, "never reaps" would pass for a tree that
        nothing can stop at all.

        The actor is in exactly one team's children, so the team's teardown
        reaches it — which is the whole of its lifetime now.
        """
        card_for(orchestrator_proxy, "alice")
        name = workspace_actor_name(WORKSPACE_PATH)
        assert name in orchestrator_proxy.children

        orchestrator_proxy.stop_all()

        assert name not in orchestrator_proxy.children
