"""Story 51-3: a hosted ``#Workspace`` owns its lifetime — sweep, grace, self-stop.

Real threads throughout, except for the defaults at the bottom: a real
``ActorSystem`` with both hosts running (the end-to-end fixture's shape), the
workspace created through the ``WorkspaceHost`` with a config that ticks every
50 ms and reaps 300 ms after its last holder stopped, and real ``Akgent``
holders started and stopped through the system.

**Every positive assertion waits on a bounded poll; two negative ones sleep.**
"The actor is still alive" cannot be waited for — it is true at every instant
until it is false — so those two sleep for a stated number of whole ticks past
the point that matters, with the arithmetic written beside the number.

**``find_by_class`` is an exact-class lookup.** The probes below subclass
``WorkspaceActor``, so ``find_by_class(WorkspaceActor)`` answers ``[]`` for a
live probe; every lookup of a probe asks for its own class, and the fixture's
teardown asks for all three.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from akgentic.core import ActorRegistry
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.orchestrator import STOP_TIMEOUT
from akgentic.core.resource_host import ResourceHost, StateDelta
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import ValidationError

from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.execution import LEASE_GRACE_S, MAX_EXEC_BUDGET_S
from akgentic.tool.workspace.host import WorkspaceHost
from akgentic.tool.workspace.models import (
    DEFAULT_REAP_GRACE_S,
    DEFAULT_SWEEP_INTERVAL_S,
    SweepTick,
    WorkspaceConfig,
)
from tests.workspace.conftest import (
    FAST_REAP_GRACE_S,
    FAST_SWEEP_INTERVAL_S,
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_PATH,
    FakeOrchestratorProxy,
    MortalAddress,
    card_for,
    children_ahead,
    fast_config,
    wait_until,
)

PATH = "u-alice/swept"
"""The tree every live spec here hosts — one per spec, since each spec has its own system."""

HOST_LOGGER = "akgentic.core.resource_host"


##
## Test doubles
##
class Poke(SerializableBaseModel):
    """A payload with no fields, told to a holder whose handler for it raises."""


class _SilentHolder(Akgent[BaseConfig, BaseState]):
    """An agent that holds a tree and does nothing else."""


class _RaisingHolder(Akgent[BaseConfig, BaseState]):
    """An agent whose ``Poke`` handler raises — and which, per pykka, keeps running.

    ``raised`` is set immediately before the raise, so a spec can prove the
    handler genuinely ran rather than assume it: a "raising holder is kept" spec
    whose holder never raised would be green on any sweep at all.
    """

    raised: ClassVar[threading.Event] = threading.Event()

    def receiveMsg_Poke(self, msg: Poke) -> None:
        _RaisingHolder.raised.set()
        raise RuntimeError("the holder's handler blew up")


class _ProbeWorkspace(WorkspaceActor):
    """A workspace that records the thread every tick ran on, and reads its holders out.

    pykka proxies expose no underscore attribute, which is why the readers are
    public methods. The tick is recorded **before** chaining to the real
    handler, so a tick that reaps is recorded too.
    """

    def on_start(self) -> None:
        self.ticks: list[int] = []
        super().on_start()

    def receiveMsg_SweepTick(self, msg: SweepTick) -> None:
        self.ticks.append(threading.get_ident())
        super().receiveMsg_SweepTick(msg)

    def holder_ids(self) -> list[str]:
        """The agent ids this tree currently counts as holders, in attach order."""
        return list(self._holders)

    def tick_idents(self) -> list[int]:
        """The thread ident of every tick so far, in order."""
        return list(self.ticks)

    def tick_count(self) -> int:
        """How many ticks have run."""
        return len(self.ticks)

    def thread_ident(self) -> int:
        """The ident of the thread this call runs on — the actor's own."""
        return threading.get_ident()


class _GatedWorkspace(_ProbeWorkspace):
    """A probe whose ``on_stop`` waits for the spec before it runs, and says when it has run.

    Holding ``on_stop`` open is holding open the window the real one has: pykka
    has already set the stopped flag, so the host answers a miss for the path,
    while the exec drain has not finished. ``released`` is set by the fixture,
    so a gated actor no spec is holding stops like any other; the wait is
    bounded, so a spec that fails before releasing cannot hang the teardown.
    """

    released: ClassVar[threading.Event] = threading.Event()
    stopped: ClassVar[threading.Event] = threading.Event()

    def on_stop(self) -> None:
        _GatedWorkspace.released.wait(timeout=HANDSHAKE_TIMEOUT_S)
        super().on_stop()
        _GatedWorkspace.stopped.set()


##
## Harness
##
@pytest.fixture
def system(workspaces_root: Path) -> Generator[ActorSystem, None, None]:
    """A real actor system with both hosts, torn down whatever the spec did.

    The teardown proves nothing hosted outlived it, asking by every class
    because the lookup is exact.
    """
    actor_system = ActorSystem()
    try:
        actor_system.createActor(
            WorkspaceHost, config=BaseConfig(name="#WorkspaceHost", role="ResourceHost")
        )
        actor_system.createActor(
            ResourceHost, config=BaseConfig(name="#ResourceHost", role="ResourceHost")
        )
        yield actor_system
    finally:
        actor_system.shutdown(timeout=10)
        ActorRegistry.stop_all()
        assert ActorSystem.find_by_class(WorkspaceActor) == [], "a workspace outlived its test"
        assert ActorSystem.find_by_class(_ProbeWorkspace) == [], "a probe outlived its test"
        assert ActorSystem.find_by_class(_GatedWorkspace) == [], "a gated probe outlived its test"


@pytest.fixture
def gate() -> Generator[type[_GatedWorkspace], None, None]:
    """The gated probe, released and unstopped, and released again whatever the spec did.

    Requested after ``system``, so it is torn down first: every gated actor is
    released before the system's shutdown asks it to stop.
    """
    _GatedWorkspace.released.set()
    _GatedWorkspace.stopped.clear()
    try:
        yield _GatedWorkspace
    finally:
        _GatedWorkspace.released.set()


def _hosted(
    system: ActorSystem,
    config: WorkspaceConfig,
    actor_class: type[WorkspaceActor] = _ProbeWorkspace,
) -> ActorAddress:
    """Get-or-create *config*'s actor through the process's ``WorkspaceHost``."""
    [host] = ActorSystem.find_by_class(WorkspaceHost)
    return system.proxy_ask(host, WorkspaceHost).getResourceOrCreate(actor_class, config)


def _probe(system: ActorSystem, address: ActorAddress) -> _ProbeWorkspace:
    """An ask proxy onto a hosted probe."""
    return system.proxy_ask(address, _ProbeWorkspace)


def _holder(
    system: ActorSystem, name: str, holder_class: type[Akgent[Any, Any]] = _SilentHolder
) -> ActorAddress:
    """Start one real holder as a root actor, the way the system starts any agent."""
    return system.createActor(holder_class, config=BaseConfig(name=name, role="Holder"))


def _stop(system: ActorSystem, address: ActorAddress) -> None:
    """Stop a holder through ``Akgent.stop``, as its team's teardown would."""
    system.proxy_ask(address, Akgent).stop()


##
## AC 1 — the sweep drops a stopped holder and keeps a raising one
##
class TestTheSweepDropsAStoppedHolderAndKeepsARaisingOne:
    """``is_alive()`` is pykka's stopped flag, and it is the only thing the sweep asks."""

    def test_a_stopped_holder_is_dropped_and_the_live_one_is_kept(
        self, system: ActorSystem
    ) -> None:
        address = _hosted(system, fast_config(PATH))
        probe = _probe(system, address)
        stopped = _holder(system, "@stopped")
        kept = _holder(system, "@kept")
        probe.attach(stopped, "stopped")
        probe.attach(kept, "kept")
        assert probe.holder_ids() == [str(stopped.agent_id), str(kept.agent_id)]

        _stop(system, stopped)

        assert wait_until(lambda: str(stopped.agent_id) not in probe.holder_ids()), (
            "the sweep never dropped a holder whose actor had stopped"
        )
        assert probe.holder_ids() == [str(kept.agent_id)]

    def test_a_holder_whose_handler_raised_is_still_running_and_is_kept(
        self, system: ActorSystem, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A crashed agent is a running agent: pykka catches ``Exception`` and continues.

        Paired with a stopped holder that **is** dropped, so the sweep is
        proven to have run and to drop — a sweep that never ran would keep the
        raising holder too, and this spec would be green for the wrong reason.
        """
        caplog.set_level(logging.ERROR)
        _RaisingHolder.raised.clear()
        address = _hosted(system, fast_config(PATH))
        probe = _probe(system, address)
        raising = _holder(system, "@raising", _RaisingHolder)
        stopped = _holder(system, "@stopped")
        probe.attach(raising, "raising")
        probe.attach(stopped, "stopped")

        raising.tell(Poke())

        assert _RaisingHolder.raised.wait(timeout=HANDSHAKE_TIMEOUT_S), "the handler never ran"
        assert wait_until(
            lambda: any(
                "@raising" in record.getMessage()
                and "ERROR processing message" in record.getMessage()
                for record in caplog.records
            )
        ), "pykka never reported the handler's failure"
        _stop(system, stopped)
        assert wait_until(lambda: str(stopped.agent_id) not in probe.holder_ids())
        after_drop = probe.tick_count()
        assert wait_until(lambda: probe.tick_count() >= after_drop + 2), "the sweep stopped"

        assert raising.is_alive()
        assert probe.holder_ids() == [str(raising.agent_id)]


##
## AC 2 — the sweep runs on the actor's thread; the timer thread only tells
##
class TestTheSweepRunsOnTheActorsThread:
    def test_every_tick_ran_on_the_actors_thread_and_none_on_a_timer_thread(
        self, system: ActorSystem
    ) -> None:
        """Compared to the actor's own ident, not merely to the test thread's.

        A sweep run inside the timer callback differs from the test thread too,
        so that comparison alone would pass it. The live timer threads are
        compared as well, and the spec insists at least one was seen.
        """
        address = _hosted(system, fast_config(PATH))
        probe = _probe(system, address)
        probe.attach(_holder(system, "@holder"), "holder")  # keeps the grace from starting
        assert wait_until(lambda: probe.tick_count() >= 2)
        timer_name = f"sweep-{PATH}"
        timer_idents: set[int | None] = set()

        def a_timer_is_armed() -> bool:
            timer_idents.update(t.ident for t in threading.enumerate() if t.name == timer_name)
            return bool(timer_idents)

        assert wait_until(a_timer_is_armed), "no sweep timer thread was ever seen"
        actor_ident = probe.thread_ident()

        assert set(probe.tick_idents()) == {actor_ident}
        assert actor_ident != threading.get_ident()
        assert actor_ident not in timer_idents


##
## AC 3 — a never-attached workspace is reaped too
##
class TestANeverAttachedWorkspaceIsReaped:
    def test_a_tree_nobody_attached_to_stops_on_its_own(self, system: ActorSystem) -> None:
        """The orphan a failed ``attach`` after a successful get-or-create would leave.

        The plain ``WorkspaceActor``, not the probe, so the exact-class lookup
        below is a real lookup of it.
        """
        address = _hosted(system, fast_config(PATH), WorkspaceActor)
        assert [a.agent_id for a in ActorSystem.find_by_class(WorkspaceActor)] == [address.agent_id]

        assert wait_until(lambda: not address.is_alive()), "the orphan was never reaped"
        assert ActorSystem.find_by_class(WorkspaceActor) == []


##
## AC 4 — the grace
##
class TestTheGrace:
    """At zero holders a grace starts; an attach cancels it; its expiry reaps."""

    def test_an_attach_during_grace_cancels_it_and_the_actor_outlives_the_deadline(
        self, system: ActorSystem
    ) -> None:
        config = fast_config(PATH)
        address = _hosted(system, config)
        probe = _probe(system, address)
        first = _holder(system, "@first")
        probe.attach(first, "first")
        _stop(system, first)
        # The drop and the deadline happen in the same tick. Attaching before the
        # drop would cancel a grace that never started.
        assert wait_until(lambda: probe.holder_ids() == []), "the first holder was never dropped"
        second = _holder(system, "@second")
        probe.attach(second, "second")

        # A negative assertion, so a real wait: the 0.3 s grace plus four 0.05 s
        # ticks. A deadline the attach left in place reaps by 0.3 + 0.05 s.
        time.sleep(config.reap_grace_s + 4 * config.sweep_interval_s)

        assert address.is_alive(), "the actor reaped with a holder attached"
        assert probe.holder_ids() == [str(second.agent_id)]
        assert [a.agent_id for a in ActorSystem.find_by_class(_ProbeWorkspace)] == [
            address.agent_id
        ]

        _stop(system, second)

        assert wait_until(lambda: not address.is_alive()), "the last holder left; nothing reaped"

    def test_the_grace_is_real_the_first_empty_tick_does_not_reap(
        self, system: ActorSystem
    ) -> None:
        config = fast_config(PATH, reap_grace_s=1.0)
        address = _hosted(system, config)
        probe = _probe(system, address)
        first = _holder(system, "@first")
        probe.attach(first, "first")
        _stop(system, first)
        assert wait_until(lambda: probe.holder_ids() == [])
        empty_at = probe.tick_count()

        # Negative again: three 0.05 s ticks, well inside the 1.0 s grace.
        time.sleep(3 * config.sweep_interval_s)

        assert address.is_alive(), "the actor reaped at the first tick that found nobody"
        assert probe.tick_count() > empty_at, "no tick ran during the wait — nothing was tested"

    def test_expiry_reaps_and_the_next_get_or_create_constructs_a_new_actor(
        self, system: ActorSystem
    ) -> None:
        config = fast_config(PATH)
        address = _hosted(system, config)
        probe = _probe(system, address)
        first = _holder(system, "@first")
        probe.attach(first, "first")
        _stop(system, first)
        assert wait_until(lambda: probe.holder_ids() == [])

        # From the drop the reap lands between grace and grace + one interval.
        assert wait_until(
            lambda: not address.is_alive(),
            timeout=config.reap_grace_s + 3 * config.sweep_interval_s,
        ), "the grace expired at zero holders and nothing reaped"

        renewed = _hosted(system, config)
        assert renewed.agent_id != address.agent_id, "the host answered the dead actor"
        assert renewed.is_alive()


class TestAnAttachRestartsTheGrace:
    """The ``attach`` clear is load-bearing: a holder no tick ever saw still restarts the grace.

    Inert, with every tick called by hand, because the interleaving is a holder
    that attaches during grace and stops before the next tick — which a live
    spec could only hope to land between two ticks. Without the clear the tree
    reaps on the *first* holder's deadline, a full grace too early for the second.
    """

    def test_a_holder_that_attaches_during_grace_and_stops_before_any_tick_restarts_it(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        grace = 0.05
        # A 30 s interval, so the real timer never fires inside the spec.
        actor = children_ahead(
            orchestrator_proxy,
            fast_config(WORKSPACE_PATH, sweep_interval_s=30.0, reap_grace_s=grace),
        )
        first = MortalAddress("first")
        actor.attach(first, "first")
        first.dead = True
        actor.receiveMsg_SweepTick(SweepTick())
        first_deadline = actor._reap_deadline
        assert first_deadline is not None, "the first holder's stop started no grace"

        second = MortalAddress("second")
        actor.attach(second, "second")
        second.dead = True  # stopped before any tick saw it alive
        time.sleep(2 * grace)  # past the first deadline; a sleep never returns early
        assert time.monotonic() >= first_deadline
        before = time.monotonic()

        actor.receiveMsg_SweepTick(SweepTick())

        assert str(second.agent_id) not in actor._holders
        assert actor._sweep_timer is not None, "the tree reaped on the first holder's deadline"
        assert actor._reap_deadline is not None
        assert actor._reap_deadline >= before + grace, "the grace did not restart at the stop"


##
## The self-stop tells the host nothing — the 51-3 review's ruling
##
class TestTheSelfStopTellsTheHostNothing:
    """The host keeps a dead entry until the next get-or-create replaces it, and that is enough.

    A late ``ResourceStopped`` would drop the registry entry by *name*, which by
    then may be a successor's. Both specs read the host through its public
    surface and its own logger; neither reaches into its registry.
    """

    def test_the_host_keeps_the_dead_entry_and_the_next_get_or_create_starts_exactly_one(
        self,
        system: ActorSystem,
        gate: type[_GatedWorkspace],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """After a self-stop the dead entry is still there, and one new actor replaces it.

        The entry is read through ``notify_delta``: a delta for a scope with no
        entry is dropped with a warning on the host's logger, and one for a scope
        whose entry is still there — dead or not — is not. No store is
        registered, so nothing is written either way.
        """
        caplog.set_level(logging.INFO, logger=HOST_LOGGER)
        config = fast_config(PATH)
        address = _hosted(system, config, gate)
        assert gate.stopped.wait(timeout=HANDSHAKE_TIMEOUT_S), "the orphan never reaped"
        assert not address.is_alive()

        [host] = ActorSystem.find_by_class(WorkspaceHost)
        system.proxy_ask(host, WorkspaceHost).notify_delta(config.name, StateDelta())

        heard = [
            record.getMessage()
            for record in caplog.records
            if record.name == HOST_LOGGER and config.name in record.getMessage()
        ]
        assert heard == [], f"the host heard about the stop, or lost the entry: {heard}"
        renewed = _hosted(system, fast_config(PATH, reap_grace_s=60.0), gate)
        assert renewed.agent_id != address.agent_id, "the host answered the dead actor"
        assert [a.agent_id for a in ActorSystem.find_by_class(gate)] == [renewed.agent_id]

    def test_a_successor_started_during_teardown_is_the_one_the_next_bind_gets(
        self, system: ActorSystem, gate: type[_GatedWorkspace]
    ) -> None:
        """The race, driven in-process and in order, with nothing left to timing.

        The doomed actor reaps itself and its ``on_stop`` is held open: the
        stopped flag is set, so the host answers a miss and a get-or-create
        starts the successor. Then the teardown is let finish. Anything it tells
        the host is queued ahead of the next get-or-create, so a
        ``ResourceStopped`` would drop the successor's entry by name and that
        ask would start a third actor beside a live successor.
        """
        gate.released.clear()
        doomed = _hosted(system, fast_config(PATH), gate)
        assert wait_until(lambda: not doomed.is_alive()), "the orphan never reaped"
        # A long grace, so neither live actor reaps inside the spec.
        lasting = fast_config(PATH, reap_grace_s=60.0)
        successor = _hosted(system, lasting, gate)
        assert successor.agent_id != doomed.agent_id
        assert not gate.stopped.is_set(), "the doomed actor's teardown was not held open"

        gate.released.set()
        assert gate.stopped.wait(timeout=HANDSHAKE_TIMEOUT_S), "the teardown never finished"
        again = _hosted(system, lasting, gate)

        assert again.agent_id == successor.agent_id, "the host lost the live successor's entry"
        assert [a.agent_id for a in ActorSystem.find_by_class(gate)] == [successor.agent_id]


##
## AC 10 — the defaults sit where the traps say
##
class TestTheDefaults:
    def test_the_grace_sits_above_the_backstop_and_the_interval_below_it(self) -> None:
        """Arithmetic, not taste: each inequality is a trap the story names.

        The grace keeps a hosted tree alive through a stopping team's last
        handlers, so it sits above the orchestrator's stop backstop; a run
        released by the lease grace has reported or been released before the
        tree can be reaped.
        """
        config = WorkspaceConfig(workspace_path="u/x")

        assert config.sweep_interval_s == DEFAULT_SWEEP_INTERVAL_S
        assert config.reap_grace_s == DEFAULT_REAP_GRACE_S
        assert config.reap_grace_s > STOP_TIMEOUT
        # "No longer than", not "below": the proposed 30 s interval equals the
        # 30 s backstop, and nothing about the sweep needs it strictly shorter.
        assert config.sweep_interval_s <= STOP_TIMEOUT
        assert config.sweep_interval_s < config.reap_grace_s
        assert config.reap_grace_s > MAX_EXEC_BUDGET_S + LEASE_GRACE_S

    @pytest.mark.parametrize("field", ["sweep_interval_s", "reap_grace_s"])
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_non_positive_interval_or_grace_is_refused(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            WorkspaceConfig(workspace_path="u/x", **{field: value})

    def test_a_stored_config_without_the_two_keys_loads_with_the_defaults(self) -> None:
        stored = WorkspaceConfig(
            workspace_path="u/x", sweep_interval_s=1.0, reap_grace_s=2.0
        ).model_dump()
        del stored["sweep_interval_s"]
        del stored["reap_grace_s"]

        loaded = WorkspaceConfig.model_validate(stored)

        assert loaded.sweep_interval_s == DEFAULT_SWEEP_INTERVAL_S
        assert loaded.reap_grace_s == DEFAULT_REAP_GRACE_S
        assert loaded.workspace_path == "u/x"

    def test_a_host_created_config_is_what_a_later_card_binds(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The card never learns the two fields; a test sets them by creating the actor first."""
        actor = children_ahead(orchestrator_proxy, fast_config(WORKSPACE_PATH))

        _card, _observer = card_for(orchestrator_proxy, "alice")

        _, bound = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]
        assert bound is actor
        assert bound.config.sweep_interval_s == FAST_SWEEP_INTERVAL_S
        assert bound.config.reap_grace_s == FAST_REAP_GRACE_S


# ---------------------------------------------------------------------------
# Story 52-5, AC 19: a bound card keeps its tree alive past the reap grace
# ---------------------------------------------------------------------------


class TestABoundCardKeepsItsTreeAlive:
    """``attach`` survives the card's move, and this is what would notice if it did not.

    **The failure it guards is delayed and silent, which is the whole reason it
    exists.** The liveness sweep and the 120 s grace stay until 52-6; they fire
    at *zero holders*, and the card's ``attach`` is the only thing that records
    one. Drop that call and every workspace actor in a live session stops two
    minutes after the last bind — taking its sandbox backend, its exec worker and
    its retrieval pipeline with it — with nothing raised and nothing logged that
    names the cause. No other spec in this suite would go red: the bind still
    succeeds, the gate still gates, and the tree is only gone by the time anybody
    runs a command.

    **Mutation**: remove ``workspace.attach(...)`` from
    ``WorkspaceTool._bind_workspace_actor``. Every spec below goes red — the
    first immediately, on the holder count; the second after the grace, on the
    actor being stopped.
    """

    def test_the_bind_records_this_agent_as_a_holder(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Immediate half: the holder is there the moment ``observer()`` returns."""
        card, observer = card_for(orchestrator_proxy, "alice")

        _address, actor = orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)]

        assert isinstance(actor, WorkspaceActor)
        assert set(actor._holders) == {str(observer.myAddress.agent_id)}
        # And the display name went with it, which is what a busy refusal prints.
        assert actor._name_of(str(observer.myAddress.agent_id)) == str(observer.myAddress.name)
        assert card._agent_id == str(observer.myAddress.agent_id)

    def test_a_bound_card_that_does_nothing_outlives_the_grace(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Delayed half, driven by ticks rather than by waiting out a real grace.

        The actor is created ahead of the card with a fast grace, so the card's
        bind is a hit on it — the only way a spec can watch a grace that is two
        minutes long in production. Then more ticks than the grace is worth: with
        a holder recorded, not one of them reaps.
        """
        actor = children_ahead(
            orchestrator_proxy,
            fast_config(WORKSPACE_PATH, sweep_interval_s=30.0, reap_grace_s=0.01),
        )
        # Both are held for the whole spec: the card holds its observer weakly,
        # and an observer that went out of scope would take its agent with it —
        # the holder would then be dropped for being *dead* rather than for
        # never having been recorded, and this spec would fail for the wrong
        # reason while an ``attach`` regression went unnoticed.
        card, observer = card_for(orchestrator_proxy, "alice")
        assert actor._holders == {str(observer.myAddress.agent_id): observer.myAddress}, (
            "the bind recorded no holder — attach was not called"
        )

        for _ in range(5):
            time.sleep(0.02)  # comfortably past a 0.01 s grace, every time
            actor.receiveMsg_SweepTick(SweepTick())

        assert actor._reap_deadline is None, "a grace started while a live agent held the tree"
        assert actor._sweep_timer is not None, "the tree reaped itself under a live holder"
        # And it is intact rather than merely un-stopped: the surfaces a reap
        # would have taken down are still there.
        assert actor._executor is not None
        assert actor._workspace._root == workspace_tree.resolve()
        assert card._workspace_proxy is not None

    def test_the_same_tree_reaps_once_its_holder_stops(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The positive control, without which the spec above proves nothing.

        A sweep that had simply stopped reaping — a broken ``_grace_expired``, a
        holder map nothing prunes — would pass the previous spec for entirely the
        wrong reason. So the holder is stopped here and the reap must follow.
        """
        actor = children_ahead(
            orchestrator_proxy,
            fast_config(WORKSPACE_PATH, sweep_interval_s=30.0, reap_grace_s=0.01),
        )
        holder = MortalAddress("alice")
        actor.attach(holder, "alice")

        holder.dead = True
        actor.receiveMsg_SweepTick(SweepTick())  # drops the holder, starts the grace
        assert actor._holders == {}
        assert actor._reap_deadline is not None
        time.sleep(0.02)

        assert actor._grace_expired() is True
