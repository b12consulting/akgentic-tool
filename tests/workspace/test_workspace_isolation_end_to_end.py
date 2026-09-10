"""The four things epic 48 promised, held by guards rather than by memory.

``test_workspace_path_resolution.py`` tests the resolver *as a function*. This
file tests the **wiring**: a real ``ActorSystem``, a real core ``Orchestrator``,
real spawned members that bind their own cards inside their own ``on_start``.
Nothing here hands the resolver its inputs — the identity arrives the way it
arrives in production, through ``ActorSystem.createActor(user_id=…)`` and then
``Akgent.createActor``'s propagation to every child. A guard that constructs a
fake observer carrying the values it is about to assert on cannot catch a
propagation regression, which is the failure these specs exist for.

**What this file does NOT prove, and must not be read as proving.** The module
boundary forbids ``akgentic-tool`` from importing ``akgentic-team``, so the
create/resume guard below reproduces the *two lifecycle orderings* over core's
own surface. It does not prove that ``TeamFactory.build`` and ``TeamRestorer``
actually use those orderings — that guard belongs to ``akgentic-team``'s epic 36
and must not be re-implemented here. **A green suite in this package is not
evidence that the team package is correct.**

``FakeActorToolObserver`` is deliberately absent. It is the right tool almost
everywhere else in this suite and the wrong one here, for the one reason above.
The shared ``tool_named`` helper is imported rather than re-spelled: it carries
no identity, and a second copy of it would be the same one-rule-two-spellings
defect this epic exists to remove.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pykka
import pytest
from akgentic.core import ActorRegistry
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.orchestrator import EventMessage, StartMessage
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.resource_host import ResourceHost
from akgentic.core.utils.deserializer import deserialize_object
from akgentic.core.utils.serializer import SerializableBaseModel

from akgentic.tool.errors import RetriableError
from akgentic.tool.vector_store.protocol import VectorStoreParam
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.event import WorkspaceAttached
from akgentic.tool.workspace.host import WorkspaceHost
from akgentic.tool.workspace.models import WorkspaceConfig
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    SandboxScript,
    fast_config,
    tool_named,
)

SPAWN_TIMEOUT_S = 15.0
"""Upper bound on a member's spawn — never a delay, only a failure budget.

Every wait below is on a ``threading.Event``, never a ``sleep``: a bare sleep
would make these specs slow when they pass and flaky when they do not.
"""


class CaseMetadata(SerializableBaseModel):
    """Stand-in for the team metadata model this package may not import.

    The resolver reads declared keys off it by attribute and never learns its
    type — the key names are configuration, the model is data.
    """

    customer_id: str | None = None
    case_id: str | None = None


@dataclass
class Bind:
    """What one member recorded while binding its own card, inside its own spawn.

    ``error`` is asserted on as well as ``path``: a swallowed exception must not
    be able to read as a pass.
    """

    done: threading.Event = field(default_factory=threading.Event)
    path: PurePosixPath | None = None
    card: WorkspaceTool | None = None
    error: BaseException | None = None
    orchestrator: ActorAddress | None = None
    """The team's own orchestrator — whose stream the bind's event must land on."""
    member: ActorAddress | None = None
    """The member that bound — whose id the bind's event must name."""


_BINDS: dict[str, Bind] = {}
"""Every member's recording, keyed by the ``bind_key`` its config carries.

Module-level because a spawned actor receives only its config, and a config is a
Pydantic model — a ``threading.Event`` cannot travel in one.
"""


class MemberConfig(BaseConfig):
    """A member's card declaration, plus the slot it records into.

    ``rag_in_memory`` and ``exec_local`` switch on the two capabilities whose
    teardown story 51-3 guards: the in-memory store child and the exec runner.
    Both default off, so every earlier spec binds exactly the card it did.
    """

    bind_key: str = ""
    workspace_id: str | None = None
    workspace_metadata_keys: list[str] = []
    rag_in_memory: bool = False
    exec_local: bool = False


def _resolved_path(card: WorkspaceTool) -> PurePosixPath:
    """The two-segment path *card* actually bound to, read back off its backend.

    ``Filesystem`` exposes no public root — ``_root`` is what the rest of this
    suite reads for the same reason. Reading it back off the bound card, rather
    than calling the resolver again, is the point: a spec that re-derives the
    path proves only that the resolver is deterministic.
    """
    root = Path(os.environ["AKGENTIC_WORKSPACES_ROOT"]).resolve()
    return PurePosixPath(card.workspace._root.relative_to(root).as_posix())


def _capabilities(config: MemberConfig) -> dict[str, Any]:
    """The card fields *config* switches on — nothing at all for a plain member."""
    fields: dict[str, Any] = {}
    if config.rag_in_memory:
        fields["workspace_rag_index"] = True
        fields["vector_store"] = VectorStoreParam(backend="inmemory")
    if config.exec_local:
        # A tight poll so a completed run is answered in the call that started it.
        fields["workspace_exec"] = WorkspaceExec(
            mode="local", poll_attempts=500, poll_delay_seconds=0.01
        )
    return fields


class RecordingMember(Akgent[MemberConfig, BaseState]):
    """A team member that binds a ``WorkspaceTool`` against **itself**, at spawn.

    The bind happens in ``on_start`` and nowhere else. An assertion written after
    ``createActor`` returns is satisfied by an ordering in which the team's
    identity and metadata arrive *after* the agents that depend on them — which
    is precisely the regression the create/resume guard exists to catch.

    ``super().on_start()`` runs **inside** the ``try``. A propagation regression
    can raise there rather than at the bind, and outside the ``try`` that leaves
    ``done`` unset — so the guard would report a fifteen-second timeout instead
    of the exception that caused it. A guard that goes red for the wrong reason
    is the failure mode this whole file is built to avoid.
    """

    def on_start(self) -> None:
        record = _BINDS[self.config.bind_key]
        try:
            super().on_start()
            card = WorkspaceTool(
                workspace_id=self.config.workspace_id,
                workspace_metadata_keys=list(self.config.workspace_metadata_keys),
                # Off deliberately: these specs are about which directory a card
                # reaches, and a journal would put a git subprocess on every one.
                git_journal=False,
                **_capabilities(self.config),
            )
            card.observer(self)
            record.card = card
            record.path = _resolved_path(card)
        except Exception as exc:  # noqa: BLE001 — the failure IS the signal
            record.error = exc
        finally:
            record.done.set()


def spawn_member(
    system: ActorSystem,
    *,
    bind_key: str,
    user_id: str,
    team_id: uuid.UUID,
    metadata: SerializableBaseModel | None = None,
    restoring: bool = False,
    workspace_id: str | None = None,
    keys: list[str] | None = None,
    rag_in_memory: bool = False,
    exec_local: bool = False,
) -> Bind:
    """Build a team the way the lifecycle does, and return what the member recorded.

    The shape is core's, and it is the shape both lifecycle paths share:

    * the Orchestrator is created through ``ActorSystem.createActor`` carrying the
      principal and the team id — ``TeamFactory.build`` and ``TeamRestorer`` both
      do exactly this;
    * the team's metadata is set on it **before** any member spawns — resume's
      step 2b-bis, and the create path's mirror of it;
    * the member is spawned through ``orchestrator.createActor``, which propagates
      ``user_id``, ``user_email``, ``team_id``, ``parent`` and ``orchestrator``
      from the orchestrator. That propagation is the only way the member's card
      learns whose tree it is binding.

    ``restoring=True`` is the resume path's spelling of the same call.

    The wait on ``done`` is where ``TeamFactory.build`` returns: by then every
    member has bound. Moving the ``set_metadata`` call below it reproduces the
    pre-epic-36 create ordering exactly, and deterministically.
    """
    record = Bind()
    _BINDS[bind_key] = record

    orch_addr = system.createActor(
        Orchestrator,
        restoring=restoring,
        user_id=user_id,
        team_id=team_id,
        config=BaseConfig(name="@Orchestrator", role="Orchestrator"),
    )
    record.orchestrator = orch_addr
    orch_proxy = system.proxy_ask(orch_addr, Orchestrator)
    if metadata is not None:
        orch_proxy.set_metadata(metadata)

    record.member = orch_proxy.createActor(
        RecordingMember,
        config=MemberConfig(
            name=f"@Member-{bind_key}",
            role="Member",
            bind_key=bind_key,
            workspace_id=workspace_id,
            workspace_metadata_keys=list(keys or []),
            rag_in_memory=rag_in_memory,
            exec_local=exec_local,
        ),
    )

    assert record.done.wait(timeout=SPAWN_TIMEOUT_S), (
        f"member {bind_key!r} never finished on_start"
    )
    return record


def bound(record: Bind) -> tuple[WorkspaceTool, PurePosixPath]:
    """Assert the bind succeeded and hand back the card and its path.

    ``error is None`` is asserted first and separately: a bind that raised leaves
    ``path`` unset, and "unset" must never be mistaken for "did not match".
    """
    assert record.error is None, f"the card failed to bind: {record.error!r}"
    assert record.card is not None
    assert record.path is not None
    return record.card, record.path


def _start_host(system: ActorSystem, host_class: type[ResourceHost], name: str) -> ActorAddress:
    """Create one host the way wiring does: once, right after the system."""
    return system.createActor(host_class, config=BaseConfig(name=name, role="ResourceHost"))


def _tear_down(actor_system: ActorSystem) -> None:
    """Stop everything, then prove no hosted workspace outlived the system."""
    actor_system.shutdown(timeout=10)
    ActorRegistry.stop_all()
    _BINDS.clear()
    assert ActorSystem.find_by_class(WorkspaceActor) == [], "a hosted workspace outlived its test"


@pytest.fixture
def system() -> Generator[ActorSystem, None, None]:
    """A real actor system with **both** hosts running, torn down whatever the test did.

    The ``WorkspaceHost`` is what every card binds through. The base
    ``ResourceHost`` beside it is the transitional infra wiring, and it is what
    makes the host class observable: with only one host running, a card that
    named the wrong class would fail on "no host" and never on routing.
    """
    actor_system = ActorSystem()
    try:
        _start_host(actor_system, WorkspaceHost, "#WorkspaceHost")
        _start_host(actor_system, ResourceHost, "#ResourceHost")
        yield actor_system
    finally:
        _tear_down(actor_system)


@pytest.fixture
def base_only_system() -> Generator[ActorSystem, None, None]:
    """A process wired with the base ``ResourceHost`` alone — infra's wiring until it switches."""
    actor_system = ActorSystem()
    try:
        _start_host(actor_system, ResourceHost, "#ResourceHost")
        yield actor_system
    finally:
        _tear_down(actor_system)


##
## AC 7 — two principals, one workspace name, two trees, neither reachable
##


class TestTwoPrincipalsOneNameTwoTrees:
    def test_a_file_written_by_one_principal_is_invisible_to_the_other(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """The exposure this epic exists to close, crossed at the capability surface.

        Two resolved strings differing is what the resolver's own specs already
        prove. What has to hold end to end is that the *capability* cannot cross:
        a file written through Alice's workspace is not readable, listable or
        matchable through Bob's, even though both cards declare the identical
        ``workspace_id``. Asserting on the two paths instead would stay green if
        both cards landed in one directory by a different route.
        """
        alice = spawn_member(
            system,
            bind_key="alice",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            workspace_id="notes",
        )
        bob = spawn_member(
            system,
            bind_key="bob",
            user_id="u-bob",
            team_id=uuid.uuid4(),
            workspace_id="notes",
        )
        alice_card, alice_path = bound(alice)
        bob_card, bob_path = bound(bob)

        tool_named(alice_card, "workspace_write")("secret.txt", "alice-only\n")

        # Through Bob's own capabilities, the file simply is not there.
        with pytest.raises(RetriableError, match="not found"):
            tool_named(bob_card, "workspace_read")("secret.txt")
        assert "secret.txt" not in str(tool_named(bob_card, "workspace_list")(""))
        assert "secret.txt" not in str(tool_named(bob_card, "workspace_glob")("**/*.txt"))
        assert "alice-only" not in str(tool_named(bob_card, "workspace_glob")("**/*"))

        # And Alice's own read still works, so the absence above is isolation
        # rather than a write that never landed.
        assert "alice-only" in str(tool_named(alice_card, "workspace_read")("secret.txt"))

    def test_the_two_trees_are_siblings_under_two_scopes_on_disk(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """Two real directories, differing in the scope segment and nowhere else."""
        alice = spawn_member(
            system,
            bind_key="alice",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            workspace_id="notes",
        )
        bob = spawn_member(
            system,
            bind_key="bob",
            user_id="u-bob",
            team_id=uuid.uuid4(),
            workspace_id="notes",
        )
        _, alice_path = bound(alice)
        _, bob_path = bound(bob)

        assert alice_path == PurePosixPath("u-alice/notes")
        assert bob_path == PurePosixPath("u-bob/notes")
        assert (workspaces_root / "u-alice" / "notes").is_dir()
        assert (workspaces_root / "u-bob" / "notes").is_dir()
        # Neither is a prefix of the other — containment, not collision, is the
        # hazard the fixed two-segment depth removes.
        assert not str(bob_path).startswith(f"{alice_path}/")
        assert not str(alice_path).startswith(f"{bob_path}/")


##
## AC 8, 11 — the bare card, with the identity propagated rather than injected
##


class TestTheBareCardResolvesUnderItsOwner:
    def test_a_bare_card_reaches_user_id_over_team_id(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """``WorkspaceTool()`` with nothing declared is ``<user_id>/<team_id>``.

        The ``anonymous`` assertion is the loud one: that is exactly what a
        propagation regression looks like — core's ``user_id`` property going
        away, or an orchestrator built without a principal — and it is a silent
        merge of every user's tree into one if nothing watches for it.
        """
        team_id = uuid.uuid4()
        record = spawn_member(
            system, bind_key="bare", user_id="u-alice", team_id=team_id
        )
        _, path = bound(record)

        assert path == PurePosixPath(f"u-alice/{team_id}")
        assert path.parts[0] != "anonymous"
        assert (workspaces_root / "u-alice" / str(team_id)).is_dir()

    def test_the_member_never_received_the_identity_from_the_test(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """The member's own ``user_id`` is the orchestrator's, by propagation.

        Nothing in this file constructs an observer carrying a principal. The
        value reaches the card because ``Akgent.createActor`` hands it to every
        child at construction time, which is what makes the assertion above a
        wiring guard rather than a restatement of its own fixture.
        """
        record = spawn_member(
            system, bind_key="bare", user_id="u-carol", team_id=uuid.uuid4()
        )
        card, path = bound(record)

        assert path.parts[0] == "u-carol"
        assert card.workspace._root.parent.name == "u-carol"


##
## AC 9 — the metadata card shares, and the sharing is the point
##


class TestTheMetadataCardSharesDeliberately:
    def test_two_teams_with_the_same_key_values_reach_the_same_tree(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """**This sharing is the layout's whole purpose — do not "fix" it.**

        Two teams, two team ids, two principals, two orchestrators, one resolved
        path. Everywhere else in this epic a shared tree is the defect; here it
        is the feature, which is why it is pinned as tightly as the isolation is.
        """
        keys = ["customer_id", "case_id"]
        first = spawn_member(
            system,
            bind_key="acme-a",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=keys,
        )
        second = spawn_member(
            system,
            bind_key="acme-b",
            user_id="u-bob",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=keys,
        )
        first_card, first_path = bound(first)
        second_card, second_path = bound(second)

        assert first_path == second_path
        assert first_path == PurePosixPath("_meta/customer_id-ACME__case_id-42")

        # One tree, reached from two teams: what one writes, the other reads.
        tool_named(first_card, "workspace_write")("shared.txt", "case-42\n")
        assert "case-42" in str(tool_named(second_card, "workspace_read")("shared.txt"))

    def test_a_different_key_value_reaches_a_different_tree(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        keys = ["customer_id", "case_id"]
        forty_two = spawn_member(
            system,
            bind_key="case-42",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=keys,
        )
        forty_three = spawn_member(
            system,
            bind_key="case-43",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="43"),
            keys=keys,
        )
        _, path_42 = bound(forty_two)
        _, path_43 = bound(forty_three)

        assert path_42 != path_43
        assert path_42.parts[0] == "_meta"
        assert path_43.parts[0] == "_meta"


##
## AC 10 — create and resume agree, observed during the spawn
##


class TestCreateAndResumeResolveTheSamePath:
    """The highest-value regression in this epic.

    A create/resume disagreement leaves an agent staring at an empty workspace
    where its files were, after a restart, with no error anywhere.

    Both runs record **inside the member's own ``on_start``**. An assertion made
    after the spawn call returns passes against the ordering in which metadata
    and identity arrive too late, and proves nothing.
    """

    def test_the_bare_card_resolves_the_same_path_on_both_paths(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        team_id = uuid.uuid4()
        created = spawn_member(
            system, bind_key="created", user_id="u-alice", team_id=team_id
        )
        resumed = spawn_member(
            system,
            bind_key="resumed",
            user_id="u-alice",
            team_id=team_id,
            restoring=True,
        )
        _, created_path = bound(created)
        _, resumed_path = bound(resumed)

        assert created_path == resumed_path == PurePosixPath(f"u-alice/{team_id}")

    def test_the_metadata_card_resolves_the_same_path_on_both_paths(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """And the bind must SUCCEED on both, not merely agree.

        The ordering this guards against does not produce a different path; it
        produces no path at all, because the card raises "the team carries no
        metadata" during the spawn. Asserting equality alone would compare two
        ``None``s and pass.
        """
        team_id = uuid.uuid4()
        metadata = CaseMetadata(customer_id="ACME", case_id="42")
        created = spawn_member(
            system,
            bind_key="created",
            user_id="u-alice",
            team_id=team_id,
            metadata=metadata,
            keys=["customer_id", "case_id"],
        )
        resumed = spawn_member(
            system,
            bind_key="resumed",
            user_id="u-alice",
            team_id=team_id,
            metadata=metadata,
            restoring=True,
            keys=["customer_id", "case_id"],
        )
        _, created_path = bound(created)
        _, resumed_path = bound(resumed)

        assert created_path == resumed_path
        assert created_path == PurePosixPath("_meta/customer_id-ACME__case_id-42")

    def test_the_files_written_before_a_restart_are_there_after_it(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """The consequence, stated as the user experiences it.

        A path disagreement is invisible as a path; it is visible as an agent
        finding an empty workspace where its work was.
        """
        team_id = uuid.uuid4()
        created = spawn_member(
            system, bind_key="created", user_id="u-alice", team_id=team_id
        )
        created_card, _ = bound(created)
        tool_named(created_card, "workspace_write")("before.txt", "written-before\n")

        resumed = spawn_member(
            system,
            bind_key="resumed",
            user_id="u-alice",
            team_id=team_id,
            restoring=True,
        )
        resumed_card, _ = bound(resumed)

        assert "written-before" in str(
            tool_named(resumed_card, "workspace_read")("before.txt")
        )


##
## Story 51-2 — two teams, one tree: one hosted actor, two events
##

SHARED_PATH = "_meta/customer_id-ACME__case_id-42"
"""The metadata tree both teams below resolve to."""

# The envelope key set of the frontend's 52-2 wire fixture —
# akgentic-frontend src/app/components/process/selectors/workspace-registry.selector.spec.ts:657-680
# at 4f94179. Copied here, never read from the other repository. The sender's own
# key set and the timestamp format are deliberately not pinned: the fixture is
# derived rather than captured, it disagrees with core on both, and the fold
# reads neither.
FRONTEND_ENVELOPE_KEYS = frozenset(
    {
        "id",
        "parent_id",
        "team_id",
        "timestamp",
        "sender",
        "recipient",
        "display_type",
        "__model__",
        "event",
    }
)


def _two_teams_on_one_tree(system: ActorSystem) -> tuple[Bind, Bind]:
    """Two teams, two principals, two team ids, two orchestrators — one metadata tree."""
    keys = ["customer_id", "case_id"]
    first = spawn_member(
        system,
        bind_key="acme-a",
        user_id="u-alice",
        team_id=uuid.uuid4(),
        metadata=CaseMetadata(customer_id="ACME", case_id="42"),
        keys=keys,
    )
    second = spawn_member(
        system,
        bind_key="acme-b",
        user_id="u-bob",
        team_id=uuid.uuid4(),
        metadata=CaseMetadata(customer_id="ACME", case_id="42"),
        keys=keys,
    )
    return first, second


def _stream_of(system: ActorSystem, record: Bind) -> list[object]:
    """Everything on the member's own team's stream, in order."""
    assert record.orchestrator is not None
    return list(system.proxy_ask(record.orchestrator, Orchestrator).get_messages(None, None))


def _attached_events(system: ActorSystem, record: Bind) -> list[EventMessage]:
    """The ``EventMessage``s on the member's own team's stream carrying a ``WorkspaceAttached``."""
    return [
        message
        for message in _stream_of(system, record)
        if isinstance(message, EventMessage) and isinstance(message.event, WorkspaceAttached)
    ]


class TestTwoTeamsOnOneTreeShareOneHostedActor:
    """The headline guard: hosting, observed across two real orchestrators.

    What would have to be true for these to be false, and is reachable: a card
    that binds through ``getChildrenOrCreate`` gives each orchestrator its own
    child — two actors, and no event on either stream — and a card that names
    the base ``ResourceHost`` puts the actor in the wrong registry, which the
    base host running beside the ``WorkspaceHost`` makes observable.
    """

    def test_both_binds_succeed_and_exactly_one_actor_lives_in_the_workspace_hosts_registry(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        first, second = _two_teams_on_one_tree(system)
        _, first_path = bound(first)
        _, second_path = bound(second)
        assert first_path == second_path == PurePosixPath(SHARED_PATH)

        [workspace] = ActorSystem.find_by_class(WorkspaceActor)

        # Asked of the WorkspaceHost's OWN registry: a hit answers the same actor.
        # Had the card bound through the base host, this ask is a miss, the
        # WorkspaceHost constructs a second actor, and the count below is two.
        [host] = ActorSystem.find_by_class(WorkspaceHost)
        answered = system.proxy_ask(host, WorkspaceHost).getResourceOrCreate(
            WorkspaceActor,
            WorkspaceConfig(
                name=workspace_actor_name(SHARED_PATH),
                role=WORKSPACE_ACTOR_ROLE,
                workspace_path=SHARED_PATH,
            ),
        )
        assert answered.agent_id == workspace.agent_id
        assert len(ActorSystem.find_by_class(WorkspaceActor)) == 1

    def test_each_teams_stream_carries_one_event_naming_its_own_agent(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        first, second = _two_teams_on_one_tree(system)
        bound(first)
        bound(second)

        payload_ids = []
        for record in (first, second):
            assert record.member is not None
            assert record.orchestrator is not None
            [message] = _attached_events(system, record)
            assert message.event == WorkspaceAttached(
                agent_id=record.member.agent_id, workspace_path=SHARED_PATH
            )
            assert isinstance(message.event.agent_id, uuid.UUID)
            # The envelope is the orchestrator's; the payload names the member.
            assert message.sender is not None
            assert message.sender.agent_id == record.orchestrator.agent_id
            assert message.sender.agent_id != record.member.agent_id
            payload_ids.append(message.event.agent_id)
        assert payload_ids[0] != payload_ids[1]

    def test_the_hosted_actor_is_in_no_team(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """No ``StartMessage``, no roster entry, no orchestrator — on either team."""
        first, second = _two_teams_on_one_tree(system)
        bound(first)
        bound(second)
        name = workspace_actor_name(SHARED_PATH)

        for record in (first, second):
            assert record.orchestrator is not None
            assert record.member is not None
            starts = [m for m in _stream_of(system, record) if isinstance(m, StartMessage)]
            # The positive beside the negative: the member's own start IS there.
            assert any(m.sender == record.member for m in starts)
            assert [m for m in starts if isinstance(m.config, WorkspaceConfig)] == []
            orchestrator = system.proxy_ask(record.orchestrator, Orchestrator)
            assert orchestrator.get_team_member(record.member.name) is not None
            assert orchestrator.get_team_member(name) is None

        [workspace] = ActorSystem.find_by_class(WorkspaceActor)
        assert system.proxy_ask(workspace, WorkspaceActor).orchestrator is None


class TestTheWireShapeIsTheFrontends:
    """A real serialisation of a real envelope, against the fold's contract."""

    def test_the_serialised_envelope_and_payload(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        first, second = _two_teams_on_one_tree(system)
        bound(first)
        bound(second)
        assert first.member is not None
        [message] = _attached_events(system, first)

        wire = json.loads(message.model_dump_json())

        assert set(wire) == FRONTEND_ENVELOPE_KEYS
        assert wire["__model__"] == "akgentic.core.messages.orchestrator.EventMessage"
        assert "content" not in wire
        assert wire["recipient"] is None
        assert wire["display_type"] == "other"
        # Whole-dict equality: an extra field — a metadata key list — fails it.
        assert wire["event"] == {
            "__model__": "akgentic.tool.workspace.event.WorkspaceAttached",
            "agent_id": str(first.member.agent_id),
            "workspace_path": SHARED_PATH,
        }
        agent_id = wire["event"]["agent_id"]
        assert str(uuid.UUID(agent_id)) == agent_id
        assert agent_id != wire["sender"]["agent_id"]
        assert wire["sender"]["__actor_type__"] == "akgentic.core.orchestrator.Orchestrator"

    def test_the_envelope_round_trips_to_the_same_payload(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        first, second = _two_teams_on_one_tree(system)
        bound(first)
        bound(second)
        [message] = _attached_events(system, first)

        restored = deserialize_object(json.loads(message.model_dump_json()))

        assert isinstance(restored, EventMessage)
        assert type(restored.event) is WorkspaceAttached
        assert restored.event == message.event
        assert isinstance(restored.event.agent_id, uuid.UUID)


class TestABaseOnlyProcessFailsTheFirstBind:
    """Infra's wiring until it switches: core's designed error, and nothing created."""

    def test_the_bind_fails_loudly_and_creates_and_emits_nothing(
        self, base_only_system: ActorSystem, workspaces_root: Path
    ) -> None:
        record = spawn_member(
            base_only_system,
            bind_key="base-only",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=["customer_id", "case_id"],
        )

        assert isinstance(record.error, RuntimeError)
        assert "No WorkspaceHost is running" in str(record.error)
        assert ActorSystem.find_by_class(WorkspaceActor) == []
        stream = _stream_of(base_only_system, record)
        # The stream was read: the member's own start is on it.
        assert any(isinstance(message, StartMessage) for message in stream)
        assert _attached_events(base_only_system, record) == []


##
## Story 51-3 — a tree nobody's team owns is still reclaimed, whole
##

REAPED_PATH = "u-alice/notes"
"""What ``workspace_id="notes"`` resolves to for ``u-alice``; the specs below pre-create it."""


def _wait_until(predicate: Callable[[], bool], timeout: float = HANDSHAKE_TIMEOUT_S) -> bool:
    """Poll *predicate* every 10 ms until it holds or *timeout* elapses — a budget, not a delay."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _host_ahead(system: ActorSystem, config: WorkspaceConfig) -> ActorAddress:
    """Create *config*'s workspace through the real host before any card binds it.

    The host ignores ``config`` on a hit, so the member that binds afterwards
    gets this actor — ticking at this config's speed — rather than one built
    from its card's defaults.
    """
    [host] = ActorSystem.find_by_class(WorkspaceHost)
    return system.proxy_ask(host, WorkspaceHost).getResourceOrCreate(WorkspaceActor, config)


class TestTheSelfStopTakesTheStoreChildAndTheExecutorDown:
    """Driven by the grace, not by a fixture: the team stops and the tree reaps itself.

    The only thing that stops the workspace here is its own sweep. The team's
    teardown never reaches a hosted actor, and no fixture stops it before the
    assertions run — a fixture already goes through ``Akgent.stop``, so a spec
    that let one stop the actor could not tell the self-stop's path from it.
    """

    def test_the_team_stops_and_the_workspace_reaps_with_its_store_child_and_backend(
        self, system: ActorSystem, workspaces_root: Path, sandbox_script: SandboxScript
    ) -> None:
        pytest.importorskip("numpy", reason="the [vector_search] extra is not installed")
        from akgentic.tool.vector_store.actor import VectorStoreActor

        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []
        sandbox_script.gate.set()
        workspace = _host_ahead(system, fast_config(REAPED_PATH))
        record = spawn_member(
            system,
            bind_key="reaped",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            workspace_id="notes",
            rag_in_memory=True,
            exec_local=True,
        )
        card, path = bound(record)
        assert path == PurePosixPath(REAPED_PATH)
        assert _wait_until(lambda: len(pykka.ActorRegistry.get_by_class(VectorStoreActor)) == 1)
        [store_ref] = pykka.ActorRegistry.get_by_class(VectorStoreActor)
        # The runner is built and live: a run answers, and nothing has stopped it.
        answer = str(tool_named(card, "workspace_exec")(cmd="echo hi"))
        assert "ok" in answer, answer
        assert sandbox_script.commands == [("echo hi", "")]
        assert sandbox_script.stops == 0
        assert record.orchestrator is not None

        stopped = system.proxy_ask(record.orchestrator, Orchestrator).stop(5.0)
        assert stopped.wait(timeout=SPAWN_TIMEOUT_S), "the team never finished stopping"

        assert _wait_until(lambda: not workspace.is_alive()), "the workspace never reaped"
        assert store_ref.actor_stopped.wait(timeout=HANDSHAKE_TIMEOUT_S), (
            "the store child outlived its workspace"
        )
        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []
        # ``is_alive`` turns false before ``on_stop`` runs, so its last step is waited for.
        assert _wait_until(lambda: sandbox_script.stops == 1), "the backend was never released"
        assert sandbox_script.events[-1] == ("stop",)
        assert ActorSystem.find_by_class(WorkspaceActor) == []


class TestActorSystemShutdownStillReachesAHostedWorkspace:
    def test_shutdown_runs_the_workspaces_on_stop_and_nothing_raises(
        self,
        system: ActorSystem,
        workspaces_root: Path,
        sandbox_script: SandboxScript,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Default config, so no tick fires: only ``ActorRegistry.stop_all`` can stop it.

        ``stop_all`` stops in reverse start order, so the workspace — started
        after both hosts — stops while its host is still alive, and the
        announcement in ``on_stop`` is sent to a live host.
        """
        caplog.set_level(logging.ERROR)
        sandbox_script.gate.set()
        record = spawn_member(
            system,
            bind_key="shutdown",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            workspace_id="notes",
            exec_local=True,
        )
        card, _ = bound(record)
        assert "ok" in str(tool_named(card, "workspace_exec")(cmd="echo hi"))
        [workspace] = ActorSystem.find_by_class(WorkspaceActor)
        assert sandbox_script.stops == 0

        system.shutdown(timeout=10)

        assert not workspace.is_alive()
        assert sandbox_script.stops == 1
        assert ("stop",) in sandbox_script.events
        assert ActorSystem.find_by_class(WorkspaceActor) == []
        name = workspace_actor_name(REAPED_PATH)
        assert [r.getMessage() for r in caplog.records if name in r.getMessage()] == []
