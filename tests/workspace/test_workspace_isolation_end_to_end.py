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
import uuid
from collections.abc import Generator
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
from akgentic.core.utils.deserializer import deserialize_object
from akgentic.core.utils.serializer import SerializableBaseModel

from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import (
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.event import WorkspaceAttached
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    SandboxScript,
    tool_named,
    wait_until,
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

    ``rag_enabled`` and ``exec_local`` switch on the two capabilities whose
    teardown story 51-3 guards: the vector store the card resolves and the exec
    runner. Both default off, so every earlier spec binds exactly the card it did.

    ``workspace_sharable`` is the card's own field, carried verbatim, and defaults
    off exactly as the card's does.
    """

    bind_key: str = ""
    workspace_id: str | None = None
    workspace_metadata_keys: list[str] = []
    workspace_sharable: bool = False
    rag_enabled: bool = False
    exec_local: bool = False


def _resolved_path(card: WorkspaceTool) -> PurePosixPath:
    """The three-segment path *card* actually bound to, read back off its backend.

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
    if config.rag_enabled:
        # No ``vector_store``: a card naming the in-actor backend is refused at
        # bind, and one that names none resolves to the file-backed one.
        fields["workspace_rag_index"] = True
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
                workspace_sharable=self.config.workspace_sharable,
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
    sharable: bool = False,
    rag_enabled: bool = False,
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
            workspace_sharable=sharable,
            rag_enabled=rag_enabled,
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


def _tear_down(actor_system: ActorSystem) -> None:
    """Stop everything, then prove no hosted workspace outlived the system."""
    actor_system.shutdown(timeout=10)
    ActorRegistry.stop_all()
    _BINDS.clear()
    assert ActorSystem.find_by_class(WorkspaceActor) == [], "a workspace outlived its test"


@pytest.fixture
def system() -> Generator[ActorSystem, None, None]:
    """A real actor system running **no host of any kind**, torn down whatever the test did.

    Story 52-6 collapsed two fixtures into this one. There used to be a system
    with a ``WorkspaceHost`` and core's base ``ResourceHost`` beside it — which
    made the host class observable, since a card that named the wrong one failed
    on "no host" rather than on routing — and a ``base_only_system`` with the
    base alone, standing for infra's wiring before it switched. Neither
    distinction exists: this package declares no host, and every card binds its
    actor as an ordinary team child.
    """
    actor_system = ActorSystem()
    try:
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

        assert alice_path == PurePosixPath("u-alice/_id/notes")
        assert bob_path == PurePosixPath("u-bob/_id/notes")
        assert (workspaces_root / "u-alice" / "_id" / "notes").is_dir()
        assert (workspaces_root / "u-bob" / "_id" / "notes").is_dir()
        # Neither is a prefix of the other — containment, not collision, is the
        # hazard the fixed three-segment depth removes.
        assert not str(bob_path).startswith(f"{alice_path}/")
        assert not str(alice_path).startswith(f"{bob_path}/")


##
## AC 8, 11 — the bare card, with the identity propagated rather than injected
##


class TestTheBareCardResolvesUnderItsOwner:
    def test_a_bare_card_reaches_user_id_over_team_id(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """``WorkspaceTool()`` with nothing declared is ``<user_id>/_team/<team_id>``.

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

        assert path == PurePosixPath(f"u-alice/_team/{team_id}")
        assert path.parts[0] != "anonymous"
        assert (workspaces_root / "u-alice" / "_team" / str(team_id)).is_dir()

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
        assert card.workspace._root.parent.name == "_team"
        assert card.workspace._root.parent.parent.name == "u-carol"


##
## AC 9 — the metadata card shares when it says so, and only then
##


class TestTheMetadataCardSharesDeliberately:
    """**Deliberately** now means *declared*: ``workspace_sharable=True``.

    The metadata layout used to share by construction — every metadata tree sat
    under one reserved scope. Sharing is an axis of its own now, orthogonal to
    the kind, and the default for every kind is per-principal. So the property
    the first spec pins — two principals, one tree — is reached by the
    declaration, and the second spec pins the other side of the same decision:
    without it, the same two principals get two trees.
    """

    def test_two_teams_with_the_same_key_values_reach_the_same_tree(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """**This sharing is what the declaration is for — do not "fix" it.**

        Two teams, two team ids, two principals, two orchestrators, one resolved
        path. Everywhere else in this epic a shared tree is the defect; here it
        is the feature the card asked for, which is why it is pinned as tightly
        as the isolation is.
        """
        keys = ["customer_id", "case_id"]
        first = spawn_member(
            system,
            bind_key="acme-a",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=keys,
            sharable=True,
        )
        second = spawn_member(
            system,
            bind_key="acme-b",
            user_id="u-bob",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=keys,
            sharable=True,
        )
        first_card, first_path = bound(first)
        second_card, second_path = bound(second)

        assert first_path == second_path
        assert first_path == PurePosixPath("_shared/_meta/customer_id-ACME__case_id-42")

        # One tree, reached from two teams: what one writes, the other reads.
        tool_named(first_card, "workspace_write")("shared.txt", "case-42\n")
        assert "case-42" in str(tool_named(second_card, "workspace_read")("shared.txt"))

    def test_without_the_declaration_two_principals_reach_two_trees(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """Fail-closed: the same key values, undeclared, isolate as every kind does.

        The inversion of the old implicit sharing, crossed at the capability
        surface rather than asserted on two strings: what Alice's team writes
        through its metadata tree is not there through Bob's.
        """
        keys = ["customer_id", "case_id"]
        alice = spawn_member(
            system,
            bind_key="acme-alice",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=keys,
        )
        bob = spawn_member(
            system,
            bind_key="acme-bob",
            user_id="u-bob",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=keys,
        )
        alice_card, alice_path = bound(alice)
        bob_card, bob_path = bound(bob)

        assert alice_path == PurePosixPath("u-alice/_meta/customer_id-ACME__case_id-42")
        assert bob_path == PurePosixPath("u-bob/_meta/customer_id-ACME__case_id-42")

        tool_named(alice_card, "workspace_write")("case.txt", "alice-only\n")
        with pytest.raises(RetriableError, match="not found"):
            tool_named(bob_card, "workspace_read")("case.txt")
        assert "alice-only" in str(tool_named(alice_card, "workspace_read")("case.txt"))

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
        assert path_42 == PurePosixPath("u-alice/_meta/customer_id-ACME__case_id-42")
        assert path_43 == PurePosixPath("u-alice/_meta/customer_id-ACME__case_id-43")


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

        assert created_path == resumed_path == PurePosixPath(f"u-alice/_team/{team_id}")

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
        assert created_path == PurePosixPath("u-alice/_meta/customer_id-ACME__case_id-42")

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

SHARED_PATH = "_shared/_meta/customer_id-ACME__case_id-42"
"""The shared metadata tree both teams below resolve to — both cards declare it sharable."""

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
    """Two teams, two principals, two team ids, two orchestrators — one shared metadata tree."""
    keys = ["customer_id", "case_id"]
    first = spawn_member(
        system,
        bind_key="acme-a",
        user_id="u-alice",
        team_id=uuid.uuid4(),
        metadata=CaseMetadata(customer_id="ACME", case_id="42"),
        keys=keys,
        sharable=True,
    )
    second = spawn_member(
        system,
        bind_key="acme-b",
        user_id="u-bob",
        team_id=uuid.uuid4(),
        metadata=CaseMetadata(customer_id="ACME", case_id="42"),
        keys=keys,
        sharable=True,
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


class TestTwoTeamsOnOneTreeGetTwoActors:
    """**Premise reversed by decision** (ADR-051, *Ruling A*), across two real orchestrators.

    This class held the headline guard of epic 51: two teams resolving one tree
    were handed **one** hosted actor, because that actor held the exec lease, the
    document cache, the retrieval index and the write gate — state two instances
    could not have held consistently. Every one of those is a file under
    ``<meta>`` now, serialised by the filesystem across processes as well as
    teams, so the actor holds nothing two of it could disagree about and is an
    ordinary team child again.

    The assertions are inverted rather than dropped, and what they pin is what
    still has to be true: **the tree stays one tree**, each team gets its own
    event naming its own agent, and each team's actor is in its own roster and no
    other. The negative that used to sit beside them — the process needs no
    ``WorkspaceHost`` — is structural since 52-6 deleted the class, and is pinned
    by ``test_workspace_event.py`` instead.
    """

    def test_both_binds_succeed_and_each_team_gets_its_own_actor_on_one_tree(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        first, second = _two_teams_on_one_tree(system)
        _, first_path = bound(first)
        _, second_path = bound(second)
        assert first_path == second_path == PurePosixPath(SHARED_PATH)

        workspaces = ActorSystem.find_by_class(WorkspaceActor)

        # Two actors, one per team, both named for the one tree they share.
        assert len(workspaces) == 2
        assert {address.name for address in workspaces} == {workspace_actor_name(SHARED_PATH)}
        assert workspaces[0].agent_id != workspaces[1].agent_id

    def test_the_tree_is_still_one_tree_and_the_gate_still_spans_both_teams(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """Why two actors is safe: the gate is the live file, not the actor.

        Alice's team writes a file it has never read through Bob's team, and Bob
        is refused — across two orchestrators, two actors and two cards that
        share nothing but the directory.
        """
        first, second = _two_teams_on_one_tree(system)
        alice_card, _ = bound(first)
        bob_card, _ = bound(second)

        assert tool_named(alice_card, "workspace_write")("shared.md", "alice\n") == (
            "Written: shared.md"
        )

        with pytest.raises(RetriableError, match="read it before overwriting"):
            tool_named(bob_card, "workspace_write")("shared.md", "bob\n")
        tree = workspaces_root / SHARED_PATH
        assert (tree / "shared.md").read_text(encoding="utf-8") == "alice\n"

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
            # **The envelope's sender moved with the emitter, and the payload did
            # not.** The orchestrator forward used to emit; the card emits
            # through ``notify_event`` now, so ``sender`` is the
            # binding member — which is what the payload already named. The
            # frontend folds on the event, and the event is byte-identical.
            assert message.sender is not None
            assert message.sender.agent_id == record.member.agent_id
            payload_ids.append(message.event.agent_id)
        assert payload_ids[0] != payload_ids[1]

    def test_each_teams_actor_is_that_teams_own_member(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """**Inverted.** It was in no team\'s roster; it is in its own team\'s and no other\'s.

        The guard that matters is unchanged in substance: a team must not be able
        to reach — or stop — another team\'s workspace actor. It used to hold
        because the actor was in nobody\'s roster; it holds now because it is in
        exactly one.
        """
        first, second = _two_teams_on_one_tree(system)
        bound(first)
        bound(second)
        name = workspace_actor_name(SHARED_PATH)

        rosters = []
        for record in (first, second):
            assert record.orchestrator is not None
            assert record.member is not None
            starts = [m for m in _stream_of(system, record) if isinstance(m, StartMessage)]
            # The positive beside the negative: the member's own start IS there.
            assert any(m.sender == record.member for m in starts)
            orchestrator = system.proxy_ask(record.orchestrator, Orchestrator)
            assert orchestrator.get_team_member(record.member.name) is not None
            workspace = orchestrator.get_team_member(name)
            assert workspace is not None
            rosters.append(workspace.agent_id)

        # Each team's roster names a **different** actor: neither can reach the
        # other's, which is the property the hosted arrangement bought by
        # putting the actor in nobody's.
        assert rosters[0] != rosters[1]


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
        # The **payload** is what the frontend folds on and it is byte-identical
        # above. The envelope's ``sender`` moved with the emitter: it was the
        # orchestrator while the forward emitted, and it is the binding member
        # now that the card does — the same agent the payload already named, so
        # the two agree rather than duplicating.
        assert agent_id == wire["sender"]["agent_id"]

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


OWN_METADATA_PATH = "u-alice/_meta/customer_id-ACME__case_id-42"
"""An undeclared metadata card's tree for ``u-alice`` — per-principal, like every kind."""


class TestAProcessWithNoHostBindsGatesAndMutates:
    """**Premise reversed by decision: story 52-5's headline guard, kept by 52-6.**

    It used to assert that a process running only core's base ``ResourceHost``
    **failed** the first bind with *"No WorkspaceHost is running"* — which was
    correct while the card forwarded to one, and is exactly the error that left
    two of ``akgentic-infra`` story 69-1's specs red waiting on that story.

    The card forwards to no host at all, so the same process binds, gates and
    mutates. The three halves are asserted together on purpose: a bind that
    succeeded but gated nothing would pass a weaker spec, and the gate is the
    whole point of the seam.

    **This is the guard that makes deleting ``WorkspaceHost`` safe**, and it is
    unchanged in substance by that deletion — only the fixture is, since there is
    no longer a second kind of system to distinguish this one from.
    """

    def test_the_bind_succeeds_gates_and_mutates_with_no_host_in_the_process(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        record = spawn_member(
            system,
            bind_key="base-only",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            metadata=CaseMetadata(customer_id="ACME", case_id="42"),
            keys=["customer_id", "case_id"],
        )

        assert record.error is None
        card, path = bound(record)
        assert path == PurePosixPath(OWN_METADATA_PATH)
        # No resource host of **any** kind runs in this system — core's base
        # included, which is the one this package could still have named — and
        # the bind neither needed nor made one. Core's class is imported here
        # and nowhere under ``src/``: that asymmetry is the assertion.
        from akgentic.core.resource_host import ResourceHost

        assert ActorSystem.find_by_class(ResourceHost) == []
        assert len(ActorSystem.find_by_class(WorkspaceActor)) == 1

        # It gates: a create lands, and a second write to the same path without
        # a read between is refused exactly as it is anywhere else.
        assert tool_named(card, "workspace_write")("notes.md", "first\n") == "Written: notes.md"
        tree = workspaces_root / OWN_METADATA_PATH
        assert (tree / "notes.md").read_text(encoding="utf-8") == "first\n"
        (tree / "notes.md").write_text("somebody else\n", encoding="utf-8")
        with pytest.raises(RetriableError, match="changed since you read it"):
            tool_named(card, "workspace_write")("notes.md", "second\n")

        # And the event still reaches the team's stream, emitted by the member.
        [message] = _attached_events(system, record)
        assert message.event.workspace_path == OWN_METADATA_PATH


##
## Story 52-6 — the team's teardown reclaims the tree, whole
##

REAPED_PATH = "u-alice/_id/notes"
"""What ``workspace_id="notes"`` resolves to for ``u-alice``; the name is 51-3's."""


def _exec_workers(path: str) -> list[threading.Thread]:
    """The live worker threads of *path*'s exec executor, named by its ``thread_name_prefix``."""
    return [t for t in threading.enumerate() if t.name.startswith(f"exec-{path}_")]


class TestTheTeamsTeardownTakesTheExecutorAndTheStoreDown:
    """**Re-pointed from ``TestTheSelfStopTakesTheExecutorDownAndTheTeamTakesItsStore``.**

    51-3's version was driven by the grace: the only thing that could stop a
    hosted workspace was its own sweep, so the spec pre-created the actor with a
    fast config and waited for it to reap. The sweep, the grace and the self-stop
    are all deleted, and what replaces them is the plainest possible answer — the
    workspace is in its team's roster, so ``Orchestrator.stop`` reaches it.

    **What the spec asserts is unchanged, and it is the pair that matters**: one
    teardown ends both owners. The store is the *team's* member since 52-4, the
    exec runner and its worker thread belong to the workspace, and after the team
    stops there must be neither — a backend left released by nobody is a
    container nobody stops, which is exactly the leak a reap used to prevent.
    """

    def test_the_team_stops_the_store_and_the_workspace_reaps_with_its_backend(
        self, system: ActorSystem, workspaces_root: Path, sandbox_script: SandboxScript
    ) -> None:
        pytest.importorskip("numpy", reason="the [vector_search] extra is not installed")
        from akgentic.tool.vector_store.actor import VectorStoreActor

        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []
        sandbox_script.gate.set()
        record = spawn_member(
            system,
            bind_key="reaped",
            user_id="u-alice",
            team_id=uuid.uuid4(),
            workspace_id="notes",
            rag_enabled=True,
            exec_local=True,
        )
        card, path = bound(record)
        assert path == PurePosixPath(REAPED_PATH)
        [workspace] = ActorSystem.find_by_class(WorkspaceActor)
        assert wait_until(lambda: len(pykka.ActorRegistry.get_by_class(VectorStoreActor)) == 1)
        [store_ref] = pykka.ActorRegistry.get_by_class(VectorStoreActor)
        # The runner is built and live: a run answers, and nothing has stopped it.
        answer = str(tool_named(card, "workspace_exec")(cmd="echo hi"))
        assert "ok" in answer, answer
        assert sandbox_script.commands == [("echo hi", "")]
        assert sandbox_script.stops == 0
        # The run spawned the executor's one worker, and an idle worker never exits
        # on its own: only the teardown's ``shutdown`` ends it.
        assert _exec_workers(REAPED_PATH), "the run left no exec worker to shut down"
        assert record.orchestrator is not None

        stopped = system.proxy_ask(record.orchestrator, Orchestrator).stop(5.0)
        assert stopped.wait(timeout=SPAWN_TIMEOUT_S), "the team never finished stopping"

        assert wait_until(lambda: not workspace.is_alive()), (
            "the team stopped and its workspace actor outlived it"
        )
        assert store_ref.actor_stopped.wait(timeout=HANDSHAKE_TIMEOUT_S), (
            "the team's vector store outlived the team that created it"
        )
        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []
        # ``is_alive`` turns false before ``on_stop`` runs, so its last step is waited for.
        assert wait_until(lambda: sandbox_script.stops == 1), "the backend was never released"
        assert sandbox_script.events[-1] == ("stop",)
        assert wait_until(lambda: not _exec_workers(REAPED_PATH)), (
            "the executor was never shut down"
        )
        assert ActorSystem.find_by_class(WorkspaceActor) == []


class TestActorSystemShutdownStillReachesTheWorkspace:
    def test_shutdown_runs_the_workspaces_on_stop_and_nothing_raises(
        self,
        system: ActorSystem,
        workspaces_root: Path,
        sandbox_script: SandboxScript,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The other stop path: no team teardown, only ``ActorRegistry.stop_all``.

        ``stop_all`` is a graceful stop, so ``on_stop`` runs in full — the backend
        is released — and nothing it does is logged as an error.
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
