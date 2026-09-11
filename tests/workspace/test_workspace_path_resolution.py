"""The one resolver: three segments, scoped to an owner or shared, and no silent fallback.

What these specs are really guarding is that **nothing else derives a workspace
directory**. The resolver's own behaviour is checkable here; that it is the only
derivation is checkable only by the sites that no longer have one, which the
wiring suites assert from the other end.

Every path is ``<scope>/<kind>/<leaf>`` (ADR-052 Decisions 1 and 2). The six cells
of that layout — three kinds, per-principal or shared — are pinned twice below,
against **string literals**: once through the resolver and once through a card
bound by ``observer()``. The second pass is not redundant. A resolver specced in
isolation stays green when the card never hands it the field, and that is
exactly the "resolver nobody called at bind time" defect epic 52 shipped.

The property tests use a seeded ``random.Random`` rather than ``hypothesis``,
which is not a dependency of this package. The seed is fixed, so a failure is
reproducible from the test name alone.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

import pytest
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import ValidationError

from akgentic.tool.workspace import workspace as workspace_module
from akgentic.tool.workspace.models import GIT_DIR_SUFFIX, META_DIR_SUFFIX
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import (
    ANONYMOUS,
    ID_KIND,
    METADATA_KIND,
    RESERVED_KINDS,
    RESERVED_SCOPES,
    SHARED_KINDS_ENV,
    SHARED_SCOPE,
    TEAM_KIND,
    leaf_segment,
    resolve_workspace_path,
    user_segment,
)
from tests.workspace.conftest import FakeActorToolObserver, FakeOrchestratorProxy

# Spelled by escape, never pasted: an editor or a normaliser can rewrite the glyph
# silently, and the specs using it would then prove nothing.
LONG_S = chr(0x017F)  # LATIN SMALL LETTER LONG S

# A stand-in for the team metadata model this package is not allowed to import.
# The resolver reads declared keys off it by attribute and never learns its type,
# which is the whole point: the key names are configuration, the model is data.


class Metadata(SerializableBaseModel):
    """Two business keys and one nullable field, as a real metadata model has."""

    customer_id: str | None = None
    case_id: str | None = None
    region: str | None = None


def resolve(
    *,
    workspace_id: str | None = None,
    keys: list[str] | None = None,
    team_id: str = "11111111-2222-3333-4444-555555555555",
    user_id: str | None = "u-alice",
    metadata: SerializableBaseModel | None = None,
    sharable: bool = False,
) -> PurePosixPath:
    """Call the resolver with the defaults every spec below shares."""
    return resolve_workspace_path(
        workspace_id=workspace_id,
        workspace_metadata_keys=keys or [],
        team_id=team_id,
        user_id=user_id,
        metadata=metadata,
        workspace_sharable=sharable,
    )


##
## The six cells — the layout, pinned against literals
##


@dataclass(frozen=True)
class Cell:
    """One cell of the layout: a card declaration and the literal path it reaches.

    ``expected`` is a **literal**, never built from a constant or a helper: the
    table is what pins the layout independently of everything that derives it.
    ``{team_id}`` is the one placeholder, because a bound card's team id is the
    observer's and only exists once the observer does.
    """

    name: str
    workspace_id: str | None
    keys: tuple[str, ...]
    sharable: bool
    expected: str

    def card(self) -> WorkspaceTool:
        """The card this cell declares."""
        return WorkspaceTool(
            workspace_id=self.workspace_id,
            workspace_metadata_keys=list(self.keys),
            workspace_sharable=self.sharable,
        )


SIX_CELLS = (
    Cell("team", None, (), False, "alice/_team/{team_id}"),
    Cell("id", "notes", (), False, "alice/_id/notes"),
    Cell("meta", None, ("customer_id",), False, "alice/_meta/customer_id-ACME"),
    Cell("team-shared", None, (), True, "_shared/_team/{team_id}"),
    Cell("id-shared", "notes", (), True, "_shared/_id/notes"),
    Cell("meta-shared", None, ("customer_id",), True, "_shared/_meta/customer_id-ACME"),
)
"""ADR-052 Decision 2's table, for principal ``alice`` over metadata ``customer_id=ACME``."""

CELL_METADATA = Metadata(customer_id="ACME")
"""The team metadata every cell is resolved against."""


class Permit(Enum):
    """``bind_cell``'s default permission: exactly the kind the cell binds, if shared."""

    OWN_KIND = "own-kind"


def own_kind_token(cell: Cell) -> str:
    """The ``AKGENTIC_WORKSPACE_SHARED_KINDS`` token for *cell*'s kind — a literal, per name.

    Read off the cell's own literal name (``id-shared`` → ``id``) rather than
    derived the way the source derives it, so a spec granting it cannot agree
    with a derivation that drifted.
    """
    return cell.name.removesuffix("-shared")


def unbound_cell(
    cell: Cell, orchestrator_proxy: FakeOrchestratorProxy
) -> tuple[WorkspaceTool, FakeActorToolObserver]:
    """*cell*'s card and principal ``alice``'s observer, before any bind.

    ``user_id="alice"`` is set **explicitly** — never the fake's default
    principal and never ``None`` — so a shared cell's ``_shared`` is
    distinguishable from anything the fake would have supplied on its own. The
    team carries :data:`CELL_METADATA`, so a metadata cell resolves rather than
    failing on "the team carries no metadata".

    Separate from :func:`bind_cell` only so a spec about a **refused** bind can
    hold the card and the observer afterwards: ``bind_cell`` cannot hand back
    what it raised out of.
    """
    orchestrator_proxy.metadata = CELL_METADATA
    return cell.card(), FakeActorToolObserver(orchestrator_proxy, user_id="alice")


def bind_cell(
    cell: Cell,
    orchestrator_proxy: FakeOrchestratorProxy,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared_kinds: str | Permit | None = Permit.OWN_KIND,
) -> tuple[WorkspaceTool, str, FakeActorToolObserver]:
    """Bind *cell*'s card through ``observer()`` for principal ``alice``.

    The observer comes back with the card because the card holds it weakly.

    ``AKGENTIC_WORKSPACE_SHARED_KINDS`` is set through ``monkeypatch.context()``
    around ``observer()`` **only**, and never ambient. By default a shared cell
    permits exactly its own kind and a per-principal cell leaves the variable
    unset; *shared_kinds* overrides that with a value to set (``""`` included)
    or ``None`` to delete it.

    Returns:
        The bound card, the cell's literal path with the team id filled in, and
        the observer.
    """
    card, observer = unbound_cell(cell, orchestrator_proxy)
    if shared_kinds is Permit.OWN_KIND:
        shared_kinds = own_kind_token(cell) if cell.sharable else None
    with monkeypatch.context() as patch:
        if shared_kinds is None:
            patch.delenv(SHARED_KINDS_ENV, raising=False)
        else:
            patch.setenv(SHARED_KINDS_ENV, shared_kinds)
        card.observer(observer)
    return card, cell.expected.format(team_id=observer.team_id), observer


def _cell_ids(cell: Cell) -> str:
    return cell.name


class TestTheSixCells:
    """AC 1 — every cell through the resolver, against its literal."""

    @pytest.mark.parametrize("cell", SIX_CELLS, ids=_cell_ids)
    def test_each_cell_resolves_to_its_literal_path(self, cell: Cell) -> None:
        path = resolve(
            workspace_id=cell.workspace_id,
            keys=list(cell.keys),
            team_id="team-9",
            user_id="alice",
            metadata=CELL_METADATA,
            sharable=cell.sharable,
        )

        assert str(path) == cell.expected.format(team_id="team-9")

    def test_the_six_cells_are_six_distinct_paths(self) -> None:
        """AC 7 — one input tuple, six declarations, six trees.

        The team id is **equal to** the ``workspace_id`` here, deliberately: with
        two different leaves the six paths would be distinct even with the kind
        segment deleted, and the spec would prove nothing the kind is for. With
        one shared leaf, only ``_team`` versus ``_id`` keeps a named workspace off
        the team's own tree.
        """
        paths = {
            resolve(
                workspace_id=cell.workspace_id,
                keys=list(cell.keys),
                team_id="notes",
                user_id="alice",
                metadata=CELL_METADATA,
                sharable=cell.sharable,
            )
            for cell in SIX_CELLS
        }

        assert len(paths) == 6

    def test_a_workspace_named_after_the_team_id_does_not_reach_the_teams_tree(self) -> None:
        """The collision the kind segment removes, stated directly.

        A team id is a UUID an author can read and type. Under two segments,
        ``workspace_id=<that uuid>`` resolved to ``alice/<uuid>`` — the team's own
        default tree. The kind keeps the two apart.
        """
        team_id = "11111111-2222-3333-4444-555555555555"

        named = resolve(workspace_id=team_id, team_id=team_id, user_id="alice")
        default = resolve(team_id=team_id, user_id="alice")

        assert named == PurePosixPath(f"alice/_id/{team_id}")
        assert default == PurePosixPath(f"alice/_team/{team_id}")

    def test_the_flag_is_required_so_an_unaware_caller_fails_loudly(self) -> None:
        """No default: a caller that does not know about sharing gets a ``TypeError``.

        A default of ``False`` would resolve a shared card's tree to a
        per-principal path at every such call site — an empty directory read
        after passing authorisation, and nothing raised.
        """
        with pytest.raises(TypeError, match="workspace_sharable"):
            resolve_workspace_path(  # type: ignore[call-arg]
                workspace_id="notes",
                workspace_metadata_keys=[],
                team_id="team-9",
                user_id="alice",
                metadata=None,
            )


class TestTheSixCellsBind:
    """AC 2 — every cell through a card bound by ``observer()``, never the resolver alone.

    What a hard-coded ``workspace_sharable=False`` in the card turns red is
    **these** specs' three shared cells; every resolver-level spec above stays
    green under it, because the resolver is never the part that forgot.
    """

    @pytest.mark.parametrize("cell", SIX_CELLS, ids=_cell_ids)
    def test_the_card_binds_its_cells_path_everywhere_it_hands_one(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, expected, _observer = bind_cell(cell, orchestrator_proxy, monkeypatch)

        assert card._workspace_path == expected
        tree = workspaces_root / expected
        assert card.workspace._root == tree.resolve()
        assert tree.is_dir()
        assert f"#Workspace-{expected}" in orchestrator_proxy.children

    @pytest.mark.parametrize("cell", SIX_CELLS, ids=_cell_ids)
    def test_the_bind_creates_exactly_that_one_actor(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An equality over the created set — a second, wrong tree cannot hide beside it."""
        _card, expected, _observer = bind_cell(cell, orchestrator_proxy, monkeypatch)

        assert set(orchestrator_proxy.children) == {f"#Workspace-{expected}"}


class TestSharableRoundTrip:
    """AC 3 — a card that went through a store keeps its value and its tree.

    ``SerializableBaseModel``'s serializer emits every declared field, and the
    agent-card store round-trips a card on every team resume. A design reading
    the author's intent off ``model_fields_set`` answers correctly on the first
    load and wrongly on every resume after it — the defect story 52-4 fixed.
    """

    @pytest.mark.parametrize(
        ("declared", "sharable"),
        [
            # The author never wrote the field — the exact shape story 52-4's
            # defect had: correct on first load, wrong on every resume after.
            ({}, False),
            ({"workspace_sharable": False}, False),
            ({"workspace_sharable": True}, True),
        ],
        ids=["never-written", "written-false", "written-true"],
    )
    def test_the_value_and_the_path_survive_the_round_trip(
        self,
        declared: dict[str, bool],
        sharable: bool,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = WorkspaceTool(workspace_id="notes", **declared)
        reloaded = WorkspaceTool.model_validate(original.model_dump())

        # The trip really did inflate the naive discriminator — without this the
        # spec could pass while proving nothing.
        assert "workspace_sharable" in reloaded.model_fields_set
        assert reloaded.workspace_sharable is sharable

        # Both observers held here: a card holds its observer weakly. ``id`` is
        # the kind every row binds, so it is permitted around both binds — and a
        # per-principal row still lands under ``alice``: a permission never
        # makes a card shared.
        observers = [FakeActorToolObserver(orchestrator_proxy, user_id="alice") for _ in range(2)]
        with monkeypatch.context() as patch:
            patch.setenv(SHARED_KINDS_ENV, "id")
            original.observer(observers[0])
            reloaded.observer(observers[1])

        literal = "_shared/_id/notes" if sharable else "alice/_id/notes"
        assert original._workspace_path == literal
        assert reloaded._workspace_path == literal

    def test_the_default_is_a_real_false_not_an_absence(self) -> None:
        """A plain ``bool`` with a real default — never ``None``, never "unset"."""
        card = WorkspaceTool()

        assert card.workspace_sharable is False
        assert WorkspaceTool.model_fields["workspace_sharable"].annotation is bool

    @pytest.mark.parametrize(
        "layout", [{}, {"workspace_id": "notes"}, {"workspace_metadata_keys": ["customer_id"]}]
    )
    def test_it_is_orthogonal_to_the_layout_fields(self, layout: dict[str, object]) -> None:
        """Valid with each of the three layouts: it is not part of ``_one_layout``."""
        card = WorkspaceTool.model_validate({**layout, "workspace_sharable": True})

        assert card.workspace_sharable is True


##
## AC 8 — the shared cell does not consult the principal
##


class TestTheSharedCellIgnoresThePrincipal:
    def test_no_principal_resolves_to_the_shared_scope_not_anonymous(self) -> None:
        assert resolve(workspace_id="notes", user_id=None, sharable=True) == PurePosixPath(
            "_shared/_id/notes"
        )

    @pytest.mark.parametrize("unusable", ["acme/x", "", ".hidden", "_shared"])
    def test_an_unusable_principal_still_resolves_a_shared_tree(self, unusable: str) -> None:
        """A resolver must not refuse on an input it does not use."""
        assert resolve(workspace_id="notes", user_id=unusable, sharable=True) == PurePosixPath(
            "_shared/_id/notes"
        )

    def test_the_same_principal_is_still_refused_per_principal(self) -> None:
        """The other half: the guard is skipped only where the principal is unused."""
        with pytest.raises(ValueError, match="not usable as a workspace directory name"):
            resolve(workspace_id="notes", user_id="acme/x", sharable=False)


##
## AC 10, 11 — the scope segment
##


class TestUserSegment:
    """The user id goes in verbatim; only what cannot be a directory name is refused."""

    @pytest.mark.parametrize(
        "produced",
        [
            # The Azure AD `sub`: 43 characters of base64url, whose alphabet
            # includes `_` — which is why the reservation is exact-match.
            "2R0bQV8j9zX8CEsBl6APi7MXgAn4_laOa8vd9ZoIHIQ",
            # No auth configured.
            "anonymous",
            # An M2M service principal: an Azure AD app id.
            "1f9e4c2a-7b3d-4e5f-8a9b-2c3d4e5f6a7b",
            # An API key's admin-supplied owner_id. `@` and `.` are legal in a
            # filename, so an email needs no encoding.
            "alice@acme.example",
            # A team built through the SDK names no user.
            "cli",
        ],
    )
    def test_every_configured_producers_value_survives_unchanged(self, produced: str) -> None:
        assert user_segment(produced) == produced

    def test_no_user_id_at_all_is_the_anonymous_scope(self) -> None:
        assert user_segment(None) == ANONYMOUS

    def test_an_oidc_token_carrying_no_sub_raises_rather_than_hiding_a_tree(self) -> None:
        """``str(claims.get("sub", ""))`` defaults to empty — reachable, not an attack.

        Nobody has to misbehave to produce this, and it must still raise: an
        empty segment would give every affected user one shared hidden
        directory, which is a silent isolation failure.
        """
        with pytest.raises(ValueError, match="not usable as a workspace directory name"):
            user_segment("")

    @pytest.mark.parametrize(
        "unusable",
        [
            ".hidden",  # a dot-directory
            ".",
            "..",
            "acme/alice",  # an admin typing an org-scoped owner_id
            "back\\slash",
            "nul\x00byte",
        ],
    )
    def test_a_value_that_cannot_be_a_directory_name_raises(self, unusable: str) -> None:
        with pytest.raises(ValueError):
            user_segment(unusable)

    def test_the_reserved_names_are_pinned_to_their_literals(self) -> None:
        # Pinned against the literals, not against the constants: the module
        # derives the sets from the constants, so comparing them would be a
        # tautology and a rename of both at once would slip through unnoticed.
        assert SHARED_SCOPE == "_shared"
        assert TEAM_KIND == "_team"
        assert ID_KIND == "_id"
        assert METADATA_KIND == "_meta"
        assert RESERVED_KINDS == frozenset({"_team", "_id", "_meta"})
        assert RESERVED_SCOPES == frozenset({"_shared", "_team", "_id", "_meta"})

    @pytest.mark.parametrize("reserved", ["_shared", "_team", "_id", "_meta"])
    def test_a_reserved_scope_is_refused(self, reserved: str) -> None:
        """``_shared`` above all: a principal of that name would **be** the shared cell."""
        with pytest.raises(ValueError, match="reserved scope"):
            user_segment(reserved)

    def test_a_principal_named_shared_cannot_land_in_the_shared_cell(self) -> None:
        """Its per-principal ``_shared/_id/notes`` is the shared ``_shared/_id/notes``."""
        with pytest.raises(ValueError, match="reserved scope"):
            resolve(workspace_id="notes", user_id="_shared")

    @pytest.mark.parametrize("spelling", ["_SHARED", "_Shared", "_TEAM", "_Id", "_META", "_Meta"])
    def test_the_reservation_is_case_insensitive(self, spelling: str) -> None:
        """macOS and Windows are case-insensitive: ``_SHARED/`` **is** ``_shared/`` there."""
        with pytest.raises(ValueError, match="reserved scope"):
            user_segment(spelling)

    def test_a_long_s_principal_is_refused_because_the_match_case_folds(self) -> None:
        """A long-s ``_shared`` **is** ``_shared/`` on a case-folding volume; ``lower()`` misses it.

        U+017F is already lowercase, so ``lower()`` leaves it alone and only
        ``casefold()`` maps it onto ``s``. The premise is asserted first: a Python
        that changed either answer says so here, rather than the spec silently
        proving nothing.
        """
        principal = f"_{LONG_S}hared"
        assert principal.lower() == principal
        assert principal.casefold() == "_shared"

        with pytest.raises(ValueError, match="reserved scope"):
            user_segment(principal)

    def test_a_long_s_principal_is_refused_out_of_the_resolver(self) -> None:
        principal = f"_{LONG_S}hared"
        assert principal.lower() == principal
        assert principal.casefold() == "_shared"

        with pytest.raises(ValueError, match="reserved scope"):
            resolve(workspace_id="notes", user_id=principal)

    def test_a_long_s_principal_fails_the_bind_and_creates_nothing(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspaces_root: Path
    ) -> None:
        """The agent-side path: the card's own bind, not only the resolver it calls.

        Nothing may be left on disk. ``get_workspace`` creates the tree eagerly,
        so a guard that let this principal through would leave its long-s
        directory behind — which a case-folding volume opens as ``_shared``.
        """
        principal = f"_{LONG_S}hared"
        assert principal.lower() == principal
        assert principal.casefold() == "_shared"
        card = WorkspaceTool(workspace_id="notes")
        observer = FakeActorToolObserver(orchestrator_proxy, user_id=principal)

        with pytest.raises(ValueError, match="reserved scope"):
            card.observer(observer)

        assert list(workspaces_root.iterdir()) == []

    @pytest.mark.parametrize(
        "accepted",
        [
            "_shared2",
            "_share",
            "shared",
            "_teams",
            "_ids",
            "_i",
            "x_team",
            "_metadata",
            "_meta2",
            "_met",
            "_",
            "x_meta",
        ],
    )
    def test_the_reservation_is_exact_match_not_an_underscore_prefix(
        self, accepted: str
    ) -> None:
        """Reserving the whole ``_`` namespace looks tidier and is wrong.

        The Azure AD ``sub`` alphabet is base64url and includes ``_``, so a
        prefix rule would refuse roughly one user in sixty-four at team creation.
        """
        assert user_segment(accepted) == accepted


##
## AC 12 — the leaf segment
##


class TestLeafSegment:
    """The same guard, reserving the kind names rather than the scope names."""

    @pytest.mark.parametrize(
        "accepted",
        [
            "notes",
            "11111111-2222-3333-4444-555555555555",
            "customer_id-ACME__case_id-42",
            "a.b.c",
        ],
    )
    def test_a_usable_leaf_survives_unchanged(self, accepted: str) -> None:
        assert leaf_segment(accepted) == accepted

    @pytest.mark.parametrize(
        "unusable", ["", ".", "..", ".hidden", "a/b", "../x", "a\\b", "a\x00b"]
    )
    def test_a_leaf_that_is_not_one_segment_raises(self, unusable: str) -> None:
        with pytest.raises(ValueError, match="not usable as a directory name"):
            leaf_segment(unusable)

    @pytest.mark.parametrize("kind", ["_team", "_id", "_meta"])
    def test_a_workspace_named_after_a_kind_is_refused(self, kind: str) -> None:
        """**Reversed by decision** (ADR-052 reversed ADR-048 Decision 4's leaf rule).

        This spec used to assert the opposite — that a workspace named ``_meta``
        was accepted, because ``_meta`` was a scope and a leaf of that name
        collided with nothing. Under the three-segment layout a kind-named leaf
        still collides with nothing at its own depth; what it reaches is a tree
        of a **different** depth. ``alice/_meta``, legal under the old layout, is
        the parent of every ``alice/_meta/*`` tree minted now, and an agent
        anchored there reads and writes all of them as ordinary in-tree activity.
        Refusing the name means nothing from here on can mint such a parent.
        """
        with pytest.raises(ValueError, match="reserved kind"):
            leaf_segment(kind)
        with pytest.raises(ValueError, match="reserved kind") as excinfo:
            resolve(workspace_id=kind)

        # The refused value appears verbatim, so the admin can see what to rename.
        assert kind in str(excinfo.value)

    @pytest.mark.parametrize("spelling", ["_TEAM", "_Team", "_ID", "_Id", "_META", "_Meta"])
    def test_the_kind_reservation_is_case_insensitive(self, spelling: str) -> None:
        """``alice/_META`` is ``alice/_meta`` on the platform this is developed on."""
        with pytest.raises(ValueError, match="reserved kind"):
            leaf_segment(spelling)

    def test_the_kind_match_case_folds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``casefold()``, not ``lower()`` — observable only through a widened table.

        No shipped kind name has a letter with a non-ASCII fold: ``_team``, ``_id``
        and ``_meta`` carry no ``s``, so no real input tells the two comparisons
        apart today. Widening :data:`RESERVED_KINDS` with ``_scratch`` makes the
        fold observable now, rather than on the day a kind with an ``s`` is added.
        """
        leaf = f"_{LONG_S}cratch"
        assert leaf.lower() == leaf
        assert leaf.casefold() == "_scratch"
        monkeypatch.setattr(workspace_module, "RESERVED_KINDS", RESERVED_KINDS | {"_scratch"})

        with pytest.raises(ValueError, match="reserved kind"):
            leaf_segment(leaf)

    def test_the_suffix_match_case_folds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same, for the sidecar suffixes — neither ``.git`` nor ``.index`` has an ``s``.

        A ``.snap`` sidecar is invented for the purpose: the table is widened so
        a suffix comparison put back to ``lower()`` goes red here, instead of
        passing a long-s ``.snap`` leaf the day a real suffix carries that letter.
        """
        leaf = f"x.{LONG_S}nap"
        assert leaf.lower() == leaf
        assert leaf.casefold() == "x.snap"
        widened = {**workspace_module._SIDECAR_SUFFIXES, ".snap": "snapshot"}
        monkeypatch.setattr(workspace_module, "_SIDECAR_SUFFIXES", widened)

        with pytest.raises(ValueError, match="snapshot"):
            leaf_segment(leaf)

    @pytest.mark.parametrize(
        "accepted", ["_shared", "_shared2", "_teams", "_ids", "_i", "x_team", "_metadata"]
    )
    def test_the_kind_reservation_is_exact_match(self, accepted: str) -> None:
        """``_shared`` included: it is a scope, and a third-position leaf of that name is inert."""
        assert leaf_segment(accepted) == accepted
        assert resolve(workspace_id=accepted, user_id="alice") == PurePosixPath(
            f"alice/_id/{accepted}"
        )

    def test_a_leaf_ending_in_the_journal_suffix_is_refused(self) -> None:
        """``notes.git`` **is** workspace ``notes``'s repository, not a name near it.

        The journal is a sibling of the tree in the same directory, so a card
        declaring this roots its own ``Filesystem`` at another workspace's
        history and reads, writes and deletes inside it as ordinary in-tree
        activity. Nothing downstream raises: ``_validate_path`` refuses only what
        resolves *outside* the root, and that root is a real directory.
        """
        with pytest.raises(ValueError, match="journal directory"):
            leaf_segment(f"notes{GIT_DIR_SUFFIX}")

    def test_the_suffix_match_is_case_insensitive(self) -> None:
        """``git_dir_for`` emits lowercase, and that is not what decides this.

        macOS and Windows filesystems are case-insensitive by default, so
        ``notes.GIT`` and ``notes.git`` are one directory there — an exact-match
        guard would pass the collision straight through on the platform most of
        this is developed on.
        """
        for spelling in ("notes.GIT", "notes.Git", "notes.gIt"):
            with pytest.raises(ValueError, match="journal directory"):
                leaf_segment(spelling)

    def test_the_bare_suffix_is_refused_too(self) -> None:
        """``.git`` alone is already refused as a dot-directory, and must stay refused."""
        with pytest.raises(ValueError):
            leaf_segment(GIT_DIR_SUFFIX)

    @pytest.mark.parametrize(
        "accepted",
        [
            "gitnotes",  # the four characters, in the wrong place
            "notes.github",  # a longer suffix that merely starts the same way
            "notes.git.txt",  # the suffix mid-name, which names no repository
            "notesgit",
            "git",
            "notes",
        ],
    )
    def test_only_the_suffix_is_refused_never_the_substring(self, accepted: str) -> None:
        """The guard is a suffix rule, and over-refusing costs a real workspace its name."""
        assert leaf_segment(accepted) == accepted

    def test_the_rule_is_spelled_once_against_the_journals_own_constant(self) -> None:
        """One rule, one spelling — the defect 48-1's review caught, one scale down.

        ``git_dir_for`` builds the directory from ``GIT_DIR_SUFFIX``; this guard
        refuses the same constant. A second ``".git"`` literal here is a rule
        that has to agree with another one, and the two would drift silently.
        """
        assert leaf_segment(f"notes{GIT_DIR_SUFFIX}x") == f"notes{GIT_DIR_SUFFIX}x"
        with pytest.raises(ValueError):
            leaf_segment(f"x{GIT_DIR_SUFFIX}")

    def test_it_refuses_rather_than_renaming(self) -> None:
        """No suffix-stripping, no relocate-and-log: a raise, and no path back.

        A deployment that genuinely has a workspace named ``foo.git`` must be
        told at team creation, in front of the admin who caused it. Silently
        moving the tree is the failure this guard exists to prevent, not a
        gentler form of it.
        """
        with pytest.raises(ValueError) as excinfo:
            resolve(workspace_id="notes.git", user_id="alice")

        # The refused value appears verbatim, so the admin can see what to rename.
        assert "notes.git" in str(excinfo.value)


##
## The journal-suffix guard is asymmetric: the scope needs no equivalent, and
## that must stay pinned.  (Numbered banners in this file cite *epic* ACs; this
## one has no epic AC of its own, so it is named after the behaviour instead.)
##


class TestTheScopeGuardIsNotSymmetric:
    """A *scope* ending in ``.git`` collides with nothing, and must be accepted."""

    @pytest.mark.parametrize("principal", ["alice.git", "alice.GIT", "git", "_meta.git"])
    def test_a_principal_whose_id_ends_in_the_journal_suffix_is_accepted(
        self, principal: str
    ) -> None:
        """Journals hang off the **leaf**, so there is no journal beside a scope.

        ``git_dir_for(root)`` returns ``root.parent / f"{root.name}.git"`` — a
        sibling of the *tree*, inside ``<scope>/<kind>/``. A scope named
        ``alice.git`` would be a sibling of a hypothetical ``alice`` scope's
        nothing, because scopes have no journals. Adding the guard symmetrically
        is the obvious next edit for anyone reading the diff, and an Azure AD
        ``sub`` or an admin-typed ``owner_id`` could plausibly end in those four
        characters — so this spec exists to break instead of a real principal
        being refused at team creation.
        """
        assert user_segment(principal) == principal

    def test_a_principal_ending_in_the_suffix_resolves_a_whole_path(self) -> None:
        assert resolve(workspace_id="notes", user_id="alice.git") == PurePosixPath(
            "alice.git/_id/notes"
        )


##
## AC 5 — the two per-user layouts
##


class TestPerUserLayouts:
    def test_no_workspace_id_puts_the_team_tree_under_its_owner(self) -> None:
        assert resolve(team_id="team-9", user_id="alice") == PurePosixPath("alice/_team/team-9")

    def test_a_named_workspace_is_one_of_this_principals_trees(self) -> None:
        assert resolve(workspace_id="notes", user_id="alice") == PurePosixPath("alice/_id/notes")

    def test_two_principals_declaring_one_name_reach_two_paths(self) -> None:
        """The exposure, in one assertion.

        ``notes`` used to be a global key: whoever typed it got the same tree.
        """
        alice = resolve(workspace_id="notes", user_id="alice")
        bob = resolve(workspace_id="notes", user_id="bob")

        assert alice != bob
        assert alice == PurePosixPath("alice/_id/notes")
        assert bob == PurePosixPath("bob/_id/notes")

    def test_no_principal_resolves_to_the_anonymous_scope(self) -> None:
        assert resolve(workspace_id="notes", user_id=None) == PurePosixPath("anonymous/_id/notes")

    def test_an_empty_workspace_id_raises_rather_than_falling_back_to_the_team(self) -> None:
        """A card carrying ``workspace_id=""`` named a workspace and got it wrong.

        Falling through to the team id would answer that mistake silently, and
        the card would then look as though it had never named one.
        """
        with pytest.raises(ValueError, match="not usable as a directory name"):
            resolve(workspace_id="", team_id="team-9")

    @pytest.mark.parametrize("traversal", ["../x", "a/b", ".", ".."])
    def test_a_hand_typed_workspace_id_that_is_not_one_segment_raises(
        self, traversal: str
    ) -> None:
        """``<user>/_id/../x`` is one segment *outside* the kind directory.

        ``Filesystem`` does not validate the name it is given at all, so this
        guard is the only thing holding the three-segment invariant for a value
        an author types by hand.
        """
        with pytest.raises(ValueError):
            resolve(workspace_id=traversal)

    def test_an_unusable_principal_raises_out_of_the_resolver(self) -> None:
        with pytest.raises(ValueError, match="not usable as a workspace directory name"):
            resolve(workspace_id="notes", user_id="acme/alice")


##
## AC 6, 7 — the metadata layout
##


class TestMetadataLayout:
    def test_the_declared_keys_key_a_tree_under_its_owner(self) -> None:
        path = resolve(
            keys=["customer_id", "case_id"],
            metadata=Metadata(customer_id="ACME", case_id="42"),
        )
        assert path == PurePosixPath("u-alice/_meta/customer_id-ACME__case_id-42")

    def test_it_is_per_principal_unless_declared_sharable(self) -> None:
        """**Reversed by decision**: a metadata tree no longer shares by construction.

        It used to sit under a reserved scope and be one tree for every user —
        sharing as a side effect of which layout field an author filled in. The
        default is per-principal for every kind now, and sharing is declared.
        """
        metadata = Metadata(customer_id="ACME", case_id="42")
        keys = ["customer_id", "case_id"]
        alice = resolve(keys=keys, user_id="alice", metadata=metadata)
        bob = resolve(keys=keys, user_id="bob", metadata=metadata)

        assert alice == PurePosixPath("alice/_meta/customer_id-ACME__case_id-42")
        assert bob == PurePosixPath("bob/_meta/customer_id-ACME__case_id-42")

        shared_alice = resolve(keys=keys, user_id="alice", metadata=metadata, sharable=True)
        shared_bob = resolve(keys=keys, user_id="bob", metadata=metadata, sharable=True)

        assert shared_alice == shared_bob
        assert shared_alice == PurePosixPath("_shared/_meta/customer_id-ACME__case_id-42")

    def test_the_declaration_is_a_sequence_so_the_order_names_the_scope(self) -> None:
        """Two orders address two workspaces, and that is the decision, not a defect.

        The list is an ordered refinement path — first key coarsest — so the
        order is part of what was declared. Two cards naming the same keys
        differently declared different scopes, and unlike a silent collision the
        difference is **visible in the directory name**.

        Both exact strings are asserted rather than mere inequality: a spec that
        only checked ``forwards != backwards`` would stay green under a sort
        applied to one side.
        """
        metadata = Metadata(customer_id="ACME", case_id="42")
        forwards = resolve(keys=["customer_id", "case_id"], metadata=metadata)
        backwards = resolve(keys=["case_id", "customer_id"], metadata=metadata)

        assert forwards == PurePosixPath("u-alice/_meta/customer_id-ACME__case_id-42")
        assert backwards == PurePosixPath("u-alice/_meta/case_id-42__customer_id-ACME")
        assert forwards != backwards

    def test_refining_a_key_list_extends_the_name_so_ls_groups_the_family(self) -> None:
        """The property change A exists for: ``ls <scope>/_meta/`` groups a customer's trees.

        Sorted, ``["customer_id"]`` and ``["customer_id", "case_id"]`` produced two
        unrelated names. In declaration order the second **string-**extends the
        first, so a customer's workspaces sit beside each other in a listing.
        """
        metadata = Metadata(customer_id="ACME", case_id="42")
        coarse = resolve(keys=["customer_id"], metadata=metadata)
        refined = resolve(keys=["customer_id", "case_id"], metadata=metadata)

        assert coarse == PurePosixPath("u-alice/_meta/customer_id-ACME")
        assert refined == PurePosixPath("u-alice/_meta/customer_id-ACME__case_id-42")
        assert refined.name.startswith(coarse.name)

    def test_a_different_key_set_reaches_a_different_tree(self) -> None:
        """And a *sibling* one, never a parent — the antichain.

        One leaf name is now a **string** prefix of the other, which is the point
        of declaration order. It is not a **path** prefix: both are single leaves
        under one ``u-alice/_meta``, and a sibling cannot contain a sibling.
        Containment is what the antichain guards, so nothing here weakens it —
        and a reader who "fixes" the shared string prefix breaks this spec.
        """
        metadata = Metadata(customer_id="ACME", case_id="42")
        one = resolve(keys=["customer_id"], metadata=metadata)
        both = resolve(keys=["customer_id", "case_id"], metadata=metadata)

        assert one != both
        assert str(both.name).startswith(str(one.name))  # string prefix: deliberate
        assert not str(both).startswith(f"{one}/")  # path prefix: never
        assert not str(one).startswith(f"{both}/")
        assert one.parent == both.parent  # siblings under the one scope and kind

    def test_a_repeated_key_names_the_same_tree_as_naming_it_once(self) -> None:
        """First-occurrence dedupe, not set semantics — a repeated key adds no scope.

        With the sort gone the dedupe has to be explicit *and* stable:
        ``["a", "b", "a"]`` would otherwise either double ``a`` in the leaf or
        move ``b`` ahead of it, depending on how it was written.
        """
        metadata = Metadata(customer_id="ACME", case_id="42")
        assert resolve(keys=["customer_id", "customer_id"], metadata=metadata) == resolve(
            keys=["customer_id"], metadata=metadata
        )
        assert resolve(
            keys=["customer_id", "case_id", "customer_id"], metadata=metadata
        ) == PurePosixPath("u-alice/_meta/customer_id-ACME__case_id-42")

    @pytest.mark.parametrize(
        ("value", "encoded"),
        [
            ("ACME", "ACME"),
            ("a-b", "a%2Db"),
            ("a_b", "a%5Fb"),
            ("a/b", "a%2Fb"),
            ("a%b", "a%25b"),
            ("a~b", "a%7Eb"),
            ("a b", "a%20b"),
            ("é", "%C3%A9"),
            ("a.b", "a.b"),
        ],
    )
    def test_every_character_outside_the_safe_set_is_percent_encoded(
        self, value: str, encoded: str
    ) -> None:
        """Including ``-``, ``_`` and ``~``, which ``urllib.parse.quote`` cannot encode.

        ``quote``'s always-safe set is hard-coded ``ascii_letters + digits +
        "_.-~"`` and no argument — ``safe=""`` included — forces those three to
        encode. They are exactly the separator characters, so ``quote`` would
        leave the join forgeable while looking like it had solved the problem.
        """
        path = resolve(keys=["customer_id"], metadata=Metadata(customer_id=value))
        assert path == PurePosixPath(f"u-alice/_meta/customer_id-{encoded}")

    def test_the_join_is_not_forgeable(self) -> None:
        """The concrete forgery the encoding exists to make impossible.

        A ``customer_id`` of ``ACME__case_id-42`` under one declared key would
        otherwise yield the *same* string as two keys over ``ACME`` and ``42``,
        and two key sets would silently address one tree.
        """
        forged = resolve(
            keys=["customer_id"],
            metadata=Metadata(customer_id="ACME__case_id-42"),
        )
        genuine = resolve(
            keys=["customer_id", "case_id"],
            metadata=Metadata(customer_id="ACME", case_id="42"),
        )

        assert forged != genuine

    def test_a_metadata_value_ending_in_the_journal_suffix_raises(self) -> None:
        """The same collision, reached from business data rather than a card field.

        ``.`` is inside the encoder's safe set, so a ``customer_id`` of ``x.git``
        survives encoding whole and the joined leaf is ``customer_id-x.git`` —
        which **is** the journal directory of ``<scope>/_meta/customer_id-x``, a
        perfectly ordinary metadata workspace. Nobody typed it; a business record
        did, which is what makes it likelier than the hand-typed case.
        """
        with pytest.raises(ValueError, match="journal directory") as excinfo:
            resolve(keys=["customer_id"], metadata=Metadata(customer_id="x.git"))

        # The joined leaf, not the raw value: the refusal names the directory
        # that would have been opened, which is what an admin has to act on.
        assert "customer_id-x.git" in str(excinfo.value)

    def test_the_suffix_is_refused_from_the_last_key_of_a_join(self) -> None:
        """Only the *joined* leaf's ending matters — a mid-join ``.git`` is harmless.

        Which key lands last is now the card's own declaration order: under
        ``["customer_id", "case_id"]`` a ``case_id`` of ``42.git`` ends the join
        and collides, while the same value in ``customer_id`` does not, because
        ``case_id-…`` follows it.
        """
        metadata = Metadata(customer_id="ACME", case_id="42.git")
        with pytest.raises(ValueError, match="journal directory"):
            resolve(keys=["customer_id", "case_id"], metadata=metadata)

        harmless = Metadata(customer_id="ACME.git", case_id="42")
        assert resolve(keys=["customer_id", "case_id"], metadata=harmless) == PurePosixPath(
            "u-alice/_meta/customer_id-ACME.git__case_id-42"
        )

    def test_the_metadata_branch_newly_rejects_nothing_but_the_suffix(self) -> None:
        """Routing through ``leaf_segment`` must not narrow a branch that accepted everything.

        The joined leaf is percent-encoded, so it holds no ``/``, ``\\`` or NUL;
        it starts with a declared pydantic field name, which can be neither empty
        nor a leading ``.``; it always contains ``-``, so it never equals a kind
        name; and the keys list is non-empty on this branch by construction. The
        characters below are exactly the ones the encoder lets through or
        escapes, and every one of them still resolves.
        """
        for value in ["a-b", "a_b", "a/b", "a%b", "a~b", "a b", "é", "a.b", ".hidden", "_x"]:
            path = resolve(keys=["customer_id"], metadata=Metadata(customer_id=value))
            assert path.parts[:2] == ("u-alice", "_meta")
            assert len(path.parts) == 3


##
## AC 9 — the four metadata conditions, none of which may fall back
##


class TestMetadataFailuresAlwaysRaise:
    def test_a_team_carrying_no_metadata_raises(self) -> None:
        """Falling back to another kind would silently re-home the tree.

        The card declared a metadata tree; landing it on the team's or a named
        tree instead is this feature's own failure mode, and just as quiet.
        """
        with pytest.raises(ValueError, match="carries no metadata"):
            resolve(keys=["customer_id"], metadata=None)

    def test_the_refusal_names_the_keys_as_the_card_declared_them(self) -> None:
        """The message reads the same ordered list the join does — nothing re-derives.

        Two spellings of one rule is what this branch removed: the raise and the
        join each said ``sorted(set(keys))``, and they can only stay in step
        because one local now feeds both. Nothing guarded that half — reverting
        the *message* alone to a sorted set left the whole suite green, which is
        the same silent drift one scale down. Asserting the declared, deduped
        order here reddens a sort restored in either place.
        """
        with pytest.raises(ValueError) as excinfo:
            resolve(keys=["customer_id", "case_id", "customer_id"], metadata=None)

        assert "['customer_id', 'case_id']" in str(excinfo.value)

    def test_a_key_that_is_not_a_field_of_the_model_raises(self) -> None:
        """The card names a field that does not exist; a typo must not reach a tree."""
        with pytest.raises(ValueError, match="not a field"):
            resolve(keys=["custmoer_id"], metadata=Metadata(customer_id="ACME"))

    @pytest.mark.parametrize("blank", [None, ""])
    def test_a_key_whose_value_is_missing_raises(self, blank: str | None) -> None:
        """``case_id-`` would be one real directory shared by every team that left it blank."""
        with pytest.raises(ValueError, match="carries no value"):
            resolve(keys=["case_id"], metadata=Metadata(case_id=blank))

    def test_a_leaf_over_the_filename_limit_raises_rather_than_truncating(self) -> None:
        """Truncating collides, and a collision here is an isolation failure that looks fine."""
        with pytest.raises(ValueError, match="exceeds 255 bytes"):
            resolve(keys=["customer_id"], metadata=Metadata(customer_id="A" * 300))

    def test_the_limit_is_bytes_not_characters(self) -> None:
        """A multibyte value blows a 255-**byte** name well before 255 characters.

        Each ``é`` percent-encodes to six ASCII bytes, so 60 of them are already
        past the limit while being only 60 characters.
        """
        with pytest.raises(ValueError, match="exceeds 255 bytes"):
            resolve(keys=["customer_id"], metadata=Metadata(customer_id="é" * 60))


##
## AC 8 — the two fields are mutually exclusive, at construction
##


class TestTheTwoFieldsAreMutuallyExclusive:
    def test_declaring_both_fails_at_card_construction(self) -> None:
        """A validation error, not a precedence rule.

        "Metadata wins" would be a silent answer to a question the author got
        wrong; two ways to name one tree on one card is worth surfacing where
        the person who wrote it is looking.
        """
        with pytest.raises(ValidationError, match="mutually exclusive"):
            WorkspaceTool(workspace_id="notes", workspace_metadata_keys=["customer_id"])

    def test_either_field_alone_is_fine(self) -> None:
        assert WorkspaceTool(workspace_id="notes").workspace_metadata_keys == []
        assert WorkspaceTool(workspace_metadata_keys=["customer_id"]).workspace_id is None

    def test_a_bare_card_declares_neither(self) -> None:
        card = WorkspaceTool()
        assert card.workspace_id is None
        assert card.workspace_metadata_keys == []

    def test_the_field_is_serialisable_and_round_trips(self) -> None:
        """It is a plain list of strings — no ``arbitrary_types_allowed`` anywhere near it."""
        card = WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"])
        restored = WorkspaceTool.model_validate(card.model_dump())
        assert restored.workspace_metadata_keys == ["customer_id", "case_id"]


##
## AC 2, 7 — the invariants, over generated input rather than hand-picked rows
##

# Fixed so a failure is reproducible from the test name alone.
_SEED = 48_1
_ALPHABET = string.ascii_letters + string.digits + "-_/%.~ é@+="


def _values(count: int, *, length: int = 12) -> list[str]:
    """Deterministic values drawn from an alphabet full of the hazardous characters."""
    rng = random.Random(_SEED)
    return [
        "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, length)))
        for _ in range(count)
    ]


def _usable(value: str) -> bool:
    """Whether *value* is one directory name at all — no ``/``, no leading ``.``."""
    return bool(value) and not value.startswith(".") and not set(value) & set("/\\\x00")


def _usable_scope(value: str) -> bool:
    """Whether a card could legally carry *value* as a principal."""
    return _usable(value) and value.casefold() not in RESERVED_SCOPES


def _usable_leaf(value: str) -> bool:
    """Whether a card could legally carry *value* as a ``workspace_id`` or a team id."""
    folded = value.casefold()
    return (
        _usable(value)
        and folded not in RESERVED_KINDS
        and not folded.endswith((GIT_DIR_SUFFIX, META_DIR_SUFFIX))
    )


def _generated_paths() -> list[PurePosixPath]:
    """Principals × leaves × metadata values × both sharable values × all three kinds.

    Every kind is exercised with every principal and both scopes: a leaf drawn
    from the generator is used once as a ``workspace_id`` (``_id``) and once as a
    team id (``_team``), and every metadata value keys a ``_meta`` tree.
    """
    users = [user for user in _values(16) if _usable_scope(user)]
    leaves = [leaf for leaf in _values(16, length=8) if _usable_leaf(leaf)]
    values = _values(10)
    paths: list[PurePosixPath] = []
    for user in users:
        for sharable in (False, True):
            for leaf in leaves:
                paths.append(resolve(workspace_id=leaf, user_id=user, sharable=sharable))
                paths.append(resolve(team_id=leaf, user_id=user, sharable=sharable))
            for value in values:
                paths.append(
                    resolve(
                        keys=["customer_id"],
                        metadata=Metadata(customer_id=value),
                        user_id=user,
                        sharable=sharable,
                    )
                )
    return paths


class TestTheShapeInvariant:
    """Every path is exactly three parts, clean, and an antichain with its siblings."""

    def test_every_resolved_path_is_three_clean_segments(self) -> None:
        paths = _generated_paths()
        kinds = set()
        scopes = set()
        for path in paths:
            assert len(path.parts) == 3, path
            assert ".." not in path.parts
            assert "." not in path.parts
            assert not path.is_absolute()
            scopes.add(path.parts[0] == "_shared")
            kinds.add(path.parts[1])
        # The generator reached every cell, or the spec proves less than it says.
        assert kinds == {"_team", "_id", "_meta"}
        assert scopes == {True, False}
        assert len(paths) > 500, "the generator produced too few usable cases to mean anything"

    def test_no_resolved_path_is_a_prefix_of_another(self) -> None:
        """Containment, not collision, is the hazard the fixed depth removes.

        ``Filesystem._validate_path`` rejects only what resolves *outside* the
        root, so a workspace at ``ACME/`` would read and write everything under
        ``ACME/42/`` as ordinary in-tree activity. At a fixed depth of three no
        path can be a proper prefix of another — a proper prefix has strictly
        fewer segments — so this states the invariant, and the three-parts spec
        above is what carries it under mutation.
        """
        paths = {str(path) for path in _generated_paths()}
        assert len(paths) > 300
        for path in paths:
            for other in paths:
                if path != other:
                    assert not other.startswith(f"{path}/"), (path, other)


class TestTheJoinIsInjective:
    """Distinct key/value sets reach distinct identifiers, over generated values."""

    def test_distinct_value_pairs_never_collide(self) -> None:
        seen: dict[str, tuple[str, str]] = {}
        values = _values(50, length=10)
        for customer in values:
            for case in values[:20]:
                path = str(
                    resolve(
                        keys=["customer_id", "case_id"],
                        metadata=Metadata(customer_id=customer, case_id=case),
                    )
                )
                previous = seen.get(path)
                assert previous in (None, (customer, case)), (previous, (customer, case))
                seen[path] = (customer, case)
        assert len(seen) > 400

    def test_one_key_never_forges_the_two_key_identifier(self) -> None:
        """Whatever a single value contains, it cannot impersonate a two-key join."""
        two_key = {
            str(
                resolve(
                    keys=["customer_id", "case_id"],
                    metadata=Metadata(customer_id=customer, case_id="42"),
                )
            )
            for customer in _values(40)
        }
        one_key = {
            str(resolve(keys=["customer_id"], metadata=Metadata(customer_id=value)))
            for value in _values(40)
        }
        assert not (two_key & one_key)

    def test_no_encoded_value_can_contain_a_separator(self) -> None:
        """The property both of the above rest on, asserted directly."""
        for value in _values(60):
            path = resolve(keys=["customer_id"], metadata=Metadata(customer_id=value))
            assert path.parent == PurePosixPath("u-alice/_meta")
            encoded = path.name.removeprefix("customer_id-")
            assert "-" not in encoded
            assert "_" not in encoded
            assert "/" not in encoded
