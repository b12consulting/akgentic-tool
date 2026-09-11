"""The platform permits, the card requests: ``AKGENTIC_WORKSPACE_SHARED_KINDS`` at bind.

``workspace_sharable=True`` puts a card's tree under ``_shared``; whether a
shared tree of that **kind** may exist on this process is the platform's to say,
through one variable read by the process that binds agents. A request the
platform does not permit **fails the bind** — naming what was asked for and what
is permitted — and never degrades to the per-principal tree.

Every refusal below is asserted **through a card bound by** ``observer()``,
never through the parser alone: epic 52 lost a real defect to a resolver specced
in isolation that nothing called at bind time. The variable is never ambient —
the root ``conftest`` deletes it for every test — and a spec that wants it sets
it through ``monkeypatch``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path, PurePosixPath

import pytest

import akgentic.tool.workspace.card as card_module
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import (
    ID_KIND,
    METADATA_KIND,
    SHARED_KINDS_ENV,
    TEAM_KIND,
    permitted_shared_kinds,
    resolve_workspace_path,
)
from tests.workspace.conftest import FakeOrchestratorProxy
from tests.workspace.test_workspace_path_resolution import (
    CELL_METADATA,
    SIX_CELLS,
    Cell,
    bind_cell,
    own_kind_token,
    unbound_cell,
)

SHARED_CELLS = tuple(cell for cell in SIX_CELLS if cell.sharable)
"""The three ``_shared/<kind>/<leaf>`` cells — the only ones the permission gates."""

PER_PRINCIPAL_CELLS = tuple(cell for cell in SIX_CELLS if not cell.sharable)
"""The three ``alice/<kind>/<leaf>`` cells — which no permission may make shared."""

TOKENS = ("team", "id", "meta")
"""The accepted vocabulary, pinned against literals **once**, here."""


def _cell_ids(cell: Cell) -> str:
    return cell.name


def _gate_refusal(token: str) -> str:
    """The phrase only the bind-time gate writes — never the parser's error.

    A refusal spec that matched only the variable's name or a quoted token would
    also be satisfied by a *parse* error, which names the variable and lists every
    accepted token: such a spec would pass whenever the parser refused the value,
    whether or not the gate ever ran.
    """
    return f"requests a shared {token!r} workspace"


def _parse(monkeypatch: pytest.MonkeyPatch, value: str) -> frozenset[str]:
    """Parse *value* as the variable, set through ``monkeypatch`` and never ambient."""
    with monkeypatch.context() as patch:
        patch.setenv(SHARED_KINDS_ENV, value)
        return permitted_shared_kinds()


##
## AC 1, 2 — the variable and its parse
##


class TestTheParse:
    """AC 1's table, row by row, and AC 2's message."""

    def test_unset_permits_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SHARED_KINDS_ENV, raising=False)

        assert permitted_shared_kinds() == frozenset()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # An empty value is none, not an error — identical to unset. It is
            # what a compose file interpolating an unset ``${VAR}`` produces.
            ("", frozenset()),
            ("   ", frozenset()),
            (",", frozenset()),
            (" , ", frozenset()),
            # Each token, alone, to its own constant.
            ("team", frozenset({TEAM_KIND})),
            ("id", frozenset({ID_KIND})),
            ("meta", frozenset({METADATA_KIND})),
            # Split on ``,``, each token stripped; order does not matter.
            ("id,meta", frozenset({ID_KIND, METADATA_KIND})),
            (" id , meta ", frozenset({ID_KIND, METADATA_KIND})),
            ("meta,id", frozenset({ID_KIND, METADATA_KIND})),
            ("team,id,meta", frozenset({TEAM_KIND, ID_KIND, METADATA_KIND})),
            # Compared lower-cased.
            ("META", frozenset({METADATA_KIND})),
            ("Meta", frozenset({METADATA_KIND})),
            # Duplicates collapse; empty tokens are skipped.
            ("meta,meta", frozenset({METADATA_KIND})),
            ("meta,", frozenset({METADATA_KIND})),
            (",meta", frozenset({METADATA_KIND})),
            ("id,,meta", frozenset({ID_KIND, METADATA_KIND})),
        ],
    )
    def test_an_accepted_value_parses_to_kind_constants(
        self, value: str, expected: frozenset[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _parse(monkeypatch, value) == expected

    @pytest.mark.parametrize(
        ("value", "offending"),
        [
            ("metadata", "metadata"),
            # The reserved path segment is not the word: one spelling per kind.
            ("_meta", "_meta"),
            # No wildcard — it would silently widen the day a fourth kind lands.
            ("all", "all"),
            ("*", "*"),
            ("none", "none"),
            # The separator is ``,`` only: each of these is one token.
            ("id meta", "id meta"),
            ("id;meta", "id;meta"),
            # One bad token refuses the whole value — ``meta`` is never permitted
            # by half a value.
            ("meta,metadata", "metadata"),
        ],
    )
    def test_an_unknown_token_refuses_the_whole_value(
        self, value: str, offending: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            _parse(monkeypatch, value)

        assert repr(offending) in str(excinfo.value)

    def test_the_refusal_names_the_variable_the_token_and_the_vocabulary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 2 — everything an admin needs to fix it without reading source.

        The accepted list is asserted **whole and in order** against literals:
        it is the one place the vocabulary is pinned, so a fourth token, a lost
        one, or a respelled one goes red here.
        """
        with pytest.raises(ValueError) as excinfo:
            _parse(monkeypatch, "metadata")

        message = str(excinfo.value)
        assert SHARED_KINDS_ENV in message
        assert "'metadata'" in message
        assert "comma-separated" in message
        assert message.endswith(", ".join(repr(token) for token in sorted(TOKENS)))

    def test_the_offending_token_is_named_as_written_not_lower_cased(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The admin searches their config for what they typed, not for our normal form."""
        with pytest.raises(ValueError) as excinfo:
            _parse(monkeypatch, "META,Metadata")

        assert "'Metadata'" in str(excinfo.value)
        assert "'metadata'" not in str(excinfo.value)


##
## AC 6 — kind by kind, never "sharing on/off"
##


def _subsets() -> list[tuple[str, ...]]:
    """The eight subsets of the vocabulary, the empty one included."""
    return [subset for size in range(len(TOKENS) + 1) for subset in combinations(TOKENS, size)]


def _subset_id(subset: tuple[str, ...]) -> str:
    return ",".join(subset) or "unset"


class TestTheRefusalIsKindByKind:
    """The 3 × 8 truth table: a shared cell binds **iff** its kind is in the subset.

    ``meta`` does not permit ``id``; ``team,id`` does not permit ``meta``. The
    empty subset is spelled as a **deleted** variable, not an empty one.
    """

    @pytest.mark.parametrize("subset", _subsets(), ids=_subset_id)
    @pytest.mark.parametrize("cell", SHARED_CELLS, ids=_cell_ids)
    def test_the_bind_succeeds_iff_the_cells_kind_is_permitted(
        self,
        cell: Cell,
        subset: tuple[str, ...],
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        value = ",".join(subset) if subset else None

        if own_kind_token(cell) in subset:
            card, expected, _observer = bind_cell(
                cell, orchestrator_proxy, monkeypatch, shared_kinds=value
            )
            assert card._workspace_path == expected
        else:
            # The gate's own phrase, never a bare token or the variable's name: a
            # parse error names the variable and lists every token too, so a
            # value the parser refused would satisfy those and read as a pass.
            with pytest.raises(ValueError) as excinfo:
                bind_cell(cell, orchestrator_proxy, monkeypatch, shared_kinds=value)
            assert _gate_refusal(own_kind_token(cell)) in str(excinfo.value)


##
## AC 7 — unset refuses every shared kind, and so does empty
##


class TestUnsetRefusesEverySharedKind:
    """A deployment that has never heard of sharing cannot be given it by a card.

    Unset and empty are pinned **separately**: they are the two cases story
    52-1's review found conflated, and here both must mean none.
    """

    @pytest.mark.parametrize("cell", SHARED_CELLS, ids=_cell_ids)
    def test_a_deleted_variable_refuses(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with pytest.raises(ValueError, match=f"{SHARED_KINDS_ENV} permits none"):
            bind_cell(cell, orchestrator_proxy, monkeypatch, shared_kinds=None)

    @pytest.mark.parametrize("cell", SHARED_CELLS, ids=_cell_ids)
    def test_an_empty_variable_refuses(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with pytest.raises(ValueError, match=f"{SHARED_KINDS_ENV} permits none"):
            bind_cell(cell, orchestrator_proxy, monkeypatch, shared_kinds="")


##
## AC 5 — a refused bind creates nothing
##


class TestARefusedBindCreatesNothing:
    """Raising is half of it; what did **not** happen is the other half.

    ``Filesystem.__init__`` creates its directory eagerly, so a gate placed after
    ``get_workspace`` would refuse the bind and still leave an empty ``_shared``
    tree — created by the refusal itself. And a fall-back would leave ``alice/``.
    """

    @pytest.mark.parametrize("cell", SHARED_CELLS, ids=_cell_ids)
    def test_no_directory_no_actor_no_event_and_no_bound_state(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
    ) -> None:
        card, observer = unbound_cell(cell, orchestrator_proxy)
        # Unset by the root conftest — asserted, so the spec cannot pass on a
        # runner that exports the variable only because the fixture was lost.
        assert SHARED_KINDS_ENV not in os.environ

        # The refusal itself, by its message: a metadata error or any other
        # ``ValueError`` raised for another reason must not read as a pass.
        with pytest.raises(ValueError, match=f"{SHARED_KINDS_ENV} permits none"):
            card.observer(observer)

        assert list(workspaces_root.iterdir()) == []  # no ``_shared/``, and no ``alice/``
        assert orchestrator_proxy.children == {}
        assert observer.events == []
        assert card._workspace is None
        assert card._workspace_path == ""
        assert card._meta_dir is None
        with pytest.raises(RuntimeError):
            _ = card.workspace


##
## AC 4 — the message names both sides
##


class _ReverseIterating(frozenset[str]):
    """A permitted set that iterates reverse-sorted: the order an unsorted rendering shows."""

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(frozenset.__iter__(self), reverse=True))


class TestTheMessageNamesBothSides:
    """A refusal reading "sharing is not enabled" sends an admin to the wrong file."""

    def test_the_requested_kind_the_permitted_kinds_the_path_and_both_remedies(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Request ``id`` with ``meta`` permitted.

        ``'meta'`` appears **only** through the permitted list — the request is
        ``id`` and the path is ``_shared/_id/notes`` — so asserting it proves the
        list is rendered.
        """
        [id_shared] = [cell for cell in SHARED_CELLS if cell.name == "id-shared"]

        with pytest.raises(ValueError) as excinfo:
            bind_cell(id_shared, orchestrator_proxy, monkeypatch, shared_kinds="meta")

        message = str(excinfo.value)
        assert f"{_gate_refusal('id')} (_shared/_id/notes)" in message
        assert f"{SHARED_KINDS_ENV} permits only 'meta' on this process" in message
        assert "on the process that binds agents" in message
        assert "workspace_sharable=False" in message

    def test_two_permitted_kinds_are_both_named(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Through the real parse. The gate's phrase, not ``'meta', 'team'`` alone.

        The parser's own error ends ``'id', 'meta', 'team'``, which contains
        ``'meta', 'team'``: with that substring alone, a parser that refused the
        two-token value passed this spec without the gate ever running.
        """
        [id_shared] = [cell for cell in SHARED_CELLS if cell.name == "id-shared"]

        with pytest.raises(ValueError) as excinfo:
            bind_cell(id_shared, orchestrator_proxy, monkeypatch, shared_kinds="team,meta")

        message = str(excinfo.value)
        assert _gate_refusal("id") in message
        assert f"{SHARED_KINDS_ENV} permits only 'meta', 'team' on this process" in message

    def test_the_permitted_list_is_rendered_sorted_whatever_the_set_iterates(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sorted, not the order the permitted set happens to iterate in.

        A ``frozenset`` iterates in an order the process's hash seed chooses, so
        with the real parse a lost ``sorted()`` goes red about one run in two —
        a guard that passes half the time it should fail. Here the parse the gate
        calls hands back a set that iterates **reverse**-sorted, so an unsorted
        rendering reads ``'team', 'meta'`` every time. The bind is still the real
        ``observer()``; only the iteration order is fixed.
        """
        [id_shared] = [cell for cell in SHARED_CELLS if cell.name == "id-shared"]
        reverse_iterating = _ReverseIterating({TEAM_KIND, METADATA_KIND})
        monkeypatch.setattr(card_module, "permitted_shared_kinds", lambda: reverse_iterating)
        assert [kind.removeprefix("_") for kind in reverse_iterating] == ["team", "meta"]

        with pytest.raises(ValueError) as excinfo:
            bind_cell(id_shared, orchestrator_proxy, monkeypatch, shared_kinds=None)

        assert f"{SHARED_KINDS_ENV} permits only 'meta', 'team' on this process" in str(
            excinfo.value
        )

    def test_nothing_permitted_says_none(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        [team_shared] = [cell for cell in SHARED_CELLS if cell.name == "team-shared"]

        with pytest.raises(ValueError) as excinfo:
            bind_cell(team_shared, orchestrator_proxy, monkeypatch, shared_kinds=None)

        message = str(excinfo.value)
        assert "'team'" in message
        assert f"{SHARED_KINDS_ENV} permits none" in message
        assert "_shared/_team/" in message


##
## AC 9 — the bind-time process decides, not the card
##


LAYOUTS: list[dict[str, object]] = [
    {},
    {"workspace_id": "notes"},
    {"workspace_metadata_keys": ["customer_id"]},
]
"""The three layouts, each declared ``workspace_sharable=True`` below."""


class TestTheBindTimeProcessDecides:
    """A card is validated in more places than it is bound.

    The stateless API server validates a catalog write under **its** environment
    while a worker with a different allow-list creates the tree, so no
    ``model_validator`` may judge the request: only the binding process can.
    """

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_a_sharable_card_constructs_and_round_trips_with_nothing_permitted(
        self, layout: dict[str, object]
    ) -> None:
        assert SHARED_KINDS_ENV not in os.environ

        card = WorkspaceTool.model_validate({**layout, "workspace_sharable": True})
        again = WorkspaceTool.model_validate(card.model_dump())

        assert again.workspace_sharable is True

    @pytest.mark.parametrize("cell", SHARED_CELLS, ids=_cell_ids)
    def test_unpermitted_at_construction_and_permitted_at_bind_binds(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert SHARED_KINDS_ENV not in os.environ
        _card, observer = unbound_cell(cell, orchestrator_proxy)
        card = WorkspaceTool.model_validate(cell.card().model_dump())

        with monkeypatch.context() as patch:
            patch.setenv(SHARED_KINDS_ENV, own_kind_token(cell))
            card.observer(observer)

        assert card._workspace_path == cell.expected.format(team_id=observer.team_id)

    @pytest.mark.parametrize("cell", SHARED_CELLS, ids=_cell_ids)
    def test_permitted_at_construction_and_unset_at_bind_raises(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with monkeypatch.context() as patch:
            patch.setenv(SHARED_KINDS_ENV, own_kind_token(cell))
            card = WorkspaceTool.model_validate(cell.card().model_dump())
        _card, observer = unbound_cell(cell, orchestrator_proxy)
        assert SHARED_KINDS_ENV not in os.environ

        with pytest.raises(ValueError, match=f"{SHARED_KINDS_ENV} permits none"):
            card.observer(observer)


##
## AC 10 — the resolver stays environment-free
##


class TestTheResolverReadsNoEnvironment:
    """``akgentic-infra`` calls the resolver on the API server — the wrong process to decide.

    A **malformed** value is the sharpest probe: a resolver that consulted the
    variable at all would raise on it.
    """

    @pytest.mark.parametrize("value", [None, "metadata"], ids=["unset", "malformed"])
    @pytest.mark.parametrize("cell", SHARED_CELLS, ids=_cell_ids)
    def test_a_shared_request_resolves_whatever_the_variable_says(
        self, cell: Cell, value: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with monkeypatch.context() as patch:
            if value is None:
                patch.delenv(SHARED_KINDS_ENV, raising=False)
            else:
                patch.setenv(SHARED_KINDS_ENV, value)
            path = resolve_workspace_path(
                workspace_id=cell.workspace_id,
                workspace_metadata_keys=list(cell.keys),
                team_id="team-9",
                user_id="alice",
                metadata=CELL_METADATA,
                workspace_sharable=True,
            )

        assert path == PurePosixPath(cell.expected.format(team_id="team-9"))


##
## AC 11 — read at every bind, never cached
##


class TestNoCache:
    """Permitted, then deleted, then permitted again — three fresh cards, one process."""

    def test_each_bind_reads_the_variable_as_it_is_then(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        [meta_shared] = [cell for cell in SHARED_CELLS if cell.name == "meta-shared"]

        first, expected, _first_observer = bind_cell(
            meta_shared, orchestrator_proxy, monkeypatch, shared_kinds="meta"
        )
        assert first._workspace_path == expected

        with pytest.raises(ValueError, match=f"{SHARED_KINDS_ENV} permits none"):
            bind_cell(meta_shared, orchestrator_proxy, monkeypatch, shared_kinds=None)

        third, expected, _third_observer = bind_cell(
            meta_shared, orchestrator_proxy, monkeypatch, shared_kinds="meta"
        )
        assert third._workspace_path == expected


##
## AC 12 — a malformed value fails every bind
##


class TestAMalformedValueFailsEveryBind:
    """Per-principal binds included: a typo surfaces at the next team start, not weeks later."""

    @pytest.mark.parametrize("cell", PER_PRINCIPAL_CELLS, ids=_cell_ids)
    def test_a_per_principal_bind_raises_the_parse_error(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with pytest.raises(ValueError, match="names 'metadata', which is not a workspace kind"):
            bind_cell(cell, orchestrator_proxy, monkeypatch, shared_kinds="metadata")


##
## AC 13 — the permission never makes a card shared
##


class TestPermissionNeverShares:
    """The card requests; the platform only permits."""

    @pytest.mark.parametrize("cell", PER_PRINCIPAL_CELLS, ids=_cell_ids)
    def test_every_kind_permitted_and_a_per_principal_card_stays_per_principal(
        self,
        cell: Cell,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        card, expected, _observer = bind_cell(
            cell, orchestrator_proxy, monkeypatch, shared_kinds="team,id,meta"
        )

        assert expected.startswith("alice/")
        assert card._workspace_path == expected
        assert set(orchestrator_proxy.children) == {f"#Workspace-{expected}"}
