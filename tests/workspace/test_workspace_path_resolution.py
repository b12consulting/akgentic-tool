"""The one resolver: two segments, scoped to an owner, and no silent fallback.

What these specs are really guarding is that **nothing else derives a workspace
directory**. The resolver's own behaviour is checkable here; that it is the only
derivation is checkable only by the seven sites that no longer have one, which
the wiring suites assert from the other end.

The two property tests use a seeded ``random.Random`` rather than ``hypothesis``,
which is not a dependency of this package. The seed is fixed, so a failure is
reproducible from the test name alone.
"""

from __future__ import annotations

import random
import string
from pathlib import PurePosixPath

import pytest
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import ValidationError

from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import (
    ANONYMOUS,
    METADATA_SCOPE,
    RESERVED_SCOPES,
    leaf_segment,
    resolve_workspace_path,
    user_segment,
)

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
) -> PurePosixPath:
    """Call the resolver with the defaults every spec below shares."""
    return resolve_workspace_path(
        workspace_id=workspace_id,
        workspace_metadata_keys=keys or [],
        team_id=team_id,
        user_id=user_id,
        metadata=metadata,
    )


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
            "geoffroy.piroux@weareyuma.com",
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
            "acme/geoffroy",  # an admin typing an org-scoped owner_id
            "back\\slash",
            "nul\x00byte",
        ],
    )
    def test_a_value_that_cannot_be_a_directory_name_raises(self, unusable: str) -> None:
        with pytest.raises(ValueError):
            user_segment(unusable)

    def test_the_reserved_scope_is_refused(self) -> None:
        assert RESERVED_SCOPES == frozenset({METADATA_SCOPE})
        with pytest.raises(ValueError, match="reserved scope"):
            user_segment(METADATA_SCOPE)

    @pytest.mark.parametrize("accepted", ["_meta2", "_met", "_", "_metadata", "x_meta"])
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
    """The same guard, minus the reservation — because ``_meta`` is a scope."""

    @pytest.mark.parametrize(
        "accepted",
        [
            "notes",
            "11111111-2222-3333-4444-555555555555",
            "case_id-42__customer_id-ACME",
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

    def test_a_workspace_legitimately_named_meta_is_accepted(self) -> None:
        """``_meta`` under a principal collides with nothing — it is not a scope there."""
        assert leaf_segment(METADATA_SCOPE) == METADATA_SCOPE
        assert resolve(workspace_id=METADATA_SCOPE) == PurePosixPath("u-alice/_meta")


##
## AC 5 — the two per-user layouts
##


class TestPerUserLayouts:
    def test_no_workspace_id_puts_the_team_tree_under_its_owner(self) -> None:
        assert resolve(team_id="team-9", user_id="alice") == PurePosixPath("alice/team-9")

    def test_a_named_workspace_is_one_of_this_principals_trees(self) -> None:
        assert resolve(workspace_id="notes", user_id="alice") == PurePosixPath("alice/notes")

    def test_two_principals_declaring_one_name_reach_two_paths(self) -> None:
        """The exposure, in one assertion.

        ``notes`` used to be a global key: whoever typed it got the same tree.
        """
        alice = resolve(workspace_id="notes", user_id="alice")
        bob = resolve(workspace_id="notes", user_id="bob")

        assert alice != bob
        assert alice == PurePosixPath("alice/notes")
        assert bob == PurePosixPath("bob/notes")

    def test_no_principal_resolves_to_the_anonymous_scope(self) -> None:
        assert resolve(workspace_id="notes", user_id=None) == PurePosixPath("anonymous/notes")

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
        """``<user>/../x`` is one segment *outside* the principal's directory.

        ``Filesystem`` does not validate the name it is given at all, so this
        guard is the only thing holding the two-segment invariant for a value an
        author types by hand.
        """
        with pytest.raises(ValueError):
            resolve(workspace_id=traversal)

    def test_an_unusable_principal_raises_out_of_the_resolver(self) -> None:
        with pytest.raises(ValueError, match="not usable as a workspace directory name"):
            resolve(workspace_id="notes", user_id="acme/geoffroy")


##
## AC 6, 7 — the metadata layout
##


class TestMetadataLayout:
    def test_the_declared_keys_key_a_shared_tree_under_the_reserved_scope(self) -> None:
        path = resolve(
            keys=["customer_id", "case_id"],
            metadata=Metadata(customer_id="ACME", case_id="42"),
        )
        assert path == PurePosixPath("_meta/case_id-42__customer_id-ACME")

    def test_it_is_shared_across_principals_by_design(self) -> None:
        """The one layout that must *not* differ per user — that is its purpose."""
        metadata = Metadata(customer_id="ACME", case_id="42")
        alice = resolve(keys=["customer_id", "case_id"], user_id="alice", metadata=metadata)
        bob = resolve(keys=["customer_id", "case_id"], user_id="bob", metadata=metadata)

        assert alice == bob
        assert alice.parts[0] == METADATA_SCOPE

    def test_the_declaration_is_a_set_so_key_order_cannot_split_a_tree(self) -> None:
        metadata = Metadata(customer_id="ACME", case_id="42")
        forwards = resolve(keys=["customer_id", "case_id"], metadata=metadata)
        backwards = resolve(keys=["case_id", "customer_id"], metadata=metadata)

        assert forwards == backwards

    def test_a_different_key_set_reaches_a_different_tree(self) -> None:
        """And a *sibling* one, never a parent — Decision 1's antichain."""
        metadata = Metadata(customer_id="ACME", case_id="42")
        one = resolve(keys=["customer_id"], metadata=metadata)
        both = resolve(keys=["customer_id", "case_id"], metadata=metadata)

        assert one != both
        assert not str(both).startswith(f"{one}/")
        assert not str(one).startswith(f"{both}/")

    def test_a_repeated_key_names_the_same_tree_as_naming_it_once(self) -> None:
        metadata = Metadata(customer_id="ACME")
        assert resolve(keys=["customer_id", "customer_id"], metadata=metadata) == resolve(
            keys=["customer_id"], metadata=metadata
        )

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
        assert path == PurePosixPath(f"_meta/customer_id-{encoded}")

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


##
## AC 9 — the four metadata conditions, none of which may fall back
##


class TestMetadataFailuresAlwaysRaise:
    def test_a_team_carrying_no_metadata_raises(self) -> None:
        """Falling back to the user path would silently *un-share* the workspace.

        That is this feature's own failure mode in reverse, and just as quiet.
        """
        with pytest.raises(ValueError, match="carries no metadata"):
            resolve(keys=["customer_id"], metadata=None)

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

    def test_the_deprecated_exec_card_gains_no_metadata_field(self) -> None:
        """A card epic 41 deletes must not ship a capability that is then deleted again."""
        from akgentic.tool.sandbox.tool import ExecTool

        assert "workspace_metadata_keys" not in ExecTool.model_fields


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
    """Whether *value* is one a card could legally carry as a leaf."""
    return bool(value) and not value.startswith(".") and not set(value) & set("/\\\x00")


class TestTheShapeInvariant:
    """Every path is exactly two parts, clean, and an antichain with its siblings."""

    def test_every_resolved_path_is_two_clean_segments(self) -> None:
        checked = 0
        for user in _values(40):
            for leaf in _values(40, length=8):
                if not _usable(user) or not _usable(leaf) or user in RESERVED_SCOPES:
                    continue
                path = resolve(workspace_id=leaf, user_id=user)
                assert len(path.parts) == 2, path
                assert ".." not in path.parts
                assert "." not in path.parts
                assert not path.is_absolute()
                checked += 1
        assert checked > 100, "the generator produced too few usable cases to mean anything"

    def test_a_metadata_path_is_two_clean_segments_too(self) -> None:
        checked = 0
        for value in _values(60):
            path = resolve(keys=["customer_id"], metadata=Metadata(customer_id=value))
            assert len(path.parts) == 2, path
            assert path.parts[0] == METADATA_SCOPE
            assert ".." not in path.parts and "." not in path.parts
            checked += 1
        assert checked > 40

    def test_no_resolved_path_is_a_prefix_of_another(self) -> None:
        """Containment, not collision, is the hazard the fixed depth removes.

        ``Filesystem._validate_path`` rejects only what resolves *outside* the
        root, so a workspace at ``ACME/`` would read and write everything under
        ``ACME/42/`` as ordinary in-tree activity.
        """
        paths: set[str] = set()
        for user in _values(30):
            for leaf in _values(30, length=6):
                if not _usable(user) or not _usable(leaf) or user in RESERVED_SCOPES:
                    continue
                paths.add(str(resolve(workspace_id=leaf, user_id=user)))
        for path in paths:
            for other in paths:
                if path is not other and path != other:
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
            leaf = str(resolve(keys=["customer_id"], metadata=Metadata(customer_id=value)))
            encoded = leaf.removeprefix(f"{METADATA_SCOPE}/customer_id-")
            assert "-" not in encoded
            assert "_" not in encoded
            assert "/" not in encoded
