"""A ``workspace_id`` has one grammar, and it is refused where the author writes it.

What these specs guard is that the card and the platform's workspace routes can
no longer disagree about which ids exist. Before, the card accepted anything
``leaf_segment`` accepted — ``"Customer Notes"`` included — and got a working
tree for its agents, while every route answered 400 for the same name.

The grammar is :func:`validate_workspace_id`: a strict charset and a 128-character
bound, then every ``leaf_segment`` rule, **called rather than copied**. The card
runs it as a field validator, so a refused id fails at construction, at catalog
save and at team resume — never at bind, where nobody is looking. ``None``, the
default card's value, passes through every one of those paths.

The resolver keeps ``leaf_segment`` as its only leaf rule; the bind-time refusal
of a leaf is pinned in ``test_workspace_path_resolution.py``, not here.
"""

from __future__ import annotations

import pytest
from akgentic.core.utils.deserializer import deserialize_object
from pydantic import ValidationError

import akgentic.tool.workspace as workspace_package
from akgentic.tool.workspace import WorkspaceTool
from akgentic.tool.workspace.workspace import leaf_segment, validate_workspace_id

# Spelled by escape, never pasted: an editor or a normaliser can rewrite the glyph
# silently, and the spec using it would then prove nothing.
LONG_S = chr(0x017F)  # LATIN SMALL LETTER LONG S


##
## AC 1 — the function exists, is exported, and returns the value unchanged
##


ACCEPTED = [
    # Real ids from the catalogs this platform ships and runs.
    "shared_workspace",
    "claude_skills",
    "insurance-call-center-policy-db",
    "cli-git",
    "cli-catalog",
    "proj-42",
    "123e4567-e89b-12d3-a456-426614174000",
    # Dots mid-name, and the journal suffix anywhere but at the end.
    "a.b.c",
    "notes.github",
    "notes.git.txt",
    # Underscore names that are not a kind name: the reservation is exact.
    "_shared",
    "_teams",
    "_metadata",
    # The bounds.
    "A",
    "0",
    "x" * 128,
    # The metadata suffix's neighbours: it is refused as a suffix, never a substring.
    "index",
    "search-index",
    "search_index",
    "notes.indexes",
    "notes.index.md",
    "reindex",
]


class TestTheFunctionIsPublic:
    def test_it_is_exported_from_the_workspace_package(self) -> None:
        """Exported where ``leaf_segment`` is, and listed in that package's ``__all__``."""
        assert workspace_package.validate_workspace_id is validate_workspace_id
        assert "validate_workspace_id" in workspace_package.__all__


class TestAnAcceptedIdIsReturnedUnchanged:
    @pytest.mark.parametrize("accepted", ACCEPTED, ids=lambda value: value[:40])
    def test_the_same_object_comes_back(self, accepted: str) -> None:
        """Identity, not only equality: nothing normalises, strips or folds the value."""
        assert validate_workspace_id(accepted) is accepted


##
## AC 2 — the charset and the length, naming the value and the rule
##


REFUSED_BY_THE_CHARSET = [
    pytest.param("", id="empty"),
    pytest.param("Customer Notes", id="a-space"),
    pytest.param(" notes", id="leading-space"),
    pytest.param("notes ", id="trailing-space"),
    # ``$`` matches before a trailing newline, so a ``^…$`` pattern would pass this.
    pytest.param("notes\n", id="trailing-newline"),
    pytest.param("\nnotes", id="leading-newline"),
    pytest.param("notes/x", id="a-slash"),
    pytest.param("a\\b", id="a-backslash"),
    pytest.param("nul\x00byte", id="a-nul"),
    pytest.param("caf" + chr(0x00E9), id="an-accent"),
    pytest.param("a%2Db", id="a-percent"),
    pytest.param("a+b", id="a-plus"),
    pytest.param("alice@acme.example", id="an-email"),
    pytest.param("x" * 129, id="129-characters"),
    # The charset makes a workspace_id immune to case folding by construction.
    pytest.param(f"_{LONG_S}hared", id="long-s-shared"),
]


class TestTheCharsetAndTheLengthAreRefused:
    @pytest.mark.parametrize("refused", REFUSED_BY_THE_CHARSET)
    def test_the_message_names_the_value_and_the_whole_rule(self, refused: str) -> None:
        with pytest.raises(ValueError, match="not a valid workspace name") as excinfo:
            validate_workspace_id(refused)

        message = str(excinfo.value)
        assert repr(refused) in message
        assert "128" in message


##
## AC 3 — every leaf_segment rule, reached through it
##


# The suffixes are **literals** on purpose, like the reserved-name pin in
# ``test_workspace_path_resolution.py``: a rename of the constant reddens this
# table, which forces somebody to look at the decision rather than follow it.
REFUSED_BY_A_LEAF_RULE = [
    pytest.param(".hidden", "not usable as a directory name", id="leading-dot"),
    pytest.param(".", "not usable as a directory name", id="dot"),
    pytest.param("..", "not usable as a directory name", id="dot-dot"),
    pytest.param(".git", "not usable as a directory name", id="bare-git-suffix"),
    pytest.param(".index", "not usable as a directory name", id="bare-index-suffix"),
    pytest.param("_team", "reserved kind", id="kind-team"),
    pytest.param("_id", "reserved kind", id="kind-id"),
    pytest.param("_meta", "reserved kind", id="kind-meta"),
    pytest.param("_TEAM", "reserved kind", id="kind-team-upper"),
    pytest.param("_Meta", "reserved kind", id="kind-meta-title"),
    pytest.param("notes.git", "journal directory", id="git-suffix"),
    pytest.param("notes.GIT", "journal directory", id="git-suffix-upper"),
    pytest.param("notes.index", "index and metadata directory", id="index-suffix"),
    pytest.param("notes.INDEX", "index and metadata directory", id="index-suffix-upper"),
    pytest.param("notes.Index", "index and metadata directory", id="index-suffix-title"),
    pytest.param("search.index", "index and metadata directory", id="search-index"),
]


class TestEveryLeafRuleIsReachedThroughIt:
    """Every value below is charset-legal, so only ``leaf_segment`` can refuse it."""

    @pytest.mark.parametrize(("refused", "rule"), REFUSED_BY_A_LEAF_RULE)
    def test_the_message_names_the_value_and_the_rule_it_broke(
        self, refused: str, rule: str
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_workspace_id(refused)

        message = str(excinfo.value)
        assert rule in message
        assert repr(refused) in message


##
## AC 3b — search.index is refused with a message an admin can act on
##


class TestTheSuffixRefusalNamesTheCollision:
    """``.index`` is an ordinary word, so the message is the whole user experience.

    ``search.index`` is where workspace ``search`` keeps its exec lock, document
    records and retrieval index. An admin told only that the name "ends in
    '.index'" cannot tell what to rename, or why.
    """

    def test_search_index_names_the_workspace_it_would_collide_with(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_workspace_id("search.index")

        message = str(excinfo.value)
        assert "'.index'" in message
        assert "'search.index'" in message
        assert "would collide with" in message
        assert "index and metadata directory" in message
        assert "'search'" in message

    def test_the_resolver_and_the_card_give_the_same_account(self) -> None:
        """The route guards its selector with ``leaf_segment``, so it inherits this too."""
        with pytest.raises(ValueError, match="would collide with the index and metadata"):
            leaf_segment("search.index")
        with pytest.raises(ValidationError, match="would collide with the index and metadata"):
            WorkspaceTool(workspace_id="search.index")

    def test_the_journal_twin_names_its_workspace_the_same_way(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_workspace_id("notes.git")

        message = str(excinfo.value)
        assert "'.git'" in message
        assert "would collide with" in message
        assert "journal directory" in message
        assert "'notes'" in message

    def test_the_stem_is_sliced_off_the_value_as_written(self) -> None:
        """The fold decides the match; the stem the admin reads is what they typed."""
        with pytest.raises(ValueError) as excinfo:
            leaf_segment("Search.INDEX")

        assert "'Search'" in str(excinfo.value)


##
## AC 4 — the card refuses at construction, not at bind
##


ONE_PER_RULE = [
    pytest.param("Customer Notes", id="charset"),
    pytest.param("x" * 129, id="over-length"),
    pytest.param("notes\n", id="trailing-newline"),
    pytest.param("", id="empty"),
    pytest.param(".hidden", id="leading-dot"),
    pytest.param("_meta", id="kind-name"),
    pytest.param("notes.git", id="git-suffix"),
    pytest.param("notes.index", id="index-suffix"),
]


class TestTheCardRefusesAtConstruction:
    """No ``observer()`` is called anywhere below: the refusal needs no bind to exist."""

    @pytest.mark.parametrize("refused", ONE_PER_RULE)
    def test_constructing_the_card_raises(self, refused: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            WorkspaceTool(workspace_id=refused)

        self._assert_names_the_field_and_the_value(excinfo.value, refused)

    @pytest.mark.parametrize("refused", ONE_PER_RULE)
    def test_validating_a_stored_card_raises(self, refused: str) -> None:
        """``model_validate`` is the catalog's path, on save and on load."""
        with pytest.raises(ValidationError) as excinfo:
            WorkspaceTool.model_validate({"workspace_id": refused})

        self._assert_names_the_field_and_the_value(excinfo.value, refused)

    @staticmethod
    def _assert_names_the_field_and_the_value(error: ValidationError, refused: str) -> None:
        (only,) = error.errors()
        assert only["loc"] == ("workspace_id",)
        assert "workspace_id" in str(error)
        assert repr(refused) in str(error)


##
## AC 5 — None stays legal, through every path
##


class TestNoneStaysLegal:
    """Every default card's dump carries ``workspace_id: None``, and every resume re-reads it."""

    def test_a_bare_card_declares_no_workspace_id(self) -> None:
        assert WorkspaceTool().workspace_id is None

    def test_an_explicit_none_constructs(self) -> None:
        assert WorkspaceTool(workspace_id=None).workspace_id is None

    def test_a_default_card_round_trips_through_its_dump(self) -> None:
        dump = WorkspaceTool().model_dump()
        assert "workspace_id" in dump
        assert dump["workspace_id"] is None

        assert WorkspaceTool.model_validate(dump).workspace_id is None

    def test_a_valid_id_round_trips_through_its_dump(self) -> None:
        dump = WorkspaceTool(workspace_id="notes").model_dump()

        assert WorkspaceTool.model_validate(dump).workspace_id == "notes"


##
## AC 6 — a stored card with a refused id fails the resume loudly
##


def _stored_card(workspace_id: str) -> dict[str, object]:
    """A valid card's stored shape, with its ``workspace_id`` overwritten as a store could."""
    stored = WorkspaceTool(workspace_id="notes").model_dump()
    assert "__model__" in stored  # the tag the resume path dispatches on
    stored["workspace_id"] = workspace_id
    return stored


class TestAResumedCardWithARefusedIdFailsLoudly:
    """Team resume rebuilds each card through ``deserialize_object``, which validates.

    Inside a list, core drops only a card whose **class** is gone. A validation
    failure propagates, so a stored card whose id is now refused fails the resume
    in front of whoever runs it — and is never silently left out of the team.
    """

    def test_the_valid_card_resumes(self) -> None:
        """The control: what reddens below is the id, not the stored shape."""
        card = deserialize_object(WorkspaceTool(workspace_id="notes").model_dump())

        assert isinstance(card, WorkspaceTool)
        assert card.workspace_id == "notes"

    @pytest.mark.parametrize("refused", ["Customer Notes", "notes.git"])
    def test_one_card_raises(self, refused: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            deserialize_object(_stored_card(refused))

        assert repr(refused) in str(excinfo.value)

    @pytest.mark.parametrize("refused", ["Customer Notes", "notes.git"])
    def test_a_card_in_a_list_raises_rather_than_being_dropped(self, refused: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            deserialize_object([_stored_card(refused)])

        assert repr(refused) in str(excinfo.value)
