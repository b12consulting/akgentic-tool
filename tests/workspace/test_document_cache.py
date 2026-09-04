"""The extraction cache: hit/miss, LRU, the two caps, and the no-notify guard (story 45-3).

These specs address the actor **directly**, which is what keeps them a guard on
the cache itself rather than on whoever calls it. ``workspace_read`` became that
caller in 45-4; the read path's own guards live in
``test_document_read_path.py``. What is asserted here is the shape the read path
calls into and the one rule the whole design rests on: **the read path never
snapshots state**.

``notify_state_change()`` serialises the *whole* state through
``model_dump_json()`` and the actor forwards it to the orchestrator, which is an
event-store write. Reads are the majority of workspace traffic, so a notify on a
read — or on a cache hit, which a read is — would put that write back on the
path ADR-036's NFR1 exists to keep free. A *fill* is different: it is amortised
against the seconds of extraction that preceded it, and it notifies exactly once.

The caps are exercised **at** the cap and one past it, so an off-by-one in
either direction is visible. Small caps throughout: the production defaults
would make each spec build 32 entries or two megabytes of text.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from akgentic.core.agent_state import BaseState
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.documents.models import (
    DEFAULT_MAX_DOCUMENT_CHARS,
    DEFAULT_MAX_DOCUMENTS,
    EXTRACTOR_VERSION,
    DocumentExtract,
    evict_document_bodies,
)
from akgentic.tool.workspace.models import WorkspaceConfig, WorkspaceState, content_sha
from akgentic.tool.workspace.tool import WorkspaceTool

from tests.workspace.conftest import WORKSPACE_NAME, read


def start_actor(
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
) -> WorkspaceActor:
    """Build and start an actor over the test workspace, without an actor thread."""
    actor = WorkspaceActor(
        config=WorkspaceConfig(
            name=workspace_actor_name(WORKSPACE_NAME),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_name=WORKSPACE_NAME,
            max_documents=max_documents,
            max_document_chars=max_document_chars,
        )
    )
    actor.on_start()
    return actor


def sha_of(text: str) -> str:
    """The digest 45-4's caller will hand in, over the source bytes."""
    return content_sha(text.encode())


def fill(
    actor: WorkspaceActor,
    path: str,
    body: str = "body",
    source: str | None = None,
    version: int = EXTRACTOR_VERSION,
) -> None:
    """Cache *body* as the extraction of *path*, whose source is *source*."""
    actor.cache_document(path, sha_of(source if source is not None else path), version, body)


def look_up(
    actor: WorkspaceActor,
    path: str,
    source: str | None = None,
    version: int = EXTRACTOR_VERSION,
) -> str | None:
    """The ask 45-4 will make: the Markdown on a hit, ``None`` on any miss."""
    return actor.document_extract(path, sha_of(source if source is not None else path), version)


def an_extract(path: str, body: str | None, source: str | None = None) -> DocumentExtract:
    """One cache entry, built the way ``cache_document`` builds it."""
    return DocumentExtract(
        path=path,
        source_sha=sha_of(source if source is not None else path),
        extractor_version=EXTRACTOR_VERSION,
        markdown=body,
        char_count=len(body) if body is not None else 0,
        extracted_at=datetime.now(UTC),
    )


class _StateSpy:
    """Records every state-change notification the actor's state emits.

    The same shape ``test_workspace_actor.py`` uses, deliberately — a second
    observer double would be a second thing to keep true.
    """

    def __init__(self) -> None:
        self.notifications: list[BaseState] = []

    def notify_state_change(self, state: BaseState) -> None:
        self.notifications.append(state)


def watch(actor: WorkspaceActor) -> _StateSpy:
    """Attach a spy to *actor*'s state and discard the attach-time notification."""
    spy = _StateSpy()
    actor.state.observer(spy)
    spy.notifications.clear()  # attaching an observer notifies once, by design
    return spy


class _ExtractWithExtraField(DocumentExtract):
    """A persisted extract carrying a field the write path has never heard of.

    Golden Rule #12's guard, in the only formulation that works. A whole-model
    comparison would stay green against an enumerated rebuild naming every field
    that exists *today* — which is exactly the code that silently destroys the
    field added tomorrow.
    """

    extra_field: str = "sentinel"


# ---------------------------------------------------------------------------
# AC1: the model
# ---------------------------------------------------------------------------


class TestDocumentExtract:
    def test_carries_exactly_the_six_fields(self) -> None:
        assert set(DocumentExtract.model_fields) == {
            "path",
            "source_sha",
            "extractor_version",
            "markdown",
            "char_count",
            "extracted_at",
        }

    def test_round_trips_through_pydantic(self) -> None:
        entry = an_extract("notes.docx", "# Notes")
        assert DocumentExtract.model_validate(entry.model_dump()) == entry

    def test_a_dropped_body_round_trips_as_none(self) -> None:
        entry = an_extract("notes.docx", None)
        assert DocumentExtract.model_validate(entry.model_dump()).markdown is None

    def test_it_round_trips_through_json_which_is_the_persisted_encoding(self) -> None:
        # ``model_dump()`` above hands back a live ``datetime``; the snapshot
        # that ``notify_state_change()`` emits is ``model_dump_json()``, where
        # ``extracted_at`` becomes an ISO string. Only this asserts the form
        # that is actually persisted survives the trip back.
        entry = an_extract("notes.docx", "# Notes")
        assert DocumentExtract.model_validate_json(entry.model_dump_json()) == entry


# ---------------------------------------------------------------------------
# AC2: the state field
# ---------------------------------------------------------------------------


class TestWorkspaceStateDocuments:
    def test_documents_defaults_to_an_empty_map(self) -> None:
        assert WorkspaceState().documents == {}

    def test_a_filled_cache_survives_a_state_round_trip(self, workspaces_root: Path) -> None:
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        restored = WorkspaceState.model_validate(actor.state.model_dump())
        assert restored.documents["notes.docx"].markdown == "# Notes"

    def test_a_filled_cache_survives_the_json_snapshot_the_notify_emits(
        self, workspaces_root: Path
    ) -> None:
        # ``notify_state_change()`` serialises the whole state through
        # ``model_dump_json()``, so that — not the python-mode dump — is the
        # shape a resume reads back, nested ``DocumentExtract`` and all.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        restored = WorkspaceState.model_validate_json(actor.state.model_dump_json())
        assert restored.documents["notes.docx"] == actor.state.documents["notes.docx"]


# ---------------------------------------------------------------------------
# AC6: hit and miss — four distinct miss reasons, four specs
# ---------------------------------------------------------------------------


class TestHitAndMiss:
    def test_a_hit_returns_the_body(self, workspaces_root: Path) -> None:
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        assert look_up(actor, "notes.docx") == "# Notes"

    def test_an_unknown_path_is_a_miss(self, workspaces_root: Path) -> None:
        actor = start_actor()
        assert look_up(actor, "never-seen.docx") is None

    def test_a_source_sha_mismatch_is_a_miss(self, workspaces_root: Path) -> None:
        # The file changed under the cache: the extract describes bytes that are
        # no longer there.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes", source="v1")
        assert look_up(actor, "notes.docx", source="v2") is None

    def test_an_extractor_version_mismatch_is_a_miss(self, workspaces_root: Path) -> None:
        # Bumping the constant invalidates every cached extract everywhere, with
        # no sweep and no migration.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes", version=EXTRACTOR_VERSION)
        assert look_up(actor, "notes.docx", version=EXTRACTOR_VERSION + 1) is None

    def test_an_evicted_body_is_a_miss(self, workspaces_root: Path) -> None:
        # Metadata alone is not a hit — the body is what the caller asked for.
        #
        # ``is None`` on its own cannot assert that. Drop the body check from
        # the lookup and it falls through to ``return entry.markdown``, which
        # *is* ``None`` — so the row would be treated as a hit and the
        # assertion would still pass. The LRU is what separates the two, since
        # a hit reorders and a miss does not.
        actor = start_actor()
        actor.state.documents["a.docx"] = an_extract("a.docx", None)
        for name in ("b.docx", "c.docx"):
            fill(actor, name)
        assert look_up(actor, "a.docx") is None
        assert list(actor.state.documents) == ["a.docx", "b.docx", "c.docx"]

    def test_an_empty_body_is_a_hit(self, workspaces_root: Path) -> None:
        # An extraction that legitimately produced nothing is not a miss: re-running
        # it would produce nothing again.
        actor = start_actor()
        fill(actor, "empty.docx", body="")
        assert look_up(actor, "empty.docx") == ""


# ---------------------------------------------------------------------------
# AC6: the LRU order, which a plain dict's insertion order carries
# ---------------------------------------------------------------------------


class TestLruOrder:
    def test_a_hit_moves_the_entry_to_the_end(self, workspaces_root: Path) -> None:
        actor = start_actor()
        for name in ("a.docx", "b.docx", "c.docx"):
            fill(actor, name)
        look_up(actor, "a.docx")
        assert list(actor.state.documents) == ["b.docx", "c.docx", "a.docx"]

    def test_a_re_fill_refreshes_recency_rather_than_duplicating(
        self, workspaces_root: Path
    ) -> None:
        actor = start_actor()
        for name in ("a.docx", "b.docx", "c.docx"):
            fill(actor, name)
        fill(actor, "a.docx", body="re-extracted")
        assert list(actor.state.documents) == ["b.docx", "c.docx", "a.docx"]
        assert actor.state.documents["a.docx"].markdown == "re-extracted"

    def test_a_miss_reorders_nothing(self, workspaces_root: Path) -> None:
        actor = start_actor()
        for name in ("a.docx", "b.docx", "c.docx"):
            fill(actor, name)
        look_up(actor, "never-seen.docx")
        look_up(actor, "a.docx", source="a different source")
        assert list(actor.state.documents) == ["a.docx", "b.docx", "c.docx"]


# ---------------------------------------------------------------------------
# AC5: THE HEADLINE CRITERION — the no-notify guard
# ---------------------------------------------------------------------------


class TestNotifyMatrix:
    def test_a_text_read_notifies_nothing(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        # The property epic 29 shipped and this story must not spend: reads are
        # the majority of workspace traffic and none of them writes an event.
        (workspace_tree / "plain.txt").write_text("hello\n", encoding="utf-8")
        spy = watch(workspace_actor)
        read(wired_card, "plain.txt")
        assert spy.notifications == []

    def test_a_cache_hit_notifies_nothing(self, workspaces_root: Path) -> None:
        # The hit reorders the LRU in memory. Persisted recency therefore lags
        # live recency until the next fill — deliberate, and not to be "fixed".
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        spy = watch(actor)
        assert look_up(actor, "notes.docx") == "# Notes"
        assert spy.notifications == []

    def test_a_lookup_miss_notifies_nothing(self, workspaces_root: Path) -> None:
        actor = start_actor()
        spy = watch(actor)
        look_up(actor, "never-seen.docx")
        assert spy.notifications == []

    def test_a_fill_notifies_exactly_once(self, workspaces_root: Path) -> None:
        actor = start_actor()
        spy = watch(actor)
        fill(actor, "notes.docx", body="# Notes")
        assert len(spy.notifications) == 1

    def test_a_fill_that_also_evicts_still_notifies_exactly_once(
        self, workspaces_root: Path
    ) -> None:
        # Never once per evicted entry: the notify follows the insert *and* the
        # eviction, so one fill is one event however much it displaced.
        actor = start_actor(max_documents=1)
        fill(actor, "a.docx")
        spy = watch(actor)
        fill(actor, "b.docx")
        assert len(spy.notifications) == 1
        assert list(actor.state.documents) == ["b.docx"]


# ---------------------------------------------------------------------------
# AC7 / AC10: the two caps, at the cap and one past it
# ---------------------------------------------------------------------------


class TestMaxDocumentsCap:
    def test_at_the_cap_nothing_is_evicted(self, workspaces_root: Path) -> None:
        actor = start_actor(max_documents=3)
        for name in ("a.docx", "b.docx", "c.docx"):
            fill(actor, name)
        assert list(actor.state.documents) == ["a.docx", "b.docx", "c.docx"]

    def test_one_past_the_cap_removes_the_lru_oldest_entry_outright(
        self, workspaces_root: Path
    ) -> None:
        # The entry, not just its body: this cap answers the number of rows in
        # the map, and metadata kept forever is the leak it exists to stop.
        actor = start_actor(max_documents=3)
        for name in ("a.docx", "b.docx", "c.docx", "d.docx"):
            fill(actor, name)
        assert "a.docx" not in actor.state.documents
        assert list(actor.state.documents) == ["b.docx", "c.docx", "d.docx"]

    def test_a_hit_protects_an_entry_from_the_next_eviction(
        self, workspaces_root: Path
    ) -> None:
        actor = start_actor(max_documents=3)
        for name in ("a.docx", "b.docx", "c.docx"):
            fill(actor, name)
        look_up(actor, "a.docx")
        fill(actor, "d.docx")
        assert "a.docx" in actor.state.documents
        assert "b.docx" not in actor.state.documents


class TestMaxDocumentCharsCap:
    def test_at_the_cap_no_body_is_dropped(self, workspaces_root: Path) -> None:
        actor = start_actor(max_documents=100, max_document_chars=100)
        fill(actor, "a.docx", body="a" * 60)
        fill(actor, "b.docx", body="b" * 40)
        assert actor.state.documents["a.docx"].markdown == "a" * 60
        assert actor.state.documents["b.docx"].markdown == "b" * 40

    def test_one_past_the_cap_drops_the_lru_oldest_body_and_keeps_its_metadata(
        self, workspaces_root: Path
    ) -> None:
        actor = start_actor(max_documents=100, max_document_chars=100)
        fill(actor, "a.docx", body="a" * 60)
        fill(actor, "b.docx", body="b" * 40)
        before = actor.state.documents["a.docx"]
        fill(actor, "c.docx", body="c")

        dropped = actor.state.documents["a.docx"]
        assert dropped.markdown is None
        assert dropped.path == before.path
        assert dropped.source_sha == before.source_sha
        assert dropped.extractor_version == before.extractor_version
        assert dropped.char_count == before.char_count
        assert dropped.extracted_at == before.extracted_at
        # It drops one body, not every body: 40 + 1 fits under the cap.
        assert actor.state.documents["b.docx"].markdown == "b" * 40
        assert actor.state.documents["c.docx"].markdown == "c"

    def test_a_single_over_cap_document_drops_its_own_body(
        self, workspaces_root: Path
    ) -> None:
        # A permanent miss costing one re-extraction per read — correct, and not
        # a case to special-case. The loop must terminate rather than spin.
        actor = start_actor(max_documents=100, max_document_chars=100)
        fill(actor, "huge.docx", body="x" * 150)
        assert actor.state.documents["huge.docx"].markdown is None
        assert actor.state.documents["huge.docx"].char_count == 150

    def test_the_char_sum_counts_only_bodied_entries(self, workspaces_root: Path) -> None:
        # A dropped body must not keep pressing on the cap it already relieved.
        actor = start_actor(max_documents=100, max_document_chars=100)
        fill(actor, "a.docx", body="a" * 90)
        fill(actor, "b.docx", body="b" * 90)
        assert actor.state.documents["a.docx"].markdown is None
        assert actor.state.documents["b.docx"].markdown == "b" * 90


class TestEvictDocumentBodiesIsPure:
    def test_it_reports_every_path_it_touched(self) -> None:
        documents = {
            "a.docx": an_extract("a.docx", "a" * 60),
            "b.docx": an_extract("b.docx", "b" * 60),
        }
        assert evict_document_bodies(documents, max_documents=10, max_document_chars=100) == [
            "a.docx"
        ]

    def test_an_empty_map_is_a_no_op(self) -> None:
        documents: dict[str, DocumentExtract] = {}
        assert evict_document_bodies(documents, max_documents=0, max_document_chars=0) == []
        assert documents == {}

    def test_a_bodiless_map_over_the_char_cap_terminates(self) -> None:
        # No bodied entry remains, so there is nothing left to drop — the loop
        # must exit rather than spin on a cap it can no longer satisfy.
        documents = {"a.docx": an_extract("a.docx", None)}
        assert evict_document_bodies(documents, max_documents=10, max_document_chars=0) == []
        assert documents["a.docx"].markdown is None


# ---------------------------------------------------------------------------
# AC9: Golden Rule #12 — the body drop copies, it does not rebuild
# ---------------------------------------------------------------------------


class TestGoldenRule12BodyDrop:
    def test_a_body_drop_preserves_a_field_the_write_path_never_heard_of(self) -> None:
        # An enumerated rebuild returns a plain ``DocumentExtract`` and fails
        # both assertions. A whole-model comparison would fail neither, which is
        # why this is the guard the rule prescribes.
        documents: dict[str, DocumentExtract] = {
            "a.docx": _ExtractWithExtraField(
                path="a.docx",
                source_sha=sha_of("a.docx"),
                extractor_version=EXTRACTOR_VERSION,
                markdown="a" * 60,
                char_count=60,
                extracted_at=datetime.now(UTC),
            )
        }
        evict_document_bodies(documents, max_documents=10, max_document_chars=10)

        survivor = documents["a.docx"]
        assert survivor.markdown is None
        assert isinstance(survivor, _ExtractWithExtraField)
        assert survivor.extra_field == "sentinel"


# ---------------------------------------------------------------------------
# AC6: neither method raises — a document path degrades, it never propagates
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_a_lookup_of_an_absent_path_returns_none(self, workspaces_root: Path) -> None:
        actor = start_actor()
        assert look_up(actor, "") is None
        assert look_up(actor, "nested/deeply/absent.docx") is None

    def test_a_fill_of_an_empty_body_records_a_zero_char_count(
        self, workspaces_root: Path
    ) -> None:
        actor = start_actor()
        fill(actor, "empty.docx", body="")
        assert actor.state.documents["empty.docx"].char_count == 0

    def test_a_lookup_of_an_already_bodiless_entry_returns_none_twice(
        self, workspaces_root: Path
    ) -> None:
        actor = start_actor()
        actor.state.documents["a.docx"] = an_extract("a.docx", None)
        assert look_up(actor, "a.docx") is None
        assert look_up(actor, "a.docx") is None

    def test_char_count_is_computed_from_the_body_not_supplied(
        self, workspaces_root: Path
    ) -> None:
        # It cannot disagree with the body it describes, because it is never a
        # parameter.
        actor = start_actor()
        fill(actor, "notes.docx", body="12345")
        assert actor.state.documents["notes.docx"].char_count == 5
