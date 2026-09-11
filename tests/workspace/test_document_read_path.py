"""``workspace_read``'s document branch, against ``#Workspace``'s state cache (45-4).

``test_document_cache.py`` addresses the actor directly; this file drives the
**read path** — the card's two closures, the digest it takes over the source
bytes, and what the tree looks like afterwards.

The headline is one spec: a document whose source has been *replaced* reads as
the replacement. The sidecar this story deletes was keyed on the source
**filename**, so it served the old extraction for ever; the cache is keyed on the
source **bytes**, so a replacement is a miss. Everything else here defends the
cost of that: a text read still asks the actor nothing, a hit still notifies
nothing, and a document read now writes nothing to the tree at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import PrivateAttr

from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.documents.models import EXTRACTOR_VERSION
from akgentic.tool.workspace.models import content_sha
from akgentic.tool.workspace.readers import DocumentReader
from akgentic.tool.workspace.tool import WorkspaceRead, WorkspaceTool

from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    RecordingDocumentStore,
    read,
    stored_docs,
)


class _StubDocumentReader(DocumentReader):
    """An extractor that reports the bytes it was handed, and records its runs.

    Deriving the body from the *source* is what makes the stale-source spec
    meaningful: a re-extraction of replaced bytes produces a visibly different
    answer, so "the second read returned the new extraction" is an assertion
    about content rather than about a call count alone.
    """

    _runs: list[str] = PrivateAttr(default_factory=list)

    @property
    def runs(self) -> list[str]:
        """One entry per extraction, holding the source bytes as text."""
        return self._runs

    def extract_text(self, content: bytes, path: str) -> str:
        body = content.decode("utf-8", errors="replace")
        self._runs.append(body)
        return f"# extracted\n{body}\n" + "filler " * 20


class SpyingAskProxy:
    """Counts the cache calls a read makes and forwards everything to the actor.

    The property NFR1 needs is *how many asks a read makes*, which cannot be
    inferred from the resulting state: a text read and a document hit both leave
    the map untouched.
    """

    def __init__(self, target: WorkspaceActor) -> None:
        self.target = target
        self.lookups: list[str] = []
        self.fills: list[str] = []

    def document_extract(self, path: str, source_sha: str, extractor_version: int) -> str | None:
        self.lookups.append(path)
        return self.target.document_extract(path, source_sha, extractor_version)

    def cache_document(
        self, path: str, source_sha: str, extractor_version: int, markdown: str
    ) -> None:
        self.fills.append(path)
        self.target.cache_document(path, source_sha, extractor_version, markdown)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)


class RaisingCacheProxy:
    """A proxy whose every cache call fails — a dead actor, from the card's side."""

    def __init__(self) -> None:
        self.calls = 0

    def attach(self, agent: object, agent_name: str) -> None:
        """The bind-time holder registration — the actor was alive then; it died later."""

    def document_extract(self, path: str, source_sha: str, extractor_version: int) -> str | None:
        self.calls += 1
        raise RuntimeError("actor is dead")

    def cache_document(
        self, path: str, source_sha: str, extractor_version: int, markdown: str
    ) -> None:
        self.calls += 1
        raise RuntimeError("actor is dead")


def document_card(
    orchestrator_proxy: FakeOrchestratorProxy,
    reader: DocumentReader,
    workspace_proxy: object | None = None,
) -> tuple[WorkspaceTool, WorkspaceActor | None]:
    """A card that can read documents, plus the actor behind it when there is one."""
    observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=workspace_proxy)
    card = WorkspaceTool(
        workspace_id=WORKSPACE_NAME,
        workspace_read=WorkspaceRead(document_reader=reader),
    )
    card.observer(observer)
    entry = orchestrator_proxy.children.get(workspace_actor_name(WORKSPACE_PATH))
    actor = entry[1] if entry is not None else None
    return card, actor if isinstance(actor, WorkspaceActor) else None


def tree_snapshot(tree: Path) -> dict[str, bytes]:
    """Every file under *tree*, by relative path, with its bytes."""
    return {
        str(path.relative_to(tree)): path.read_bytes()
        for path in sorted(tree.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# AC5 — THE HEADLINE: backlog row 34, the defect the whole ADR exists to kill
# ---------------------------------------------------------------------------


class TestAStaleSourceIsAMiss:
    def test_a_replaced_source_reads_as_the_replacement(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # The sidecar was keyed on the filename, so this exact sequence served
        # the first extraction for ever — for the life of the workspace, with no
        # way for an agent to discover it was reading a file that no longer
        # exists. The cache is keyed on the bytes, so the replacement is a miss.
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        source = workspace_tree / "report.pdf"

        source.write_bytes(b"the original report")
        first = read(card, "report.pdf")
        source.write_bytes(b"a completely different report")
        second = read(card, "report.pdf")

        assert "the original report" in first
        assert "a completely different report" in second
        assert "the original report" not in second
        assert reader.runs == ["the original report", "a completely different report"]

    def test_rewriting_the_identical_bytes_is_still_a_hit(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # The negative half of the same property. Keying on the bytes must not
        # collapse into "always re-extract": a file rewritten with what it
        # already held has not changed, whatever its mtime now says.
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        source = workspace_tree / "report.pdf"

        source.write_bytes(b"the original report")
        read(card, "report.pdf")
        source.write_bytes(b"the original report")
        read(card, "report.pdf")

        assert reader.runs == ["the original report"]

    def test_a_deleted_source_is_not_served_from_the_cache(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # Row 34 in miniature, and a deliberate consequence of hashing the
        # source: the read fails at ``backend.read`` before any lookup. Under
        # the sidecar the extraction of a deleted file was served for ever.
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        source = workspace_tree / "report.pdf"
        source.write_bytes(b"the original report")
        read(card, "report.pdf")

        source.unlink()

        with pytest.raises(RetriableError, match="File not found: report.pdf"):
            read(card, "report.pdf")


# ---------------------------------------------------------------------------
# AC6 — an EXTRACTOR_VERSION bump invalidates every entry, through the read path
# ---------------------------------------------------------------------------


class TestTheExtractorVersion:
    def test_a_bumped_version_re_extracts_an_otherwise_valid_entry(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No sweep, no migration, no code that has to find the stale rows: a
        # closure carrying a different version simply stops hitting them. This
        # is the remedy backlog row 46 needs, and the story bumps nothing.
        reader = _StubDocumentReader()
        card, actor = document_card(orchestrator_proxy, reader)
        assert actor is not None
        (workspace_tree / "report.pdf").write_bytes(b"the original report")

        read(card, "report.pdf")
        assert stored_docs(actor)["report.pdf"].extractor_version == EXTRACTOR_VERSION

        # The version is captured when the callable is built, so a new card is
        # what a deployment carrying a bumped constant would have.
        monkeypatch.setattr("akgentic.tool.workspace.card.EXTRACTOR_VERSION", EXTRACTOR_VERSION + 1)
        bumped, _actor = document_card(orchestrator_proxy, reader)
        read(bumped, "report.pdf")

        assert reader.runs == ["the original report", "the original report"]
        assert stored_docs(actor)["report.pdf"].extractor_version == EXTRACTOR_VERSION + 1


# ---------------------------------------------------------------------------
# AC2 / AC7 — NFR1: what a read is still allowed to cost
# ---------------------------------------------------------------------------


class TestTheReadPathStaysFree:
    def test_a_text_read_asks_the_actor_nothing(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # The document branch's one ask must not leak onto the text branch,
        # which is the majority of workspace traffic. Hoisting the hash and the
        # lookup above the branch is the tidying that would do it.
        reader = _StubDocumentReader()
        _seed, actor = document_card(orchestrator_proxy, reader)  # creates the actor
        assert actor is not None
        spy = SpyingAskProxy(actor)
        spied, _actor = document_card(orchestrator_proxy, reader, workspace_proxy=spy)
        (workspace_tree / "notes.md").write_text("alpha\nbravo\n", encoding="utf-8")

        assert "alpha" in read(spied, "notes.md")

        assert spy.lookups == []
        assert spy.fills == []

    def test_a_document_read_makes_exactly_one_lookup_and_one_fill(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        reader = _StubDocumentReader()
        _seed, actor = document_card(orchestrator_proxy, reader)  # creates the actor
        assert actor is not None
        spy = SpyingAskProxy(actor)
        spied, _actor = document_card(orchestrator_proxy, reader, workspace_proxy=spy)
        (workspace_tree / "report.pdf").write_bytes(b"the original report")

        read(spied, "report.pdf")  # a miss: one lookup, one fill
        read(spied, "report.pdf")  # a hit: one lookup, no fill

        assert spy.lookups == ["report.pdf", "report.pdf"]
        assert spy.fills == ["report.pdf"]

    def test_a_document_cache_hit_through_a_live_proxy_writes_nothing(
        self,
        threaded_orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        """The no-write property against a real actor thread and a real proxy.

        A document read *would* write per read if the lookup reached a write
        site. 45-4 is the first story to put a read on a proxy call, so this is
        the story that could break it. Frozen here against the live shape —
        45-3's guard sits at the actor and cannot see this.

        The recorder is announced through ``configure_document_store`` on the
        actor's own thread, before the fills, which must each show up: a
        recorder the actor never reached could not make the hit look silent.

        **Two documents, and the hit is on the older one.** Under the LRU that
        used to matter because the hit reordered; it no longer reorders, and the
        second document stays so that a write on *any* path would still be
        visible here.
        """
        reader = _StubDocumentReader()
        (workspace_tree / "a.pdf").write_bytes(b"the first report")
        (workspace_tree / "b.pdf").write_bytes(b"the second report")
        card, _actor = document_card(threaded_orchestrator_proxy, reader)
        pykka_proxy = threaded_orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)][1]
        recorder = RecordingDocumentStore()
        pykka_proxy.configure_document_store(recorder)

        read(card, "a.pdf")  # two misses, two fills — and a fill does write
        read(card, "b.pdf")

        # The attribute fetch is itself a mailbox turn, so it lands after both
        # fills: reaching the actor at all proves they have been applied.
        pykka_proxy.state.get(timeout=HANDSHAKE_TIMEOUT_S)
        assert sorted(recorder.written) == ["a.pdf", "b.pdf"]
        fills = len(recorder.puts)

        assert "the first report" in read(card, "a.pdf")

        assert reader.runs == ["the first report", "the second report"]  # a hit
        pykka_proxy.state.get(timeout=HANDSHAKE_TIMEOUT_S)  # the hit's turn has run
        assert len(recorder.puts) == fills, "the cache hit wrote a record"


# ---------------------------------------------------------------------------
# AC3 / AC10 — the tree stops carrying the agents' cache
# ---------------------------------------------------------------------------


class TestNothingIsWrittenToTheTree:
    def test_a_miss_and_a_hit_leave_the_tree_byte_identical(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        (workspace_tree / "report.pdf").write_bytes(b"the original report")
        before = tree_snapshot(workspace_tree)

        read(card, "report.pdf")  # a miss
        read(card, "report.pdf")  # a hit

        assert tree_snapshot(workspace_tree) == before

    def test_a_document_in_a_subdirectory_writes_nothing_either(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        docs = workspace_tree / "docs"
        docs.mkdir()
        (docs / "slides.pptx").write_bytes(b"PK the deck")
        before = tree_snapshot(workspace_tree)

        read(card, "docs/slides.pptx")

        assert tree_snapshot(workspace_tree) == before

    def test_reading_an_image_sidecar_extracts_and_writes_no_markdown(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Backlog row 47(b), re-checked — and the ADR's stated reason corrected.

        ADR-045 §8 says removing the ``is_sidecar`` guard means a re-read of
        ``.photo.png.1568.png`` *now* reaches ``DocumentReader``. It already did:
        the guard tested ``name.startswith(".") and name.endswith(".md")`` and
        this name ends ``.png``, so it never matched. Removing the guard changes
        nothing here.

        What does change, and improves: the re-read no longer writes a
        sidecar-of-a-sidecar (``..photo.png.1568.png.md``). Rows 34(b) and 47(b)
        stay open — ``_maybe_resize``'s own filename-keyed image cache keeps its
        shape and its defects, because base64 image bytes do not belong in a JSON
        state document.
        """
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        (workspace_tree / ".photo.png.1568.png").write_bytes(b"resized image bytes")

        result = read(card, ".photo.png.1568.png")

        assert reader.runs == ["resized image bytes"]  # it reached DocumentReader
        assert "# extracted" in result
        assert list(workspace_tree.glob("*.md")) == []
        assert not (workspace_tree / "..photo.png.1568.png.md").exists()


# ---------------------------------------------------------------------------
# AC4 — force_document_regeneration, with its new meaning
# ---------------------------------------------------------------------------


class TestForceDocumentRegeneration:
    def test_it_ignores_a_valid_entry_re_extracts_and_re_fills(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # The old meaning was "ignore a file that happens to sit beside the
        # source", which no notion of validity governed. This one overrides a
        # correct answer — and still leaves the cache filled, not emptied.
        reader = _StubDocumentReader()
        card, actor = document_card(orchestrator_proxy, reader)
        assert actor is not None
        (workspace_tree / "report.pdf").write_bytes(b"the original report")

        read(card, "report.pdf")
        entry_before = stored_docs(actor)["report.pdf"]
        read(card, "report.pdf", force_document_regeneration=True)
        entry_after = stored_docs(actor)["report.pdf"]

        assert reader.runs == ["the original report", "the original report"]
        assert entry_after.markdown == entry_before.markdown
        assert entry_after.source_sha == content_sha(b"the original report")
        assert entry_after.extracted_at >= entry_before.extracted_at

    def test_a_forced_read_still_serves_the_current_bytes(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        (workspace_tree / "report.pdf").write_bytes(b"the original report")

        assert "the original report" in read(card, "report.pdf", force_document_regeneration=True)


# ---------------------------------------------------------------------------
# Degradation — every failure is a miss, never a failed read (ADR-045 §C5)
# ---------------------------------------------------------------------------


class TestTheCacheDegradesToAMiss:
    def test_a_card_with_no_proxy_reads_correctly_and_never_hits(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # A harness shape that binds no actor. Reaching for the private
        # attributes is the only way to produce it: ``observer()`` always binds,
        # and the point is what the closures do when the binding is absent.
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        card._workspace_proxy = None
        card._workspace_tell = None
        (workspace_tree / "report.pdf").write_bytes(b"the original report")

        assert "the original report" in read(card, "report.pdf")
        assert "the original report" in read(card, "report.pdf")

        assert reader.runs == ["the original report"] * 2  # every read is a miss

    def test_a_raising_proxy_reads_correctly_and_never_hits(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        reader = _StubDocumentReader()
        raising = RaisingCacheProxy()
        card, _actor = document_card(orchestrator_proxy, reader, workspace_proxy=raising)
        (workspace_tree / "report.pdf").write_bytes(b"the original report")

        assert "the original report" in read(card, "report.pdf")
        assert "the original report" in read(card, "report.pdf")

        assert reader.runs == ["the original report"] * 2
        assert raising.calls == 4  # two lookups and two fills, all refused


# ---------------------------------------------------------------------------
# AC3 — the sidecar is gone from the read path
# ---------------------------------------------------------------------------


class TestTheSidecarIsGone:
    def test_an_existing_sidecar_is_neither_read_nor_written(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # A tree that predates this story still holds these. They are inert: the
        # read extracts from the source and leaves the leftover untouched.
        reader = _StubDocumentReader()
        card, _actor = document_card(orchestrator_proxy, reader)
        (workspace_tree / "report.pdf").write_bytes(b"the original report")
        leftover = workspace_tree / ".report.pdf.md"
        leftover.write_text("# stale extraction from another era", encoding="utf-8")

        result = read(card, "report.pdf")

        assert "the original report" in result
        assert "another era" not in result
        assert leftover.read_text(encoding="utf-8") == "# stale extraction from another era"

    def test_the_extraction_is_reachable_only_through_the_actor(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        reader = _StubDocumentReader()
        card, actor = document_card(orchestrator_proxy, reader)
        assert actor is not None
        (workspace_tree / "report.pdf").write_bytes(b"the original report")

        read(card, "report.pdf")

        entry = stored_docs(actor)["report.pdf"]
        assert entry.source_sha == content_sha(b"the original report")
        assert entry.markdown is not None and "the original report" in entry.markdown
        with patch.object(DocumentReader, "extract_text") as never:
            assert "the original report" in read(card, "report.pdf")
            never.assert_not_called()
