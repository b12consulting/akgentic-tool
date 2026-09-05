"""``IndexWorker``: what it is, what it reads, and what it composes.

The worker is exercised as a plain object on this thread — no actor system, no
mailbox — which is what the conftest's ``ExecHarness`` does for the exec worker
and for the same reason: the properties under test are about what the worker
*produces*, and a second actor system would only add a way for them to be flaky.

The one property that needs a parser is AC20's: a table cut at the ceiling has to
come back out as a table, and the only honest way to assert that is to re-parse
the composed piece through the same ``markdown-it-py`` the splitter uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.tool.core.deferred import DeferredWorker
from akgentic.tool.workspace.actor import WorkspaceActor
from akgentic.tool.workspace.card.params import WorkspaceRagIndex
from akgentic.tool.workspace.documents.models import RagChunk, chunk_id
from akgentic.tool.workspace.documents.splitter import _parser
from akgentic.tool.workspace.documents.worker import (
    INDEX_WORKER_NAME_PREFIX,
    IndexError,
    IndexRequest,
    IndexResult,
    IndexWorker,
    compose_chunk_text,
    index_worker_name,
)
from akgentic.tool.workspace.execution import ExecWorker
from akgentic.tool.workspace.readers import DocumentReader

from tests.workspace.conftest import WORKSPACE_NAME, DeadAddress

_TABLE_PARAMS = WorkspaceRagIndex(
    chunk_chars=60, max_chunk_chars=120, min_chunk_chars=1, chunk_overlap_chars=0
)
"""Small enough that a modest table passes the ceiling and is cut at a row."""


class _Reporter:
    """Stands in for ``#Workspace``: collects whichever payload the worker sends."""

    def __init__(self) -> None:
        self.results: list[IndexResult] = []
        self.errors: list[IndexError] = []

    def receiveMsg_IndexResult(self, msg: IndexResult) -> None:  # noqa: N802
        self.results.append(msg)

    def receiveMsg_IndexError(self, msg: IndexError) -> None:  # noqa: N802
        self.errors.append(msg)


def run_worker(request: IndexRequest) -> _Reporter:
    """Run one request through a bare worker and return what it reported."""
    worker = IndexWorker()
    worker.config = BaseConfig(name=index_worker_name(request.scope, request.path))
    worker._parent = DeadAddress("#Workspace")
    worker._orchestrator = DeadAddress("orchestrator")
    worker.on_start()
    reporter = _Reporter()
    worker.proxy_tell = lambda address, actor_type=None: reporter  # type: ignore[method-assign]
    worker.stop = lambda *args, **kwargs: None  # type: ignore[method-assign,assignment]
    worker.receiveMsg_IndexRequest(request)
    return reporter


def a_request(
    path: str = "notes.md",
    markdown: str | None = "# Title\n\nA paragraph.\n",
    source_sha: str = "sha-1",
    params: WorkspaceRagIndex | None = None,
) -> IndexRequest:
    """One request, with the actor's cached body supplied unless told otherwise."""
    return IndexRequest(
        path=path,
        scope=WORKSPACE_NAME,
        source_sha=source_sha,
        markdown=markdown,
        params=params or WorkspaceRagIndex(),
        reader=DocumentReader(llm_client=None),
    )


def a_table(rows: int) -> str:
    """A GFM table with *rows* body rows, under one heading."""
    lines = ["## Numbers", "", "| left | right |", "| --- | --- |"]
    lines.extend(f"| value {index:03d} | other {index:03d} |" for index in range(rows))
    return "\n".join(lines) + "\n"


class TestWhatTheWorkerIs:
    """Structural facts that are load-bearing and would otherwise be invisible."""

    def test_it_is_a_plain_akgent_and_not_a_deferred_worker(self) -> None:
        """``DeferredWorker`` reports into the actor's **exec** result cache.

        ``DeferredWorker.deliver`` / ``fail`` land on ``DeferredResultActor[…, str,
        ExecOutcome]``, so an index result routed that way would evict a running
        agent's exec outcome and mis-type the cache's value.
        """
        assert issubclass(IndexWorker, Akgent)
        assert not issubclass(IndexWorker, DeferredWorker)

    def test_the_actors_deferred_worker_is_still_the_exec_worker(self) -> None:
        """The index worker is spawned directly and joins no deferred mechanism."""
        assert WorkspaceActor.worker_class(WorkspaceActor) is ExecWorker  # type: ignore[arg-type]

    def test_the_worker_name_starts_with_the_teardown_marker(self) -> None:
        """Only the leading ``#`` is load-bearing — it classifies a tool actor."""
        name = index_worker_name("team-7", "deep/nested/report.pdf")
        assert name.startswith("#")
        assert name.startswith(f"{INDEX_WORKER_NAME_PREFIX}team-7-")

    def test_the_name_is_stable_for_one_path_and_differs_across_paths(self) -> None:
        """A path may contain anything a filesystem allows; the digest may not."""
        assert index_worker_name("t", "a.md") == index_worker_name("t", "a.md")
        assert index_worker_name("t", "a.md") != index_worker_name("t", "b.md")

    def test_the_three_payloads_are_not_messages(self) -> None:
        """A ``Message`` payload surfaces every transient worker as a busy member."""
        from akgentic.core.messages import Message

        for model in (IndexRequest, IndexResult, IndexError):
            assert not issubclass(model, Message)


class TestTheBodyItSplits:
    """Where the Markdown comes from, and what happens when it cannot be had."""

    def test_a_supplied_body_is_reused_and_reported_as_not_extracted(self) -> None:
        """The actor's cache already paid for it; extracting again would be waste."""
        reporter = run_worker(a_request(markdown="# Title\n\nBody.\n"))

        [result] = reporter.results
        assert result.extracted is False
        assert result.markdown == "# Title\n\nBody.\n"

    def test_a_text_file_is_read_through_the_backend_when_no_body_is_supplied(
        self, workspace_tree: Path
    ) -> None:
        """Every path goes through ``Filesystem``, never a join onto its private root."""
        (workspace_tree / "notes.md").write_text("# Read me\n\nFrom disk.\n", encoding="utf-8")

        reporter = run_worker(a_request(markdown=None))

        [result] = reporter.results
        assert result.extracted is True
        assert "From disk." in result.markdown

    def test_a_missing_file_reports_an_index_error_rather_than_raising(
        self, workspace_tree: Path
    ) -> None:
        """The actor is the only party that can record it against the file."""
        reporter = run_worker(a_request(path="gone.md", markdown=None))

        assert reporter.results == []
        [failure] = reporter.errors
        assert failure.path == "gone.md"
        assert failure.source_sha == "sha-1"
        assert "FileNotFoundError" in failure.reason

    def test_an_escaping_path_reports_an_index_error(self, workspace_tree: Path) -> None:
        """``Filesystem`` validates internally; the worker turns the refusal into a row."""
        reporter = run_worker(a_request(path="../outside.md", markdown=None))

        assert reporter.results == []
        [failure] = reporter.errors
        assert "escapes workspace root" in failure.reason

    def test_an_extractor_that_raises_reports_an_index_error(
        self, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A document path degrades to a ``FAILED`` row, never to a dead worker."""
        (workspace_tree / "report.pdf").write_bytes(b"%PDF-1.4 not really")

        def boom(self: DocumentReader, content: bytes, path: str) -> str:
            raise RuntimeError("markitdown exploded")

        monkeypatch.setattr(DocumentReader, "extract_text", boom)

        reporter = run_worker(a_request(path="report.pdf", markdown=None))

        [failure] = reporter.errors
        assert failure.reason == "RuntimeError: markitdown exploded"

    def test_a_document_extension_goes_through_the_reader(
        self, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extraction configuration lives on the card and rides in with the request."""
        (workspace_tree / "report.pdf").write_bytes(b"%PDF-1.4 not really")
        seen: list[str] = []

        def fake_extract(self: DocumentReader, content: bytes, path: str) -> str:
            seen.append(path)
            return "# Extracted\n\nFrom the reader.\n"

        monkeypatch.setattr(DocumentReader, "extract_text", fake_extract)

        reporter = run_worker(a_request(path="report.pdf", markdown=None))

        assert seen == ["report.pdf"]
        [result] = reporter.results
        assert result.extracted is True
        assert "From the reader." in result.markdown


class TestTheChunksItMints:
    """Identity and alignment — the two things the actor cannot re-derive."""

    def test_every_chunk_carries_the_id_minted_from_the_scope_and_the_digest(self) -> None:
        """The actor stores these ids verbatim as ``VectorEntry.ref_id``."""
        reporter = run_worker(a_request(markdown="# A\n\nOne.\n\n## B\n\nTwo.\n"))

        [result] = reporter.results
        assert len(result.chunks) >= 2
        for ordinal, chunk in enumerate(result.chunks):
            assert chunk.ordinal == ordinal
            assert chunk.chunk_id == chunk_id(WORKSPACE_NAME, "notes.md", "sha-1", ordinal)

    def test_texts_are_index_aligned_with_chunks(self) -> None:
        """The actor zips the two lists into ``VectorEntry`` records, strictly."""
        reporter = run_worker(a_request(markdown="# A\n\nOne.\n\n## B\n\nTwo.\n"))

        [result] = reporter.results
        assert len(result.texts) == len(result.chunks)

    def test_an_empty_document_produces_no_chunks_and_still_reports(self) -> None:
        """Nothing to embed is a settled file, not a failure."""
        reporter = run_worker(a_request(markdown="   \n\n"))

        [result] = reporter.results
        assert result.chunks == []
        assert result.texts == []


class TestComposition:
    """What is embedded is re-derived at embed time and never stored."""

    def test_the_heading_path_leads_the_text_when_configured(self) -> None:
        """The prefix is composed from ``heading_path`` and written back nowhere."""
        chunk = RagChunk(chunk_id="c", ordinal=0, start=0, end=4, heading_path=["Invoice", "Fees"])

        composed = compose_chunk_text("body", chunk, prepend_heading_path=True)

        assert composed == "Invoice > Fees\n\nbody"

    def test_the_heading_path_is_omitted_when_the_card_turns_it_off(self) -> None:
        """``prepend_heading_path`` is read by the embedder and by nothing else."""
        chunk = RagChunk(chunk_id="c", ordinal=0, start=0, end=4, heading_path=["Invoice"])

        assert compose_chunk_text("body", chunk, prepend_heading_path=False) == "body"

    def test_a_chunk_with_no_heading_path_gets_no_prefix(self) -> None:
        """Content before the first heading has a legal path, and it is empty."""
        chunk = RagChunk(chunk_id="c", ordinal=0, start=0, end=4)

        assert compose_chunk_text("body", chunk, prepend_heading_path=True) == "body"

    def test_a_cut_tables_continuation_reparses_as_a_table(self) -> None:
        """AC20: the header alone is not a table — the delimiter row is generated.

        ``Span`` points at the header *row*, and the GFM delimiter line is not a
        ``tr``, so it falls outside those offsets. Without the generated line the
        composed piece is a header followed by rows, which ``markdown-it`` reads as
        a paragraph. The assertion is the re-parse, not the string.
        """
        markdown = a_table(rows=12)
        reporter = run_worker(a_request(markdown=markdown, params=_TABLE_PARAMS))

        [result] = reporter.results
        continuations = [
            (chunk, text)
            for chunk, text in zip(result.chunks, result.texts, strict=True)
            if chunk.header_start is not None
        ]
        assert continuations, "the table was not cut — the fixture no longer exercises rule 4"

        for chunk, text in continuations:
            body = text.split("\n\n", 1)[-1] if chunk.heading_path else text
            tokens = _parser().parse(body)
            assert any(token.type == "table_open" for token in tokens), (
                f"the continuation piece did not re-parse as a table:\n{body}"
            )

    def test_the_continuation_carries_the_header_row_verbatim(self) -> None:
        """The header is a slice of the document, not a copy stored on the chunk."""
        markdown = a_table(rows=12)
        reporter = run_worker(a_request(markdown=markdown, params=_TABLE_PARAMS))

        [result] = reporter.results
        for chunk, text in zip(result.chunks, result.texts, strict=True):
            if chunk.header_start is None:
                continue
            assert chunk.header_end is not None
            assert markdown[chunk.header_start : chunk.header_end] in text

    def test_the_first_piece_of_a_cut_table_carries_no_header_offsets(self) -> None:
        """It already contains the header inside its own slice."""
        reporter = run_worker(a_request(markdown=a_table(rows=12), params=_TABLE_PARAMS))

        [result] = reporter.results
        assert result.chunks[0].header_start is None


class TestReporting:
    """The worker reports once and stops, whatever happened."""

    def test_it_stops_itself_on_the_success_path(self) -> None:
        """A worker holds its parent's teardown open for as long as it lives."""
        stopped: list[bool] = []
        worker = IndexWorker()
        worker.config = BaseConfig(name="#index-t-abc")
        worker._parent = DeadAddress("#Workspace")
        worker._orchestrator = DeadAddress("orchestrator")
        worker.on_start()
        worker.proxy_tell = lambda address, actor_type=None: _Reporter()  # type: ignore[method-assign]
        worker.stop = lambda *args, **kwargs: stopped.append(True)  # type: ignore[method-assign,assignment]

        worker.receiveMsg_IndexRequest(a_request())

        assert stopped == [True]

    def test_it_stops_itself_on_the_failure_path(self, workspace_tree: Path) -> None:
        """The ``finally`` is what makes that true of both paths."""
        stopped: list[bool] = []
        worker = IndexWorker()
        worker.config = BaseConfig(name="#index-t-abc")
        worker._parent = DeadAddress("#Workspace")
        worker._orchestrator = DeadAddress("orchestrator")
        worker.on_start()
        worker.proxy_tell = lambda address, actor_type=None: _Reporter()  # type: ignore[method-assign]
        worker.stop = lambda *args, **kwargs: stopped.append(True)  # type: ignore[method-assign,assignment]

        worker.receiveMsg_IndexRequest(a_request(path="gone.md", markdown=None))

        assert stopped == [True]

    def test_a_lost_parent_does_not_raise(self) -> None:
        """There is nobody left to record the result against, and that is not fatal."""
        worker = IndexWorker()
        worker.config = BaseConfig(name="#index-t-abc")
        worker._parent = None
        worker._orchestrator = DeadAddress("orchestrator")
        worker.on_start()
        worker.stop = lambda *args, **kwargs: None  # type: ignore[method-assign,assignment]

        worker.receiveMsg_IndexRequest(a_request())  # must not raise

    def test_a_reporting_failure_does_not_raise(self) -> None:
        """A parent that stopped mid-extraction must not produce a traceback."""

        class _Dead:
            def receiveMsg_IndexResult(self, msg: Any) -> None:  # noqa: N802
                raise RuntimeError("actor is gone")

        worker = IndexWorker()
        worker.config = BaseConfig(name="#index-t-abc")
        worker._parent = DeadAddress("#Workspace")
        worker._orchestrator = DeadAddress("orchestrator")
        worker.on_start()
        worker.proxy_tell = lambda address, actor_type=None: _Dead()  # type: ignore[method-assign]
        worker.stop = lambda *args, **kwargs: None  # type: ignore[method-assign,assignment]

        worker.receiveMsg_IndexRequest(a_request())  # must not raise
