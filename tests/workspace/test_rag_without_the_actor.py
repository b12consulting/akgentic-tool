"""A search and a listing answer with **no actor at all** (story 57-1).

The claim ADR-053 Decision 6 makes, applied to retrieval's read side: what needs
a mailbox is a run whose report must land somewhere and the ``#index-`` /
``#embed-`` children a Pydantic card cannot parent. A search reads the document
records; a listing renders them. Neither needs one, and after story 57-1 neither
has one.

**Why the raising stand-in is the row that matters.** The other two rows here —
a card whose ``_workspace_proxy`` is ``None``, and a list-only card — assert the
answer and would stay green if the ask came back on some *other* path, or if a
future refactor reached the actor for one field. The raising stand-in asserts the
property behaviourally: any attribute access at all is a failure, so the moment a
line reads ``proxy.anything`` this module reddens and names the attribute. It is
the row a re-introduced ask trips over, and a source-reading assertion would not
be.

Nothing here starts a thread, and nothing here reaches a network: the vector leg
is degraded by replacing ``build_embedding_service`` at its source module, so the
searches below answer from the keyword leg over records seeded on disk.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from akgentic.tool.workspace.documents.models import (
    EXTRACTOR_VERSION,
    DocumentExtract,
    RagChunk,
    RagFile,
    RagStatus,
    chunk_id,
)
from akgentic.tool.workspace.documents.store import DocumentEntry, YamlDocumentStore
from akgentic.tool.workspace.models import content_sha
from akgentic.tool.workspace.rag.params import WorkspaceRagList
from akgentic.tool.workspace.tool import WorkspaceTool
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    tool_named,
)

_UNAVAILABLE = "Retrieval indexing is not available for this workspace."

_BODY = "# Invoice\n\nPayment terms are net thirty.\n"
"""One document, indexed whole, so the keyword leg has exactly one chunk to hit."""


class _ActorTouched(BaseException):
    """Raised by :class:`_RaisingProxy`, and deliberately **not** an ``Exception``.

    Every retrieval closure wraps its body in ``except Exception`` and answers the
    unavailable sentence, which is correct — an LLM-facing callable must not hand
    a model a traceback. It also means an ``AssertionError`` from a stand-in is
    swallowed, and the row below then reddens on the *sentence* instead of naming
    the attribute that was read. Deriving from ``BaseException`` puts the cause in
    the failure report, which is the difference between "something degraded" and
    "``rag_snapshot`` was called".
    """


class _RaisingProxy:
    """A stand-in actor that raises on **every** attribute access.

    Not "a proxy whose ``rag_search`` raises": that would only pin the one name a
    reader thought of. Any read of any attribute is the failure, which is what
    makes "no actor at all" a behavioural claim rather than a source-reading one.
    """

    def __getattr__(self, name: str) -> Any:
        raise _ActorTouched(f"the card reached the actor for {name!r} — it must not")


def _seed(path: str = "invoice.md") -> None:
    """Put one fully indexed record on disk for the tree the cards below resolve.

    Written through a plain :class:`YamlDocumentStore` keyed by the resolved
    workspace path — the same file a card's own cache reads — so nothing here
    depends on an actor having been created.
    """
    sha = content_sha(_BODY.encode("utf-8"))
    store = YamlDocumentStore()
    store.put_document(
        WORKSPACE_PATH,
        DocumentEntry(
            path=path,
            extract=DocumentExtract(
                path=path,
                source_sha=sha,
                extractor_version=EXTRACTOR_VERSION,
                markdown=_BODY,
                char_count=len(_BODY),
                extracted_at=datetime.now(UTC),
            ),
            row=RagFile(
                path=path,
                status=RagStatus.EMBEDDED,
                indexed_sha=sha,
                chunk_count=1,
                chunks=[
                    RagChunk(
                        chunk_id=chunk_id(WORKSPACE_PATH, path, sha, 0),
                        ordinal=0,
                        start=0,
                        end=len(_BODY),
                        heading_path=["Invoice"],
                    )
                ],
                updated_at=datetime.now(UTC),
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Degrade the vector leg, so every answer below is the keyword leg's.

    Replaced at its **source module**, which is where the capability's
    function-level import finds it. Without this the leg would build a real
    embedding client and attempt a round trip.
    """

    def _explodes(model: str, provider: str) -> object:
        raise RuntimeError("no embedder in this spec")

    monkeypatch.setattr(
        "akgentic.tool.vector_store.embedding_actor.build_embedding_service", _explodes
    )


def _bind(orchestrator_proxy: FakeOrchestratorProxy, **fields: Any) -> WorkspaceTool:
    """Bind a real card over the shared tree and hand it back."""
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, **fields)
    card.observer(FakeActorToolObserver(orchestrator_proxy))
    return card


class TestTheSnapshotRendersWithNoActor:
    """Three rows, and the third is the one a re-introduced ask would redden."""

    def test_a_card_whose_proxy_is_none_renders_its_rows(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The bare-observer shape the closures' own comments already document."""
        _seed()
        card = _bind(orchestrator_proxy, workspace_rag_list=True)
        card._workspace_proxy = None

        rendered = card.get_commands()[WorkspaceRagList]()

        assert "invoice.md" in rendered
        assert rendered != _UNAVAILABLE

    def test_a_list_only_card_renders_its_rows(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Neither indexing nor searching is enabled; the listing still answers."""
        _seed()
        card = _bind(
            orchestrator_proxy,
            workspace_rag_list=True,
            workspace_rag_index=False,
            workspace_rag_search=False,
        )

        rendered = card.get_commands()[WorkspaceRagList]()
        [provider] = card.get_context_states()
        state = provider()

        assert "invoice.md" in rendered
        assert state is not None
        assert [row.path for row in state.rows] == ["invoice.md"]

    def test_a_proxy_that_raises_on_every_access_is_never_touched(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The mutation-proof row: both the command and the provider render anyway."""
        _seed()
        card = _bind(orchestrator_proxy, workspace_rag_list=True)
        card._workspace_proxy = _RaisingProxy()

        rendered = card.get_commands()[WorkspaceRagList]()
        [provider] = card.get_context_states()
        state = provider()

        assert "invoice.md" in rendered
        assert state is not None
        assert [row.path for row in state.rows] == ["invoice.md"]


class TestTheSearchRunsWithNoActor:
    """The same third row, for the search."""

    def test_a_proxy_that_raises_on_every_access_is_never_touched(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """It returns its hits, so the assertion is the answer and not merely "no raise"."""
        _seed()
        card = _bind(orchestrator_proxy, workspace_rag_search=True)
        assert card._retrieval_bound(), "this bind resolved nothing — the gate would answer first"
        card._workspace_proxy = _RaisingProxy()

        answer = str(tool_named(card, "workspace_rag_search")("payment"))

        assert "invoice.md" in answer
        assert "keyword match" in answer

    def test_a_card_whose_proxy_is_none_still_returns_its_hits(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A bound card whose actor was never created, or died — it searches anyway.

        That is the point of story 57-1 rather than a side effect: the records are
        on disk, and reading them was never the actor's to own.
        """
        _seed()
        card = _bind(orchestrator_proxy, workspace_rag_search=True)
        card._workspace_proxy = None

        answer = str(tool_named(card, "workspace_rag_search")("payment"))

        assert "invoice.md" in answer
        assert "keyword match" in answer
