"""The human globs, greps and searches the workspace from slash commands.

``/workspace_glob``, ``/workspace_grep`` and ``/workspace_rag_search`` are the
read-side commands of story 62-3. The first two answer **exactly** what their
tools answer, through one shared render, and return the tool's refusal as text
rather than raising it. The third runs the tool's own search and renders the hits
for the human — a rendering the **tool** must never reach: the model keeps
receiving a :class:`RagSearchResult`.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from akgentic.tool.core import COMMAND, TOOL_CALL, CommandRegistry, ToolFactory
from akgentic.tool.errors import RetriableError
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
from akgentic.tool.workspace.rag import _UNAVAILABLE, _render_hits_for_human
from akgentic.tool.workspace.rag.params import WorkspaceRagSearch
from akgentic.tool.workspace.rag.search import MatchKind, RagSearchHit, RagSearchResult
from akgentic.tool.workspace.read.params import WorkspaceGlob, WorkspaceGrep
from akgentic.tool.workspace.tool import WorkspaceTool
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    read,
    tool_named,
)

SEARCH_COMMANDS = {"workspace_glob", "workspace_grep", "workspace_rag_search"}

_BODY = "# Invoice\n\nPayment terms are net thirty.\n"


def registry(card: WorkspaceTool) -> CommandRegistry:
    """The agent's command registry over *card* alone, as ``BaseAgent`` builds it."""
    return ToolFactory(tool_cards=[card]).get_command_registry()


def run(card: WorkspaceTool, text: str) -> str | None:
    """Dispatch *text* exactly as ``BaseAgent._dispatch_command`` does."""
    return registry(card).dispatch(text)


def command_names(card: WorkspaceTool) -> set[str]:
    return {fn.__name__ for fn in card.get_commands().values()}


def populate(tree: Path, files: dict[str, str]) -> None:
    for relative, body in files.items():
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def _bind(orchestrator_proxy: FakeOrchestratorProxy, **fields: Any) -> WorkspaceTool:
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, **fields)
    card.observer(FakeActorToolObserver(orchestrator_proxy))
    return card


def _seed_index(path: str = "invoice.md") -> None:
    """One fully indexed record on disk, so the keyword leg has a chunk to hit."""
    sha = content_sha(_BODY.encode("utf-8"))
    YamlDocumentStore().put_document(
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
    """Degrade the vector leg, so a search answers from the keyword leg, offline."""

    def _explodes(model: str, provider: str) -> object:
        raise RuntimeError("no embedder in this spec")

    monkeypatch.setattr(
        "akgentic.tool.vector_store.embedding_actor.build_embedding_service", _explodes
    )


@pytest.fixture
def search_card(orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path) -> WorkspaceTool:
    """A card with retrieval search on, bound over the shared tree."""
    return _bind(orchestrator_proxy, workspace_rag_search=True)


# ---------------------------------------------------------------------------
# AC 1, AC 2 — the three commands, their signatures, and when they are served
# ---------------------------------------------------------------------------


class TestTheSearchCommandSurface:
    EXPECTED = {
        "workspace_glob": [("pattern", True), ("path", False)],
        "workspace_grep": [("pattern", True), ("path", False), ("include", False)],
        "workspace_rag_search": [("query", True)],
    }

    def test_the_descriptors_name_each_command_and_its_arguments(
        self, search_card: WorkspaceTool
    ) -> None:
        descriptors = {d.name: d for d in registry(search_card).descriptors()}
        for name, args in self.EXPECTED.items():
            assert [(arg.name, arg.required) for arg in descriptors[name].args] == args, name

    def test_every_description_carries_an_example(self, search_card: WorkspaceTool) -> None:
        descriptors = {d.name: d for d in registry(search_card).descriptors()}
        for name in self.EXPECTED:
            assert f"Example: /{name}" in descriptors[name].description, name

    def test_the_grep_example_single_quotes_its_regex_and_says_why(
        self, search_card: WorkspaceTool
    ) -> None:
        descriptors = {d.name: d for d in registry(search_card).descriptors()}
        description = descriptors["workspace_grep"].description
        assert r"Example: /workspace_grep 'def \w+_factory' src" in description
        assert "single quotes" in description

    def test_command_is_on_by_default_for_the_three_params(self) -> None:
        for param in (WorkspaceGlob, WorkspaceGrep, WorkspaceRagSearch):
            assert param().expose == {TOOL_CALL, COMMAND}, param.__name__

    def test_a_default_card_serves_glob_and_grep_and_no_search(
        self, wired_card: WorkspaceTool
    ) -> None:
        # Retrieval stays opt-in: ``workspace_rag_search`` is ``False`` by default.
        names = command_names(wired_card)
        assert {"workspace_glob", "workspace_grep"} <= names
        assert "workspace_rag_search" not in names

    def test_a_read_only_card_serves_all_three(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = _bind(orchestrator_proxy, read_only=True, workspace_rag_search=True)
        assert SEARCH_COMMANDS <= command_names(card)

    @pytest.mark.parametrize("field", ["workspace_glob", "workspace_grep", "workspace_rag_search"])
    def test_a_persisted_param_on_tool_call_alone_serves_no_command(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path, field: str
    ) -> None:
        param_cls = {
            "workspace_glob": WorkspaceGlob,
            "workspace_grep": WorkspaceGrep,
            "workspace_rag_search": WorkspaceRagSearch,
        }[field]
        stored = WorkspaceTool(workspace_id=WORKSPACE_NAME).model_dump()
        stored[field] = {**param_cls().model_dump(), "expose": ["tool_call"]}
        card = WorkspaceTool.model_validate(stored)
        card.observer(FakeActorToolObserver(orchestrator_proxy))
        assert field not in command_names(card)
        assert field in {tool.__name__ for tool in card.get_tools()}

    @pytest.mark.parametrize(
        ("field", "param"),
        [
            ("workspace_glob", WorkspaceGlob(expose={COMMAND})),
            ("workspace_grep", WorkspaceGrep(expose={COMMAND})),
            ("workspace_rag_search", WorkspaceRagSearch(expose={COMMAND})),
        ],
    )
    def test_command_alone_serves_the_command_and_no_tool(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        field: str,
        param: Any,
    ) -> None:
        card = _bind(orchestrator_proxy, **{field: param})
        assert field in command_names(card)
        assert field not in {tool.__name__ for tool in card.get_tools()}


# ---------------------------------------------------------------------------
# AC 3 — /workspace_glob is the tool's answer, as text
# ---------------------------------------------------------------------------


def _aged(tree: Path, ages: dict[str, int]) -> None:
    """Write each file and set its mtime *age* seconds into the past."""
    now = datetime.now(UTC).timestamp()
    for relative, age in ages.items():
        populate(tree, {relative: f"{relative}\n"})
        os.utime(tree / relative, (now - age, now - age))


class TestGlob:
    def test_the_command_is_byte_equal_to_the_tool_ordering_and_truncation_included(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # mtime order (newest first) is deliberately not alphabetical, and the
        # third match is past ``max_results`` so the truncation notice is present.
        _aged(workspace_tree, {"docs/a.md": 30, "docs/b.md": 10, "docs/c.md": 20})
        card = _bind(orchestrator_proxy, workspace_glob=WorkspaceGlob(max_results=2))

        expected = tool_named(card, "workspace_glob")("*.md", "docs")

        assert run(card, "/workspace_glob '*.md' docs") == expected
        assert expected == ("docs/b.md\ndocs/c.md\n[... truncated: 3 total, showing first 2 ...]")

    def test_no_match_is_the_tools_sentence(self, wired_card: WorkspaceTool) -> None:
        assert run(wired_card, "/workspace_glob '*.nothing'") == "No files found."

    @pytest.mark.parametrize(
        ("text", "args"),
        [
            ("/workspace_glob '../*'", ("../*",)),
            ("/workspace_glob '*' ../neighbour", ("*", "../neighbour")),
        ],
    )
    def test_an_escape_is_the_tools_refusal_as_text(
        self, wired_card: WorkspaceTool, workspace_tree: Path, text: str, args: tuple[str, ...]
    ) -> None:
        (workspace_tree.parent / "neighbour").mkdir()
        with pytest.raises(RetriableError) as refused:
            tool_named(wired_card, "workspace_glob")(*args)
        assert run(wired_card, text) == str(refused.value)


# ---------------------------------------------------------------------------
# AC 4 — /workspace_grep, the same way
# ---------------------------------------------------------------------------


class TestGrep:
    def test_the_command_is_byte_equal_to_the_tool(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # One matching file: ``rg`` searches files in parallel, so the order
        # *across* files is the engine's and is not this spec's to pin.
        populate(workspace_tree, {"src/a.txt": "TODO one\nnope\nTODO two\n", "b.md": "done\n"})
        expected = tool_named(wired_card, "workspace_grep")("TODO")
        assert run(wired_card, "/workspace_grep TODO") == expected
        assert expected == "src/a.txt:1: TODO one\nsrc/a.txt:3: TODO two"

    def test_a_single_quoted_backslash_survives_the_registrys_shlex(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"src/a.txt": "x12\n", "src/b.txt": "xd\n"})
        assert run(wired_card, r"/workspace_grep 'x\d+' src") == "src/a.txt:1: x12"

    def test_include_restricts_the_search_to_matching_files(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"src/a.py": "# TODO py\n", "src/b.txt": "TODO txt\n"})
        assert run(wired_card, "/workspace_grep TODO src '*.py'") == "src/a.py:1: # TODO py"

    def test_no_match_is_the_tools_sentence(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"a.md": "alpha\n"})
        assert run(wired_card, "/workspace_grep zulu") == "No matches found."

    @pytest.mark.parametrize(
        ("text", "args"),
        [
            ("/workspace_grep '('", ("(",)),
            ("/workspace_grep x ../neighbour", ("x", "../neighbour")),
            ("/workspace_grep x '' '../*'", ("x", "", "../*")),
        ],
        ids=["invalid-regex", "escaping-path", "escaping-include"],
    )
    def test_a_refusal_is_returned_as_text(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        text: str,
        args: tuple[str, ...],
    ) -> None:
        populate(workspace_tree, {"a.md": "x\n"})
        (workspace_tree.parent / "neighbour").mkdir()
        with pytest.raises(RetriableError) as refused:
            tool_named(wired_card, "workspace_grep")(*args)
        assert run(wired_card, text) == str(refused.value)

    def test_an_invalid_regex_says_so(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"a.md": "x\n"})
        assert str(run(wired_card, "/workspace_grep '('")).startswith("Invalid regex pattern:")


# ---------------------------------------------------------------------------
# AC 5, AC 6 — /workspace_rag_search takes the rest of the line and renders text
# ---------------------------------------------------------------------------


class TestRagSearchCommand:
    def test_the_whole_rest_of_the_line_is_the_query_with_the_params_knobs(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[tuple[str, dict[str, Any]]] = []

        def capture(*args: Any, **kwargs: Any) -> RagSearchResult:
            seen.append((args[3], kwargs))
            return RagSearchResult(note="captured")

        monkeypatch.setattr("akgentic.tool.workspace.rag.search_documents", capture)
        card = _bind(
            orchestrator_proxy,
            workspace_rag_search=WorkspaceRagSearch(top_k=3, alpha=0.4, score_threshold=0.2),
        )

        assert run(card, "/workspace_rag_search how is the gate refused?") == "captured"

        [(query, kwargs)] = seen
        assert query == "how is the gate refused?"
        assert kwargs["top_k"] == 3
        assert kwargs["alpha"] == 0.4
        assert kwargs["score_threshold"] == 0.2
        assert kwargs["path_prefix"] == ""

    def test_the_hits_are_rendered_for_the_human(
        self, search_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        _seed_index()
        rendered = str(run(search_card, "/workspace_rag_search payment"))
        assert rendered.startswith("1. invoice.md")
        assert "(keyword)" in rendered
        assert "Payment terms are net thirty." in rendered

    def test_the_tool_still_returns_the_model(
        self, search_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        _seed_index()
        answer = tool_named(search_card, "workspace_rag_search")("payment")
        assert isinstance(answer, RagSearchResult)
        assert [hit.path for hit in answer.hits] == ["invoice.md"]

    def test_unbound_retrieval_answers_the_unavailable_sentence(
        self, search_card: WorkspaceTool
    ) -> None:
        search_card._resolved_store = None
        assert not search_card._retrieval_bound()
        assert run(search_card, "/workspace_rag_search anything at all") == _UNAVAILABLE


def _hit(**fields: Any) -> RagSearchHit:
    defaults: dict[str, Any] = {
        "path": "docs/a.md",
        "ordinal": 1,
        "chunk_count": 4,
        "start_line": 3,
        "end_line": 9,
        "score": 0.8312,
        "match": MatchKind.HYBRID,
        "text": "the gate refuses",
    }
    return RagSearchHit(**{**defaults, **fields})


class TestRenderHitsForHuman:
    def test_a_hit_with_a_range_and_a_score(self) -> None:
        result = RagSearchResult(hits=[_hit()])
        assert _render_hits_for_human(result) == (
            "1. docs/a.md:3-9 (hybrid, score 0.83)\n   the gate refuses"
        )

    def test_a_hit_with_no_range_names_its_chunk(self) -> None:
        result = RagSearchResult(hits=[_hit(start_line=None, end_line=None)])
        assert _render_hits_for_human(result) == (
            "1. docs/a.md (chunk 2/4) (hybrid, score 0.83)\n   the gate refuses"
        )

    def test_a_hit_with_no_score_omits_it(self) -> None:
        result = RagSearchResult(hits=[_hit(score=None, match=MatchKind.KEYWORD)])
        assert _render_hits_for_human(result) == ("1. docs/a.md:3-9 (keyword)\n   the gate refuses")

    def test_an_empty_result_with_a_note_is_the_note(self) -> None:
        assert _render_hits_for_human(RagSearchResult(note="Nothing matched.")) == (
            "Nothing matched."
        )

    def test_an_empty_result_with_no_note(self) -> None:
        assert _render_hits_for_human(RagSearchResult()) == "No passages found."

    def test_the_text_is_one_line_cut_to_300_characters(self) -> None:
        long_text = "line one\nline two\n" + "x" * 400
        rendered = _render_hits_for_human(RagSearchResult(hits=[_hit(text=long_text)]))
        _header, body = rendered.split("\n", 1)
        assert "\n" not in body
        assert body == "   " + ("line one line two " + "x" * 400)[:300] + "…"

    def test_hits_are_numbered_best_first(self) -> None:
        result = RagSearchResult(hits=[_hit(path="first.md"), _hit(path="second.md")])
        lines = _render_hits_for_human(result).splitlines()
        assert lines[0].startswith("1. first.md")
        assert lines[2].startswith("2. second.md")


# ---------------------------------------------------------------------------
# AC 7 — no observations
# ---------------------------------------------------------------------------


class TestNoObservations:
    @pytest.mark.parametrize(
        "text",
        [
            "/workspace_glob '*.md'",
            "/workspace_grep alpha",
            "/workspace_rag_search payment terms",
        ],
    )
    def test_no_command_touches_the_observation_map(
        self, search_card: WorkspaceTool, workspace_tree: Path, text: str
    ) -> None:
        populate(workspace_tree, {"a.md": "alpha\n", "docs/d.md": "delta\n"})
        _seed_index()
        read(search_card, "a.md")
        before = dict(search_card._observations)

        run(search_card, text)

        assert dict(search_card._observations) == before
        assert search_card.observation_for("docs/d.md") is None
