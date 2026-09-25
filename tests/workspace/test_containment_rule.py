"""One containment rule, written on ``Filesystem``, obeyed by every glob a caller supplies.

``Filesystem`` has always refused a *path* that resolves outside the tree. It never
saw a *pattern*: ``_expand_media_refs`` globbed the private root with a string taken
verbatim from the agent's prompt, and ``workspace_glob``/``workspace_grep`` validated
their ``path`` argument while handing ``pattern`` and ``include`` straight to
``Path.glob`` and ``Path.rglob``. Both traverse ``..`` without normalising it, so the
two sibling directories ADR-051 Decision 9 placed *beside* the tree — precisely so no
read capability could name them — were reachable, and so was anything else the
process could open.

Three habits run through the specs below:

- **the card, never a hand-built backend.** A spec that constructs a ``Filesystem``
  and calls it directly proves the helper and not the capability; every leak recorded
  in the audit arrived through a bound ``WorkspaceTool``;
- **the planted file is asserted readable first.** ``_hands_back_nothing`` returns
  ``True`` for an empty answer by design, which is the same shape a typo'd fixture
  path produces — so every probe checks its own bait before checking the closure;
- **the two grep engines are exercised separately.** ``_grep_rg`` refuses a ``..``
  glob of its own accord and ``_grep_python`` did not, so a suite that only ever ran
  the first would have called the second's content leak green.
"""

from __future__ import annotations

import ast
import inspect
import shutil
from pathlib import Path

import pytest

import akgentic.tool.workspace.read as read_module
from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.card import WorkspaceTool
from akgentic.tool.workspace.models import GIT_DIR_SUFFIX, META_DIR_SUFFIX, PERM_ERR_MSG
from akgentic.tool.workspace.readers import MediaContent
from akgentic.tool.workspace.workspace import Filesystem, PathEscapeError, get_workspace
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeOrchestratorProxy,
    card_for,
    tool_named,
    workspace_path_for,
)
from tests.workspace.test_metadata_directory import (
    LOCK_NAME,
    META_LEAF,
    _hands_back_nothing,
    _seeded_meta,
)

GIT_LEAF = f"{WORKSPACE_NAME}{GIT_DIR_SUFFIX}"
"""The journal sibling's own leaf, beside ``WORKSPACE_NAME`` in one kind directory."""

OUTSIDE_DIR = "outside"
"""An ordinary directory beside the tree — neither sibling, just *not the tree*."""

OUTSIDE_IMAGE = "outside-sibling.png"
OUTSIDE_IMAGE_BYTES = b"OUTSIDE-SIBLING-BYTES"
OUTSIDE_TEXT = "outside-secret.txt"
OUTSIDE_TEXT_BODY = "SECRET-OUTSIDE"

META_IMAGE = "meta-sibling.png"
META_IMAGE_BYTES = b"META-SIBLING-BYTES"
RAG_RECORD = "record.yaml"
RAG_RECORD_BODY = "extract: SECRET-EXTRACTED-TEXT"

GIT_DOCUMENT = "journal-sibling.pdf"
GIT_DOCUMENT_BYTES = b"%PDF-JOURNAL-SIBLING"

FAR_DIR = "far"
FAR_IMAGE = "far-above.png"
FAR_IMAGE_BYTES = b"FOUR-LEVELS-ABOVE-BYTES"

SECRET_RE = "SECRET"
"""The regex every grep probe searches for. Both baits carry it; nothing in the tree does."""


class Bait:
    """Every file planted outside the tree, with the paths that name each one.

    A class rather than a tuple because each probe wants a different pair of
    (pattern, planted file) and a reader has to be able to tell which bait a
    failing assertion caught.
    """

    def __init__(self, tree: Path, workspaces_root: Path) -> None:
        self.tree = tree
        self.sibling_image = tree.parent / OUTSIDE_DIR / OUTSIDE_IMAGE
        self.sibling_text = tree.parent / OUTSIDE_DIR / OUTSIDE_TEXT
        self.deep_text = workspaces_root / OUTSIDE_DIR / OUTSIDE_TEXT
        self.meta_image = tree.parent / META_LEAF / META_IMAGE
        self.rag_record = tree.parent / META_LEAF / "rag" / RAG_RECORD
        self.git_document = tree.parent / GIT_LEAF / GIT_DOCUMENT
        self.far_image = workspaces_root.parent / FAR_DIR / FAR_IMAGE

    def plant(self) -> Bait:
        """Create every bait file, then prove each one is readable.

        The proof is the point: an empty answer from a read closure is
        indistinguishable from a refusal, so a probe whose bait was never
        written passes green while testing nothing at all.
        """
        for path, payload in (
            (self.sibling_image, OUTSIDE_IMAGE_BYTES),
            (self.deep_text, OUTSIDE_TEXT_BODY.encode()),
            (self.sibling_text, OUTSIDE_TEXT_BODY.encode()),
            (self.meta_image, META_IMAGE_BYTES),
            (self.rag_record, RAG_RECORD_BODY.encode()),
            (self.git_document, GIT_DOCUMENT_BYTES),
            (self.far_image, FAR_IMAGE_BYTES),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            assert path.read_bytes() == payload, f"bait {path} was not planted"
        assert not self.tree.joinpath(OUTSIDE_DIR).exists()
        return self


@pytest.fixture
def bait(workspace_tree: Path, workspaces_root: Path) -> Bait:
    """Images and documents planted in a sibling directory, in both siblings, and above."""
    _seeded_meta()
    return Bait(workspace_tree, workspaces_root).plant()


@pytest.fixture
def card(orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path) -> WorkspaceTool:
    """A real bound card for agent ``alice`` over the shared test tree."""
    built, _observer = card_for(orchestrator_proxy, "alice")
    return built


def _spoken(parts: list[str | MediaContent], pattern: str) -> str:
    """Everything the command says back, minus the agent's own pattern echoed to it.

    The refusal names the token it refused, in the family of the three strings
    ``_expand_media_refs`` already emits, so the pattern the agent typed comes
    back verbatim — and a pattern naming ``<leaf>.index`` would trip a plain
    substring check on nothing but its author's own words. What is left after
    removing the echo is exactly what the command *added*, which is the only
    place a disclosure can come from.
    """
    rendered = "".join(part for part in parts if isinstance(part, str))
    return rendered.replace(pattern, "")


def _leaks(answer: object) -> list[str]:
    """Every bait marker present in *answer* — empty when nothing escaped.

    Returned rather than asserted so a failure names *which* bait came back,
    which is the difference between a red run that documents the leak and one
    that only says ``False is not True``.
    """
    rendered = str(answer)
    markers = (
        OUTSIDE_IMAGE,
        OUTSIDE_TEXT,
        OUTSIDE_TEXT_BODY,
        META_IMAGE,
        RAG_RECORD,
        RAG_RECORD_BODY,
        GIT_DOCUMENT,
        FAR_IMAGE,
        LOCK_NAME,
        META_DIR_SUFFIX,
        GIT_DIR_SUFFIX,
    )
    return [marker for marker in markers if marker in rendered]


##
## AC 1 — the rule is stated once, publicly, on Filesystem
##


class TestFilesystemStatesTheRule:
    """``root``, ``contains`` and ``resolve_path`` — one comparison, two shapes of caller."""

    def test_root_is_the_resolved_absolute_root(self, workspaces_root: Path) -> None:
        """The property answers exactly what the constructor resolved."""
        workspace = get_workspace(WORKSPACE_PATH)

        assert workspace.root.is_absolute()
        assert workspace.root == (workspaces_root / WORKSPACE_PATH).resolve()

    def test_root_is_read_only(self, workspaces_root: Path) -> None:
        """A settable root would be a second way to move the tree under the gate."""
        workspace = get_workspace(WORKSPACE_PATH)

        with pytest.raises(AttributeError):
            workspace.root = Path("/tmp")  # type: ignore[misc]

    def test_contains_accepts_what_is_under_the_root(self, workspace_tree: Path) -> None:
        """A file, a directory and the root itself all lie under the root."""
        workspace = get_workspace(WORKSPACE_PATH)
        (workspace_tree / "sub").mkdir()
        (workspace_tree / "sub" / "notes.md").write_text("hello", encoding="utf-8")

        assert workspace.contains(workspace.root)
        assert workspace.contains(workspace_tree / "sub")
        assert workspace.contains(workspace_tree / "sub" / "notes.md")

    def test_contains_refuses_a_sibling_and_a_traversal(self, workspace_tree: Path) -> None:
        """Both siblings and a plain ``..`` are outside, and an unresolved one is too."""
        workspace = get_workspace(WORKSPACE_PATH)

        assert not workspace.contains(workspace_tree.parent / META_LEAF)
        assert not workspace.contains(workspace_tree.parent / GIT_LEAF)
        assert not workspace.contains(workspace_tree / ".." / OUTSIDE_DIR)

    def test_contains_compares_components_not_a_string_prefix(self, workspaces_root: Path) -> None:
        """``team-1`` must not contain ``team-11`` — the prefix-sibling trap.

        Two workspaces of one principal and one kind, whose names share a
        prefix. A containment written as ``str.startswith`` answers that the
        second is inside the first, and an agent anchored on ``team-1`` reads
        and writes ``team-11``'s whole tree as ordinary in-tree activity, with
        the per-path write gate none the wiser. ``is_relative_to`` compares path
        components and answers correctly.
        """
        short = get_workspace(workspace_path_for("team-1"))
        long = get_workspace(workspace_path_for("team-11"))
        (long.root / "notes.md").write_text("the neighbour's file", encoding="utf-8")

        assert str(long.root).startswith(str(short.root))
        assert not short.contains(long.root)
        assert not short.contains(long.root / "notes.md")
        with pytest.raises(PathEscapeError):
            short.resolve_path("../team-11/notes.md")

    def test_resolve_path_returns_the_resolved_path(self, workspace_tree: Path) -> None:
        """The public spelling of what every backend method already did."""
        workspace = get_workspace(WORKSPACE_PATH)

        assert workspace.resolve_path("sub/notes.md") == workspace.root / "sub" / "notes.md"

    def test_resolve_path_raises_on_an_escape(self, workspace_tree: Path) -> None:
        """``PathEscapeError``, so an escaping path stays tellable from an OS denial."""
        workspace = get_workspace(WORKSPACE_PATH)

        with pytest.raises(PathEscapeError):
            workspace.resolve_path(f"../{META_LEAF}/{LOCK_NAME}")

    def test_resolve_path_is_written_in_terms_of_contains(self, workspace_tree: Path) -> None:
        """One comparison, not two: weakening ``contains`` must weaken ``resolve_path``.

        Asserted by substitution rather than by reading the source, because the
        defect this story removes is precisely a second copy of the comparison
        that agrees today and drifts later.
        """
        workspace = get_workspace(WORKSPACE_PATH)
        escaping = f"../{META_LEAF}/{LOCK_NAME}"

        workspace.contains = lambda candidate: True  # type: ignore[method-assign]
        assert workspace.resolve_path(escaping) == (workspace.root / escaping).resolve()

    def test_the_private_validator_is_renamed_not_aliased(self) -> None:
        """No compatibility shim: a second spelling is a second caller to keep in step."""
        assert not hasattr(Filesystem, "_validate_path")


##
## AC 2 and AC 3 — no module outside Filesystem reaches into it, and no copy survives
##


def _tool_sources() -> list[Path]:
    """Every module under ``src/akgentic/tool/``, from the installed package itself."""
    package_root = Path(inspect.getfile(Filesystem)).parent.parent
    return sorted(package_root.rglob("*.py"))


def _cross_object_reaches(source_file: Path, names: set[str]) -> list[str]:
    """Attribute accesses to *names* on anything but ``self``, as ``file:line`` strings."""
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    return [
        f"{source_file.name}:{node.lineno} ({node.attr})"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in names
        and not (isinstance(node.value, ast.Name) and node.value.id == "self")
    ]


class TestNothingReachesIntoTheFilesystem:
    """The private attribute stops being part of the backend's contract."""

    def test_no_module_reaches_a_private_attribute_of_the_filesystem(self) -> None:
        """AST, not a text search: a docstring naming ``_root`` is documentation.

        ``GitJournal._root`` is that class's own attribute over a different tree,
        reached as ``self._root`` — which is why the sweep exempts ``self`` rather
        than exempting a file.
        """
        sources = _tool_sources()
        offenders = [
            reach
            for source_file in sources
            for reach in _cross_object_reaches(source_file, {"_root", "_validate_path"})
        ]

        assert offenders == []
        assert len(sources) > 40, "the sweep found almost no modules — the root is wrong"

    def test_the_read_closures_hold_no_second_copy_of_the_comparison(self) -> None:
        """``is_relative_to`` belongs to ``Filesystem.contains`` and nowhere else here.

        ``_glob_factory`` and ``_grep_factory`` each hand-rolled
        ``(backend._root / path).resolve()`` followed by their own
        ``is_relative_to`` — two copies of a rule that has one owner, and the
        shape that let the media-ref glob skip it entirely.
        """
        tree = ast.parse(Path(inspect.getfile(read_module)).read_text(encoding="utf-8"))
        copies = [
            f"read/__init__.py:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "is_relative_to"
        ]

        assert copies == []

    def test_an_escaping_path_still_reads_exactly_as_it_did(
        self, card: WorkspaceTool, bait: Bait
    ) -> None:
        """The refusal the two closures already gave, byte-identical after the rewrite."""
        for name in ("workspace_glob", "workspace_grep"):
            with pytest.raises(RetriableError) as excinfo:
                tool_named(card, name)("*", f"../{META_LEAF}")
            assert str(excinfo.value) == PERM_ERR_MSG


##
## AC 4, AC 5 and AC 6 — the media-ref glob obeys the rule
##

ESCAPING_MEDIA_PATTERNS = [
    f"../{OUTSIDE_DIR}/*.png",
    f"../{META_LEAF}/*.png",
    f"../{GIT_LEAF}/*.pdf",
    "../../../../*/*.png",
    f"../{META_LEAF}/rag/*.yaml",
]
"""The audit's probe shape, verbatim, plus the record glob its extension filter hid."""


class TestTheMediaRefGlobObeysTheRule:
    """A pattern out of the prompt is a pattern, not a licence."""

    @pytest.mark.parametrize("pattern", ESCAPING_MEDIA_PATTERNS)
    def test_an_escaping_pattern_yields_no_media_and_no_name(
        self, card: WorkspaceTool, bait: Bait, pattern: str
    ) -> None:
        """No ``MediaContent``, no planted filename, neither sibling's suffix.

        The document-hint branch is checked too: ``!!name[=> Use workspace_read
        tool]`` discloses a name, which is the whole of what the journal probe
        returned before this story.
        """
        parts = card._expand_media_refs(f'look at !!"{pattern}" please')

        assert not [part for part in parts if isinstance(part, MediaContent)]
        assert _leaks(_spoken(parts, pattern)) == []

    @pytest.mark.parametrize("pattern", ESCAPING_MEDIA_PATTERNS)
    def test_the_refusal_says_the_pattern_escaped(
        self, card: WorkspaceTool, bait: Bait, pattern: str
    ) -> None:
        """Not "no image found": an author told that rewrites the same path for ever.

        This is the spec the explicit shape branch exists for, and it asserts the
        **shape** refusal specifically. Deleting that branch leaves the match test
        catching every one of these patterns — safely, and with the *other*
        message — so a spec that accepted either wording would pass against a
        rule that no longer looks at the pattern at all.
        """
        parts = card._expand_media_refs(f'!!"{pattern}"')
        rendered = "".join(part for part in parts if isinstance(part, str))

        assert "no image found in the workspace" not in rendered
        assert read_module._REF_ESCAPE_MSG in rendered

    def test_an_absolute_pattern_returns_a_list_rather_than_raising(
        self, card: WorkspaceTool, bait: Bait
    ) -> None:
        """Today an uncaught ``NotImplementedError`` leaves the COMMAND channel.

        Decided **before** ``Path.glob`` is reached, never by catching what it
        raises: CPython 3.12 raises ``NotImplementedError`` here and 3.13 raises
        ``ValueError``, so a guard written as an ``except`` stops guarding on an
        interpreter upgrade.
        """
        for pattern in ("/etc/passwd", str(bait.sibling_image), f"{bait.tree}/*.png"):
            parts = card._expand_media_refs(f'!!"{pattern}"')

            assert isinstance(parts, list)
            assert not [part for part in parts if isinstance(part, MediaContent)]
            assert _leaks(_spoken(parts, pattern)) == []

    def test_a_match_that_escapes_through_a_symlink_is_refused(
        self, card: WorkspaceTool, bait: Bait, workspace_tree: Path
    ) -> None:
        """Defence in depth: the shape is legal and the match still leaves the tree.

        A symlink planted *inside* the tree satisfies both shape rules — no
        ``..``, not absolute — and resolves onto the sibling directory. Only the
        match filter catches it, which is why the filter is not redundant with
        the two branches above.
        """
        (workspace_tree / "shortcut").symlink_to(bait.sibling_image.parent)

        parts = card._expand_media_refs('!!"shortcut/*.png"')
        rendered = _spoken(parts, "shortcut/*.png")

        assert not [part for part in parts if isinstance(part, MediaContent)]
        assert _leaks(rendered) == []
        assert read_module._REF_OUTSIDE_MSG in rendered
        assert "no image found in the workspace" not in rendered


class TestLegitimateMediaRefsAreUntouched:
    """The four result classes the command already had, unchanged."""

    def test_an_image_inside_the_tree_still_expands(
        self, card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        """The ordinary case, through the real card rather than a hand-built backend."""
        (workspace_tree / "photo.png").write_bytes(b"inside-bytes")

        parts = card._expand_media_refs("look at !!photo.png")

        assert MediaContent(data=b"inside-bytes", media_type="image/png") in parts

    def test_a_pattern_matching_nothing_still_says_no_image_found(
        self, card: WorkspaceTool, bait: Bait
    ) -> None:
        """A missing file is not an escape, and the two must not answer alike."""
        parts = card._expand_media_refs("!!missing.png")

        assert "!!missing.png[Error: no image found in the workspace]" in parts

    def test_a_document_inside_the_tree_still_gets_its_hint(
        self, card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        """The hint discloses a name, so it must survive for names inside the tree."""
        (workspace_tree / "report.pdf").write_bytes(b"%PDF-inside")

        parts = card._expand_media_refs("!!report.pdf")

        assert "!!report.pdf[=> Use workspace_read tool]" in parts


##
## AC 7 — workspace_glob's pattern
##

ESCAPING_GLOB_PATTERNS = [
    f"../{META_LEAF}/*",
    f"../{GIT_LEAF}/*",
    f"../{OUTSIDE_DIR}/*",
    "../../../../*/*",
    f"{{..,src}}/{META_LEAF}/*",
    "/etc/*",
]
"""Including the brace form, which only escapes *after* ``_expand_braces`` has run."""


class TestTheGlobPatternObeysTheRule:
    """``pattern`` is validated like ``path``, and after brace expansion."""

    @pytest.mark.parametrize("pattern", ESCAPING_GLOB_PATTERNS)
    def test_an_escaping_pattern_enumerates_nothing(
        self, card: WorkspaceTool, bait: Bait, pattern: str
    ) -> None:
        """Refused, or empty — never a ``../`` path rendered back to the agent."""
        glob = tool_named(card, "workspace_glob")

        try:
            answer = str(glob(pattern))
        except RetriableError as refusal:
            assert str(refusal) == PERM_ERR_MSG
            return
        assert _leaks(answer) == []

    def test_a_recursive_glob_inside_the_tree_still_works(
        self, card: WorkspaceTool, bait: Bait, workspace_tree: Path
    ) -> None:
        """The pattern an agent writes when it is looking around must not be collateral."""
        (workspace_tree / "sub").mkdir()
        (workspace_tree / "sub" / "notes.md").write_text("hello", encoding="utf-8")

        answer = str(tool_named(card, "workspace_glob")("**/*"))

        assert "notes.md" in answer
        assert _leaks(answer) == []

    def test_a_brace_pattern_inside_the_tree_still_expands(
        self, card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        """Brace expansion is the reason the check runs after it, not instead of it."""
        (workspace_tree / "a.py").write_text("x = 1", encoding="utf-8")
        (workspace_tree / "b.ts").write_text("const x = 1", encoding="utf-8")

        answer = str(tool_named(card, "workspace_glob")("*.{py,ts}"))

        assert "a.py" in answer
        assert "b.ts" in answer


##
## AC 8 — workspace_grep's include glob, which reads content
##

ESCAPING_INCLUDE_GLOBS = [
    f"../{META_LEAF}/rag/*.yaml",
    "../../../outside/*.txt",
    f"../{OUTSIDE_DIR}/*.txt",
    "/etc/*",
]

GREP_ENGINES = ["python", "rg"]
"""Both, because containment must not depend on whether ``rg`` is installed."""


@pytest.fixture(params=GREP_ENGINES)
def grep_engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Force ``_grep_python`` by hiding ``rg``, or leave the real lookup alone.

    Hiding it is the case that matters: ``rg`` refuses a ``..`` glob on its own
    account, so a suite that only ever ran the ripgrep leg would have called the
    Python leg's content leak green.
    """
    if request.param == "python":
        monkeypatch.setattr(read_module.shutil, "which", lambda _name: None)
    elif shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed on this host")
    return str(request.param)


class TestTheGrepIncludeGlobObeysTheRule:
    """``include`` reaches ``rglob`` and then ``read_text`` — a content leak, not a listing."""

    @pytest.mark.parametrize("include", ESCAPING_INCLUDE_GLOBS)
    def test_an_escaping_include_returns_no_content_from_outside_the_tree(
        self, card: WorkspaceTool, bait: Bait, grep_engine: str, include: str
    ) -> None:
        """The records ADR-051 put beside the tree, and any readable file above it."""
        grep = tool_named(card, "workspace_grep")

        try:
            answer = str(grep(SECRET_RE, "", include))
        except RetriableError as refusal:
            assert str(refusal) == PERM_ERR_MSG
            return
        assert _leaks(answer) == [], f"{grep_engine} engine leaked for include={include!r}"

    def test_an_include_inside_the_tree_still_restricts_the_search(
        self, card: WorkspaceTool, bait: Bait, grep_engine: str, workspace_tree: Path
    ) -> None:
        """The capability keeps working: the rule refuses escapes, not globs."""
        (workspace_tree / "kept.py").write_text("SECRET inside\n", encoding="utf-8")
        (workspace_tree / "skipped.md").write_text("SECRET inside\n", encoding="utf-8")

        answer = str(tool_named(card, "workspace_grep")(SECRET_RE, "", "*.py"))

        assert "kept.py" in answer
        assert "skipped.md" not in answer

    def test_the_two_engines_agree_on_an_escaping_include(
        self, card: WorkspaceTool, bait: Bait, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whatever the answer is, it is the same one with and without ``rg``.

        **Skipped rather than run when ``rg`` is absent.** Without the skip this
        spec still passes on such a host — ``with_rg`` resolves to the Python
        engine too, so it compares that engine with itself and proves nothing
        while its name says otherwise. That is the failure mode the whole module
        is written against: a check narrowed until it agrees. The sibling specs
        skip loudly through ``grep_engine``; this one has to say so itself,
        because it drives both engines from inside one test rather than through
        that fixture.
        """
        if shutil.which("rg") is None:
            pytest.skip("ripgrep is not installed on this host — the comparison would be vacuous")
        grep = tool_named(card, "workspace_grep")
        include = f"../{META_LEAF}/rag/*.yaml"

        def answer_of() -> str:
            try:
                return str(grep(SECRET_RE, "", include))
            except RetriableError as refusal:
                return f"refused: {refusal}"

        with_rg = answer_of()
        monkeypatch.setattr(read_module.shutil, "which", lambda _name: None)
        without_rg = answer_of()

        assert with_rg == without_rg
        assert _leaks(with_rg) == []


##
## AC 6 again, at the surface the metadata story guarded — nothing reaches the siblings
##


class TestNeitherSiblingIsReachableFromAnyReadClosure:
    """The property ADR-051 Decision 9 chose sibling placement to obtain, end to end."""

    def test_no_read_closure_hands_back_either_sibling(
        self, card: WorkspaceTool, bait: Bait
    ) -> None:
        """List, glob and grep, each asked for both siblings by name and by traversal."""
        assert bait.meta_image.read_bytes() == META_IMAGE_BYTES
        assert bait.git_document.read_bytes() == GIT_DOCUMENT_BYTES

        glob = tool_named(card, "workspace_glob")
        grep = tool_named(card, "workspace_grep")
        listing = tool_named(card, "workspace_list")

        assert _hands_back_nothing(glob, f"../{META_LEAF}/*")
        assert _hands_back_nothing(glob, f"../{GIT_LEAF}/*")
        assert _hands_back_nothing(glob, "*", f"../{META_LEAF}")
        assert _hands_back_nothing(listing, f"../{META_LEAF}")
        assert _hands_back_nothing(grep, SECRET_RE, "", f"../{META_LEAF}/rag/*.yaml")
