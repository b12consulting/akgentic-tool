# WorkspaceTool

Team-scoped filesystem access for LLM agents: read, list, glob, grep, view images, write, edit,
patch, delete and mkdir — every path anchored to one workspace root that nothing can escape.

```python
from akgentic.tool import WorkspaceTool
```

| | |
|---|---|
| Module | `akgentic.tool.workspace.tool` |
| Actor | none — the backend is an in-process `Filesystem`, not an actor |
| Channels used | `TOOL_CALL` (11 callables), `COMMAND` (`expand_media_refs`) |
| Optional extras | `[docs]` for binary reads, `[vision]` for image resizing |
| Environment | `AKGENTIC_WORKSPACES_ROOT` (default `./workspaces`) |

---

## The ToolCard

```python
class WorkspaceTool(ToolCard):
    # Where the files live
    workspace_id: str | None = None

    # Read-side capabilities
    workspace_read: WorkspaceRead | bool = True
    workspace_view: WorkspaceView | bool = True
    workspace_list: WorkspaceList | bool = True
    workspace_glob: WorkspaceGlob | bool = True
    workspace_grep: WorkspaceGrep | bool = True
    expand_media_refs: ExpandMediaRefs | bool = True

    # The write gate
    read_only: bool = False

    # Write-side capabilities
    workspace_write: WorkspaceWrite | bool = True
    workspace_delete: WorkspaceDelete | bool = True
    workspace_edit: WorkspaceEdit | bool = True
    workspace_multi_edit: WorkspaceMultiEdit | bool = True
    workspace_patch: WorkspacePatch | bool = True
    workspace_mkdir: WorkspaceMkdir | bool = True

    # Files seeded at wiring time
    resources: list[Resource] = []

    _workspace: Filesystem | None = PrivateAttr(default=None)
```

**One card, two modes.** There is no separate read-only class. `read_only` is a gate applied in
`get_tools()`: read-side callables are always built (subject to their own field), write-side ones
are built only when `read_only is False`. A capability field left at `True` on a `read_only=True`
card is therefore not a contradiction — it simply never reaches the model.

**The backend is bound in `observer()`, not in `__init__`.** `observer()` resolves the workspace
name (`workspace_id or str(observer.team_id)`), calls `get_workspace(name)` to build a
`Filesystem` rooted at `<AKGENTIC_WORKSPACES_ROOT>/<name>`, and then seeds `resources`. Reading
`card.workspace` before that raises `RuntimeError` — a wiring error, not something an LLM can
correct. The `Filesystem` handle lives in a `PrivateAttr`, so it never appears in `model_dump()`
and the card stays catalog-serializable.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `workspace_id` | `str \| None` | `None` | Directory name under the workspaces root. `None` ⇒ the team id, so each team gets its own tree. Set it to a fixed string to **share one directory across teams**, or to pair the card with `ExecTool(workspace_id=...)` so shell commands and file tools see the same files. |
| `read_only` | `bool` | `False` | `True` removes every write-side callable from the tool list. The read side is unaffected. |
| `resources` | `list[Resource]` | `[]` | Files written into the workspace at `observer()` time, before the agent's first turn. Seeding is **idempotent**: a resource whose `file_name` already exists is skipped, so restoring a team never clobbers a file the agent has since edited. |
| `workspace_read` | `WorkspaceRead \| bool` | `True` | Read a file with line-number pagination. |
| `workspace_view` | `WorkspaceView \| bool` | `True` | Return an image as `BinaryContent` for the model's vision endpoint. |
| `workspace_list` | `WorkspaceList \| bool` | `True` | List a directory, flat or as an ASCII tree. |
| `workspace_glob` | `WorkspaceGlob \| bool` | `True` | Find files by glob pattern. |
| `workspace_grep` | `WorkspaceGrep \| bool` | `True` | Regex search across file contents. |
| `expand_media_refs` | `ExpandMediaRefs \| bool` | `True` | `COMMAND`-only: expand `!!pattern` tokens in a prompt into image content. |
| `workspace_write` | `WorkspaceWrite \| bool` | `True` | Create or overwrite a file. |
| `workspace_delete` | `WorkspaceDelete \| bool` | `True` | Delete a file. |
| `workspace_edit` | `WorkspaceEdit \| bool` | `True` | Single find-and-replace. |
| `workspace_multi_edit` | `WorkspaceMultiEdit \| bool` | `True` | Ordered sequence of find-and-replace edits. |
| `workspace_patch` | `WorkspacePatch \| bool` | `True` | Apply a unified diff. |
| `workspace_mkdir` | `WorkspaceMkdir \| bool` | `True` | Create a directory tree. |

Every capability field follows the package-wide `ParamModel | bool` convention: `True` enables it
with defaults, `False` removes it from every channel, and an instance enables it with custom
parameters. Every param model also inherits `instructions: str | None` (appended to the tool
docstring the model sees) and `expose: set[Channels]` from `BaseToolParam`.

---

## Capability parameters

### `WorkspaceRead` — `workspace_read(path, offset=1, limit=…, force_document_regeneration=…)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | Channels the capability reaches. |
| `default_limit` | `int` | `2000` | Maximum lines returned per call. Becomes the **default value of the `limit` argument in the tool signature**, so the model can still ask for fewer or more; it is a budget, not a hard cap. |
| `force_document_regeneration` | `bool` | `False` | Default for the same-named argument: re-extract a binary file even when a cached sidecar exists. |
| `document_reader` | `DocumentReader \| bool` | `True` | Binary extraction policy. `True` ⇒ a default `DocumentReader()` (MarkItDown, Pass 1, no LLM). `False` ⇒ reading a binary extension raises `ValueError` with an install hint. An instance ⇒ custom extraction, including the LLM fallback. |

The callable returns file contents with 1-indexed line numbers prefixed and a trailing notice when
truncated. A missing path or a path escaping the root surfaces as `RetriableError`, so the model
can correct itself.

**Binary reads** (`.pdf`, `.docx`, `.xlsx`, `.xls`, `.pptx`, `.msg`, `.epub`, and image
extensions) go through the `DocumentReader` and are cached in a sidecar next to the source. A
sidecar is a dotfile ending in `.md` and reading one directly returns it as plain text rather than
re-extracting it.

#### `DocumentReader`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `llm_client` | `Literal["openai"] \| None` | `"openai"` | Enables the Pass-2 vision fallback. `None` disables it: extraction is Pass 1 only. |
| `llm_model` | `str` | `"gpt-5.4-mini"` | Model used for the Pass-2 fallback. |
| `extensions` | `ClassVar[frozenset[str]]` | see above | The extension set treated as binary. A `ClassVar`, not a field — not configurable per instance. |

Two passes: MarkItDown alone first; if the result holds fewer than 50 non-whitespace characters
**and** `llm_client` is set, an `OpenAI()` client is constructed lazily and MarkItDown retried with
vision. If both passes come up short the reader returns `<!-- markitdown: no text extracted -->`.
The client is only built when Pass 1 falls short, so a successful text extraction never requires
credentials. Requires `akgentic-tool[docs]`; without it `extract_text` raises `ImportError` with
the install command.

### `WorkspaceView` — `workspace_view(path) -> BinaryContent`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `max_dimension` | `int` | `1568` | Longest-side pixel cap. Larger images are resized with LANCZOS, aspect ratio preserved, and the resized bytes cached in a sidecar named `.{stem}{ext}.{max_dimension}{ext}` beside the source. **`0` disables resizing** and returns the raw bytes. |

Supported formats are PNG, JPEG, GIF, WebP and BMP; anything else raises `RetriableError`.
Resizing needs `akgentic-tool[vision]` (Pillow). Without Pillow the tool still works — it logs a
one-time warning and sends the image unresized, which costs vision tokens rather than failing.

Use `workspace_view` to *look at* an image and `workspace_read` to *extract text from* a document.

### `WorkspaceList` — `workspace_list(path="", depth=…)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `max_depth` | `int` | `1` | Default for the `depth` argument. `1` ⇒ flat list of immediate children; `0` ⇒ unlimited recursive tree; `N > 1` ⇒ tree N levels deep. Despite the name it is a default, not a ceiling — the model may pass any depth. |

Directories render as `name/`, files as `name (N bytes)`. An empty directory returns
`"Empty directory."`. Pointing at a file rather than a directory is a `RetriableError`.

### `WorkspaceGlob` — `workspace_glob(pattern, path="")`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `max_results` | `int` | `100` | Hard cap on returned paths. Unlike the two fields above this is **not** exposed as an argument — exceeding it appends a truncation notice. |

Patterns are standard globs (`**/*.py`, `src/**/*.ts`) with brace expansion (`*.{py,ts}`), matched
**case-insensitively**. Results are ordered by modification time, newest first, which is what
makes "what changed recently" answerable in one call, and what makes the `max_results` cut keep
the most relevant files rather than an arbitrary alphabetical slice.

### `WorkspaceGrep` — `workspace_grep(pattern, path="", include="")`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `max_results` | `int` | `100` | Cap on reported matches; truncation is announced. |
| `max_line_length` | `int` | `2000` | Longer matching lines are clipped, so one minified file cannot flood the context. |

`pattern` is Python `re` syntax; `include` restricts the file set by glob. `ripgrep` is used when
`rg` is on PATH, with a pure-Python fallback otherwise — the output format is the same either way,
and the fallback walks files newest-first so the `max_results` cut keeps recent matches. An
invalid regex comes back as `RetriableError`.

### `ExpandMediaRefs` — `expand_media_refs(prompt) -> list[str | MediaContent]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{COMMAND}` | **COMMAND only.** It is never offered to the LLM: it is a pre-processing step applied to a prompt *before* the model sees it. |

Replaces `!!pattern` (or `!!"pattern with spaces"`) tokens with the matching images as
`MediaContent`. Image matches are sorted by path; a match whose extension is a readable document
but not an image becomes the hint `!!name[=> Use workspace_read tool]`; a pattern matching nothing
becomes `!!_pattern_[Error: no image found]`. A prompt with no `!!` token returns `[prompt]`
unchanged. The result may contain trailing empty strings when the prompt ends on a token — filter
with `[p for p in parts if p != ""]`.

Note this capability ignores `read_only` and the `TOOL_CALL` gate: `get_commands()` registers it
whenever the field resolves to anything other than `False`.

### Write-side params

`WorkspaceWrite`, `WorkspaceDelete`, `WorkspaceEdit`, `WorkspaceMultiEdit`, `WorkspacePatch` and
`WorkspaceMkdir` add **no fields of their own**. Each carries only the inherited `expose`
(`{TOOL_CALL}`) and `instructions`. They exist so a deployment can disable one operation, move it
to another channel, or attach policy text to its description:

```python
WorkspaceTool(
    workspace_delete=False,                       # no deletions, ever
    workspace_patch=WorkspacePatch(
        instructions="Patches must apply cleanly; do not force overlapping hunks.",
    ),
)
```

---

## The callables

| Callable | Signature | Notes |
|---|---|---|
| `workspace_read` | `(path, offset=1, limit=2000, force_document_regeneration=False)` | 1-indexed line numbers; binary formats via MarkItDown |
| `workspace_list` | `(path="", depth=1)` | flat list or ASCII tree |
| `workspace_glob` | `(pattern, path="")` | brace expansion; mtime-ordered |
| `workspace_grep` | `(pattern, path="", include="")` | `rg` when available |
| `workspace_view` | `(path) -> BinaryContent` | vision input |
| `workspace_write` | `(path, content)` | preserves the file's existing CRLF/LF ending |
| `workspace_edit` | `(path, old_string, new_string, replace_all=False)` | 7-strategy match cascade |
| `workspace_multi_edit` | `(edits: list[EditItem])` | sequential; **stops on first failure, no rollback** |
| `workspace_patch` | `(patch_text)` | GNU unified diff; add / update / delete |
| `workspace_delete` | `(path)` | |
| `workspace_mkdir` | `(path)` | creates parents, idempotent |

**Editing is fuzzy on purpose.** `EditMatcher` runs a cascade from exact match through
line-trimmed and whitespace-normalised variants to a similarity match at threshold 0.85, because a
model reproducing a snippet rarely reproduces its indentation byte for byte. When `old_string`
matches nothing the call returns `"[ERROR] old_string not found in <path>"` rather than raising —
the model reads the string and retries.

**`workspace_multi_edit` has no transaction.** Each `EditItem` sees the result of the previous one
and the sequence stops at the first failure, leaving earlier edits applied. Prefer
`workspace_patch` when atomicity matters.

---

## Configuration

### Where the files live

```
$AKGENTIC_WORKSPACES_ROOT/          # default ./workspaces
└── <workspace_id or team_id>/      # the root every path is anchored to
```

`Filesystem._validate_path` resolves each path against that root and rejects anything landing
outside it with `PermissionError`, which the tools surface as `RetriableError`. The check is
component-level (`Path.is_relative_to`), so a sibling workspace whose name shares a prefix —
`team-1` vs `team-11` — cannot be reached. Symlinks are resolved before the check, so a symlink
pointing out of the tree does not escape either.

### Seeding files

```python
from akgentic.tool.workspace import Resource, ResourceType, WorkspaceTool

WorkspaceTool(
    resources=[
        Resource(file_name="brief.md", content="# Engagement brief\n..."),
        Resource(file_name="logo.png", file_type=ResourceType.IMAGE, content="<base64>"),
    ],
)
```

`ResourceType` is the **encoding discriminator, never a MIME type and never inferred from the
extension**: `TEXT` writes `content.encode("utf-8")`, `IMAGE` writes `base64.b64decode(content)`.
Both fields are primitives, so a seeded resource round-trips through a catalog entry unchanged.

### Recipes

```python
WorkspaceTool()                                   # full read/write, per-team directory
WorkspaceTool(read_only=True)                     # analyst: reads only
WorkspaceTool(workspace_id="shared-corpus")       # one directory shared across teams
WorkspaceTool(read_only=True, workspace_glob=False)  # drop one capability

# Pair file tools with shell execution over the same directory
WorkspaceTool(workspace_id="proj-42"), ExecTool(workspace_id="proj-42")

# Documents without the LLM fallback (no OpenAI credentials needed)
WorkspaceTool(workspace_read=WorkspaceRead(document_reader=DocumentReader(llm_client=None)))

# Ship full-resolution images to the model
WorkspaceTool(workspace_view=WorkspaceView(max_dimension=0))
```

### Degradation without extras

| Missing extra | Effect |
|---|---|
| `[docs]` (markitdown) | `workspace_read` on a binary extension raises `ImportError` with the install command. Text files are unaffected. |
| `[vision]` (Pillow) | `workspace_view` logs one warning and returns unresized bytes. Nothing fails. |

### Failure modes worth knowing

- Reading `card.workspace` before `observer()` raises `RuntimeError` — a wiring bug, deliberately
  not retriable.
- `document_reader=False` turns a binary read into `ValueError`, raised **outside** the retry
  wrapper: it is a configuration error, not something the model can fix by trying another path.
- Sidecars (`.report.pdf.md`, `.diagram.png.1568.png`) live beside their sources and show up in
  `workspace_list` and `workspace_glob` results.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the shared error-handling contract.
