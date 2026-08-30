# WorkspaceTool

Team-scoped filesystem access for LLM agents: read, list, glob, grep, view images, write, edit,
patch, delete, mkdir and run shell commands — every path anchored to one workspace root that
nothing can escape.

```python
from akgentic.tool import WorkspaceTool
```

| | |
|---|---|
| Module | `akgentic.tool.workspace.tool` |
| Actor | `#Workspace-<workspace_id or team_id>` — one singleton per **tree**, not per team. Plus `#SandboxActor-<workspace>` and a short-lived `#defer-<run_id>` worker when `workspace_exec` is on |
| Channels used | `TOOL_CALL` (11 callables, 13 with `workspace_exec`), `COMMAND` (`expand_media_refs`) |
| Optional extras | `[docs]` for binary reads, `[vision]` for image resizing |
| Environment | `AKGENTIC_WORKSPACES_ROOT` (default `./workspaces`) |
| External tools | `git` (optional — the journal, see below), `rg` (optional — accelerates `workspace_grep`) |

---

## Three features, degrading independently

Everything below hangs off one distinction, and reading the card as a single feature is the usual
way to get it wrong:

| Feature | When it is on | What it costs you if it is off |
|---|---|---|
| **The write gate** | always — pure Python, no dependency, not configurable | nothing: it cannot be turned off |
| **The journal** | when `git` is on `PATH` and `git_journal=True` | history, attribution, and out-of-band *detection* — **not** the gate |
| **Sandboxed exec** | only when the card asks for it (`workspace_exec=…`, default off) | the two exec callables; nothing else changes |

The workspace does **not** need `git`. It does not need a sandbox backend. With neither, the gate
still refuses every stale write, because the gate hashes the file rather than consulting a record
of who wrote it.

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

    # The read-only gate
    read_only: bool = False

    # The journal
    git_journal: bool = False

    # Write-side capabilities
    workspace_write: WorkspaceWrite | bool = True
    workspace_delete: WorkspaceDelete | bool = True
    workspace_edit: WorkspaceEdit | bool = True
    workspace_multi_edit: WorkspaceMultiEdit | bool = True
    workspace_patch: WorkspacePatch | bool = True
    workspace_mkdir: WorkspaceMkdir | bool = True

    workspace_exec: WorkspaceExec | bool = False        # off unless asked for

    # Files seeded at wiring time
    resources: list[Resource] = []

    _workspace: Filesystem | None = PrivateAttr(default=None)
    _workspace_proxy: WorkspaceActor | None = PrivateAttr(default=None)   # mutations — ask
    _workspace_tell: WorkspaceActor | None = PrivateAttr(default=None)    # observations — tell
    _agent_id: str = PrivateAttr(default="")
```

**One card, two modes.** There is no separate read-only class. `read_only` is a gate applied in
`get_tools()`: read-side callables are always built (subject to their own field), write-side ones
are built only when `read_only is False`. A capability field left at `True` on a `read_only=True`
card is therefore not a contradiction — it simply never reaches the model. `workspace_exec` is on
the **write** side: a command mutates the tree whatever it happens to be, so
`WorkspaceTool(read_only=True, workspace_exec=True)` registers neither exec callable.

**The backend and the actor are both bound in `observer()`, not in `__init__`.** `observer()`
resolves the workspace name (`workspace_id or str(observer.team_id)`), calls `get_workspace(name)`
to build a `Filesystem` rooted at `<AKGENTIC_WORKSPACES_ROOT>/<name>`, seeds `resources`, then
binds the `#Workspace-<name>` singleton that owns the tree and — only if exec is enabled — the
sandbox backend. Reading `card.workspace` before that raises `RuntimeError`; calling a mutation
before it raises `RuntimeError` too, because there is deliberately no ungated path to fall back to.
Every runtime handle lives in a `PrivateAttr`, so none appears in `model_dump()` and the card stays
catalog-serializable.

**The actor's name carries the workspace, and that is load-bearing.** Two cards with different
`workspace_id` values in one team get **two** actors, each owning its own tree. A fixed name would
collapse them onto one actor owning one of the two trees, silently. The same rule names the sandbox
actor `#SandboxActor-<workspace>`.

**Two teams sharing one `workspace_id` get two actors over one tree**, and their writes are
therefore *not* ordered. They are still *checked*: the gate hashes the live file, so a cross-team
collision is detected and refused rather than lost. This is a stated limit, not an oversight.

---

## The write gate

**Every mutation is refused unless the file is still what the writing agent last read.** That is
the whole rule. It is a precondition, not a lock: nothing is held while the agent thinks, and the
check happens at the moment of the write.

### What an agent sees

A refusal arrives as a `RetriableError`, which is the package's "recoverable, retry with corrected
input" signal — so the whole text lands in the model's **next turn** and the model can act on it
without anyone writing recovery logic. A refused write does not land, and nothing is left half
done.

```
Refused to modify spec.md: it changed since you read it.
It was last written by agent 'reviewer'.
Read the file again, reconsider your change against what is there now, then retry.
Your content would have replaced the live file:
--- live/spec.md
+++ proposed/spec.md
@@ ... @@
```

Three ingredients, in order of value to the agent: **what to do next**, **who else wrote**, and
**what would have been destroyed**.

- The other writer is **named only when it is known** — that is, while the live bytes still hash to
  what that agent wrote. Once anything else has touched the path, the last accepted writer is no
  longer the author of what is on disk, so the refusal says the change came from outside the
  workspace tools (an upload, a sandbox run, or another team) rather than guessing a name.
- The diff is **live against proposed** — what your write would have replaced — not a diff against
  what you read. Your own read is still in your context; the live state is what you have not seen.
  It is capped at 200 lines, with a one-line notice saying how many were elided, because an
  uncapped diff of a large file would make the *refusal* the thing that breaks the turn.
- `workspace_edit` and `workspace_delete` have no proposed whole-file content, so their refusals
  carry the live file's line count and digest instead of a diff.

### The rules, as a user reads them

| Situation | `write` / `delete` | `edit` / `multi_edit` / `patch` |
|---|---|---|
| You have not read the file, and it does not exist | ✅ creates it | ⛔ read it before editing |
| You have not read the file, and it exists | ⛔ read it before overwriting | ⛔ read it before editing |
| You read it whole, and it has not changed | ✅ | ✅ full 7-strategy match cascade |
| You read only a **page** of it, and it has not changed | ⛔ a page is not a licence to replace the file | ✅ — the anchor is the precondition |
| It **changed** since you read it | ⛔ refused, with the diff | ✅ admitted, but matching drops to **exact only** |
| It was **deleted** since you read it | ⛔ refused once, then your next write is judged as a create | ⛔ refused once, same |

`workspace_mkdir` is routed through the actor but **not gated**: a directory has no content to
clobber, and directory creation is idempotent by design.

`workspace_patch` sits on the anchored column because its hunks' context is verified before they
are spliced. A patch whose context still matches applies on a changed file; one whose context
cannot be found anywhere is refused as stale. A **pure-add** patch over an existing file replaces
that file wholesale, so it answers to the whole-file column instead — otherwise a patch would be a
way around the gate.

### Four things worth knowing before you configure anything

- **No digest appears in any tool signature.** `workspace_write(path, content)` takes exactly what
  it always took. The precondition is derived server-side from what the agent was observed to read.
  Do not look for an `expected`, a `digest` or a `force` parameter: there is deliberately no bypass,
  because an escape hatch a model can reach for destroys the mechanism the first time a rejection is
  not understood.
- **Prefer `workspace_edit` to `workspace_write` for changing part of a file.** An anchored edit
  survives a teammate's concurrent change to an unrelated region of the same file, where replacing
  the whole file cannot. On a file that changed under it, the fuzzy cascade drops to exact matching
  — approximate matching against text somebody just rewrote is how a plausible edit lands in the
  wrong place — and a missed exact match is then reported as a *refusal* rather than the usual
  `[ERROR] old_string not found`, so the agent learns the file moved instead of retrying blind.
- **A paginated read does not license a whole-file write.** `workspace_read(path, offset=…)` records
  that a *page* was seen. The way through is `workspace_edit` on a still-matching anchor, not a
  bigger `limit`.
- **Observations do not survive a team resume.** They are actor instance state, not persisted. After
  a resume the first write to any path is refused until it is re-read. That is the safe direction and
  it is deliberate.

### What the gate catches that a registry would not

The hash is read from disk on **every** check and never cached. That is what makes the gate correct
against writers that never pass through the card at all: a frontend upload, a sandboxed command,
ADR-026 resource seeding, and a second team sharing the same `workspace_id`. None of the four
announces itself; all four are caught, because the check consults the file.

---

## The journal

When `git` is available and `git_journal` is on, **every accepted mutation is one commit**,
authored by the agent that made it.

```
$ git --git-dir workspaces/proj-42.git --work-tree workspaces/proj-42 log --oneline
9c1f0aa exec: 3 files          (builder)
41b0d3e out-of-band: changes from outside the tools   (out-of-band)
a77e214 edit: src/main.py      (reviewer)
1d5b8c2 write: spec.md         (planner)
```

- **Linear on `master`. No branches, no merges, no conflict resolution, no LLM in the loop.** A
  three-way merge of two agents' concurrent edits resolves *textually* while leaving the file
  *semantically* contradictory — and it fires precisely when the system is contended, by which time
  the agent who could have resolved it correctly is gone. A refusal is strictly better.
- **The repository is a sibling of the tree, never inside it**: workspace `foo` journals to
  `foo.git`. So it is not listable by `workspace_list`, not matchable by `workspace_glob`, not
  readable, and — the point — not inside any sandbox mount, where `git reset --hard` would destroy
  it.
- **A dirty tree is committed first, as `out-of-band`**, before the next agent's mutation touches
  disk. An upload or a previously timed-out run is never misattributed to whoever writes next.
- **Read paths dirty the tree.** A document read writes a `.report.pdf.md` sidecar; an image view
  writes a resized one; an atomic write stages a `.spec.md.<hex>.tmp` beside its target. A
  `.gitignore` covering all three is seeded once at init, and it is **not** optional hygiene —
  without it every agent's commit would be preceded by an `out-of-band` commit of regenerable noise.
  An existing `.gitignore` is never overwritten.

### When the journal is off

It degrades off — with **one** warning naming the workspace and the reason, and no further output —
in these situations:

| Condition | Deliberate? |
|---|---|
| `git_journal=False` on the card that created the actor | yes |
| `git` is not on `PATH` | yes — an environment fact |
| The workspace is itself named `<name>.git`, colliding with workspace `<name>`'s journal | no — an operator mistake |
| A sibling `<name>.git` exists and is **not** a repository (it is another workspace's tree) | no — refusing here costs one workspace's history; not refusing scatters git internals through another team's tree |
| `git init` or `git config core.bare false` fails, so there is no usable repository | no — an old or broken git |
| An earlier `git` invocation exceeded its 15 s budget, or could not be spawned at all, in this actor's life | no |

The first four are the ones a configuration can cause; the last two are a git that will not work on
this machine.

The warning goes to the `akgentic.tool.workspace.journal` logger at bind time, and there is nothing
else to read: no card field, no tool output, no state that says "this tree has no history". If you
care about the last three, check the logs when a workspace is created. **The gate is unaffected in
every one of the five cases** — no failure in the journal can fail a mutation, because the bytes are
already on disk by the time a commit is attempted.

Note also that the **first** card to create the actor for a workspace decides its configuration. A
second card arriving with `git_journal=False` does not turn off a journal that is already running.

---

## Sandboxed execution — `workspace_exec`

Off by default, and that is a security decision rather than a style one: `True` would give every
`WorkspaceTool()` in existence sandboxed shell execution through a dependency bump, probe the host
for docker at wiring time, and bring up a sandbox actor in teams that never asked for one.

```python
WorkspaceTool(workspace_exec=True)                          # auto backend, 15 s commands
WorkspaceTool(workspace_exec=WorkspaceExec(mode="docker"))  # deterministic toolchain
```

Enabling it registers **two** callables — `workspace_exec` and `workspace_exec_result` — from one
field. They go together deliberately: a result collector with nothing to collect is a foot-gun.

**A command's write set is unknowable before it runs, so exec is fenced rather than gated.** For
the duration of a run the tree is held under an **exclusive lease**:

- every **mutation**, from every agent, is refused immediately with
  `workspace busy — exec run <id> is in progress (agent '<name>')`. Immediately, not after a stall:
  ten seconds of silence inside a tool call is indistinguishable from a hang and gives the model
  nothing to react to, whereas a refusal naming the holder lets it read a file or answer the user;
- every **read** keeps working, throughout. The price of that is honest: a read during a run may see
  a half-written build artefact;
- a second `workspace_exec` is refused the same way.

Afterwards the write set is **discovered** — `git status --porcelain -uall` — and committed as one
commit attributed to the requesting agent, with the command in the body. That is where multi-file
atomicity comes from: a build touching nine files lands as one attributable unit. With the journal
off, the run still works; nothing is recorded.

**Long commands hand back a run id.** The agent's own thread waits about 5 s inside the call; a
command still going then returns `Run <id> is still in progress …`, and the agent calls
`workspace_exec_result('<id>')` on its next turn. An id nothing was issued under does not raise —
it comes back with that agent's recent run ids, so a mistyped one is correctable.

**Three budgets, and they are three different things:**

| Budget | Bounds | Default |
|---|---|---|
| `timeout_s` | the **subprocess** — reaches `subprocess.run(timeout=…)` in the backend | 15 s, clamped to the worker's 20 s |
| `poll_attempts` × `poll_delay_seconds` | how long the **agent's own thread** waits before being handed a run id | 12 × 0.4 s ≈ 5 s, clamped so it can never outlast the run |
| the backend's own default | a caller that passes no budget at all | 30 s |

A run outlives the second and is collected next turn; it never outlives the first.

**`git` is not in the command allowlist**, and `.git` is never inside a sandbox mount. The second is
the guarantee — only the first token of a command is checked and `bash` is on the list, so
`bash -c "…"` walks straight past the allowlist. The filesystem placement is what a sandboxed run
cannot argue with.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `workspace_id` | `str \| None` | `None` | Directory name under the workspaces root, **and** the suffix of the actor's name. `None` ⇒ the team id, so each team gets its own tree. Set it to a fixed string to share one directory across teams (checked by the gate, not ordered by an actor), or to give two agents in one team two separate trees. |
| `read_only` | `bool` | `False` | `True` removes every write-side callable from the tool list, `workspace_exec` included. The read side is unaffected. |
| `git_journal` | `bool` | `False` | Whether accepted mutations are recorded in the git journal. **Off by default**, because nothing in the system consumes the record: the gate re-hashes live and never consults it, and an agent's exec result carries only `exit_code`/`stdout`/`stderr`, so the journal is a human-facing audit trail you opt into. A plain field, not a capability param: it exposes no tool and nothing about it is expressible by a model. Turning it off loses history, attribution and out-of-band detection — it does **not** loosen the gate by one row. Read by the **first** card to create the actor for a workspace. |
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
| `workspace_exec` | `WorkspaceExec \| bool` | **`False`** | Run a sandboxed shell command. The one capability that is off by default, and the one field that registers **two** callables. |

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

### `WorkspaceExec` — `workspace_exec(cmd, cwd="")` and `workspace_exec_result(run_id)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | Taking exec off this channel withholds both callables **and** skips the wiring entirely — no host probe, no sandbox actor. |
| `mode` | `"local" \| "bwrap" \| "seatbelt" \| "docker" \| "auto"` | `"auto"` | The isolation backend. `"auto"` probes the host at wiring time (`bwrap` → `seatbelt` → `docker` → `local`) and warns when it falls through to `local`. A mode naming no registered backend raises `KeyError` at wiring time, deliberately. |
| `timeout_s` | `float` | `15.0` | Budget for the **subprocess**, handed to the backend. Clamped to the worker's own 20 s, which sits below the orchestrator's 30 s stop backstop. |
| `poll_attempts` | `int` | `12` | How many times the agent's own thread looks for a result before being handed a run id. |
| `poll_delay_seconds` | `float` | `0.4` | Seconds between those looks. `poll_attempts × poll_delay_seconds` is clamped so the wait can never outlast the run it waits for — past that point there is nothing left to wait for. |

None of these reaches an LLM-facing signature: nothing lets a model name a mode, a timeout, or a
git argument. See the [`ExecTool` reference](../sandbox/README.md) for what each backend actually
isolates, the bundled Docker image, and how to register a backend of your own.

---

## The callables

| Callable | Signature | Notes |
|---|---|---|
| `workspace_read` | `(path, offset=1, limit=2000, force_document_regeneration=False)` | 1-indexed line numbers; binary formats via MarkItDown |
| `workspace_list` | `(path="", depth=1)` | flat list or ASCII tree |
| `workspace_glob` | `(pattern, path="")` | brace expansion; mtime-ordered |
| `workspace_grep` | `(pattern, path="", include="")` | `rg` when available |
| `workspace_view` | `(path) -> BinaryContent` | vision input |
| `workspace_write` | `(path, content)` | gated; preserves the file's existing CRLF/LF ending |
| `workspace_edit` | `(path, old_string, new_string, replace_all=False)` | gated; 7-strategy match cascade, exact-only on a changed file |
| `workspace_multi_edit` | `(edits: list[EditItem])` | gated; **all-or-nothing** across every file it names |
| `workspace_patch` | `(patch_text)` | gated; GNU unified diff, add / update / delete; **all-or-nothing** |
| `workspace_delete` | `(path)` | gated |
| `workspace_mkdir` | `(path)` | routed but not gated; creates parents, idempotent |
| `workspace_exec` | `(cmd, cwd="")` | only when `workspace_exec` is on; takes the tree lease |
| `workspace_exec_result` | `(run_id)` | only when `workspace_exec` is on; collects a run started earlier |

The read side runs on the calling agent's own thread against its own `Filesystem`, exactly as it
always has. The mutations run on the `#Workspace-<workspace>` actor, which checks and writes in one
mailbox turn — returning a verdict and letting the agent write would reopen the window the gate
exists to close.

**Editing is fuzzy on purpose.** `EditMatcher` runs a cascade from exact match through
line-trimmed and whitespace-normalised variants to a similarity match at threshold 0.85, because a
model reproducing a snippet rarely reproduces its indentation byte for byte. When `old_string`
matches nothing on an **unchanged** file the call returns `"[ERROR] old_string not found in
<path>"` rather than raising — the model reads the string and retries. On a **changed** file the
same miss is a refusal instead, because the agent has to be told the file moved under it.

**`workspace_multi_edit` and `workspace_patch` are both all-or-nothing.** Every path is gated and
every substitution or hunk computed in memory before anything is published, so a refusal or a
missing anchor anywhere leaves every file in the batch untouched on disk. Later edits on one path
still see the result of earlier ones. One consequence is visible in the return value: where a
partial patch used to return the summary lines it had managed plus the failing file's `[ERROR]`, it
now returns only the failure — nothing was applied, so there is nothing to report.

---

## Configuration

### Where the files live

```
$AKGENTIC_WORKSPACES_ROOT/          # default ./workspaces
├── <workspace_id or team_id>/      # the root every path is anchored to
└── <workspace_id or team_id>.git/  # the journal — a SIBLING, never inside the root
```

`Filesystem._validate_path` resolves each path against that root and rejects anything landing
outside it with `PermissionError`, which the tools surface as `RetriableError`. The check is
component-level (`Path.is_relative_to`), so a sibling workspace whose name shares a prefix —
`team-1` vs `team-11` — cannot be reached. Symlinks are resolved before the check, so a symlink
pointing out of the tree does not escape either. The journal directory is outside the root by that
same rule, which is precisely why it is there.

Writes are **atomic**: bytes are staged in a `.spec.md.<32 hex>.tmp` file in the target's own
directory and published with `os.replace`, so a concurrent reader resolves the path to either the
complete previous file or the complete new one, never to a prefix. Same-directory staging is
load-bearing — `os.replace` is atomic only within one filesystem. Permission bits are preserved;
ownership, extended attributes and hardlinks are not, because publishing by rename replaces the
inode. That matters where the workspace is bind-mounted into a container running as another uid.
Orphaned staging files left by a hard kill are swept once, at actor start.

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

# A coding agent: file tools and a shell over ONE tree, ONE gate, ONE history.
# Exec used to be a second card sharing a workspace_id; it is a capability now,
# which is what puts the shell and the writes in the same serialization domain.
WorkspaceTool(workspace_id="proj-42", workspace_exec=True)

# Two agents, two trees, one team — two actors, each owning its own directory
WorkspaceTool(workspace_id="proj-42"), WorkspaceTool(workspace_id="scratch")

# No history: the gate still refuses every stale write
WorkspaceTool(git_journal=True)

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
| `git` off `PATH` | The journal degrades off with one warning. The gate is unaffected. |
| No isolation backend | `mode="auto"` falls through to `local` with a `DeprecationWarning`: commands run as a plain subprocess with no filesystem isolation. |

### What it costs

Measured on ten concurrent agents against a 27 MB tree (Apple M3 Max, Python 3.12):

- **The read path** pays one digest of the bytes it already loaded, plus one fire-and-forget
  message. Reads of 2 KB and 200 KB are unchanged inside the run-to-run spread. A **5 MB** read
  costs about **+6 ms** at p95, and the decomposition matters: ~5.1 ms of that is the digest and
  ~0.9 ms the message. It is `filesize / sha256 throughput`, not contention — it does not grow with
  the team.
- **The mutation path** pays one full file read plus, with the journal on, three short `git` forks.
  That is roughly **50–66 ms per mutation**, and those forks are serialized on one actor thread: at
  ten agents `workspace_write` measures **632 ms** at p50 against 7 ms with the journal off. That
  cost, against a record nothing in the system reads, is why the journal is off by default;
  `git_journal=True` is the lever if you want the history and can pay for it.

### Failure modes worth knowing

- Reading `card.workspace` before `observer()` raises `RuntimeError` — a wiring bug, deliberately
  not retriable. So does calling a mutation on a card that was never wired to an orchestrator:
  there is no ungated fallback path to take.
- `document_reader=False` turns a binary read into `ValueError`, raised **outside** the retry
  wrapper: it is a configuration error, not something the model can fix by trying another path.
- Sidecars (`.report.pdf.md`, `.diagram.png.1568.png`) live beside their sources and show up in
  `workspace_list` and `workspace_glob` results — and they dirty the tree, which is what the seeded
  `.gitignore` is for.
- **Two `PermissionError`s that mean opposite things** are told apart on the write path. *"Path
  escapes workspace root"* means the path is illegal. *"The change was not published: the operating
  system refused this process permission to replace the file … it did not escape it"* means the path
  was fine and the file is not replaceable — what a root-owned file created inside a container
  produces, since publication is by rename. Told the first when the second is true, an agent rewrites
  a correct path for ever. The five **read** callables do not yet make this distinction.
- A staged file that vanishes before it can be published — the sub-millisecond window in which two
  teams share one tree — is refused with *"retry exactly the same change"*, deliberately **not** with
  a staleness reason: nothing about the file changed, and telling the agent it did would send it
  redoing work that was already correct.
- A `workspace_list` during another agent's write can show a live `.<name>.<hex>.tmp` staging file.
  Harmless — it is unique-named and about to disappear — but it is visible.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the shared error-handling contract.
