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
| Actor | `#Workspace-<scope>/<leaf>` — the **resolved two-segment path**, slash included, so two principals' `notes` are two actors over two trees. One singleton per **tree**, not per team. Plus `#SandboxActor-<scope>/<leaf>` when `workspace_exec` is on |
| Channels used | `TOOL_CALL` (11 callables, 13 with `workspace_exec`), `COMMAND` (`expand_media_refs`) |
| Optional extras | `[docs]` for binary reads, `[vision]` for image resizing |
| Environment | `AKGENTIC_WORKSPACES_ROOT` (default `./workspaces`) |
| External tools | `git` (optional — the journal, see below), `rg` (optional — accelerates `workspace_grep`) |

---

## Four features, degrading independently

Everything below hangs off one distinction, and reading the card as a single feature is the usual
way to get it wrong:

| Feature | When it is on | What it costs you if it is off |
|---|---|---|
| **The write gate** | always — pure Python, no dependency, not configurable | nothing: it cannot be turned off |
| **The journal** | when `git` is on `PATH` and `git_journal=True` | history, attribution, and out-of-band *detection* — **not** the gate |
| **Sandboxed exec** | only when the card asks for it (`workspace_exec=…`, default off) | the two exec callables; nothing else changes |
| **Retrieval** | only when the card asks for it (any `workspace_rag_*`, all default off) | the three retrieval callables and the upload handler's indexing; nothing else changes |

The workspace does **not** need `git`. It does not need a sandbox backend. It does not need a
vector store. With none of them, the gate still refuses every stale write, because the gate hashes
the file rather than consulting a record of who wrote it.

Retrieval degrades further **within** itself: with the capability on and no `#VectorStore` reachable,
every retrieval callable answers one sentence rather than raising, and a search whose embedding call
fails falls back to its keyword leg. That is deliberate rather than defensive — this actor owns the
write gate, and a misconfigured vector store must not be a way to take the gate down with it.

---

## The ToolCard

```python
class WorkspaceTool(ToolCard):
    # Where the files live — two mutually exclusive ways to name the <leaf>
    workspace_id: str | None = None
    workspace_metadata_keys: list[str] = []

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

**The backend and the actor are both bound in `observer()`, not in `__init__`.** `observer()` makes
**one** call to `resolve_workspace_path(...)` — the single place a workspace directory is derived —
and hands the result down as an already-resolved value: a `Filesystem` rooted at
`<AKGENTIC_WORKSPACES_ROOT>/<scope>/<leaf>`, the `#Workspace-<scope>/<leaf>` singleton that owns the
tree, and — only if exec is enabled — the sandbox backend. Nothing below re-derives it, which is what
makes it impossible for a backend to open a different directory from the one the gate and the journal
are guarding. `resources` are seeded in between. Reading `card.workspace` before that raises
`RuntimeError`; calling a mutation before it raises `RuntimeError` too, because there is deliberately
no ungated path to fall back to. Every runtime handle lives in a `PrivateAttr`, so none appears in
`model_dump()` and the card stays catalog-serializable.

**A card declaring `workspace_metadata_keys` makes one bind-time `get_metadata()` ask; a bare card
makes none.** The team's metadata is fetched only when this card names keys, so the overwhelming
majority of `WorkspaceTool()` instances gain no round trip at wiring time.

**The actor's name carries the resolved path, and that is load-bearing.** Two cards with different
`workspace_id` values in one team get **two** actors, each owning its own tree — and so do two
*principals* whose cards both say `workspace_id="notes"`, because the name carries the scope as well
as the leaf. A fixed name would collapse them onto one actor owning one of the trees, silently. The
same rule names the sandbox actor `#SandboxActor-<scope>/<leaf>`.

**Two teams of the same principal sharing one `workspace_id` get two actors over one tree**, and
their writes are therefore *not* ordered. They are still *checked*: the gate hashes the live file, so
the collision is detected and refused rather than lost. This is a stated limit, not an oversight.
Two teams of **different** principals do not share a tree at all: `notes` resolves under each owner's
own scope. Sharing across principals is `workspace_metadata_keys`, and nothing else.

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
ADR-026 resource seeding, and a second team of the same principal sharing the same `workspace_id`.
None of the four announces itself; all four are caught, because the check consults the file. The
narrowing of the fourth is a fact about the *layout*, not about the gate: the gate never knew which
team wrote, and does not need to.

---

## The journal

When `git` is available and `git_journal` is on, **every accepted mutation is one commit**,
authored by the agent that made it.

```
$ git --git-dir workspaces/$SCOPE/proj-42.git --work-tree workspaces/$SCOPE/proj-42 log --oneline
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
- **One read path still dirties the tree.** An **image view** writes a resized sidecar beside its
  source, and an atomic write stages a `.spec.md.<hex>.tmp` beside its target. A **document read
  writes nothing at all** since ADR-045: its extraction is cached in `#Workspace`'s own state, which
  is why the cache is bounded and why an eviction costs one re-extraction rather than a file. A
  `.gitignore` covering both live patterns is seeded once at init — plus `.*.md`, kept for the
  vestigial extraction sidecars in trees that predate the change — and it is **not** optional
  hygiene: without it every agent's commit would be preceded by an `out-of-band` commit of
  regenerable noise. An existing `.gitignore` is never overwritten.

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
the duration of a run the tree is held **exclusively** by that run:

- every **mutation**, from every agent, is refused immediately with
  `workspace busy — exec run <id> is in progress (agent '<name>')`. Immediately, not after a stall:
  ten seconds of silence inside a tool call is indistinguishable from a hang and gives the model
  nothing to react to, whereas a refusal naming the holder lets it read a file or answer the user;
- every **read** keeps working, throughout. The price of that is honest: a read during a run may see
  a half-written build artefact;
- a second `workspace_exec` is **queued, never refused**. It is handed a run id of its own and takes
  the tree when the head releases it.

**Commands never refuse each other, and that is the whole of the queue.** A model emits several
`workspace_exec` calls in one response and pydantic-ai runs them **concurrently**, so they race for
the tree. Refusing the losers threw the work away and — worse — published the *winner's* run id in
the refusal, which the model could then collect: an answer to a question it never asked, and one
that was byte-indistinguishable from its own. So admission has a third answer:

| At `request_exec` | Answer | What exists for it |
|---|---|---|
| tree free | run id, `RUNNING` | the hold, and one request sent to the sandbox |
| tree held | run id, `QUEUED` | **nothing** — the entry is inert bookkeeping |
| queue full (`MAX_QUEUED_RUNS`) | a refusal naming **nobody** | — |

Three properties hold this together and none of them is optional:

- **The run id is issued at enqueue time, in every branch.** A caller leaves `request_exec` holding
  a handle to its *own* work whether or not the tree was free, so no message on this path can name
  another agent's run.
- **The hold and the clock are taken at dequeue.** A request handed to the sandbox at enqueue would
  run out of order in the sandbox's own mailbox, and a clock started at enqueue would measure the
  wait rather than the run. `#Workspace` sends the sandbox one command at a time, because only it
  knows when the tree has been committed — a second run started before the first run's write set is
  committed would sweep the first run's files into its own discovery.
- **Only the head is on the sandbox**, which is what bounds teardown **on the actor side**: three
  queued 15 s runs are 45 s of work but never more than one run's worth of liveness, comfortably
  inside the orchestrator's 30 s stop backstop. `WorkspaceActor.on_stop` drops every queued entry —
  they never ran and produced nothing to report. It says nothing about the **caller's** thread,
  which waits out its turn on the agent's side of the boundary; see the accepted costs below.

**Every ordinary exit reports, so releasing the tree is ordinarily not a decision at all.** The
sandbox's tell handler reports in a `finally`: a command that ran, one the budget killed, a backend
that raised, a binary the allowlist refused — all four arrive as a report, and the report hands the
tree on. Two cases have no report to wait for, and each releases the tree and **starts the queue
head**:

| No report because | Noticed by | What happens |
|---|---|---|
| the **sandbox stopped** mid-run — no send primitive can see this, the request was delivered to an actor that was alive at the time | `workspace_exec_result`'s liveness check, the one place `is_alive` is consulted | the run is recorded `FAILED` naming the sandbox, **nothing is committed as its agent**, and the next admission resolves a *new* sandbox — the orchestrator skips a child that is no longer alive |
| the **child ignored the kill** and is still running past `budget + LEASE_GRACE_S` | the next mutation or poll, against the run's own clock | the tree is handed on with one WARNING; the late report that may follow commits nothing and clears nothing, though its owner can still collect the outcome |

**A release drains the queue, and nothing overtakes it.** In both rows the queue **head** is started
— not whichever request happened to notice. Only an empty queue lets the arriving request run
immediately. Without that, the entries would sit with nothing scheduled to run them, and the next
request to arrive would find the tree free and start itself, breaking FIFO on exactly the path the
release created.

Because the actor is passive — nothing releases on a timer — both checks also run on
`workspace_exec_result`'s read path. That is the one message guaranteed to arrive, since every
queued caller is polling its own run by construction; without it a head that will never report
strands the work behind it until some unrelated request happens along. It costs one clock read and
one flag read.

The queue is FIFO and nothing else. No priorities, no fairness weighting: a team singleton's queue
that needs a scheduling policy is a design smell, not a feature. A queued entry belonging to an
agent that has since stopped still **runs** — the actor holds no liveness signal it could ask, and
guessing from an evicted display name would discard live work. Its output is simply never
collected, which costs one command and loses nothing.

**A run belongs to the agent that started it.** `workspace_exec_result` answers only the asking
agent's runs: an id from somebody else comes back as the existing recoverable `UNKNOWN`, carrying
the *asker's* own recent ids. No new state was introduced for it — "exists but is not yours" is a
distinction a model cannot act on differently. This is defence in depth rather than the fix: with
the queue in place nothing publishes a foreign id any more. Ownership is read from a map capped at
`MAX_TRACKED_RUNS = 32` **per agent**, so an agent past its 33rd run can no longer collect its
oldest; the answer is a recoverable `UNKNOWN`, and the alternative — a second map keyed by run —
would leak for the life of the team.

A collected result now **names its run and its command**:

```
Run 60e4db01 (`git rev-parse HEAD`):
exit_code: 0 (OK)
stdout: acf1942f5389dd…
```

The provenance line is added by `format_status`, not by `format_outcome`: the outcome body is one
shared rendering, and the caller that knows the run is the one that names it.

**Mutations still refuse while a run holds the tree, and that asymmetry is deliberate.** `workspace_write`,
`_edit`, `_patch`, `_delete` and `_mkdir` are *gated*, not fenced: they declare their write set, so
a refusal is a precondition failure that costs nothing to re-issue and that the agent can act on
immediately (read the file, answer the user, ask the holder). Exec is the only operation whose write
set is discovered after the fact, which is why it is the only one whose work would be *lost* by a
refusal and therefore the only one worth a slot in a queue. Queuing mutations would buy nothing and
would put a stale precondition in a deque. The busy refusal still names the holder's run id, which
is safe precisely because that id is now uncollectable by anyone but its owner.

Afterwards the write set is **discovered** — `git status --porcelain -uall` — and committed as one
commit attributed to the requesting agent, with the command in the body. That is where multi-file
atomicity comes from: a build touching nine files lands as one attributable unit. With the journal
off, the run still works; nothing is recorded.

**The call waits for the command, and a run id to collect later is the exception.** The agent's own
thread polls until the run reports, so an ordinary command returns its own output and the model is
never left holding a handle it has to redeem. That is deliberate: an agent inside a tool call cannot
do anything else — the call is
synchronous from the model's point of view, and it cannot yield and be resumed — so a short poll
does not save that latency, it converts it into LLM round-trips against an answer that cannot
change, and ends by telling the model to come back on a next turn it does not have. The tree is
held for the run's duration either way, so the *team* waits identically; only the requesting
agent's turn count differs.

**Under the default the wait covers your turn as well as your run**, so a command that had to queue
still returns its own output on the call that asked for it — a batch behaves as if it had been
issued one command at a time. That makes the poll **deadline-driven** rather than attempt-driven: a
run at queue position `p` has at most `p + 1` run budgets left to wait, so that is the deadline, and
it **re-arms on every queued look**, so a caller that is still advancing up the queue is never
abandoned mid-climb — a run ahead costs its budget *plus* the grace and the sandbox resolve, so a
deadline pinned to the position first seen gives a caller less time than its wait honestly takes.
What bounds it is a separate **ceiling fixed on entry and never re-armed**,
`(MAX_QUEUED_RUNS + 1) × run_budget + margin`: without it, a position that stopped decreasing would
re-start the clock for ever. The `+ 1` is the caller's own run, the same arithmetic as the 1-based
position — at the deepest legal slot you wait for the 16 ahead of you *and then for yourself*.

The thread parked by that wait is the **caller's own tool thread**, never the actor's, and the
mailbox goes on draining — so reads, mutations and other agents' polls are unaffected. On the
ordinary path a poll costs **one clock read and one flag read**: the sandbox is alive, and the run
is inside `budget + LEASE_GRACE_S`. It is not unconditionally O(1), and the difference is worth
stating rather than glossing — a poll *can* fork git and send the queue head to the sandbox, but
only on the two no-report paths above. That is the anomaly path, not the poll path, and the work is
what the tree needs done by whoever arrives first: nothing releases on a timer, so the queued
caller's own poll is the message guaranteed to arrive. Two costs come with it and are accepted: latency is serial (three 15 s commands mean the
third result lands ~45 s in, which is the point), and a parked thread can outlive the orchestrator's
30 s stop backstop during teardown — the same exposure one long run already has, not a new one.

A run that outlives even that deadline comes back saying it passed its budget and naming its id, and
`workspace_exec_result('<id>')` collects the output whenever it does land; a run still queued at the
deadline says so instead. That degraded path stops being the normal outcome but does not disappear —
it is what answers a head that hangs past its budget. An id nothing was issued under, or one
belonging to another agent, does not raise — it comes back with that agent's own recent run ids, so
a mistyped one is correctable.

A **positive** `poll_attempts` is unchanged: it asked for an explicitly bounded look and still gets
a run id when the count runs out.

**`poll_attempts` has three settings, each bounded by a different thing:**

| Setting | Meaning | Bounded by |
|---|---|---|
| `-1` (default) | wait out your **turn and** your run | a deadline, not a count: the effective run budget **plus** a report margin (~1 s), re-armed to `(p + 1)` budgets on every look while queued at position `p`, and clamped by a ceiling of `(MAX_QUEUED_RUNS + 1) × run_budget + margin` fixed on entry. So a command killed at its budget still arrives as a readable `exit_code: 124`, and one that had to queue still returns its own output |
| a positive count | a bounded look, then a run id | the effective run budget alone — no margin |
| `0` | no polling; the run id comes back immediately | — |

Anything below `-1` is a validation error rather than a second spelling of the sentinel.

**One budget bounds the run, and two other numbers are routinely mistaken for it:**

| Number | What it is | Default |
|---|---|---|
| `timeout_s` | **the run budget** — the only one that stops anything. It reaches `subprocess.run(timeout=…)` in the backend | 15 s, capped at `MAX_EXEC_BUDGET_S` (20 s) |
| `poll_attempts` × `poll_delay_seconds` | **a caller-side wait**, not a budget of the run: how long the agent's own thread stays inside the tool call before it hands back a run id | wait out the turn and the run, at 0.5 s granularity |
| the backend's own default | **a fallback**, reached only by a caller that passes no budget at all — which `workspace_exec` never does | 30 s |

Raising the wait cannot raise the budget: the poll buys more looking, never more running. The cap
exists because a Python thread cannot be cancelled — the sandbox's thread holds the orchestrator's
blocking `stop_children` open until the subprocess returns, so a run allowed past 20 s is a team
that cannot shut down inside its own 30 s backstop.

**`git` is not in the command allowlist**, and `.git` is never inside a sandbox mount. The second is
the guarantee — only the first token of a command is checked and `bash` is on the list, so
`bash -c "…"` walks straight past the allowlist. The filesystem placement is what a sandboxed run
cannot argue with.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `workspace_id` | `str \| None` | `None` | The `<leaf>` — a directory name **under the caller's own principal**, not under the workspaces root, and the second half of the actor's name. `None` ⇒ the team id, so each team gets its own tree. A fixed string names a **second tree of your own**: two agents in one team can hold two separate trees, and two teams of the *same* principal reach one tree (checked by the gate, not ordered by an actor). It does **not** share across principals — two users declaring `notes` get `<alice>/notes` and `<bob>/notes`. For sharing that actually shares, use `workspace_metadata_keys`. |
| `workspace_metadata_keys` | `list[str]` | `[]` | The `<leaf>` derived from the team's own metadata, under the reserved `_meta` scope: `["customer_id", "case_id"]` over `ACME` and `42` resolves to `_meta/customer_id-ACME__case_id-42`. This is the layout that is **shared across teams and across users** — which is why it sits under a reserved scope rather than under anybody's principal. Keys are a **sequence**: joined in declaration order, so the list reads as a refinement path from the coarsest scope down and `ls _meta/` groups a customer's trees together — the trade being that two cards naming the same keys in different orders address different workspaces, which is visible in the directory name rather than silent. Values are percent-encoded, which is what keeps the join unforgeable. The declared list also travels on the wire as `WorkspaceConfig.metadata_keys`, so a client attributes an agent to a workspace by plain list equality against this field, with nothing to normalise on either side. **Mutually exclusive with `workspace_id`** — declaring both is a `ValidationError` at card construction, not a precedence rule. Every failure is a hard error at bind time, never a fallback to a user path: no metadata on the team, a key that is not a field of the metadata model, a value that is `None` or empty, or a joined leaf over 255 bytes. |
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
| `workspace_exec` | `WorkspaceExec \| bool` | **`False`** | Run a sandboxed shell command. Off by default, and the one field that registers **two** callables. |
| `workspace_rag_index` | `WorkspaceRagIndex \| bool` | **`False`** | Queue workspace files for retrieval indexing. On the **read** side of `read_only`: indexing derives from the tree and writes nothing into it. |
| `workspace_rag_list` | `WorkspaceRagList \| bool` | **`False`** | Where every file stands in the index. `COMMAND` + `LLM_CONTEXT`, never `TOOL_CALL` — it is pushed into the context tail as a per-turn delta, so a tool call for it would be a round trip for what the model already has. |
| `workspace_rag_search` | `WorkspaceRagSearch \| bool` | **`False`** | Hybrid search over the indexed chunks. `TOOL_CALL` only — a search is something the model *does*, not something it is *shown*. Read side, like its two siblings. |
| `rag_collection` | `CollectionConfig` | `CollectionConfig()` | Backend, dimension and tenant of the one `workspace_chunks` collection. Named `rag_collection` rather than the house's bare `collection`: on a card whose other twenty fields are workspace operations, a bare `collection` reads as "the workspace's collection of files". |
| `max_documents` | `int \| None` | `None` | Row cap on the extraction cache. `None` is **not** zero and not "use the default" — it means *derive it* from the vector backend and whether retrieval is on. An explicit value always wins. |
| `max_document_chars` | `int \| None` | `None` | Character cap on the bodies the cache holds. Same three-way meaning. |

**The three retrieval capabilities are off by default for the reason `workspace_exec` is, one notch
weaker.** Every file capability defaults to on because the card already implies file access; these
three do not, because they reach the vector store and can spend embedding credits — a whole tree for
the indexer, one query embed per call for the search. A capability that costs money on somebody
else's account is opt-in.

**Enabling any one of them turns retrieval on for the tree.** The three are read by one predicate,
which three sites consult and must never disagree about: the backend-derived document caps, the
Weaviate configuration check, and the bind-time announcement to the actor. A card enabling only
`workspace_rag_search` therefore still creates the collection and still shrinks the cache on an
in-memory backend.

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
| `force_document_regeneration` | `bool` | `False` | Default for the same-named argument: **ignore a valid cache entry** and re-extract. The meaning is new in ADR-045 — it used to mean "ignore a file that happens to sit beside the source", which no notion of validity governed at all. A cache entry knows whether it describes the current bytes, so forcing now means overriding a *correct* answer: coherent, and rarely needed. A forced read still fills the cache with what it extracted. |
| `document_reader` | `DocumentReader \| bool` | `True` | Binary extraction policy. `True` ⇒ a default `DocumentReader()` (MarkItDown, Pass 1, no LLM). `False` ⇒ reading a binary extension raises `ValueError` with an install hint. An instance ⇒ custom extraction, including the LLM fallback. |

The callable returns file contents with 1-indexed line numbers prefixed and a trailing notice when
truncated. A missing path or a path escaping the root surfaces as `RetriableError`, so the model
can correct itself.

**Binary reads** (`.pdf`, `.docx`, `.xlsx`, `.xls`, `.pptx`, `.msg`, `.epub`, and image
extensions) go through the `DocumentReader`, and the extracted Markdown is cached **in
`#Workspace`'s state** — not in a file beside the source (ADR-045 §3). The sidecar it replaced is
gone, and so is the read-path rule that used to return a dotfile ending in `.md` as plain text.

The cache is keyed by path and hits only when the entry was produced from *these* source bytes by
*this* extractor version, so it can never serve a stale body: a changed file misses and re-extracts.
It is bounded on two dimensions — a row count and a character total — because it is re-serialised
into the team's event store on every fill. Over the character cap the least-recently-used entry
keeps its metadata and **drops its body**; over the row cap the entry goes entirely. Every byte is
regenerable from the tree, so an eviction costs one re-extraction and never a wrong answer.

An evicted body does **not** de-index its file: index membership lives in its own map and is never
inferred from this one.

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
| `timeout_s` | `float` | `15.0` | Budget for the **subprocess**, handed to the backend. Capped at `MAX_EXEC_BUDGET_S` (20 s), which sits below the orchestrator's 30 s stop backstop. |
| `poll_attempts` | `int` | `-1` | How many times the agent's own thread looks for a result. `-1` is the sentinel for "wait out my turn **and** my run" — a deadline of the effective run budget plus a report margin, re-armed from the queue position on every look and clamped by a ceiling fixed on entry, so a queued command still returns its own output; a positive count is a bounded look clamped to that budget without the margin, and is unaffected by the queue; `0` opts out of polling and takes the run id immediately. Below `-1` is a validation error. |
| `poll_delay_seconds` | `float` | `0.5` | Seconds between those looks — the granularity of the wait, not its length. The length comes from `poll_attempts` resolved against the run budget, and can never outlast the run it waits for: past that point there is nothing left to wait for. |

None of these reaches an LLM-facing signature: nothing lets a model name a mode, a timeout, or a
git argument. See the [sandbox backend reference](../sandbox/README.md) for what each backend
actually isolates, the bundled Docker image, and how to register a backend of your own.

---

### `WorkspaceRagIndex` — `workspace_rag_index(path="", force=False)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL, COMMAND}` | |
| `chunk_chars` | `int` | `1200` | **Target** chunk size in characters. Soft: packing stops at the first block that would take the chunk past it, so a chunk lands near this size from below. |
| `chunk_overlap_chars` | `int` | `150` | Overlap **budget**, honoured in whole blocks, so a chunk never starts mid-sentence. `0` disables overlap. |
| `max_chunk_chars` | `int` | `4000` | **Hard ceiling**, and the only point at which an atomic block — a table, a fenced code block, a list — is ever cut. It is what keeps a chunk inside the embedding model's input limit. |
| `min_chunk_chars` | `int` | `200` | Below this a chunk merges **forward**, and only under the same heading path. A chunk never crosses a heading boundary, which outranks this. |
| `prepend_heading_path` | `bool` | `True` | Embed `"Invoice > Payment terms > Late fees"` ahead of the chunk's slice. Read by the **embedder**, never by the splitter: the heading context is composed at embed time and never stored, which is what keeps a stored chunk a pair of offsets rather than a copy of the document. |

Sizes are in characters rather than tokens so the splitter stays uncoupled from any one model, at
roughly four characters per token. A configuration the splitter could not honour — a target outside
its own bounds, or an overlap at or above the target — is a validation error at configuration time
rather than a surprise at index time.

The callable **returns immediately** with three counts: queued, already current, and unsupported.
Everything behind it is bounded work on the actor's thread — a tree walk through the backend, one
read per candidate to hash it, and up to four worker spawns. No extraction, split or embedding
happens on the calling agent's thread or on the actor's. Watch `workspace_rag_list` for progress.

`force` re-indexes a file that is already current at its own bytes; without it, a file already
indexed at its live digest — or one whose run over those same bytes is still in flight — is a no-op.

### `WorkspaceRagList` — `workspace_rag_list()`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{COMMAND, LLM_CONTEXT}` | **Deliberately not `TOOL_CALL`.** |
| `max_pending_shown` | `int` | `20` | How many `pending` rows the render may carry. Everything that is **not** pending is always shown, because each of those rows says something different; pending rows all say the same thing, so past this count they collapse into one `…and N more pending` line. |

A file moves `pending` → `extraction` or `splitting` → `embedding` → `embedded`, and can reach
`failed` or `stale` from anywhere. `failed` keeps whatever chunks the file already had, so a
previously indexed file stays searchable at its previous content — a failure is a degradation, not a
loss. `stale` means the tree changed underneath an indexed file.

### `WorkspaceRagSearch` — `workspace_rag_search(query, top_k=…, path_prefix="")`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `top_k` | `int` | `5` | How many passages to render. Applied **after** filtering — the scope and prefix predicates go to the backend, so the budget is never spent on another workspace's chunks. |
| `alpha` | `float` | `0.7` | Weight of the vector leg; the keyword leg gets `1 - alpha`. `1.0` is pure vector, `0.0` pure keyword. Mirrors the value `weaviate-client` sends for `hybrid(alpha=…)`. |
| `score_threshold` | `float` | `0.0` | Minimum **raw** cosine for a vector hit, applied before normalisation so the number keeps its absolute meaning. |

Two legs, combined by the fusion rule the whole package shares: a scoped similarity search against
`workspace_chunks`, and a case-insensitive term match over the extraction bodies the actor already
holds. Each hit renders its path, its heading path, a score label — `(hybrid: 0.90)`,
`(semantic: 0.85)` or `(keyword match)` — and the chunk's text.

**A hit's text comes from the vector store, not from the cache**, which is what keeps a file whose
extraction body was evicted both searchable and renderable. That file loses only its lexical leg:
the keyword search skips an evicted body rather than slicing it, and its stored offsets degrade to
provenance. The consequence is worth stating plainly — `max_documents` bounds the size of the
actor's state, **not** the size of the searchable corpus.

**`path_prefix` must not contain `*` or `?`, and a prefix carrying either is refused with a
sentence.** Both are legal in a POSIX filename; both are wildcards in Weaviate's `Like` operator,
which is what a prefix filter compiles to there, while the in-memory backend uses `startswith` and
treats them literally. The v4 filter API offers no escape, so the same query would otherwise mean
two different things depending on where the collection happens to live. A prefix is a filter rather
than a filename, so a shorter wildcard-free prefix still reaches the file.

**Every failure degrades and none raises.** No vector store, an embedding call that fails or returns
nothing, a search that fails: each yields one warning and the keyword leg alone. With retrieval off
entirely the callable answers the same sentence its two siblings do.

### `NewFileMessage` — the handler an upload addresses

Not a capability and not a callable: a message that whatever accepted an upload **tells**
`#Workspace-<workspace>`, carrying `paths`, a free-form `source` (`"upload"` today) and `force`.

```python
tell.receiveMsg_NewFileMessage(NewFileMessage(paths=["reports/q3.pdf"]))
```

Seven properties, each of them load-bearing:

- **Tell, never ask.** The handler returns `None` and the sender does not wait — a 500-page PDF must
  not hold an HTTP request open. Progress is observed through `workspace_rag_list`.
- **It never raises.** It is reachable from *outside* the framework, and an exception on this
  mailbox turn would kill the actor that owns the write gate for the whole team.
- **Every path is validated** through `Filesystem`, never joined onto the backend's root. An upload
  handler taking caller-supplied paths is the most escape-prone surface the card has.
- **A path that escapes, is missing, or is of an unsupported type is skipped with a log line**, not
  an error. Missing means "not yet": the message routinely races the upload's own write.
- **It is idempotent** at the live content digest, unless `force=True`.
- **It indexes, where a gate write only marks `stale`.** The asymmetry is the decision, not an
  omission: an upload is one deliberate human act, while an agent write is a stream of them, and
  auto-indexing each would spend embedding credits on content about to change again.
- **With no `workspace_rag_*` capability enabled it records the paths as `pending` and spawns
  nothing**, so enabling retrieval later picks them up and an upload never spends embedding credits
  in a team that never opted in.

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
| `workspace_exec` | `(cmd, cwd="")` | only when `workspace_exec` is on; takes the tree for the run |
| `workspace_exec_result` | `(run_id)` | only when `workspace_exec` is on; collects a run started earlier |
| `workspace_rag_index` | `(path="", force=False)` | returns immediately with counts; extraction, splitting and embedding happen in `#index-` workers |
| `workspace_rag_search` | `(query, top_k=…, path_prefix="")` | hybrid over the indexed chunks; degrades to keyword-only, never raises |
| `workspace_rag_list` | `()` | `COMMAND` only — the full table. The per-turn delta is a `ContextState`, not a callable |

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

**A workspace is a relative path of exactly two segments** — `<scope>/<leaf>`. `<scope>` answers
*whose is this*, `<leaf>` answers *which of theirs*, and the leaf is always a discriminator unique to
one workspace: a team id, a `workspace_id`, or a joined metadata key, never a category. There are
three layouts and no fourth:

| Card | Resolved path |
|---|---|
| `WorkspaceTool()` | `<user_id>/<team_id>` |
| `WorkspaceTool(workspace_id="notes")` | `<user_id>/notes` |
| `WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"])` | `_meta/customer_id-ACME__case_id-42` |

```
$AKGENTIC_WORKSPACES_ROOT/                # default ./workspaces
├── <user_id>/
│   ├── <team_id>/                        # the root every path is anchored to
│   ├── <team_id>.git/                    # the journal — a SIBLING, never inside the root
│   ├── notes/                            # a named workspace: a second tree of your OWN
│   └── notes.git/
└── _meta/                                # reserved: the shared, metadata-keyed layout
    ├── customer_id-ACME__case_id-42/
    └── customer_id-ACME__case_id-42.git/
```

**Depth is fixed at two, and what that buys is that no workspace path is a prefix of another.**
`Filesystem._validate_path` rejects only paths resolving *outside* the root, so a workspace at
`ACME/` would read and write everything under `ACME/42/` as ordinary in-tree activity, with the
per-path write gate none the wiser. That is containment rather than a name collision, and it is
worse. At depth two with a unique leaf the property holds by construction.

**Where there is no principal there is no isolation, by construction.** A team created through the
SDK carries `user_id="cli"` and an HTTP deployment with no authentication configured carries
`anonymous`, so every such team on one host shares that one scope. Team trees stay distinct anyway —
`cli/<team_id>`, and a team id is unique — but a **named** workspace does not: `cli/notes` is one
tree for every SDK-created team on that host. On a developer machine that is the desired behaviour;
on a shared host reached without authentication it is the old flat namespace again, narrowed to one
scope. Supplying a principal is the deployment's job.

The `<scope>` never reaches a client. The frontend sends and labels the **leaf** it read from the
tool row — `notes`, a team id, or the joined metadata key — and the server recomputes the scope from
the team's own card.

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

### Migrating a pre-48 workspaces root

A root written before the two-segment layout holds every workspace at the top level. Every one of
them moves:

| Before | After |
|---|---|
| `workspaces/<team_id>/` | `workspaces/<user_id>/<team_id>/` |
| `workspaces/<team_id>.git/` | `workspaces/<user_id>/<team_id>.git/` |
| `workspaces/<name>/` | `workspaces/<user_id>/<name>/`, **manually**, once per owning principal |
| `workspaces/<name>.git/` | `workspaces/<user_id>/<name>.git/`, with its tree |

**The journal moves with its tree — both directories or neither.** The repository is the sibling
`<name>.git` in the same directory as the tree, so a move that relocates only the tree loses that
workspace's history *silently*: nothing raises, and the actor initialises an empty repository at the
new sibling on next start.

#### Team workspaces — the script

```bash
# Dry run is the default: it prints the plan and changes nothing.
python -m akgentic.tool.workspace.migrate --root ./workspaces --owners owners.json

# Then apply it.
python -m akgentic.tool.workspace.migrate --root ./workspaces --owners owners.json --apply
```

`owners.json` is a flat object mapping team id to the owning user id:

```json
{
  "3f2b1c8e-0a4d-4c9a-9f3e-2b6d7c8a1e55": "2R0bQV8j9zX8CEsBl6APi7MXgAn4_laOa8vd9ZoIHIQ",
  "b7c4e2a1-55d9-4f30-8a1b-9c0e6d4f2a77": "geoffroy.piroux@example.com"
}
```

For a deployment with one principal for the whole root — the community and CLI tiers, where every
team carries `cli` or `anonymous` — `--owner <user_id>` covers it in one flag and no file is needed:

```bash
python -m akgentic.tool.workspace.migrate --root ./workspaces --owner anonymous --apply
```

**Where the mapping comes from.** Each team's owner is the `user_id` stored on that team's team
record — one owner per team, no judgement involved. The script does **not** read those records:
`akgentic-tool` may not import `akgentic-team`, so the lookup happens outside this package and its
result is handed in. On an `akgentic-infra` server, `GET /teams` returns `team_id` and `user_id` for
each team, but only for **the calling user's own** teams — so a root with several principals is
built from the deployment's team store directly, one entry per team, rather than from one call.

**An unmapped team id is refused, never guessed.** The run reports it and exits non-zero without
moving it. Defaulting to `anonymous` would file one user's tree under the shared anonymous scope,
which is exactly the exposure the two-segment layout exists to close.

What the plan says about each directory:

| Verdict | Meaning |
|---|---|
| `MOVE` | a team id in the mapping — tree and journal move together |
| `ALREADY_MIGRATED` | the source is gone and the destination is there: a previous run did it. Not a conflict, and it does not fail the run. If the tree moved but its journal is still at the root — a hand-migration, or a run killed between the two moves — the row says so, and that journal must be moved beside its tree by hand |
| `CONFLICT` | the destination already exists. **The whole plan refuses**: nothing moves and the exit code is non-zero |
| `UNMAPPED` | a team id absent from the mapping. Never moved; the run exits non-zero |
| `MANUAL` | a named workspace, or a `.git` the mapping did not account for. Never touched — see below. The row distinguishes the two: a journal whose tree is beside it moves **with** that tree, while one standing alone cannot be placed at all |
| `SKIPPED` | already a scope directory, not a workspace |

The whole plan is validated before anything moves, so a collision is found while the root is still
untouched rather than halfway through. Running the script twice over the same root is a clean no-op
the second time.

#### Named workspaces — the manual path

The script never touches them, and that is deliberate: the mapping from a *name* to a principal
exists nowhere on disk, and a wrong guess hands one user's files to another.

For each named workspace, decide who owns it and move both directories:

```bash
mkdir -p workspaces/<user_id>
mv workspaces/<name>      workspaces/<user_id>/<name>
mv workspaces/<name>.git  workspaces/<user_id>/<name>.git   # if it exists
```

`<user_id>` is literally the value on that user's teams' records. Where a tree really was being
shared by several principals, **copy** it once per principal — and treat that as a signal: a tree
that genuinely needs sharing should move to `workspace_metadata_keys` instead of being copied,
because copies diverge from the moment they are made. Replace
`WorkspaceTool(workspace_id="acme-case-42")` with
`WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"])` on a team whose metadata carries
those fields, and put the directory at `workspaces/_meta/customer_id-ACME__case_id-42`. The sharing
is then declared rather than implied by everyone happening to type the same string.

#### The RAG index must be rebuilt, and there is no script for it

The retrieval index is scoped by the workspace path, so a moved workspace's chunks are still filed
under the old scope. Re-index the tree after the move — `workspace_rag_index(force=True)` — and the
old scope's chunks stay where they are until a scope-wide purge primitive exists.

That costs re-embedding and no information: the index is derived data, and every chunk is recoverable
from the tree. The stranded chunks are inert rather than cross-readable — no query ever issues a
bare-leaf scope any more — so they leak storage and nothing else.

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
WorkspaceTool()                                   # <user_id>/<team_id>: this team's own tree
WorkspaceTool(read_only=True)                     # analyst: reads only
WorkspaceTool(workspace_id="scratch")             # <user_id>/scratch: a second tree of YOUR OWN,
                                                  # not a tree shared with other principals
WorkspaceTool(read_only=True, workspace_glob=False)  # drop one capability

# Shared across teams AND across users, because the sharing is DECLARED: the leaf
# is derived from the team's own metadata and lands under the reserved _meta scope,
# e.g. _meta/customer_id-ACME__case_id-42 — declaration order, so `ls _meta/` groups
# a customer's trees. Mutually exclusive with workspace_id.
WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"])

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

# Retrieval over a document corpus at <user_id>/corpus — reachable by this
# principal's other teams, and by nobody else's. Needs a #VectorStore on the team;
# without one every retrieval callable answers a sentence and nothing raises.
WorkspaceTool(
    workspace_id="corpus",
    read_only=True,
    workspace_rag_index=True,
    workspace_rag_list=True,
    workspace_rag_search=True,
)

# Search only — the model queries an index another card of the SAME principal
# fills. Enabling any one of the three still turns retrieval on for the tree, so
# this creates the collection and shrinks the extraction cache exactly as the
# indexer would. Use workspace_metadata_keys instead to index a corpus that
# several principals must share.
WorkspaceTool(workspace_id="corpus", workspace_rag_search=True)

# A durable shared index. Fails at wiring time if AKGENTIC_WEAVIATE_URL is unset,
# rather than silently giving a card that asked for a cluster a local index.
WorkspaceTool(
    workspace_rag_index=True,
    rag_collection=CollectionConfig(backend="weaviate", tenant="acme"),
)
```

### Degradation without extras

| Missing extra | Effect |
|---|---|
| `[docs]` (markitdown) | `workspace_read` on a binary extension raises `ImportError` with the install command. Text files are unaffected. |
| `[vision]` (Pillow) | `workspace_view` logs one warning and returns unresized bytes. Nothing fails. |
| `git` off `PATH` | The journal degrades off with one warning. The gate is unaffected. |
| No isolation backend | `mode="auto"` falls through to `local` with a `DeprecationWarning`: commands run as a plain subprocess with no filesystem isolation. |
| No `#VectorStore` on the team | Every retrieval callable answers *"Retrieval indexing is not available for this workspace."* — one warning at bind time, nothing raised. Add `VectorStoreTool` to the team configuration. |
| `[vector_search]` (numpy) | The in-memory vector backend cannot be built. Retrieval degrades as above; nothing else on the card changes. |

### What it costs

Retrieval adds three things worth knowing before you turn it on. **Indexing spends embedding credits
per file**, which is why all three capabilities are opt-in. **The in-memory vector backend keeps
every vector inside the store's own state**, re-serialised on every notify, which is why enabling
retrieval on that backend shrinks the extraction cache from 32 documents / 2 MB to 8 / 200 KB —
2 MB of Markdown is roughly 1,900 chunks, and 1536 floats rendered as JSON is about 23 KB each.
Weaviate keeps the vectors in the cluster and keeps the large caps. And **`workspace_rag_search`
makes one embedding round trip on the mailbox turn of the actor that owns the write gate**: bounded
to a single call, fully degrading, but it is the one external call this card puts on that thread.

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
- **Resized-image sidecars** (`.diagram.png.1568.png`) live beside their sources and show up in
  `workspace_list` and `workspace_glob` results — and they dirty the tree, which is what the seeded
  `.gitignore` is for. **Extraction sidecars (`.report.pdf.md`) are no longer written by anything**
  since ADR-045; a tree that predates it still holds its own, still ignored, and the retrieval index
  skips every dot-prefixed name so neither family is ever indexed.
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
