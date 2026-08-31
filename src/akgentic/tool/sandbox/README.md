# The sandbox backend — and `ExecTool`, deprecated

Sandboxed shell execution inside the team workspace: one `SandboxActor` per workspace tree, an
allow-listed binary set, and four interchangeable isolation backends selected by a single `mode`.

| | |
|---|---|
| Module | `akgentic.tool.sandbox` |
| Actor | a `SandboxActor` subclass, singleton named `#SandboxActor-<workspace_id or team_id>` |
| Channels used | `TOOL_CALL`, through `WorkspaceTool` |
| Optional extras | none — `bwrap` / `sandbox-exec` / `docker` are host tools, not Python packages |
| Environment | `AKGENTIC_WORKSPACES_ROOT`, `AKGENTIC_SANDBOX_IMAGE` |

---

## `ExecTool` is deprecated — use `WorkspaceTool(workspace_exec=…)`

**The card moved. The backend did not.** Sandboxed execution is now a capability of
`WorkspaceTool`, because exec and the write gate share one resource — the tree — and two cards over
one tree means two mailboxes that interleave. Everything on *this* page below the next section is
the exec **backend**, is not deprecated, and is what `workspace_exec` resolves through.

```python
# Before
ToolFactory([WorkspaceTool(workspace_id="proj-42"), ExecTool(workspace_id="proj-42")], observer=agent)

# After — one card, one tree, one gate, one history
ToolFactory([WorkspaceTool(workspace_id="proj-42", workspace_exec=True)], observer=agent)
```

| | `ExecTool` | `WorkspaceTool(workspace_exec=…)` |
|---|---|---|
| Callable | `exec_command(cmd, cwd="")` | `workspace_exec(cmd, cwd="")` + `workspace_exec_result(run_id)` |
| Mode | `ExecTool(mode=…)` | `WorkspaceExec(mode=…)` |
| Budget | not configurable | `WorkspaceExec(timeout_s=…, poll_attempts=…, poll_delay_seconds=…)` |
| Directory | `ExecTool(workspace_id=…)` | `WorkspaceTool(workspace_id=…)` |
| Journal | off by default, and **not reachable** by any `ExecTool` field | `WorkspaceTool(git_journal=…)` |

**What still works.** The class resolves, the field names are unchanged, and `exec_command` behaves
identically — same lease, same worker, same discovery, same commit. It is a shim over
`workspace_exec`, not a second implementation, so the two cannot drift.

**What warns.** A `DeprecationWarning` when the card is **wired** — at `observer()` — naming its
replacement. Deliberately not at import: an import-time warning fires for anybody who merely has
the module in a dependency's `__init__`, which is nobody's decision to change.

**What an `ExecTool`-only agent gives up.** It creates the `#Workspace-<workspace>` actor for its workspace, and
the first card to create that actor decides the configuration — so it gets the journal's default,
which is **off**. `AC2` froze the shim's three fields, so there is no `git_journal` on it and no way
to reach one. An `ExecTool` agent sharing a workspace whose actor a `WorkspaceTool(git_journal=True)`
created first does get a journal, by that same first-card-wins rule. Move to `WorkspaceTool` if you
need to decide it yourself.

### What "a deprecated card" means here

The package's [§Migration](../../../../README.md#migration-moved-import-paths) policy governs moved
**import paths** and their Stable/Internal tiers. `ExecTool` moves no module, so that policy does
not cover it. The policy for a deprecated *card* is:

- it keeps working, identically, for as long as it ships;
- it emits a `DeprecationWarning` naming its replacement;
- it leaves the package README's Tool Catalog for a migration pointer, and stops counting towards
  the number of tools the package advertises — a shim is not a distinct usable capability;
- it is removed **no earlier than the minor release after** the one that deprecated it.

---

## The `ExecTool` card, as it stands

```python
class ExecTool(ToolCard):
    exec_command: ExecCommand | bool = True
    mode: Literal["local", "bwrap", "seatbelt", "docker", "auto"] = "auto"
    workspace_id: str | None = None

    _sandbox_proxy: SandboxActor | None = PrivateAttr(default=None)
    _workspace_proxy: WorkspaceActor | None = PrivateAttr(default=None)
    _agent_id: str = PrivateAttr(default="")
```

**The backend class is resolved at `observer()` time, not at import time.** `observer()` looks
`mode` up in the module-level `SANDBOX_ACTOR_CLASSES` registry, which infrastructure packages may
extend before any card is constructed. A `mode` naming an unregistered backend raises `KeyError` —
deliberately fail-fast, at team creation.

**One actor per tree, many callers.** `getChildrenOrCreate(actor_class, config=SandboxConfig(...))`
is idempotent and keys on the actor **name**, so every agent whose card resolves to one workspace
shares one `#SandboxActor-<workspace>`. The name carries the workspace for the same reason
`#Workspace-<workspace>` does: a constant name would resolve two exec-capable cards on two
workspaces onto the *first* actor, and the second agent's commands would then run in the first
agent's tree while its own workspace actor gated an untouched one. With `mode="docker"`, note that
the **container** is still named per team (`sandbox-{team_id}`), so two workspaces in one team on
the docker backend still share one container and therefore one mount.

---

## Backend selection

### `mode`

| Mode | Platform | Isolation | Requirement |
|---|---|---|---|
| `auto` *(default)* | any | best available | probes at `observer()` time |
| `bwrap` | Linux | filesystem namespace (bubblewrap); network disabled via `--unshare-net` | `bwrap` on PATH |
| `seatbelt` | macOS | Apple Seatbelt SBPL profile — all reads allowed, writes confined to workspace + tmpdir + `/dev/null`; network allowed | `sandbox-exec` on PATH |
| `docker` | any | persistent container per team | Docker daemon on PATH |
| `local` | any | **none** — plain subprocess in the workspace directory | nothing |

**Auto-mode probe order** is `bwrap` → `seatbelt` → `docker` → `local`. The seatbelt probe is not
just a PATH check: macOS 15+ may block `sandbox_apply` even where `sandbox-exec` exists, so the
probe actually runs `sandbox-exec -p "(version 1)(allow default)" /usr/bin/true` and requires exit
code 0. When `auto` falls through to `local`, a `DeprecationWarning` is emitted — no isolation
backend was found, and the caller should know.

`local` is a development convenience, not a security boundary. The sandboxed process can read
anything the host user can read; the **command allowlist is the primary boundary** in that mode.

### `workspace_id`

| Value | Effect |
|---|---|
| `None` *(default)* | The workspace directory is named after the team id. |
| any `str` | That directory is used instead. |

Forwarded to `SandboxConfig.workspace_id`, and to the sandbox actor's own **name**, so a card
resolving to one workspace gets one backend over one directory. On `WorkspaceTool` the same value
also names the tree the file tools use and the workspace actor that gates it — one field, one tree.

The Docker **container** name still derives from the team id (`sandbox-{team_id}`): containers are
per-team execution resources. Two workspaces in one team therefore get two sandbox *actors* but one
container, whose mount is whichever tree started first — use one workspace per team on the docker
backend, or a different `mode`.

### `exec_command`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | The only field. |

`ExecCommand` adds nothing beyond the inherited `expose` and `instructions`. Use `instructions` to
attach policy the model reads with the tool description:

```python
ExecTool(exec_command=ExecCommand(
    instructions="Run the test suite with `pytest -q`. Never install packages.",
))
```

---

## The callable

```python
exec_command(cmd: str, cwd: str = "") -> str          # the shim
workspace_exec(cmd: str, cwd: str = "") -> str        # the replacement — same arguments
```

| Argument | Meaning |
|---|---|
| `cmd` | Full command string. The **first token** must be in `ALLOWED_COMMANDS`. |
| `cwd` | Subdirectory relative to the workspace root. Empty ⇒ the root. |

Both render a finished run through one formatter, so the two surfaces cannot drift:

```
exit_code: 0 (OK)
stdout:
...
stderr (note: many tools write progress to stderr even on success):
...
```

A command still running when the caller's poll budget runs out returns
`Run <id> is still in progress. …` instead, and its output is collected on the next turn — through
`workspace_exec_result(run_id)` on the capability, or by a second `exec_command` poll on the shim.

The allowed binaries are appended to the docstring at build time, so the model sees the list in
the tool description rather than discovering it by failing.

### The allowlist

`ALLOWED_COMMANDS` is a module-level `frozenset` of 32 binaries:

`python`, `python3`, `pytest`, `ruff`, `mypy`, `uv`, `pip`, `node`, `npm`, `npx`, `sh`, `bash`,
`cat`, `echo`, `ls`, `cp`, `mv`, `rm`, `mkdir`, `find`, `grep`, `sed`, `awk`, `jq`, `wc`, `xargs`,
`touch`, `make`, `git`, `kill`, `curl`, `wget`

**`git` is on it**, and that is deliberate. It was briefly removed, on the reasoning that a
`git reset --hard` from inside the sandbox would destroy the journal beside the workspace. The
reasoning does not survive the next paragraph: the check is first-token-only and `bash` is on the
list, so the removal stopped nobody while costing an agent the use of git in a directory that *is* a
git repository. The guarantee that a sandboxed run cannot reach the journal was never this list —
the repository lives at the sibling `<root>.git`, **outside the mount of every backend that
constructs one**, so it is not there to be reached.

**Only the first whitespace-delimited token is checked.** Argument-level filtering is explicitly
out of scope, and `bash` and `sh` are on the list — so `bash -c "<anything>"` walks straight past
it. The allowlist bounds *which interpreters start*, not what they subsequently do; treat it as a
usability filter that keeps an obvious mistake from running, and as a way to tell an agent what the
sandbox offers. **Nothing may rely on it for safety.**

That is why the backend matters, and why one of the four is different: on `bwrap` / `seatbelt` /
`docker` the mount is a real boundary and the journal sits outside it. On **`local` there is no
mount at all** — it runs a plain subprocess with a cwd, so the journal beside the tree is reachable
exactly as any other host path is, with or without a git binary. `local` is `auto`'s final fallback.
Use an isolating backend where that matters.

An empty `cmd`, or a first token outside the set, raises `CommandNotAllowedError` — **on the
worker's thread**, where the allowlist check now runs. It therefore reaches the agent as a reported
run *failure* rather than as an exception out of the tool call, and it still lists the allowed
commands:

```
Run 3f2ac91d failed: Command 'psql' is not in the allowed commands list.
 Allowed: ['bash', 'cat', 'cp', ...]
```

A command its budget killed is an **outcome**, not a failure — "too slow" is the ordinary case for
a shell, so it comes back collectible, with exit code 124 and a stderr saying so. Nothing
propagates as an exception from either surface: a tool call must always yield a tool response.

---

## Configuration

### Where commands run

```
$AKGENTIC_WORKSPACES_ROOT/          # default ./workspaces
├── <workspace_id or team_id>/      # created on actor start; cwd is resolved under it
└── <workspace_id or team_id>.git/  # the journal — outside every mount, deliberately
```

The directory is created by `_start_sandbox()` and recorded in `SandboxState.workspace_path`. Only
the first line is ever mounted.

### Backend specifics

| | `local` | `bwrap` | `seatbelt` | `docker` |
|---|---|---|---|---|
| Timeout when the caller passes none | 30 s | 30 s | 30 s | 30 s (`DOCKER_EXEC_TIMEOUT`) |
| Process group | new group, so a timeout kills the whole subtree | same | not applied | container-side |
| `RLIMIT_AS` | 512 MB on Linux, **skipped on Darwin** where it is not reliably enforceable | same | not applied | n/a |
| Network | host network | **disabled** (`--unshare-net`) | allowed | container network |
| Reads outside the workspace | allowed | denied — `/usr`, `/lib*`, `/tmp`, `/dev`, `/proc` bound read-only, everything else invisible | **allowed** (macOS tooling needs broad reads); writes confined | container filesystem only |

CPU-time and file-size limits are applied on all POSIX platforms for `local` and `bwrap`.

**These four are per-backend *defaults*, not the budget a run actually gets.** Every real caller
passes one: `WorkspaceExec.timeout_s` (15 s by default) travels to `subprocess.run(timeout=…)` after
being clamped to the deferred worker's own 20 s, which sits below the orchestrator's 30 s stop
backstop. A budget that stopped at the proxy would be decoration — a Python thread cannot be
cancelled, so a subprocess still running past its worker's budget holds its parent's teardown open
for the difference. The 30 s above applies only to a caller that names no budget at all, such as a
harness starting a backend directly.

**`.git` is never inside a mount.** `bwrap` binds only the workspace root; `docker` mounts only
`<root>:/workspace`; `seatbelt` confines *writes* to the workspace and the tmpdir. The journal lives
at the sibling `<root>.git`, so it is outside all three by construction — binding a parent directory
here for convenience would silently undo it.

`SeatbeltSandboxActor._start_sandbox()` emits a `DeprecationWarning`: `sandbox-exec` has been
deprecated since macOS 10.15 and may be removed. Treat seatbelt as a developer-workstation
backend.

### The Docker image

`docker` mode runs `akgentic-sandbox:latest` (the `SANDBOX_IMAGE` constant in `sandbox/docker.py`)
— Python 3.12 with pytest/ruff/mypy, uv, and Node.js 18. The image is **built automatically on
first use** from the bundled `sandbox.Dockerfile`; no manual step is required, and Docker's layer
cache makes later starts instant.

Set `AKGENTIC_SANDBOX_IMAGE=<name>` to use a pre-built or registry image. When it is set the
auto-build check is skipped entirely and that image is used directly — the recommended setup for
CI and production.

To warm the cache manually:

```bash
docker build \
  -f packages/akgentic-tool/src/akgentic/tool/sandbox/sandbox.Dockerfile \
  -t akgentic-sandbox:latest \
  packages/akgentic-tool/src/akgentic/tool/sandbox
```

### Registering another backend

`SANDBOX_ACTOR_CLASSES` is a mutable `dict[str, type[SandboxActor]]`. Infrastructure packages
inject into it at import time, before any `ExecTool` is constructed:

```python
from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES
from my_infra.e2b_actor import E2BSandboxActor

SANDBOX_ACTOR_CLASSES["e2b"] = E2BSandboxActor   # now available as ExecTool(mode="e2b")
```

A backend is a `SandboxActor` subclass implementing three methods: `_start_sandbox()`,
`_stop_sandbox()` and `_exec(cmd, cwd) -> ExecResult`. The base class owns state initialisation,
the allowlist check, and swallowing exceptions raised during teardown so a broken backend cannot
leave a Pykka actor wedged.

### Recipes

Written on `WorkspaceTool`, which is where the card now lives. Every line has an `ExecTool`
equivalent that still works and still warns.

```python
WorkspaceTool(workspace_exec=True)                              # auto: bwrap -> seatbelt -> docker -> local
WorkspaceTool(workspace_exec=WorkspaceExec(mode="docker"))      # deterministic toolchain
WorkspaceTool(workspace_exec=WorkspaceExec(mode="bwrap"))       # Linux CI runner, real isolation
WorkspaceTool(workspace_exec=WorkspaceExec(mode="local"))       # local development, no isolation
WorkspaceTool(workspace_id="proj-42", workspace_exec=True)      # named tree, shell over the same one
WorkspaceTool(workspace_exec=WorkspaceExec(timeout_s=8.0))      # tighter command budget
WorkspaceTool()                                                 # exec withheld — the default
```

### Import paths

```python
from akgentic.tool import ExecTool, WorkspaceTool          # ExecTool: deprecated, still resolves
from akgentic.tool.workspace import WorkspaceExec
from akgentic.tool.sandbox import (
    ALLOWED_COMMANDS, SANDBOX_ACTOR_NAME, CommandNotAllowedError, ExecResult,
    SandboxActor, SandboxConfig, SandboxState, sandbox_actor_name,
    LocalSandboxActor, BwrapSandboxActor, SeatbeltSandboxActor, DockerSandboxActor,
)
from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES, ExecCommand
```

`SANDBOX_ACTOR_NAME` is the `#`-prefix, **not** the actor's name; `sandbox_actor_name(workspace)`
builds the live one. Copying the constant expecting a name is the mistake this pair exists to make
hard.

---

See the [`WorkspaceTool` reference](../workspace/README.md) for the `workspace_exec` capability, the
tree lease, the discovered write set and the journal; and the
[package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery and the
tool-actor conventions.
