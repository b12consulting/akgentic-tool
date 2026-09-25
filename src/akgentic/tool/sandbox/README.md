# The sandbox backend

Sandboxed execution of one binary plus arguments inside the team workspace: one Docker backend
behind one Protocol, with no mode and no fallback, an allow-listed binary set, and no actor of its
own — `#Workspace` builds the backend for its tree and runs commands on it from its own worker
thread. This is the backend `WorkspaceTool(workspace_exec=…)` runs commands on; there is no card
here.

| | |
|---|---|
| Module | `akgentic.tool.sandbox` |
| Actor | **none.** `#Workspace-<scope>/<kind>/<leaf>` owns one backend per tree and one single-worker executor; an exec-enabled team runs the workspace actor and the agent, and nothing else |
| Channels used | `TOOL_CALL`, through `WorkspaceTool` |
| Optional extras | none — `docker` is a host tool, not a Python package |
| Environment | `AKGENTIC_WORKSPACES_ROOT`, `AKGENTIC_SANDBOX_IMAGE` |

---

## Migration: `ExecTool` was removed

**The card is gone; the backend is not.** `ExecTool` used to be the card over this backend. Sandboxed
execution is now a capability of `WorkspaceTool`, because exec and the write gate share one
resource — the tree — and two cards over one tree means two mailboxes that interleave. Everything
below this section is the exec **backend**, is not deprecated, and is what `workspace_exec` resolves
through.

```python
# Before
ToolFactory([WorkspaceTool(workspace_id="proj-42"), ExecTool(workspace_id="proj-42")], observer=agent)

# After — one card, one tree, one gate, one history
ToolFactory([WorkspaceTool(workspace_id="proj-42", workspace_exec=True)], observer=agent)
```

| Was | Is |
|---|---|
| `exec_command(cmd, cwd="")` | `workspace_exec(cmd, cwd="")` + `workspace_exec_result(run_id)` |
| `ExecTool(mode=…)` | nothing — the sandbox is Docker, and there is no mode (see [below](#migration-the-sandbox-is-docker-only)) |
| `ExecTool(workspace_id=…)` | `WorkspaceTool(workspace_id=…)`, or `workspace_metadata_keys=[…]` for a metadata-keyed tree — per-principal like the other kinds; sharing across principals is `workspace_sharable=True`, where the platform permits it |
| no budget, no journal | `WorkspaceExec(timeout_s=…, poll_attempts=…, poll_delay_seconds=…)`, `WorkspaceTool(git_journal=…)` |

**Mind the other capabilities.** `WorkspaceTool`'s file capabilities default to `True`. A card that
used to carry `ExecTool` *alone* — a shell and nothing else — becomes
`WorkspaceTool(workspace_exec=True, workspace_read=False, workspace_write=False, …)` with every
other capability turned off explicitly. A bare `model_type` swap grants read, write, delete, edit,
multi-edit, patch and mkdir to an agent that previously had only a shell.

`from akgentic.tool import ExecTool` and `from akgentic.tool.sandbox import ExecTool` raise an
`ImportError` naming `WorkspaceTool(workspace_exec=…)`; every other name on those modules behaves as
before. The row is in the package README's
[§Migration](../../../../README.md#migration-moved-import-paths), and the policy this removal
follows is [§Deprecating a card](../../../../README.md#deprecating-a-card--not-the-same-as-moving-an-import-path).

## Migration: the sandbox actor was retired

`SandboxActor`, its four subclasses (`LocalSandboxActor`, `BwrapSandboxActor`,
`SeatbeltSandboxActor`, `DockerSandboxActor`), `SandboxConfig`, `SandboxState`,
`SANDBOX_ACTOR_NAME`, `sandbox_actor_name()` and the `SANDBOX_ACTOR_CLASSES` registry are
**deleted**. The blocking call that actor existed to host runs on `#Workspace`'s own single worker
thread, on a strategy object — `DockerBackend` below. Import it where a deployment used to import
the actors, and assign a class of its own to `akgentic.tool.sandbox.SANDBOX_BACKEND` where it used
to register into the actor registry.

**Persisted records break, and the break is accepted.** A team's event stream that holds a
`#SandboxActor` start event carries `SandboxConfig` as a `__model__`-tagged nested value, and a
checkpointed sandbox state carries `SandboxState` the same way. Core's deserializer imports a tagged
class before its guarded construction, so a stale tag fails the whole record rather than dropping a
field: such a record now loads with a "corrupted event" warning and its history is unreadable.
Agents, files and the UI are unaffected. This was accepted on the grounds that the only affected
records are internal test teams, and the exemption does **not** extend to the next removal — once
anything is released and adopted, a persisted model is a migration.

## Migration: the sandbox is Docker-only

**The isolation a team gets is fixed in deployment code**, not chosen by a catalog entry or by what
happens to be installed on the host. `workspace_exec` always runs on `DockerBackend`: there is no
`mode`, no host probe and no fallback.

| Removed | What replaces it |
|---|---|
| `WorkspaceExec.mode`, `ExecConfig.mode` | nothing — there is one backend |
| `CardMode`, `SandboxMode`, `resolve_mode`, `_resolve_auto_mode` | nothing — nothing is resolved |
| `SANDBOX_BACKEND_CLASSES` | `akgentic.tool.sandbox.SANDBOX_BACKEND`, one class (see [§Replacing the backend](#replacing-the-backend)) |
| `LocalBackend`, `BwrapBackend`, `SeatbeltBackend` | nothing shipped — a deployment that needs another executor installs its own class at the slot |

`from akgentic.tool.sandbox import LocalBackend` (and `BwrapBackend`, `SeatbeltBackend`) raises an
`ImportError` saying the sandbox is Docker-only and naming the `SANDBOX_BACKEND` slot.
`SANDBOX_BACKEND_CLASSES` and the other removed names get the interpreter's ordinary
`cannot import name`; this section is where a deployment that registered into the dict finds its
replacement.

**A stored `mode:` is ignored, and no migration runs.** A persisted card or catalog entry that still
writes `mode: auto` (or any other value) loads: the key is dropped on validation and the card runs on
whatever is installed at the slot. Nothing needs rewriting, but the key does nothing any more.

---

## Backend selection

**Docker, always.** Every exec-capable card runs its commands on the class installed at
`akgentic.tool.sandbox.SANDBOX_BACKEND`, which ships as `DockerBackend`: an ephemeral read-only
container per workspace tree, created on the first command and removed when `#Workspace` stops. It
needs a Docker daemon and the `docker` CLI on `PATH`.

**No mode, no probe, no fallback.** Nothing on the card chooses a backend, and nothing inspects the
host to pick one. A host without Docker is not quietly given a weaker backend.

**It fails closed, at the first command — not at wiring.** Binding an exec-capable card constructs
nothing that touches the host: the card announces its `ExecConfig` (the tree and the run budget),
and `#Workspace.configure_exec` builds an unstarted backend from the slot. On a host without Docker
the first `workspace_exec` run is reported as **failed**, with the reason
*docker CLI not found on PATH*, and nothing else runs the command. Binding is unaffected, so a team
that never runs a command still starts there.

**The slot is read at call time, through the package.** `configure_exec` looks up
`akgentic.tool.sandbox.SANDBOX_BACKEND` on the package module each time it builds a runner, so a
class a deployment assigns after every module was imported is still the one constructed.

**One backend per tree per team, many callers.** Every agent of one team whose card resolves to one
workspace shares one `#Workspace-<scope>/<kind>/<leaf>`, and that actor holds exactly one backend and
one worker for its tree. Two exec-capable cards on two workspaces in one team get two workspace
actors, two backends, and two containers, each mounting its own tree. Two **teams** on one tree get
two actors as well, since the actor is a team child: the tree orders them, because the exec hold is
an `O_EXCL` marker file in the tree's `<leaf>.index` sibling rather than state in either actor. The
name carries the resolved path so that a second workspace can never be resolved onto the first one's
actor.

**One backend per tree per team, many callers.** Every agent of one team whose card resolves to one
workspace shares one `#Workspace-<scope>/<kind>/<leaf>`, and that actor holds exactly one backend and
one worker for its tree. Two exec-capable cards on two workspaces in one team get two workspace
actors, two backends, and — on the docker backend — two containers, each mounting its own tree. Two
**teams** on one tree get two actors as well, since the actor is a team child: the tree orders them,
because the exec hold is an `O_EXCL` marker file in the tree's `<leaf>.index` sibling rather than
state in either actor. The name
carries the resolved path so that a second workspace can never be resolved onto the first one's
actor.

### The directory

A workspace is a relative path of **exactly three segments**, `<scope>/<kind>/<leaf>`. The card's
`workspace_id` supplies the `<leaf>` and, by being set, selects the `_id` kind; left at `None`, the
kind is `_team` and the leaf is the team id. Neither chooses the `<scope>`: that is the owning
principal, or the reserved `_shared` when the card also declares `workspace_sharable=True` and the
platform permits the kind (see the [workspace README](../workspace/README.md#where-the-files-live)).

| `workspace_id` | Effect, for principal `alice` |
|---|---|
| `None` *(default)* | `alice/_team/<team_id>` — the team's own tree, under its owner. |
| any `str` | `alice/_id/<that string>` — a second tree of the **same** principal, not a tree shared with other principals. |

The card resolves that path **once**, at bind time, and hands the result to `#Workspace` inside its
`ExecConfig`; the actor passes it to `backend.start(workspace_path)` on the worker thread, before
the first command. A backend that joins a path it was handed cannot open a different directory from
the one the card, the write gate and the journal are working on. One derivation, one tree.

---

## The callable

```python
workspace_exec(cmd: str, cwd: str = "") -> str
workspace_exec_result(run_id: str) -> str
```

| Argument | Meaning |
|---|---|
| `cmd` | One binary plus arguments — tokenised POSIX-style so quoting groups, **first token** in `ALLOWED_COMMANDS`. `&&`, `\|\|`, `;`, `\|`, `>`, `$VAR` and `$(…)` are **not** interpreted; use `bash -c '…'` for shell syntax. |
| `cwd` | Subdirectory relative to the workspace root. Empty ⇒ the root. |

A finished run renders through one formatter:

```
exit_code: 0 (OK)
stdout:
...
stderr (note: many tools write progress to stderr even on success):
...
```

A command still running when the caller's poll budget runs out returns
`Run <id> is still in progress. …` instead, and its output is collected on the next turn through
`workspace_exec_result(run_id)`.

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
the repository lives at the sibling `<root>.git`, **outside the container's mount**, so it is not
there to be reached.

**Only the first shlex token is checked** — `validate_command` in `sandbox/backend.py`, the same
call the backend's `exec()` makes, so check and run agree. Argument-level filtering is explicitly
out of scope, and `bash` and `sh` are on the list — so `bash -c "<anything>"` walks straight past
it. The allowlist bounds *which interpreters start*, not what they subsequently do; treat it as a
usability filter that keeps an obvious mistake from running, and as a way to tell an agent what the
sandbox offers. **Nothing may rely on it for safety.**

That is why the backend matters: the container's mount is the real boundary, and the journal sits
outside it.

An empty `cmd`, or a first token outside the set, raises `CommandNotAllowedError`. `#Workspace`'s
`ExecRunner` validates **before** the backend is provisioned — so a refused command never builds an
image or creates a container — and the backend validates again inside `exec()`, so the tokens it
runs are the tokens that were checked. Either way the refusal reaches the agent as a reported run
*failure* rather than as an exception out of the tool call, and it still lists the allowed commands:

```
Run 3f2ac91d failed: Command 'psql' is not in the allowed commands list.
 Allowed: ['bash', 'cat', 'cp', ...]
```

A command string that cannot be tokenised at all — an unbalanced quote — comes back the same way,
as a reported run failure, but carrying `CommandParseError`'s message and no allowlist: the binary
was never the problem.

A command its budget killed is an **outcome**, not a failure — "too slow" is the ordinary case for
a shell, so it comes back collectible, with exit code 124 and a stderr saying so. Nothing
propagates as an exception from the tool call: a tool call must always yield a tool response.

---

## Configuration

### Where commands run

```
$AKGENTIC_WORKSPACES_ROOT/            # default ./workspaces
└── <scope>/                          # the owning principal, or the reserved _shared
    └── <kind>/                       # _team, _id or _meta
        ├── <leaf>/                   # created by backend.start(); cwd is resolved under it
        ├── <leaf>.git/               # the journal — outside every mount, deliberately
        └── <leaf>.index/          # <meta>: exec lock, rag/, index/, locks/ — likewise
```

`<scope>/<kind>/<leaf>` is the three-segment path the card resolved: `<scope>/_team/<team_id>` for a
default card, `<scope>/_id/<workspace_id>` for a named one, `<scope>/_meta/<joined keys>` for a
metadata-keyed one. The backend receives it already resolved and derives nothing.

The directory is created by `start()`, which the worker thread calls lazily before the first
command; the resolved host path is held on the backend instance (`workspace_path`) and nowhere
else. Only the tree is ever mounted — never the `.git` sibling, and never the `.index` one.

### Backend specifics

| | `DockerBackend` |
|---|---|
| Timeout when the caller passes none | 30 s (`DOCKER_EXEC_TIMEOUT`) |
| Process group | none on the host — the direct `docker exec` client is signalled; only `stop()`'s container removal ends the process inside |
| Network | container network |
| Reads outside the workspace | container filesystem only, and the root is read-only |

**The process group is for a host-process backend a deployment installs.** On Linux `sh -c '…'`
forks its command as a grandchild rather than exec'ing it in place, so a signal to the shell alone
leaves the command running and holding the output pipes. `ProcessBackend` — which `DockerBackend`
inherits — therefore lets a backend that runs its command as a host process put the child in a new
process group (`os.setpgrp()` in the `preexec_fn`, with `process_group=True` passed to `_run`), and
both the timeout path and `kill()` then signal that group. Docker creates no host-side group and
takes the direct-child path. A process that leaves the group of its own accord (`setsid`,
daemonising) is out of reach, as it is for any sandbox.

**The 30 s is a *default*, not the budget a run actually gets.** Every real caller
passes one: `WorkspaceExec.timeout_s` (15 s by default) travels to the process after being capped
at `MAX_EXEC_BUDGET_S` (20 s), which sits below the orchestrator's 30 s stop backstop. A budget that
stopped at the caller would be decoration — a Python thread cannot be cancelled, so a subprocess
still running past its budget holds `#Workspace`'s bounded teardown drain open for the difference.
The 30 s above applies only to a caller that names no budget at all, such as a harness starting a
backend directly.

**`.git` is never inside a mount.** `DockerBackend` mounts only `<root>:/workspace`. The journal
lives at the sibling `<root>.git`, so it is outside the container by construction — binding a
parent directory here for convenience would silently undo it.

### The Docker container

`DockerBackend` creates **one ephemeral container per workspace tree**, on the first command, and
removes it (`docker rm -f`) when `#Workspace` stops, which is when its team's teardown reaches it. The container holds nothing worth keeping: its root is read-only, `/workspace` is the
only bind mount and the only writes that outlive it, and its name (`akgentic-sandbox-<12 hex>`) is
opaque, generated per `start()`, held on the backend instance and **persisted nowhere**. A reaper
keys on the container's `akgentic.workspace_path=<scope>/<kind>/<leaf>` label, never on the name.

Three further flags are what make a read-only root usable, and they are applied together because
two of the three is a wall: `--user <host uid>:<host gid>` so files written to the mount belong to
the host user; `-e HOME=/home/agent` with a tmpfs mounted there (and at `/tmp`) so `pip`, `uv` and
`npm` have a writable cache; and git's identity plus `safe.directory=*` passed as
`GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_n` / `GIT_CONFIG_VALUE_n` — git's *command* scope, which is the
one place `safe.directory` can be set without writing a file. Both tmpfs mounts are `rw,exec` with
an explicit size, because a bare `--tmpfs` is `noexec` and unbounded.

**A test that asserts a host-visible uid, gid or mode proves nothing on macOS.** Docker Desktop's
bind-mount file sharing remaps ownership, so a container running as root still writes files the host
sees as its own user. Daemon specs must assert the container's *own* view — `id -u; id -g`, the
mount options read from `/proc/mounts` — and may use host-side ownership only as a second,
platform-dependent check. The same spec is meaningful on a Linux runner and inert here.

### The Docker image

`DockerBackend` runs `akgentic-sandbox:v3` (the `SANDBOX_IMAGE` constant in `sandbox/docker.py`) —
Python 3.12 with pytest/ruff/mypy, `uv` in `/usr/local/bin` where a non-root user can reach it, and
Node.js 18. The image is **built automatically on first use** from the bundled `sandbox.Dockerfile`,
under a ten-minute budget with its output captured; a failed or timed-out build fails that run with
a message naming both remedies (pre-build the image, or set `AKGENTIC_SANDBOX_IMAGE`). It is never
pulled: the tag is published to no registry.

Set `AKGENTIC_SANDBOX_IMAGE=<name>` to use a pre-built or registry image. When it is set the
auto-build check is skipped entirely and that image is used directly — the recommended setup for
CI and production.

To warm the cache manually:

```bash
docker build \
  -f packages/akgentic-tool/src/akgentic/tool/sandbox/sandbox.Dockerfile \
  -t akgentic-sandbox:v3 \
  packages/akgentic-tool/src/akgentic/tool/sandbox
```

**The tag moves whenever the Dockerfile does, and it has to:** `_ensure_image` skips the build
whenever *any* image carries the tag, so a host holding an image built from an older file would keep
it for ever. It moved from `:latest` to `:v2` when `uv` was relocated off `/root`, and to `:v3` when
`libreoffice-java-common` was added. A host that built an older tag keeps that image on disk until an
operator removes it — this code never removes an image it did not create.

### Replacing the backend

`SANDBOX_BACKEND` is a single `type[SandboxBackend]`, shipped as `DockerBackend`. A deployment that
needs a different executor assigns its own class to the **package attribute**, in its own wiring,
before any card binds:

```python
import akgentic.tool.sandbox as sandbox
from my_infra.e2b_backend import E2BBackend

sandbox.SANDBOX_BACKEND = E2BBackend
```

**Assign the attribute on the package module, exactly as above.** The slot is a rebindable name,
not a mutable container, so the import pattern that worked for the old registry dict silently does
nothing here:

- `from akgentic.tool.sandbox import SANDBOX_BACKEND` followed by `SANDBOX_BACKEND = E2BBackend`
  rebinds the deployment's own local name and **replaces nothing**.
- Assigning `akgentic.tool.sandbox.registry.SANDBOX_BACKEND` does not change the package attribute
  either. The module the default happens to be defined in is not the documented surface.

`#Workspace.configure_exec` reads `akgentic.tool.sandbox.SANDBOX_BACKEND` **at call time**, when it
builds its runner, so an assignment made after every module was imported is still the one
constructed. The class is constructed with no arguments. No card field, catalog entry or environment
variable reaches the slot: the choice of executor belongs to deployment code. The test suite
installs its fake the same way, on the package attribute, and restores it afterwards.

A backend is a plain class satisfying the `SandboxBackend` Protocol in `sandbox/backend.py`:

| Method | Contract |
|---|---|
| `__init__()` | constructible from the slot with no arguments; everything a backend needs arrives through `start` |
| `start(workspace_path)` | provision for the already-resolved three-segment path — called once, lazily, on the worker thread before the first command; a failure is that run's reported error and the next run retries |
| `exec(cmd, cwd, timeout) -> ExecResult` | tokenise with `validate_command(cmd)` and run; **must** hand `timeout` to the process, and **must** raise `subprocess.TimeoutExpired` when it expires — that is what becomes the agent's "too slow" answer |
| `kill()` | end the run in flight, idempotent and best-effort; called from `#Workspace`'s thread at teardown |
| `stop()` | `kill()`, then release whatever `start()` provisioned; called last at teardown and when a card re-announces a different configuration |

`DockerBackend` inherits `ProcessBackend`, which owns the `Popen` dance once — holding the handle
of the run in flight, the timeout path's kill-and-drain, and the group-aware signal — so a
deployment's own backend that runs a host process should build its argv and call
`self._run(...)`, passing `process_group=True` beside any `preexec_fn` that calls `os.setpgrp()`.
`@runtime_checkable` on the Protocol checks four method **names** and nothing more; mypy over `src/`
is the real conformance check.

### Recipes

```python
WorkspaceTool(workspace_exec=True)                              # Docker; fails the first run without it
WorkspaceTool(workspace_id="proj-42", workspace_exec=True)      # named tree, shell over the same one
WorkspaceTool(workspace_exec=WorkspaceExec(timeout_s=8.0))      # tighter command budget
WorkspaceTool()                                                 # exec withheld — the default
```

### Import paths

```python
from akgentic.tool import WorkspaceTool
from akgentic.tool.workspace import WorkspaceExec
from akgentic.tool.sandbox import (
    ALLOWED_COMMANDS, SANDBOX_BACKEND,
    CommandNotAllowedError, CommandParseError, ExecResult, ExecReport,
    SandboxBackend, ProcessBackend, validate_command,
    DockerBackend,
)
```

Import `SANDBOX_BACKEND` to **read** it. To replace it, assign the package attribute — see
[§Replacing the backend](#replacing-the-backend).

---

See the [`WorkspaceTool` reference](../workspace/README.md) for the `workspace_exec` capability, the
tree lease, the discovered write set and the journal; and the
[package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery and the
tool-actor conventions.
