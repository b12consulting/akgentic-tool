# ExecTool

Sandboxed shell execution inside the team workspace. One `SandboxActor` per team, an allow-listed
binary set, and four interchangeable isolation backends selected by a single `mode` field.

```python
from akgentic.tool import ExecTool
```

| | |
|---|---|
| Module | `akgentic.tool.sandbox.tool` |
| Actor | a `SandboxActor` subclass, singleton named `#SandboxActor` |
| Channels used | `TOOL_CALL` |
| Optional extras | none — `bwrap` / `sandbox-exec` / `docker` are host tools, not Python packages |
| Environment | `AKGENTIC_WORKSPACES_ROOT`, `AKGENTIC_SANDBOX_IMAGE` |

---

## The ToolCard

```python
class ExecTool(ToolCard):
    exec_command: ExecCommand | bool = True
    mode: Literal["local", "bwrap", "seatbelt", "docker", "auto"] = "auto"
    workspace_id: str | None = None

    _sandbox_proxy: SandboxActor | None = PrivateAttr(default=None)
```

**The backend class is resolved at `observer()` time, not at import time.** `observer()` looks
`mode` up in the module-level `SANDBOX_ACTOR_CLASSES` registry, which infrastructure packages may
extend before any card is constructed. A `mode` naming an unregistered backend raises `KeyError` —
deliberately fail-fast, at team creation.

**One actor, many callers.** `getChildrenOrCreate(actor_class, config=SandboxConfig(...))` is
idempotent, so every agent carrying an `ExecTool` shares one `#SandboxActor` and one workspace
directory. With `mode="docker"` that means one container per team, reused across every call
rather than started per command.

---

## ToolCard fields

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

Forwarded to `SandboxConfig.workspace_id`, so pairing it with `WorkspaceTool(workspace_id=…)`
gives the file tools and the shell the **same directory** — the usual setup for a coding agent
that writes a file and then runs it:

```python
ToolFactory([WorkspaceTool(workspace_id="proj-42"), ExecTool(workspace_id="proj-42")], observer=agent)
```

The Docker **container** name always derives from the team id: containers are per-team, not
per-workspace.

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
exec_command(cmd: str, cwd: str = "") -> str
```

| Argument | Meaning |
|---|---|
| `cmd` | Full command string. The **first token** must be in `ALLOWED_COMMANDS`. |
| `cwd` | Subdirectory relative to the workspace root. Empty ⇒ the root. |

Returns a combined summary, never raises:

```
exit_code: 0 (OK)
stdout:
...
stderr (note: many tools write progress to stderr even on success):
...
```

The allowed binaries are appended to the docstring at build time, so the model sees the list in
the tool description rather than discovering it by failing.

### The allowlist

`ALLOWED_COMMANDS` is a module-level `frozenset`:

`python`, `python3`, `pytest`, `ruff`, `mypy`, `git`, `uv`, `pip`, `cat`, `ls`, `find`, `grep`,
`mkdir`, `cp`, `mv`, `rm`, `echo`, `touch`, `curl`, `wget`, `make`, `bash`, `sh`, `node`, `npm`,
`npx`

**Only the first whitespace-delimited token is checked.** Argument-level filtering is explicitly
out of scope, and `bash` and `sh` are on the list — so the allowlist bounds *which interpreters
start*, not what they subsequently do. That is why the filesystem backend matters: on `local` the
allowlist is the whole boundary, on `bwrap` / `seatbelt` / `docker` it is one layer of two.

An empty `cmd`, or a first token outside the set, produces a `CommandNotAllowedError` **string**
listing the allowed commands. Any backend failure produces a `SandboxError: <Type>: <message>`
string. Nothing propagates as an exception — a tool call must always yield a tool response.

```
CommandNotAllowedError: Command 'psql' is not in the allowed commands list.
 Allowed: ['bash', 'cat', 'cp', ...]

SandboxError: TimeoutExpired: Command 'python main.py' timed out after 30s
```

---

## Configuration

### Where commands run

```
$AKGENTIC_WORKSPACES_ROOT/          # default ./workspaces
└── <workspace_id or team_id>/      # created on actor start; cwd is resolved under it
```

The directory is created by `_start_sandbox()` and recorded in `SandboxState.workspace_path`.

### Backend specifics

| | `local` | `bwrap` | `seatbelt` | `docker` |
|---|---|---|---|---|
| Per-command timeout | 30 s | 30 s | 30 s | 60 s (`DOCKER_EXEC_TIMEOUT`) |
| Process group | new group, so a timeout kills the whole subtree | same | not applied | container-side |
| `RLIMIT_AS` | 512 MB on Linux, **skipped on Darwin** where it is not reliably enforceable | same | not applied | n/a |
| Network | host network | **disabled** (`--unshare-net`) | allowed | container network |
| Reads outside the workspace | allowed | denied — `/usr`, `/lib*`, `/tmp`, `/dev`, `/proc` bound read-only, everything else invisible | **allowed** (macOS tooling needs broad reads); writes confined | container filesystem only |

CPU-time and file-size limits are applied on all POSIX platforms for `local` and `bwrap`.

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

```python
ExecTool()                        # auto: bwrap -> seatbelt -> docker -> local
ExecTool(mode="docker")           # deterministic toolchain, per-team container
ExecTool(mode="bwrap")            # Linux CI runner with real isolation
ExecTool(mode="local")            # local development, no isolation
ExecTool(workspace_id="proj-42")  # share the directory with WorkspaceTool
ExecTool(exec_command=False)      # card wired, capability withheld
```

### Import paths

```python
from akgentic.tool import ExecTool
from akgentic.tool.sandbox import (
    ALLOWED_COMMANDS, SANDBOX_ACTOR_NAME, CommandNotAllowedError, ExecResult,
    SandboxActor, SandboxConfig, SandboxState,
    LocalSandboxActor, BwrapSandboxActor, SeatbeltSandboxActor, DockerSandboxActor,
)
from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES, ExecCommand
```

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery and
the tool-actor conventions.
