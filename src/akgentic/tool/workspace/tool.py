"""Workspace ToolCards — configurable read-only or full read/write/delete/edit access.

:class:`WorkspaceTool` exposes workspace operations as LLM-callable tools.
Pass ``read_only=True`` to restrict to read-side callables only (``workspace_read``,
``workspace_list``, ``workspace_glob``, ``workspace_grep``, ``workspace_view``).
The default ``read_only=False`` also includes write-side callables (``workspace_write``,
``workspace_delete``, ``workspace_edit``, ``workspace_multi_edit``, ``workspace_patch``,
``workspace_mkdir``).

**Reads and mutations take different routes.** A read runs on the calling agent's
own thread against its own
:class:`~akgentic.tool.workspace.workspace.Filesystem`, exactly as it always has,
and reports what it saw to ``#Workspace`` through a fire-and-forget ``tell``. A
mutation is an ``ask`` to ``#Workspace``, which checks the live file against that
observation and performs the write itself, in one mailbox turn (ADR-036 §1, §3).

Nothing about the gate is visible in an LLM-facing signature: the six mutation
callables take exactly what they always took, and the precondition is derived
server-side from what the actor observed. There is no digest, no ``expected``,
and no ``force``.
"""

from __future__ import annotations

import base64
import io
import logging
import re as _re
import shutil
import subprocess
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from pydantic import PrivateAttr
from pydantic_ai.messages import BinaryContent

from akgentic.core.actor_address import ActorAddress
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import COMMAND, TOOL_CALL, BaseToolParam, Channels, ToolCard, _resolve
from akgentic.tool.core.deferred import DEFAULT_WORKER_TIMEOUT_S, poll_deferred
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.errors import RetriableError
from akgentic.tool.sandbox.actor import CardMode
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_POLL_ATTEMPTS,
    DEFAULT_EXEC_POLL_DELAY_S,
    DEFAULT_EXEC_TIMEOUT_S,
    ExecConfig,
    ExecStatus,
    format_status,
    in_progress,
    poll_attempts_within,
    resolve_mode,
    sandbox_config,
)
from akgentic.tool.workspace.models import (
    PERM_ERR_MSG,
    MutationOutcome,
    MutationStatus,
    Observation,
    WorkspaceConfig,
    content_sha,
)
from akgentic.tool.workspace.readers import _MIME_MAP, DocumentReader, MediaContent
from akgentic.tool.workspace.workspace import Filesystem, get_workspace

logger = logging.getLogger(__name__)

_PERM_ERR_MSG = PERM_ERR_MSG
_UNBOUND_MSG = (
    "The workspace actor is not bound — a mutating WorkspaceTool must be wired "
    "through observer() with a live orchestrator."
)
_REF_RE = _re.compile(r'!!"([^"]+)"|!!(\S+)')
_PILLOW_FMT: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".gif": "GIF",
    ".bmp": "BMP",
}
_PILLOW_WARN_EMITTED: bool = False  # guards the one-time Pillow-absent warning

# Binds a capability's configuration field to the factory that consumes it.
_ParamT = TypeVar("_ParamT", bound=BaseToolParam)


class WorkspaceRead(BaseToolParam):
    """Read a file from the team workspace with pagination support."""

    expose: set[Channels] = {TOOL_CALL}
    default_limit: int = 2000
    force_document_regeneration: bool = False
    document_reader: DocumentReader | bool = True


class WorkspaceList(BaseToolParam):
    """List immediate children of a directory in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
    max_depth: int = 1  # 1 = flat list (default), 0 = unlimited, N = N levels deep


class WorkspaceGlob(BaseToolParam):
    """Find files matching a glob pattern in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
    max_results: int = 100


class WorkspaceGrep(BaseToolParam):
    """Search file contents by regex in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
    max_results: int = 100
    max_line_length: int = 2000


class ExpandMediaRefs(BaseToolParam):
    """Expand ``!!glob_pattern`` tokens in a prompt into binary image content.

    COMMAND channel only — never exposed as an LLM tool.
    """

    expose: set[Channels] = {COMMAND}


class WorkspaceView(BaseToolParam):
    """View an image file from the team workspace as binary content for LLM vision."""

    expose: set[Channels] = {TOOL_CALL}
    max_dimension: int = 1568
    """Longest-side pixel cap. Images exceeding this are resized (aspect-ratio preserved, LANCZOS).
    Set to 0 to disable resizing and return raw bytes."""


def _maybe_resize(data: bytes, suffix: str, max_dim: int, root: Path, path: str) -> bytes:
    """Resize *data* if longest side exceeds *max_dim*, with sidecar cache.

    When *max_dim* is 0, returns *data* unchanged and writes no sidecar.
    When Pillow is not installed, logs a one-time warning and returns *data* unchanged.

    Sidecar naming: ``.{stem}.{ext}.{max_dim}.{ext}`` colocated with the source file.
    """
    if max_dim == 0:
        return data

    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        global _PILLOW_WARN_EMITTED  # noqa: PLW0603
        if not _PILLOW_WARN_EMITTED:
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                "Pillow not installed — sending raw image without resizing. "
                'Install with: pip install "akgentic-tool[vision]"'
            )
            _PILLOW_WARN_EMITTED = True
        return data

    p = Path(path)
    sidecar_name = f".{p.stem}{p.suffix}.{max_dim}{p.suffix}"
    sidecar_path = root / p.parent / sidecar_name

    if sidecar_path.exists():
        return sidecar_path.read_bytes()

    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_dim:
        return data  # already within limit — no resize, no sidecar

    img.thumbnail((max_dim, max_dim), Image.LANCZOS)  # type: ignore[attr-defined]
    buf = io.BytesIO()
    fmt = _PILLOW_FMT.get(suffix, "JPEG")
    img.save(buf, format=fmt)
    resized = buf.getvalue()
    sidecar_path.write_bytes(resized)
    return resized


def _paginate(raw: str, offset: int, limit: int) -> tuple[str, bool]:
    """Number *raw*'s lines within the requested window.

    Returns:
        The numbered text — with a truncation notice when the window stops short
        — and whether the window covered the **whole** file. The flag is derived
        from the same clamped bounds the text is, so what a read records about
        itself can never disagree with what the agent was shown.
    """
    lines = raw.splitlines()
    total = len(lines)
    start = max(0, offset - 1)
    end = min(start + limit, total)
    numbered = "\n".join(f"{start + i + 1:<6}{line}" for i, line in enumerate(lines[start:end]))
    if end < total:
        numbered += f"\n[... truncated: {total} lines total, showing {start + 1}-{end} ...]"
    return numbered, start == 0 and end == total


def _resolve_outcome(outcome: MutationOutcome) -> str:
    """Turn the actor's verdict into what the tool callable does.

    A refusal is raised as a :class:`RetriableError` — the package's declared
    "recoverable, retry with corrected input" signal — because that is what
    carries the rejection, and its diff, into the model's next turn. A returned
    string is easy for a model to ignore, and the entire point of the gate is
    that the write must not land.

    Args:
        outcome: What ``#Workspace`` decided.

    Returns:
        The message, for the accepted and failed statuses alike.

    Raises:
        RetriableError: When the mutation was rejected.
    """
    if outcome.status is MutationStatus.REJECTED:
        raise RetriableError(outcome.message)
    return outcome.message


def _settled_status(status: ExecStatus) -> ExecStatus | None:
    """Answer the poll only once there is something final to say.

    ``poll_deferred`` stops at the first non-``None``, so a fetch that answered
    with a *running* status would end the poll on its first attempt and hand the
    agent a run id it did not need. A failure, by contrast, is final — it is
    collected as a failure with its reason, never reported as still running.
    """
    return status if status.settled else None


def _bound(proxy: WorkspaceActor | None) -> WorkspaceActor:
    """Return the mutation proxy, refusing to fall back to an ungated write.

    Raises:
        RuntimeError: When the card was never wired to an orchestrator. There is
            deliberately no ungated path to fall back to: one would be a bypass
            of the gate reachable from any harness that skipped the binding.
    """
    if proxy is None:
        raise RuntimeError(_UNBOUND_MSG)
    return proxy


def _grep_python(
    root: Path,
    pattern: str,
    include_glob: str,
    max_results: int,
    max_line_len: int,
) -> list[tuple[Path, int, str]]:
    """Search files using Python regex — no external dependencies required.

    Args:
        root: Filesystem root to search within.
        pattern: Python regex pattern.
        include_glob: Glob to restrict which files are searched (empty = all).
        max_results: Maximum number of matching lines to return.
        max_line_len: Truncate matching lines to this many characters.

    Returns:
        List of (file_path, line_number, line_text) tuples.
    """
    compiled = _re.compile(pattern)
    results: list[tuple[Path, int, str]] = []
    candidates = sorted(
        root.rglob(include_glob or "*"),
        key=lambda p: p.stat().st_mtime if p.is_file() else 0,
        reverse=True,
    )
    for fpath in candidates:
        if not fpath.is_file():
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                results.append((fpath, lineno, line[:max_line_len]))
                if len(results) >= max_results:
                    return results
    return results


def _grep_rg(
    root: Path,
    pattern: str,
    include_glob: str,
    max_results: int,
) -> list[tuple[Path, int, str]] | None:
    """Try ripgrep; return None if rg is not on PATH or exits with error.

    Args:
        root: Filesystem root to search within.
        pattern: Python regex pattern (ripgrep uses the same RE2 syntax).
        include_glob: Glob to restrict which files are searched (empty = all).
        max_results: Maximum number of matching lines to return.

    Returns:
        List of (file_path, line_number, line_text) tuples, or None if rg
        is unavailable or encounters an error.
    """
    if shutil.which("rg") is None:
        return None
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--hidden",
        "--no-messages",
        "--max-count",
        str(max_results),
    ]
    if include_glob:
        cmd += ["--glob", include_glob]
    cmd += [pattern, str(root)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode not in (0, 1):
        return None
    matches: list[tuple[Path, int, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            try:
                matches.append((Path(parts[0]), int(parts[1]), parts[2]))
            except ValueError:
                continue
    return matches


_BRACE_RE = _re.compile(r"\{([^{}]+)\}")


def _normalize_glob_pattern(pattern: str) -> str:
    """Ensure '**' only appears as a standalone path component.

    Fixes patterns like '**.py' → '**/*.py' that are rejected by Python 3.12
    pathlib.glob() with: ValueError: '**' can only be an entire path component.
    """
    parts = pattern.split("/")
    result: list[str] = []
    for part in parts:
        if "**" in part and part != "**":
            result.append("**")
            remainder = part.replace("**", "*")
            if remainder not in ("", "*"):
                result.append(remainder)
        else:
            result.append(part)
    return "/".join(result)


def _expand_braces(pattern: str) -> list[str]:
    """Expand brace groups in a glob pattern into multiple patterns.

    Handles multiple non-nested brace groups via recursion.
    Patterns without braces are returned as-is (passthrough).

    Args:
        pattern: Glob pattern, potentially containing brace groups like ``{py,js}``.

    Returns:
        List of fully expanded patterns (one entry if no braces found).
    """
    match = _BRACE_RE.search(pattern)
    if not match:
        return [pattern]
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    alternatives = match.group(1).split(",")
    expanded: list[str] = []
    for alt in alternatives:
        expanded.extend(_expand_braces(f"{prefix}{alt.strip()}{suffix}"))
    return expanded


def _build_tree(
    root: Path,
    prefix: str = "",
    current_depth: int = 0,
    max_depth: int = 0,
) -> list[str]:
    """Render directory entries as an ASCII tree recursively.

    Args:
        root: Filesystem path of the directory to render.
        prefix: Current indentation prefix string for rendering.
        current_depth: Current recursion depth (0 = top level).
        max_depth: Max depth to recurse (0 = unlimited, N = stop at N).

    Returns:
        List of rendered lines (one per entry, no trailing newline).
    """
    try:
        children = list(root.iterdir())
    except PermissionError:
        return []

    dirs = sorted([c for c in children if c.is_dir()], key=lambda c: c.name)
    files = sorted([c for c in children if c.is_file()], key=lambda c: c.name)
    entries = dirs + files

    lines: list[str] = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            # Recurse if max_depth is unlimited (0) or we haven't reached the limit
            if max_depth == 0 or current_depth + 1 < max_depth:
                extension = "    " if is_last else "│   "
                lines.extend(_build_tree(entry, prefix + extension, current_depth + 1, max_depth))
        else:
            size = entry.stat().st_size
            lines.append(f"{prefix}{connector}{entry.name} ({size} bytes)")

    return lines


class WorkspaceWrite(BaseToolParam):
    """Write content to a file in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceDelete(BaseToolParam):
    """Delete a file from the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceEdit(BaseToolParam):
    """Apply a surgical find-and-replace edit to a workspace file."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceMultiEdit(BaseToolParam):
    """Apply a sequence of find-and-replace edits to workspace files."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspacePatch(BaseToolParam):
    """Apply a unified diff patch to the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceMkdir(BaseToolParam):
    """Create a directory (and parents) in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceExec(BaseToolParam):
    """Run a sandboxed shell command against the team workspace.

    Configuration only — nothing here duplicates an argument of the callables it
    enables. The three budgets it carries are three different things and are
    easy to conflate:

    - ``timeout_s`` bounds the **subprocess**, and reaches
      ``subprocess.run(timeout=...)`` in the backend. It is clamped to the
      worker's own budget, which sits below the orchestrator's stop backstop.
    - ``poll_attempts`` × ``poll_delay_seconds`` bounds how long the **agent's
      own thread** waits inside the tool call before it is handed a run id.

    A run outlives the second and is collected on the next turn; it never
    outlives the first.
    """

    expose: set[Channels] = {TOOL_CALL}
    mode: CardMode = "auto"
    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S
    poll_attempts: int = DEFAULT_EXEC_POLL_ATTEMPTS
    poll_delay_seconds: float = DEFAULT_EXEC_POLL_DELAY_S


class ResourceType(StrEnum):
    """Encoding of a seeded resource's ``content`` field.

    Acts as the explicit encoding discriminator for a :class:`Resource`: it
    decides how ``content`` is decoded into bytes (see :meth:`Resource.to_bytes`).
    Encoding is always explicit — never inferred from the filename extension.
    """

    TEXT = "text"  # content is UTF-8 text, written verbatim
    IMAGE = "image"  # content is base64-encoded binary, decoded before write


class Resource(SerializableBaseModel):
    """A file seeded into the team workspace at team-creation time.

    Fully Pydantic-serializable: primitive fields plus a :class:`ResourceType`
    ``StrEnum`` only, so it round-trips cleanly through ``model_dump`` /
    ``model_validate``. The file extension lives in ``file_name`` (e.g.
    ``logo.png``); ``file_type`` carries the encoding discriminator, not a MIME
    type.
    """

    file_name: str
    file_type: ResourceType = ResourceType.TEXT
    content: str

    def to_bytes(self) -> bytes:
        """Decode ``content`` into the bytes to write to the workspace.

        Returns:
            ``base64.b64decode(content)`` when ``file_type`` is
            :attr:`ResourceType.IMAGE`, else ``content.encode("utf-8")``.

        Raises:
            binascii.Error: If ``file_type`` is :attr:`ResourceType.IMAGE` and
                ``content`` is not valid base64.
        """
        if self.file_type is ResourceType.IMAGE:
            return base64.b64decode(self.content)
        return self.content.encode("utf-8")


class WorkspaceTool(ToolCard):
    """Workspace access with configurable read-only or full read/write/delete/edit mode.

    Pass ``read_only=True`` to restrict to read-side tools only.  The default
    ``read_only=False`` also exposes write-side tools (write, delete, edit,
    multi_edit, patch, mkdir).

    Binary-extraction config lives on the nested :class:`WorkspaceRead` capability
    (``workspace_read=WorkspaceRead(document_reader=...)``), co-located with the read
    capability that uses it. ``WorkspaceRead.document_reader`` controls extraction:
    - ``True`` (default): uses a default ``DocumentReader()`` (Pass 1 only, no LLM).
    - ``False``: binary reads raise ``ValueError`` with install hint.
    - ``DocumentReader(...)`` instance: custom extraction config (e.g. with LLM).
    """

    # Read capability fields (formerly in WorkspaceReadTool)
    workspace_id: str | None = None
    workspace_read: WorkspaceRead | bool = True
    workspace_view: WorkspaceView | bool = True
    workspace_list: WorkspaceList | bool = True
    workspace_glob: WorkspaceGlob | bool = True
    workspace_grep: WorkspaceGrep | bool = True
    expand_media_refs: ExpandMediaRefs | bool = True

    # Read-only gate (NEW)
    read_only: bool = False

    workspace_git: bool = True
    """Whether accepted mutations are recorded in a git journal.

    A plain field, not a capability param: it exposes no tool, appears in no
    signature, and nothing about it is expressible by a model. Turning it off
    loses history, attribution and out-of-band *detection* — it does not loosen
    the gate by one row, because the gate is pure Python and independent.

    Note what ``getChildrenOrCreate`` implies: the **first** card to create the
    actor for a workspace decides its configuration, exactly as the observation
    caps already do. A second card arriving with ``workspace_git=False`` does not
    turn off a journal that is already running.
    """

    # Write capability fields
    workspace_write: WorkspaceWrite | bool = True
    workspace_delete: WorkspaceDelete | bool = True
    workspace_edit: WorkspaceEdit | bool = True
    workspace_multi_edit: WorkspaceMultiEdit | bool = True
    workspace_patch: WorkspacePatch | bool = True
    workspace_mkdir: WorkspaceMkdir | bool = True

    workspace_exec: WorkspaceExec | bool = False
    """Sandboxed shell execution — **off unless asked for**, and that is a security
    decision rather than a style one.

    Every other capability on this card defaults to on because every other one is
    a file operation the card already implies. Exec is not: defaulting it to
    ``True`` would give every ``WorkspaceTool()`` in existence sandboxed shell
    execution through a dependency bump, probe the host for docker at wiring
    time, and bring a ``#SandboxActor`` into teams that never asked for one.
    Capability escalation must be opt-in.

    It is also the one field that registers **two** callables — ``workspace_exec``
    and ``workspace_exec_result`` — breaking the card's otherwise strict
    one-field-one-callable convention. Deliberate: the result collector is
    meaningless without the runner, and separate fields would let a team enable
    the half that cannot do anything.

    Both live on the write side of ``read_only``: exec mutates the tree, whatever
    the command happens to be, so ``WorkspaceTool(read_only=True,
    workspace_exec=True)`` registers neither.
    """

    resources: list[Resource] = []
    """Files seeded into the team workspace at observer() time, before the
    agent's first turn. Each resource is written only if its path does not
    already exist — restoring a team never clobbers edited files."""

    # Private runtime state — not part of the serialised config.
    # Default None sentinel lets the workspace property detect uninitialized state
    # reliably under both normal execution and coverage instrumentation.
    _workspace: Filesystem | None = PrivateAttr(default=None)

    # Two proxies over the one ``#Workspace-<workspace>`` singleton, and the owning
    # agent's identity as a plain string.  All three are PrivateAttr: a proxy in a
    # Pydantic field breaks the card's serialisation contract, and the id is
    # captured as a string so no closure below holds an edge back to the agent
    # (ADR-030).  The proxies point at a *different* actor, so holding them
    # strongly roots nothing.
    #
    # The split is not stylistic.  Mutations must ask — the closure needs the
    # verdict.  Observations must tell — the reader needs nothing back, and an
    # ask would let a slow actor stall a read instead of refusing a write.
    _workspace_proxy: WorkspaceActor | None = PrivateAttr(default=None)
    _workspace_tell: WorkspaceActor | None = PrivateAttr(default=None)
    _agent_id: str = PrivateAttr(default="")

    def observer(  # type: ignore[override]
        self, observer: ActorToolObserver
    ) -> WorkspaceTool:
        """Attach observer, initialise the backend, and bind the workspace singleton.

        Args:
            observer: Actor tool observer; must have a non-None orchestrator.

        Returns:
            Self, enabling method chaining.

        Raises:
            ValueError: If ``observer.orchestrator`` is None.
        """
        if observer.orchestrator is None:
            raise ValueError("WorkspaceTool requires access to the orchestrator.")
        super().observer(observer)  # store the observer weakly via the base setter
        ws_name = self.workspace_id or str(observer.team_id)
        self._workspace = get_workspace(ws_name)
        self._seed_resources()
        self._bind_workspace_actor(observer, observer.orchestrator, ws_name)
        self._bind_sandbox(observer, observer.orchestrator)
        return self

    def _enabled_exec(self) -> WorkspaceExec | None:
        """Return the exec configuration only when it will register callables.

        One predicate, because the two halves of this capability have to agree on
        what "on" means. They did not: the wiring looked at the field and
        ``read_only``, while ``_exec_tools`` also required the ``TOOL_CALL``
        channel — so a card that put exec off the tool channel still resolved the
        backend, still emitted the ``auto`` fallback warning, and still brought up
        a ``#SandboxActor`` (a running container, on the docker backend) to serve
        two callables it then never registered.

        Returns:
            The parameters, or ``None`` when nothing exec-related should happen.
        """
        params = _resolve(self.workspace_exec, WorkspaceExec)
        if params is None or self.read_only or TOOL_CALL not in params.expose:
            return None
        return params

    def _bind_sandbox(self, observer: ActorToolObserver, orchestrator: ActorAddress) -> None:
        """Bring up the team's ``#SandboxActor`` and tell ``#Workspace`` about it.

        **Nothing happens here when the capability is off** — no host probe, no
        actor, no message. That is the whole of what ``workspace_exec=False``
        buys, and it is why the check is at the top rather than inside.

        The order matters: this runs *after* ``_bind_workspace_actor``, because
        ``configure_exec`` travels over the tell proxy that method binds, and
        after ``register_agent``, so the actor can already name this agent in a
        refusal the first run causes.

        Args:
            observer: The owning agent, live at bind time.
            orchestrator: Address of the orchestrator.

        Raises:
            KeyError: If the configured mode names no registered backend —
                fail-fast at wiring time rather than at the first command.
        """
        params = self._enabled_exec()
        if params is None:
            return
        mode, actor_class = resolve_mode(params.mode)
        config = ExecConfig(
            mode=mode,
            team_id=str(observer.team_id),
            workspace_id=self.workspace_id,
            timeout_s=params.timeout_s,
        )
        orchestrator_proxy = observer.proxy_ask(orchestrator, Orchestrator)
        orchestrator_proxy.getChildrenOrCreate(actor_class, config=sandbox_config(config))
        self._announce_exec(config)

    def _announce_exec(self, config: ExecConfig) -> None:
        """Tell the actor which backend to run commands on — fire and forget.

        Guarded exactly as :meth:`_register_agent_name` is, and for the same
        reason: a stand-in proxy that does not carry the method, or an actor that
        died between the get-or-create and this line, must not take the whole card
        down. Unguarded, this one line was the harsher of two adjacent messages on
        one binding path — the registration a line earlier already degrades.

        The degradation is an exec request refused for want of a backend: visible,
        and recoverable by rebinding. A raise at wiring time is neither.
        """
        tell = self._workspace_tell
        if tell is None:
            return
        try:
            tell.configure_exec(config)
        except Exception:
            logger.debug("Could not announce the exec backend to #Workspace", exc_info=True)

    def _bind_workspace_actor(
        self, observer: ActorToolObserver, orchestrator: ActorAddress, workspace_name: str
    ) -> None:
        """Bind the ``#Workspace-<workspace_name>`` singleton that owns this tree.

        Get-or-create in one message (ADR-025): a check-then-create pair is a
        TOCTOU window that produces two singletons over one tree, which is the
        exact failure the pattern exists to prevent.

        The actor's name carries the workspace, so two cards with different
        ``workspace_id`` values in one team get two actors, each owning its own
        tree — the unicity domain of the actor equals the resource it owns.

        Two proxies are bound over the one address: an ask proxy for mutations,
        which need the verdict, and a tell proxy for observations, which need
        nothing back.

        The agent's **name** is registered here, once, over the tell proxy. What
        the card can capture without an edge back to the agent is
        ``agent_id`` — a UUID — and a journal authored by UUID, or a refusal
        naming one, is a record nobody can read. This is the only new message the
        journal adds, and it is O(1), once per card, never on the mutation path.

        Args:
            observer: The owning agent, live at bind time.
            orchestrator: Address of the orchestrator.
            workspace_name: The resolved workspace this card is anchored to.
        """
        orchestrator_proxy = observer.proxy_ask(orchestrator, Orchestrator)
        workspace_addr = orchestrator_proxy.getChildrenOrCreate(
            WorkspaceActor,
            config=WorkspaceConfig(
                name=workspace_actor_name(workspace_name),
                role=WORKSPACE_ACTOR_ROLE,
                workspace_name=workspace_name,
                workspace_git=self.workspace_git,
            ),
        )
        self._workspace_proxy = observer.proxy_ask(workspace_addr, WorkspaceActor)
        self._workspace_tell = observer.proxy_tell(workspace_addr, WorkspaceActor)
        self._agent_id = str(observer.myAddress.agent_id)
        self._register_agent_name(observer)

    def _register_agent_name(self, observer: ActorToolObserver) -> None:
        """Tell the actor this agent's display name — fire and forget.

        Never raises: a harness that hands back a stand-in proxy without the
        method, or an actor that is already gone, must not stop a card binding.
        The consequence of a lost registration is that the journal and the
        refusals fall back to the agent id, which is degraded and not broken.
        """
        proxy = self._workspace_tell
        if proxy is None:
            return
        try:
            proxy.register_agent(self._agent_id, str(observer.myAddress.name))
        except Exception:
            logger.debug("Could not register the agent's name with #Workspace", exc_info=True)

    def _observation_recorder(self) -> Callable[[str, bytes, bool], None]:
        """Build the closure a read closure uses to report what it saw.

        The **tell** proxy and the agent id are captured **here**, at
        ``get_tools`` time, as a proxy to a different actor and a plain string.
        Neither is an edge back to the owning agent, which is what keeps the read
        closures free of the retention ADR-030 forbids.

        The tell is what makes "a read never waits on the actor" a property
        rather than a hope. From this story the actor hashes files on its ask
        path, so a read that asked would queue behind another agent's mutation
        hashing a large file; the ``except`` below would not save it, because a
        fail-open guard covers a raising actor and a dead one, never a hung one.

        Returns:
            A callable taking the path, the file's raw bytes and whether the read
            covered the whole file. It never raises: a lost observation is a lost
            precondition, which the gate turns into a *refused* write — it must
            never turn into a failed read.
        """
        proxy = self._workspace_tell
        agent_id = self._agent_id

        def record(path: str, data: bytes, full: bool) -> None:
            if proxy is None:
                return  # harness shapes that wire a bare observer never bind one
            try:
                proxy.record_observation(
                    agent_id, path, Observation(sha=content_sha(data), full=full)
                )
            except Exception:
                # Deliberately blind: a lost precondition, never a lost read. The
                # gate reads a missing observation as "you have not read this" and
                # refuses the overwrite, so every failure here degrades towards
                # refusing a write rather than accepting a stale one.
                logger.debug("Could not record an observation for %s", path, exc_info=True)

        return record

    def _seed_resources(self) -> None:
        """Write each configured resource that is not already present.

        Idempotent: an existing file is never overwritten, so a team restore
        cannot clobber edits made to a seeded file since team creation.
        """
        assert self._workspace is not None
        for resource in self.resources:
            if self._workspace.exists(resource.file_name):
                continue
            self._workspace.write(resource.file_name, resource.to_bytes())

    @property
    def workspace(self) -> Filesystem:
        """Return the workspace backend (set after :meth:`observer` is called).

        Raises:
            RuntimeError: If :meth:`observer` has not been called yet.
        """
        if not isinstance(self._workspace, Filesystem):
            raise RuntimeError("WorkspaceTool.workspace accessed before observer() was called.")
        return self._workspace

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return enabled workspace tool callables.

        Read tools are always included (when their capability field is enabled).
        Write tools are only included when ``read_only=False`` (the default).

        Returns:
            List of callables for all enabled capabilities.
        """
        tools = self._read_tools()
        if not self.read_only:
            tools += self._write_tools()
        return tools

    def _read_tools(self) -> list[Callable[..., Any]]:
        """Return enabled read-side callables — included regardless of ``read_only``."""
        candidates = [
            self._tool_if_enabled(self.workspace_read, WorkspaceRead, self._read_factory),
            self._tool_if_enabled(self.workspace_list, WorkspaceList, self._list_factory),
            self._tool_if_enabled(self.workspace_glob, WorkspaceGlob, self._glob_factory),
            self._tool_if_enabled(self.workspace_grep, WorkspaceGrep, self._grep_factory),
            self._tool_if_enabled(self.workspace_view, WorkspaceView, self._view_factory),
        ]
        return [tool for tool in candidates if tool is not None]

    def _write_tools(self) -> list[Callable[..., Any]]:
        """Return enabled write-side callables — omitted entirely when ``read_only``.

        Exec lands here rather than beside the reads because a command mutates
        the tree whatever it happens to be, so it belongs on the write side of
        the ``read_only`` gate.
        """
        candidates = [
            self._tool_if_enabled(self.workspace_write, WorkspaceWrite, self._write_factory),
            self._tool_if_enabled(self.workspace_delete, WorkspaceDelete, self._delete_factory),
            self._tool_if_enabled(self.workspace_edit, WorkspaceEdit, self._edit_factory),
            self._tool_if_enabled(
                self.workspace_multi_edit, WorkspaceMultiEdit, self._multi_edit_factory
            ),
            self._tool_if_enabled(self.workspace_patch, WorkspacePatch, self._patch_factory),
            self._tool_if_enabled(self.workspace_mkdir, WorkspaceMkdir, self._mkdir_factory),
        ]
        tools = [tool for tool in candidates if tool is not None]
        return tools + self._exec_tools()

    def _exec_tools(self) -> list[Callable[..., Any]]:
        """Return both exec callables, or neither.

        The one place in this card where a single capability field yields two
        callables. ``_tool_if_enabled`` encodes the 1:1 shape and is deliberately
        not used here — ``workspace_exec_result`` can do nothing without
        ``workspace_exec``, so the pair is enabled or absent as a unit.

        Shares :meth:`_enabled_exec` with the wiring, so the callables and the
        actor they need can never disagree about whether the capability is on.
        """
        params = self._enabled_exec()
        if params is None:
            return []
        return [self._exec_factory(params), self._exec_result_factory(params)]

    @staticmethod
    def _tool_if_enabled(
        value: _ParamT | bool,
        param_cls: type[_ParamT],
        factory: Callable[[_ParamT], Callable[..., Any]],
    ) -> Callable[..., Any] | None:
        """Build a capability's callable, or ``None`` when it is off the TOOL_CALL channel.

        Pairs the configuration field with the factory that consumes it, so the type
        checker verifies each row of :meth:`_read_tools` / :meth:`_write_tools`.
        """
        params = _resolve(value, param_cls)
        if params is None or TOOL_CALL not in params.expose:
            return None
        return factory(params)

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return COMMAND-channel capabilities for this tool.

        Returns:
            Dict mapping ``ExpandMediaRefs`` to ``_expand_media_refs`` when enabled.
        """
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}
        pr = _resolve(self.expand_media_refs, ExpandMediaRefs)
        if pr is not None:
            commands[ExpandMediaRefs] = self._expand_media_refs
        return commands

    def _expand_media_refs(self, prompt: str) -> list[str | MediaContent]:
        """Expand ``!!glob_pattern`` tokens in a prompt into binary image content.

        Supports both ``!!pattern`` (no spaces) and ``!!"pattern with spaces"`` (quoted).

        For each ``!!pattern`` or ``!!"pattern"`` token:
        - Image matches (extension in ``_MIME_MAP``) → ``MediaContent`` objects (sorted by path)
        - Document-only matches (extension in ``DocumentReader.extensions`` but NOT in
          ``_MIME_MAP``) → ``"!!name[=> Use workspace_read tool]"`` hint strings
        - No matches at all → ``"!!_pattern_[Error: no image found]"``

        Pure-text prompts (no ``!!`` tokens) return ``[prompt]``.

        .. note::
            The returned list may contain trailing empty strings (``""``) when the
            prompt ends with a ``!!token`` with no text following it.  Consumers
            that only care about non-empty parts should filter out empty strings::

                parts = [p for p in result if p != ""]

        Args:
            prompt: Input prompt string potentially containing ``!!glob_pattern`` tokens.

        Returns:
            Mixed list of plain strings and ``MediaContent`` objects.  May include
            trailing ``""`` entries when the last character of *prompt* is part of
            an expanded token.

        Raises:
            RuntimeError: If :meth:`observer` has not been called yet (workspace
                not initialised).
        """
        parts: list[str | MediaContent] = []
        last = 0
        for m in _REF_RE.finditer(prompt):
            if m.start() > last:
                parts.append(prompt[last : m.start()])
            pattern = m.group(1) or m.group(2)
            all_matches = sorted(p for p in self.workspace._root.glob(pattern) if p.is_file())
            image_matches = [p for p in all_matches if p.suffix.lower() in _MIME_MAP]
            doc_matches = [
                p
                for p in all_matches
                if p.suffix.lower() in DocumentReader.extensions
                and p.suffix.lower() not in _MIME_MAP
            ]
            if image_matches:
                for path in image_matches:
                    try:
                        data = path.read_bytes()
                    except OSError:
                        parts.append(f"!!{path.name}[Error: file unreadable]")
                        continue
                    parts.append(
                        MediaContent(
                            data=data,
                            media_type=_MIME_MAP[path.suffix.lower()],
                        )
                    )
            elif doc_matches:
                for path in doc_matches:
                    parts.append(f"!!{path.name}[=> Use workspace_read tool]")
            else:
                parts.append(f"!!{pattern}[Error: no image found in the workspace]")
            last = m.end()
        parts.append(prompt[last:])
        return parts

    def _read_factory(self, params: WorkspaceRead) -> Callable[..., Any]:
        """Create the ``workspace_read`` tool callable.

        Args:
            params: Read capability configuration.

        Returns:
            Callable that reads a workspace file with pagination.
        """
        backend = self.workspace
        record = self._observation_recorder()
        _dr_cfg = params.document_reader
        if _dr_cfg is True:
            document_reader: DocumentReader | None = DocumentReader()
        elif _dr_cfg is False:
            document_reader = None
        else:
            document_reader = _dr_cfg

        def workspace_read(
            path: str,
            offset: int = 1,
            limit: int = params.default_limit,
            force_document_regeneration: bool = params.force_document_regeneration,
        ) -> str:
            """Read a file from the team workspace.

            Args:
                path: Relative path from workspace root (e.g. "src/main.py").
                offset: First line to return, 1-indexed. Defaults to 1.
                limit: Maximum lines to return. Defaults to 2000.
                force_document_regeneration: If True, re-extract binary files
                    even if a cached sidecar exists. Defaults to False.

            Returns:
                File contents with 1-indexed line numbers prefixed.
                Truncated files include a trailing notice.

            Raises:
                RetriableError: If the path does not exist or escapes the
                    workspace root.
                ValueError: If the file is a binary format and
                    ``document_reader`` is not configured.
            """
            ext = Path(path).suffix.lower()
            p = Path(path)

            # Sidecar self-read guard: .report.pdf.md -> plain text
            is_sidecar = p.name.startswith(".") and p.name.endswith(".md")

            # ValueError check outside try -- configuration error, not retryable
            if not is_sidecar and document_reader is None and ext in DocumentReader.extensions:
                raise ValueError(
                    "Binary file reading requires document_reader. "
                    'Install: pip install "akgentic-tool[docs]"'
                )

            try:
                # Raw source bytes, and only on the branch that actually read them.
                # A document read shows extracted Markdown rather than the file, and
                # on a sidecar cache hit the source is never opened at all — hashing
                # it would put a full file read back onto the path NFR1 keeps free.
                observed: bytes | None = None
                if (
                    not is_sidecar
                    and document_reader is not None
                    and ext in document_reader.extensions
                ):
                    # Binary path: sidecar cache or extraction
                    sidecar = backend._root / p.parent / f".{p.name}.md"
                    if sidecar.exists() and not force_document_regeneration:
                        raw = sidecar.read_text(encoding="utf-8")
                    else:
                        content_bytes = backend.read(path)
                        raw = document_reader.extract_text(content_bytes, path)
                        sidecar.write_text(raw, encoding="utf-8")
                else:
                    # Text path (existing logic)
                    observed = backend.read(path)
                    raw = observed.decode("utf-8")

                numbered, full = _paginate(raw, offset, limit)
                if observed is not None:
                    # One recording call per invocation, whatever the file's size.
                    record(path, observed, full)
                return numbered
            except FileNotFoundError:
                raise RetriableError(f"File not found: {path}")
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)

        workspace_read.__doc__ = params.format_docstring(workspace_read.__doc__)
        return workspace_read

    def _list_factory(self, params: WorkspaceList) -> Callable[..., Any]:
        """Create the ``workspace_list`` tool callable.

        Args:
            params: List capability configuration.

        Returns:
            Callable that lists workspace directory contents (flat or tree).
        """
        backend = self.workspace

        def workspace_list(path: str = "", depth: int = params.max_depth) -> str:
            """List the contents of a directory in the team workspace.

            Args:
                path: Relative directory path. Defaults to workspace root.
                depth: Tree depth. 1 = flat list (default), 0 = unlimited tree,
                    N > 1 = tree N levels deep.

            Returns:
                Flat list or ASCII tree of entries. Directories shown as ``name/``,
                files as ``name (N bytes)``. Returns "Empty directory." if no entries.

            Raises:
                RetriableError: If the directory does not exist, the path points at
                    a file rather than a directory, or the path escapes the
                    workspace root.
            """
            try:
                if path:
                    resolved = backend._validate_path(path)
                else:
                    resolved = backend._root

                entries = backend.list(path)
                if not entries:
                    return "Empty directory."

                if depth == 1:
                    # Flat list — no tree connectors
                    lines: list[str] = []
                    for entry in entries:
                        if entry.is_dir:
                            lines.append(f"{entry.name}/")
                        else:
                            lines.append(f"{entry.name} ({entry.size} bytes)")
                    return "\n".join(lines)
                else:
                    # ASCII tree — depth=0 means unlimited, depth>1 means N levels
                    tree_lines = _build_tree(resolved, max_depth=depth)
                    return ".\n" + "\n".join(tree_lines) if tree_lines else "Empty directory."
            except (FileNotFoundError, NotADirectoryError):
                raise RetriableError(f"Directory not found: {path}")
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)

        workspace_list.__doc__ = params.format_docstring(workspace_list.__doc__)
        return workspace_list

    def _glob_factory(self, params: WorkspaceGlob) -> Callable[..., Any]:
        """Create the ``workspace_glob`` tool callable.

        Args:
            params: Glob capability configuration.

        Returns:
            Callable that searches the workspace via glob patterns.
        """
        backend = self.workspace
        max_results = params.max_results

        def workspace_glob(pattern: str, path: str = "") -> str:
            """Find files matching a glob pattern in the team workspace.

            Args:
                pattern: Glob pattern (e.g. "**/*.py", "src/**/*.ts").
                path: Subdirectory to search within. Defaults to workspace root.

            Returns:
                Newline-separated list of relative file paths, or "No files found."
                Includes truncation notice if more than max_results files matched.

            Raises:
                RetriableError: If path escapes the workspace root.
            """
            try:
                if path:
                    search_root = (backend._root / path).resolve()
                    if not search_root.is_relative_to(backend._root):
                        raise PermissionError(f"Path '{path}' escapes workspace root")
                else:
                    search_root = backend._root
                seen: set[Path] = set()
                raw_matches: list[Path] = []
                for expanded_pattern in _expand_braces(pattern):
                    safe_pattern = _normalize_glob_pattern(expanded_pattern)
                    for m in search_root.glob(safe_pattern, case_sensitive=False):
                        if m.is_file() and m not in seen:
                            seen.add(m)
                            raw_matches.append(m)
                all_matches = sorted(
                    raw_matches,
                    key=lambda match: match.stat().st_mtime,
                    reverse=True,
                )
                truncated = len(all_matches) > max_results
                shown = [str(m.relative_to(backend._root)) for m in all_matches[:max_results]]
                if not shown:
                    return "No files found."
                result = "\n".join(shown)
                if truncated:
                    result += (
                        f"\n[... truncated: {len(all_matches)} total,"
                        f" showing first {max_results} ...]"
                    )
                return result
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)

        workspace_glob.__doc__ = params.format_docstring(workspace_glob.__doc__)
        return workspace_glob

    def _grep_factory(self, params: WorkspaceGrep) -> Callable[..., Any]:
        """Create the ``workspace_grep`` tool callable.

        Args:
            params: Grep capability configuration.

        Returns:
            Callable that searches workspace file contents by regex.
        """
        backend = self.workspace
        max_results = params.max_results
        max_line_len = params.max_line_length

        def workspace_grep(pattern: str, path: str = "", include: str = "") -> str:
            """Search file contents using a regex pattern in the team workspace.

            Args:
                pattern: Regular expression pattern (Python re syntax).
                path: Subdirectory to search within. Defaults to workspace root.
                include: Glob pattern to restrict which files are searched
                    (e.g. "*.py", "*.ts"). Empty = all files.

            Returns:
                Formatted results grouped by file, or "No matches found."

            Raises:
                RetriableError: If pattern is not a valid regex or path escapes workspace root.
            """
            try:
                if path:
                    search_root = (backend._root / path).resolve()
                    if not search_root.is_relative_to(backend._root):
                        raise PermissionError(f"Path '{path}' escapes workspace root")
                else:
                    search_root = backend._root

                raw_matches = _grep_rg(search_root, pattern, include, max_results)
                if raw_matches is None:
                    raw_matches = _grep_python(
                        search_root, pattern, include, max_results, max_line_len
                    )

                if not raw_matches:
                    return "No matches found."

                result_lines = [
                    f"{fpath.relative_to(backend._root)}:{lineno}: {line}"
                    for fpath, lineno, line in raw_matches
                ]
                return "\n".join(result_lines)
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)
            except _re.error as e:
                raise RetriableError(f"Invalid regex pattern: {e}")

        workspace_grep.__doc__ = params.format_docstring(workspace_grep.__doc__)
        return workspace_grep

    def _view_factory(self, params: WorkspaceView) -> Callable[..., Any]:
        """Create the ``workspace_view`` tool callable.

        Args:
            params: View capability configuration.

        Returns:
            Callable that reads an image from the workspace as BinaryContent.
        """
        backend = self.workspace
        max_dim = params.max_dimension

        def workspace_view(path: str) -> BinaryContent:
            """View an image file from the workspace. Returns the image for vision analysis.

            Use this when you need to visually inspect an image (screenshot, diagram, photo).
            For text extraction from documents (PDF, DOCX), use workspace_read instead.

            Supported formats: PNG, JPEG, GIF, WebP, BMP.

            Args:
                path: Relative path to the image file within the workspace.

            Returns:
                BinaryContent with the image bytes and MIME type.

            Raises:
                RetriableError: If the path does not exist, escapes the workspace root,
                    or the file extension is not a supported image format.
            """
            try:
                data = backend.read_bytes(path)
            except FileNotFoundError:
                raise RetriableError(f"File not found: {path}")
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)
            try:
                suffix = PurePosixPath(path).suffix.lower()
                mime = _MIME_MAP.get(suffix)
                if mime is None:
                    raise RetriableError(
                        f"Unsupported image format '{suffix}'. "
                        f"Supported: {', '.join(sorted(_MIME_MAP))}. "
                        f"For documents, use workspace_read instead."
                    )
                data = _maybe_resize(data, suffix, max_dim, backend._root, path)
                return BinaryContent(data=data, media_type=mime)
            except RetriableError:
                raise

        workspace_view.__doc__ = params.format_docstring(workspace_view.__doc__)
        return workspace_view

    def _write_factory(self, params: WorkspaceWrite) -> Callable[..., Any]:
        """Create the ``workspace_write`` tool callable.

        Args:
            params: Write capability configuration.

        Returns:
            Callable that writes content to a workspace file, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id

        def workspace_write(path: str, content: str) -> str:
            """Write content to a file in the team workspace.

            Refused if another writer has changed the file since you last read
            it, or if you have not read it at all — the refusal shows you what
            your content would have replaced. Read the file again and redo the
            change against what is now there.

            Args:
                path: Relative path from workspace root (e.g. "src/main.py").
                content: Text content to write.

            Returns:
                Confirmation string "Written: <path>".

            Raises:
                RetriableError: If the write is refused, or the path escapes the
                    workspace root.
            """
            return _resolve_outcome(_bound(proxy).apply_write(agent_id, path, content))

        workspace_write.__doc__ = params.format_docstring(workspace_write.__doc__)
        return workspace_write

    def _delete_factory(self, params: WorkspaceDelete) -> Callable[..., Any]:
        """Create the ``workspace_delete`` tool callable.

        Args:
            params: Delete capability configuration.

        Returns:
            Callable that deletes a file from the workspace, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id

        def workspace_delete(path: str) -> str:
            """Delete a file from the team workspace.

            Refused unless you have read the whole file and it has not changed
            since — deleting a file someone else has just rewritten destroys
            work you never saw.

            Args:
                path: Relative path from workspace root (e.g. "src/old.py").

            Returns:
                Confirmation string "Deleted: <path>".

            Raises:
                RetriableError: If the delete is refused, the path does not
                    exist, or it escapes the workspace root.
            """
            return _resolve_outcome(_bound(proxy).apply_delete(agent_id, path))

        workspace_delete.__doc__ = params.format_docstring(workspace_delete.__doc__)
        return workspace_delete

    def _edit_factory(self, params: WorkspaceEdit) -> Callable[..., Any]:
        """Create the ``workspace_edit`` tool callable.

        Args:
            params: Edit capability configuration.

        Returns:
            Callable that applies a surgical find-and-replace edit, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id

        def workspace_edit(
            path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ) -> str:
            """Apply a surgical find-and-replace edit to a workspace file.

            Preferred over workspace_write for changing part of a file: an edit
            survives a concurrent change to an unrelated region, where replacing
            the whole file cannot. On a file that changed since you read it,
            old_string must match exactly.

            Args:
                path: Relative path from workspace root.
                old_string: Exact (or approximately matching) text to replace.
                new_string: Replacement text.
                replace_all: If True, replace all occurrences (default False).

            Returns:
                Unified diff string of the change, or "[ERROR] ..." on failure.

            Raises:
                RetriableError: If the edit is refused, the path does not exist,
                    or it escapes the workspace root.
            """
            return _resolve_outcome(
                _bound(proxy).apply_edit(agent_id, path, old_string, new_string, replace_all)
            )

        workspace_edit.__doc__ = params.format_docstring(workspace_edit.__doc__)
        return workspace_edit

    def _multi_edit_factory(self, params: WorkspaceMultiEdit) -> Callable[..., Any]:
        """Create the ``workspace_multi_edit`` tool callable.

        Args:
            params: Multi-edit capability configuration.

        Returns:
            Callable that applies a batch of find-and-replace edits, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id

        def workspace_multi_edit(edits: list[EditItem]) -> str:
            """Apply a sequence of find-and-replace edits to workspace files.

            All-or-nothing: if any edit is refused or its old_string is not
            found, no file in the batch is changed on disk.

            Args:
                edits: Ordered list of EditItem objects. Each edit is applied
                    sequentially; each sees the result of the previous one.

            Returns:
                Combined unified diff of all applied edits, or "[ERROR] ..." on failure.

            Raises:
                RetriableError: If any edit is refused, a target file does not
                    exist, or a path escapes the workspace root.
            """
            return _resolve_outcome(_bound(proxy).apply_multi_edit(agent_id, edits))

        workspace_multi_edit.__doc__ = params.format_docstring(workspace_multi_edit.__doc__)
        return workspace_multi_edit

    def _patch_factory(self, params: WorkspacePatch) -> Callable[..., Any]:
        """Create the ``workspace_patch`` tool callable.

        Args:
            params: Patch capability configuration.

        Returns:
            Callable that applies a unified diff patch, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id

        def workspace_patch(patch_text: str) -> str:
            """Apply a unified diff patch to the team workspace.

            Supports add (--- /dev/null), update, and delete (+++ /dev/null).
            Every file the patch touches is checked against what you last read
            of it. Application stops at the first file that fails.

            Args:
                patch_text: GNU unified diff string.

            Returns:
                Newline-joined summary: "created: ...", "updated: ...", or
                "deleted: ...". Returns "[ERROR] ..." on failure.

            Raises:
                RetriableError: If a file's change is refused, or any path
                    escapes the workspace root.
            """
            return _resolve_outcome(_bound(proxy).apply_patch(agent_id, patch_text))

        workspace_patch.__doc__ = params.format_docstring(workspace_patch.__doc__)
        return workspace_patch

    def _mkdir_factory(self, params: WorkspaceMkdir) -> Callable[..., Any]:
        """Create the ``workspace_mkdir`` tool callable.

        Args:
            params: Mkdir capability configuration.

        Returns:
            Callable that creates a directory in the workspace, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id

        def workspace_mkdir(path: str) -> str:
            """Create a directory and all missing parents in the team workspace.

            Args:
                path: Relative directory path from workspace root (e.g. "src/utils").

            Returns:
                Confirmation string "Created: <path>".

            Raises:
                RetriableError: If the path escapes the workspace root.
            """
            return _resolve_outcome(_bound(proxy).apply_mkdir(agent_id, path))

        workspace_mkdir.__doc__ = params.format_docstring(workspace_mkdir.__doc__)
        return workspace_mkdir

    def _exec_factory(self, params: WorkspaceExec) -> Callable[..., Any]:
        """Create the ``workspace_exec`` tool callable.

        Args:
            params: Exec capability configuration.

        Returns:
            Callable that runs a sandboxed command, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id
        delay = params.poll_delay_seconds
        # Clamped, not accepted: a poll outlasting the run is a sleep with no
        # possible answer, because by then the run has reported or its own budget
        # has killed it. Bounded by the *effective* run budget, which is what
        # stops the run — a card asking for 999 s never gets more than the worker
        # allows it either.
        attempts = poll_attempts_within(
            params.poll_attempts, delay, min(params.timeout_s, DEFAULT_WORKER_TIMEOUT_S)
        )

        def workspace_exec(cmd: str, cwd: str = "") -> str:
            """Run a shell command in the team workspace, in a sandbox.

            The workspace is held exclusively for the duration of the run: your
            teammates can still read files, but every change they attempt is
            refused until it finishes. Everything the command touched — files you
            never named included — is recorded as one change attributed to you.

            A command that outlives the wait returns a run id instead of output.
            Call workspace_exec_result with that id on your next turn.

            Args:
                cmd: Full command string. The binary (first token) must be in
                    the allow-list.
                cwd: Subdirectory relative to workspace root. Defaults to root.

            Returns:
                Combined stdout, stderr and exit code, or a message naming the
                run id to collect later.

            Raises:
                RetriableError: If another agent's run holds the workspace.
            """
            start = _bound(proxy).request_exec(agent_id, cmd, cwd)
            if not start.run_id:
                raise RetriableError(start.refusal)
            run_id = start.run_id
            settled = poll_deferred(
                lambda: _settled_status(_bound(proxy).exec_status(agent_id, run_id)),
                attempts=attempts,
                delay=delay,
            )
            return format_status(settled) if settled is not None else in_progress(run_id)

        workspace_exec.__doc__ = params.format_docstring(workspace_exec.__doc__)
        return workspace_exec

    def _exec_result_factory(self, params: WorkspaceExec) -> Callable[..., Any]:
        """Create the ``workspace_exec_result`` tool callable.

        Args:
            params: Exec capability configuration.

        Returns:
            Callable that collects a finished run's output, through the actor.
        """
        proxy = self._workspace_proxy
        agent_id = self._agent_id

        def workspace_exec_result(run_id: str) -> str:
            """Collect the output of a command started by workspace_exec.

            Args:
                run_id: The id workspace_exec handed back.

            Returns:
                The command's output if it has finished, a note that it is still
                running, why it failed, or — for an id nothing was issued under —
                your recent run ids so you can retry with the right one.
            """
            return format_status(_bound(proxy).exec_status(agent_id, run_id))

        workspace_exec_result.__doc__ = params.format_docstring(workspace_exec_result.__doc__)
        return workspace_exec_result
