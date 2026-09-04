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

The twenty factory bodies live in the three sibling mixins — ``card/read.py``,
``card/write.py``, ``card/execution.py`` — and the capability parameters in
``card/params.py``. What stays here is the card itself: its fields, ``observer()``
with its private binding helpers, and the ``get_tools`` / ``get_commands``
registration (ADR-045 §1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import PrivateAttr

from akgentic.core.actor_address import ActorAddress
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import TOOL_CALL, BaseToolParam, ToolCard, _resolve
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.card.execution import ExecFactories
from akgentic.tool.workspace.card.params import (
    ExpandMediaRefs,
    Resource,
    ResourceType,
    WorkspaceDelete,
    WorkspaceEdit,
    WorkspaceExec,
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceMkdir,
    WorkspaceMultiEdit,
    WorkspacePatch,
    WorkspaceRead,
    WorkspaceView,
    WorkspaceWrite,
)
from akgentic.tool.workspace.card.read import ReadFactories
from akgentic.tool.workspace.card.write import WriteFactories
from akgentic.tool.workspace.documents.models import EXTRACTOR_VERSION
from akgentic.tool.workspace.execution import (
    ExecConfig,
    resolve_mode,
    sandbox_config,
)
from akgentic.tool.workspace.models import (
    Observation,
    WorkspaceConfig,
    content_sha,
)
from akgentic.tool.workspace.workspace import Filesystem, get_workspace

logger = logging.getLogger(__name__)

# Binds a capability's configuration field to the factory that consumes it.
_ParamT = TypeVar("_ParamT", bound=BaseToolParam)

__all__ = [
    "ExpandMediaRefs",
    "Resource",
    "ResourceType",
    "WorkspaceDelete",
    "WorkspaceEdit",
    "WorkspaceExec",
    "WorkspaceGlob",
    "WorkspaceGrep",
    "WorkspaceList",
    "WorkspaceMkdir",
    "WorkspaceMultiEdit",
    "WorkspacePatch",
    "WorkspaceRead",
    "WorkspaceTool",
    "WorkspaceView",
    "WorkspaceWrite",
]


class WorkspaceTool(ReadFactories, WriteFactories, ExecFactories, ToolCard):
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

    The three mixins carry the factory bodies and declare no Pydantic field, so
    every field of this card is declared right here.
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

    # Write capability fields
    workspace_write: WorkspaceWrite | bool = True
    workspace_delete: WorkspaceDelete | bool = True
    workspace_edit: WorkspaceEdit | bool = True
    workspace_multi_edit: WorkspaceMultiEdit | bool = True
    workspace_patch: WorkspacePatch | bool = True
    workspace_mkdir: WorkspaceMkdir | bool = True

    git_journal: bool = False
    """Whether accepted mutations are recorded in a git journal.

    A plain field, not a capability param: it exposes no tool, appears in no
    signature, and nothing about it is expressible by a model.

    **Off by default, because nothing in the system reads what it records.** The
    gate re-hashes the live file at mutation time and never consults the journal,
    and an agent's exec result carries ``exit_code``/``stdout``/``stderr`` and not
    the discovered write set — so the record exists only for a human reading
    ``git log`` afterwards. That is worth opting into, not worth three ``git``
    forks on every mutation by default. Turning it on buys history, attribution
    and out-of-band *detection*; leaving it off loosens the gate by nothing at
    all, because the gate is pure Python and independent.

    Note what ``getChildrenOrCreate`` implies: the **first** card to create the
    actor for a workspace decides its configuration, exactly as the observation
    caps already do. A second card arriving with ``git_journal=False`` does not
    turn off a journal that is already running, and a card arriving with it on
    does not start one on an actor already built without it.
    """

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
                git_journal=self.git_journal,
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
        rather than a hope — of a **text** read, which is what this recorder
        serves. From epic 29 the actor hashes files on its ask path, so a read
        that asked would queue behind another agent's mutation hashing a large
        file; the ``except`` below would not save it, because a fail-open guard
        covers a raising actor and a dead one, never a hung one.

        A **document** read is the one exception, and it is deliberate: it makes
        one bounded ask through :meth:`_extract_lookup` (ADR-045 §3), against
        O(1) dict work with no I/O behind it. It records no observation at all,
        so it never reaches this closure.

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

    def _extract_lookup(self) -> Callable[[str, str], str | None]:
        """Build the closure a document read uses to ask for a cached extraction.

        An **ask**, because the answer is the whole point: the caller extracts
        when there is nothing to serve. It is the one place a read waits on the
        actor, and what makes that acceptable is what is behind it — a dict
        lookup and an LRU reorder, no I/O, and no notify.

        :data:`EXTRACTOR_VERSION` is captured **here**, at ``get_tools`` time,
        exactly as the agent id is above. The extractor a miss would run is a
        property of the code, never of the call, so it is not a parameter of the
        tool callable and no agent can choose it. Bumping the constant therefore
        invalidates every stored entry on the next read, with no sweep.

        Returns:
            A callable taking the path and the digest of the source bytes the
            caller just read, and returning the cached Markdown or ``None``. It
            never raises: every failure degrades to a miss, which costs one
            extraction and can never be a wrong answer.
        """
        proxy = self._workspace_proxy
        version = EXTRACTOR_VERSION

        def lookup(path: str, source_sha: str) -> str | None:
            if proxy is None:
                return None  # harness shapes that wire a bare observer never bind one
            try:
                return proxy.document_extract(path, source_sha, version)
            except Exception:
                # Fail open, towards the pre-cache behaviour: a miss re-extracts
                # from the tree, which is where every byte came from anyway.
                logger.debug("Could not read the document cache for %s", path, exc_info=True)
                return None

        return lookup

    def _extract_recorder(self) -> Callable[[str, str, str], None]:
        """Build the closure a document read uses to fill the extraction cache.

        A **tell**, because nothing comes back and a slow actor must not hold a
        read that has already produced its answer. The split against
        :meth:`_extract_lookup` is a correctness requirement, not a style
        choice: collapsing both onto one proxy either makes a fill blocking or
        makes a lookup answerless.

        Returns:
            A callable taking the path, the digest of the source bytes and the
            extracted Markdown. It never raises: a lost fill is a cache that did
            not grow, never a failed read.
        """
        proxy = self._workspace_tell
        version = EXTRACTOR_VERSION

        def remember(path: str, source_sha: str, markdown: str) -> None:
            if proxy is None:
                return  # harness shapes that wire a bare observer never bind one
            try:
                proxy.cache_document(path, source_sha, version, markdown)
            except Exception:
                logger.debug("Could not fill the document cache for %s", path, exc_info=True)

        return remember

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
