"""What a hosted ``#Workspace`` must never do, pinned before it is hosted.

A hosted actor has no orchestrator: ``ResourceHost`` starts it with
``orchestrator=None``, and every ask it would make through one degrades or
raises the moment hosting lands. So the invariant for the whole ``workspace/actor/``
package is **no read of the orchestrator on ``self``** — not ``self.orchestrator``,
not ``self._orchestrator``, not the ``orchestrator_proxy_ask`` the base builds
when one exists — **and no call to a base-class helper that asks it on this
actor's behalf**, such as ``self.get_team_member``. The canary here walks every
module of the package with ``ast``.

**An AST walk, and deliberately not a text search.** ``documents.py`` names
``orchestrator`` in a comment and in docstrings, and a hosted actor may keep
doing so: what is forbidden is a *read*, which is an ``Attribute`` node whose
value is the name ``self``. A grep would either false-positive on the prose or
be narrowed until it agreed with it; the tree has no comments in it at all, so
the question it answers is the one that matters.

**The sweep floors its own size.** A canary that enumerated an empty directory
would find nothing and pass, so the module count is asserted in the same spec
that walks them, and every module is imported so an import error cannot hide one.

The live spec below is the same invariant seen from the other side: a workspace
started with no orchestrator at all still resolves its in-memory store, because
the store is now its own child, created with ``createActor`` and stopped with
its parent through ``stop_children``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pykka
import pytest

from akgentic.tool import workspace as workspace_package
from akgentic.tool.vector_store.protocol import VectorStoreParam
from akgentic.tool.workspace import actor as actor_package
from akgentic.tool.workspace.actor import workspace_actor_name
from akgentic.tool.workspace.tool import WorkspaceTool
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    tool_named,
)

_ACTOR_PACKAGE_DIR = Path(actor_package.__file__).parent
"""Where the package's modules live on disk, taken from the import and not typed."""

_ORCHESTRATOR_SLOTS: frozenset[str] = frozenset(
    {"orchestrator", "_orchestrator", "orchestrator_proxy_ask"}
)
"""The three attributes a read of which reaches the orchestrator.

``orchestrator`` is the public property, ``_orchestrator`` the slot behind it, and
``orchestrator_proxy_ask`` the ask proxy ``Akgent.__init__`` builds over it when
one was handed in. A hosted actor has none of the three to offer.
"""

_ORCHESTRATOR_ASK_HELPERS: frozenset[str] = frozenset(
    {
        "get_team",
        "get_team_member",
        "discover_catalog",
        "get_agent_card",
        "find_agents_with_skill",
        "get_available_roles",
    }
)
"""The ``Akgent`` methods that ask the orchestrator **on the caller's behalf**.

Each one reads ``orchestrator_proxy_ask`` inside core, so a module here calling
``self.get_team_member(VS_ACTOR_NAME)`` makes exactly the ask this story removed
while reading none of the three slots above. A canary that only knew the slots
would pass that line. ``createActor`` and ``send`` read the slot too, but only to
hand it to a child or to emit telemetry, and neither asks the orchestrator
anything — which is why they are not here.
"""

_FORBIDDEN_ON_SELF: frozenset[str] = _ORCHESTRATOR_SLOTS | _ORCHESTRATOR_ASK_HELPERS
"""Every attribute on ``self`` the actor package may not touch."""

_MINIMUM_MODULES = 5
"""``__init__``, ``documents``, ``execution``, ``gate``, ``observation`` — the package
as it stood when this canary was written. A sweep that finds fewer walked nothing."""


def _actor_modules() -> list[Path]:
    """Every ``.py`` file anywhere under the actor package, sorted for a stable report.

    Recursive, so a subpackage added later is swept too rather than escaping a
    walk that only looked one level down.
    """
    return sorted(_ACTOR_PACKAGE_DIR.rglob("*.py"))


def _dotted(path: Path) -> str:
    """The import path of one module file of the actor package."""
    parts = path.relative_to(_ACTOR_PACKAGE_DIR).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((actor_package.__name__, *parts))


def _orchestrator_reads(path: Path) -> list[str]:
    """Every ``self.<forbidden>`` read in *path*, as ``file:line``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{path.name}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in _FORBIDDEN_ON_SELF
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ]


class TestTheActorNeverReadsItsOrchestrator:
    """No read of the orchestrator on ``self`` anywhere under ``workspace/actor/``."""

    def test_no_module_of_the_package_reads_the_orchestrator_on_self(self) -> None:
        """Zero across the package, and the package is at least the five modules it was."""
        modules = _actor_modules()
        assert len(modules) >= _MINIMUM_MODULES, (
            f"the sweep found {len(modules)} module(s) under {_ACTOR_PACKAGE_DIR} — "
            f"an empty or moved package would pass a canary that walks nothing"
        )
        for module in modules:
            assert importlib.import_module(_dotted(module)) is not None
        hits = [hit for module in modules for hit in _orchestrator_reads(module)]
        assert hits == [], f"orchestrator read(s) under workspace/actor/: {hits}"

    def test_the_walk_sees_a_read_when_there_is_one(self, tmp_path: Path) -> None:
        """The detector itself, against a module that does read — and one that only talks."""
        reading = tmp_path / "reading.py"
        reading.write_text(
            "class A:\n"
            "    def f(self):\n"
            "        # self.orchestrator in a comment is not a read\n"
            "        '''self.orchestrator in a docstring is not one either'''\n"
            "        return self.orchestrator\n"
            "    def g(self):\n"
            "        return self.get_team_member('#VectorStore')\n",
            encoding="utf-8",
        )
        assert _orchestrator_reads(reading) == ["reading.py:5", "reading.py:7"]

    def test_every_ask_helper_it_refuses_still_exists_on_akgent(self) -> None:
        """A helper renamed in core would leave its old name here refusing nothing.

        So the list is checked against the class it describes, and a rename turns
        this red rather than turning the canary quietly blind to the new name.
        """
        from akgentic.core.agent import Akgent

        missing = sorted(name for name in _ORCHESTRATOR_ASK_HELPERS if not hasattr(Akgent, name))
        assert missing == [], f"no longer on Akgent: {missing}"


_WORKSPACE_PACKAGE_DIR = Path(workspace_package.__file__).parent
"""The whole workspace package on disk — card, actor, and everything beside them."""

_MINIMUM_WORKSPACE_MODULES = 25
"""The package had 25 modules before story 51-2 added ``host.py`` and ``event.py``.
A sweep that finds fewer walked a moved or emptied package, and found nothing."""


def _workspace_modules() -> list[Path]:
    """Every ``.py`` file anywhere under the workspace package, sorted for a stable report."""
    return sorted(_WORKSPACE_PACKAGE_DIR.rglob("*.py"))


def _workspace_dotted(path: Path) -> str:
    """The import path of one module file of the workspace package."""
    parts = path.relative_to(_WORKSPACE_PACKAGE_DIR).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((workspace_package.__name__, *parts))


def _child_binds(path: Path, root: Path) -> list[str]:
    """Every ``.getChildrenOrCreate`` attribute in *path*, as ``file:line``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{path.relative_to(root)}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "getChildrenOrCreate"
    ]


class TestTheWorkspaceIsNeverBoundAsAChild:
    """No ``getChildrenOrCreate`` anywhere in the workspace package: every bind is hosted."""

    def test_no_module_of_the_workspace_package_calls_get_children_or_create(self) -> None:
        modules = _workspace_modules()
        assert len(modules) >= _MINIMUM_WORKSPACE_MODULES, (
            f"the sweep found {len(modules)} module(s) under {_WORKSPACE_PACKAGE_DIR} — "
            f"an empty or moved package would pass a canary that walks nothing"
        )
        for module in modules:
            assert importlib.import_module(_workspace_dotted(module)) is not None
        hits = [hit for module in modules for hit in _child_binds(module, _WORKSPACE_PACKAGE_DIR)]
        assert hits == [], f"getChildrenOrCreate under workspace/: {hits}"

    def test_the_walk_sees_a_call_and_ignores_the_prose(self, tmp_path: Path) -> None:
        """The detector itself: a docstring or a comment naming it is not a call."""
        module = tmp_path / "binding.py"
        module.write_text(
            "def bind(proxy):\n"
            '    """getChildrenOrCreate in a docstring is not a call"""\n'
            "    # proxy.getChildrenOrCreate in a comment is not one either\n"
            "    return proxy.getChildrenOrCreate(object, config=None)\n",
            encoding="utf-8",
        )
        assert _child_binds(module, tmp_path) == ["binding.py:4"]


class TestAWorkspaceWithNoOrchestratorOwnsItsStore:
    """The hosted-style falsifier on real threads, through public API only."""

    def test_the_store_child_is_created_named_and_stopped_with_its_workspace(
        self, threaded_orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """A workspace started with no orchestrator indexes, and its child dies with it.

        ``FakeOrchestratorProxy`` in live mode starts the actor with no
        orchestrator at all, which is exactly what ``ResourceHost`` will do. The
        child is found through ``pykka.ActorRegistry`` and read through its own
        proxy — never through ``_children``, ``_actor_ref`` or ``_actor``.

        The workspace is stopped by ``stop_all``, which goes through
        ``Akgent.stop`` exactly as the orchestrator's teardown does. Nothing
        stops the child but its parent's ``stop_children``: a bare
        ``ActorRef.stop()`` on the workspace runs ``on_stop`` only, and would
        leave this spec red on the child outliving it.
        """
        pytest.importorskip("numpy", reason="the [vector_search] extra is not installed")
        from akgentic.tool.vector_store.actor import (
            VS_ACTOR_NAME,
            VS_ACTOR_ROLE,
            VectorStoreActor,
        )

        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []
        observer = FakeActorToolObserver(threaded_orchestrator_proxy, name="alice")
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME,
            workspace_rag_index=True,
            vector_store=VectorStoreParam(backend="inmemory"),
        )
        card.observer(observer)

        answer = tool_named(card, "workspace_rag_index")("")

        assert answer == "0 file(s) queued, 0 already current, 0 unsupported"
        [store_ref] = pykka.ActorRegistry.get_by_class(VectorStoreActor)
        store = store_ref.proxy()
        config = store.config.get()
        assert config.name == f"{VS_ACTOR_NAME}-{WORKSPACE_PATH}"
        assert config.role == VS_ACTOR_ROLE
        assert store.orchestrator.get() is None
        workspace_address, _workspace = threaded_orchestrator_proxy.hosted[
            workspace_actor_name(WORKSPACE_PATH)
        ]
        assert store.team_id.get() == workspace_address.team_id
        assert store_ref.is_alive()

        threaded_orchestrator_proxy.stop_all()

        assert not workspace_address.is_alive()
        assert store_ref.actor_stopped.wait(timeout=HANDSHAKE_TIMEOUT_S), (
            "the store child outlived its workspace"
        )
        assert not store_ref.is_alive()
        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []

    def test_the_inert_teardown_stops_the_store_child_too(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The inert half of ``stop_all``: an actor with no thread still stops its child.

        An inert ``WorkspaceActor`` runs ``enable_rag`` on the test's own thread,
        and its real ``createActor`` starts a real store thread. ``stop_all`` has
        to reach that child through the parent's ``stop_children``, as
        ``Akgent.stop`` does; ``on_stop`` alone leaves it running. Nothing else in
        the suite goes red when that half is missing: a later file's registry-wide
        ``ActorRegistry.stop_all()`` reaps the orphan, and the only symptom left is
        an interpreter that will not exit when a retrieval file runs alone.
        """
        pytest.importorskip("numpy", reason="the [vector_search] extra is not installed")
        from akgentic.tool.vector_store.actor import VS_ACTOR_NAME, VectorStoreActor

        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []
        observer = FakeActorToolObserver(orchestrator_proxy, name="alice")
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME,
            workspace_rag_index=True,
            vector_store=VectorStoreParam(backend="inmemory"),
        )
        card.observer(observer)

        [store_ref] = pykka.ActorRegistry.get_by_class(VectorStoreActor)
        assert store_ref.proxy().config.get().name == f"{VS_ACTOR_NAME}-{WORKSPACE_PATH}"

        orchestrator_proxy.stop_all()

        assert store_ref.actor_stopped.wait(timeout=HANDSHAKE_TIMEOUT_S), (
            "the store child outlived its inert workspace"
        )
        assert pykka.ActorRegistry.get_by_class(VectorStoreActor) == []
