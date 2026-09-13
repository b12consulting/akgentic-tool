"""What the write capability owes beyond the gate's own suites.

Two properties, both written against the **un-moved** tree and confirmed green
there — the discipline ``test_read_capability.py`` and ``test_card_public_api.py``
state about themselves — so what they assert is what the move preserved rather
than what the move produced.

**This module is deliberately small.** The gate's behaviour is already pinned by
``test_write_gate.py`` (15 classes), ``test_gate_locks.py`` (the cross-process
lock and the busy refusal) and ``test_live_hash.py`` (the hash is not a cache);
none of it is copied here. What a read-only bind costs on the write side is
``test_read_capability.py::TestEnablingNothingCostsNothing``, which asserts both
halves of it — ``[tool.__name__ for tool in card.get_tools()] == READ_CALLABLES``,
a list equality in order, so none of the six write names can appear, and
``not meta_dir.exists()``, which is strictly stronger than "no ``locks/``
directory" because ``_hold`` creates that lazily on the first mutation with
paths. Neither is duplicated here; this module holds only what was genuinely
unpinned.

**A stated limit of the capability, recorded here because this is where a reader
looks for it.** ``read_only`` gates closure **registration**, not the gate:
``get_tools()`` withholds the six mutation closures, but ``CardGate`` is still in
``WorkspaceTool``'s MRO, so ``card.apply_write(...)`` remains a reachable Python
method on a read-only card. Nothing in ``src/`` calls it except those six
closures and no LLM-facing surface reaches it, so the limit is a stated design
rather than a hole. It is deliberately **not** closed with a ``read_only`` check
inside the gate: that would be a change to the gate's mechanics, and it would put
the decision about whether the capability is on in a second place.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from akgentic.core.utils import deserialize_object, import_class, serialize

from akgentic.tool.core import COMMAND, TOOL_CALL
from akgentic.tool.workspace.card import WorkspaceTool
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.write import gate as gate_module

# From where they are **defined**, never through ``card/params.py``'s re-export:
# this module is what holds that re-export in place, so a spec that imported
# through it would be reddened by its own import line rather than by the
# assertions that state the property.
from akgentic.tool.workspace.write.params import WorkspaceMkdir, WorkspaceWrite
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    mutate,
    read,
)

##
## AC 9 — the lock is taken in sorted path order
##


class TestTheLocksAreTakenInSortedPathOrder:
    """``_hold``'s ``sorted(set(paths))``, which nothing asserted until now.

    The docstring on :meth:`CardGate._hold` calls sorted acquisition "*what makes
    a deadlock impossible*" — two agents touching the same two files take them in
    the same order whatever order their own batch names them in — and the audit
    cites the line. No spec did.

    **Asserted on the ``path`` argument, never on the lock filename.** A lock file
    is named ``path-<sha256 of the path>``, and a digest's sort order has nothing
    to do with its input's, so a spec asserting sorted *filenames* would assert
    nothing about the property it is named for.
    """

    @pytest.fixture
    def three_files(self, wired_card: WorkspaceTool, workspace_tree: Path) -> Path:
        """Three files the card has read, so the edits below are accepted.

        Driven through the real harness — the card's own callables — because a
        spec that called ``CardGate`` methods on a bare object would prove the
        helper and not the capability.
        """
        for name, body in (("a.md", "x = 1\n"), ("b.md", "y = 2\n"), ("c.md", "z = 3\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
            read(wired_card, name)
        return workspace_tree

    @pytest.fixture
    def locked_paths(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Every path ``_hold`` asked a lock file for, in the order it asked.

        Patched on the gate module's own namespace, which is where ``_hold``
        looks the function up, and delegating to the real implementation so the
        mutation genuinely takes the locks it is being watched taking.
        """
        seen: list[str] = []
        original = gate_module.lock_file_for

        def _recording(meta_dir: Path, path: str) -> Path:
            seen.append(path)
            return original(meta_dir, path)

        monkeypatch.setattr(gate_module, "lock_file_for", _recording)
        return seen

    def test_a_multi_edit_locks_its_paths_sorted_and_deduplicated(
        self, wired_card: WorkspaceTool, three_files: Path, locked_paths: list[str]
    ) -> None:
        """The batch names its paths unsorted and names one of them twice.

        Both halves are what make the assertion capable of failing: an unsorted
        batch is the only input that can distinguish ``sorted(...)`` from the
        order given, and the duplicate is the only input that can distinguish
        ``set(...)`` from a plain list.
        """
        edits = [
            EditItem(path="c.md", old_string="z = 3", new_string="z = 30"),
            EditItem(path="a.md", old_string="x = 1", new_string="x = 10"),
            EditItem(path="b.md", old_string="y = 2", new_string="y = 20"),
            EditItem(path="a.md", old_string="x = 10", new_string="x = 100"),
        ]
        given = [item.path for item in edits]

        mutate(wired_card, "workspace_multi_edit", edits)

        assert locked_paths == sorted(set(given))
        # Non-vacuity: the spy saw something, and what it saw is not simply the
        # order the batch was written in — either would satisfy the equality
        # above on a batch that happened to arrive sorted.
        assert locked_paths != given


##
## AC 14 — a card a deployment already persisted still loads
##

STORED_MARKER_MODULE = "akgentic.tool.workspace.card.params"
"""The module path a deployment's stored write parameters carry in ``__model__``.

**Captured by serializing a configured card on the working tree, not transcribed
from a design document** — the discipline ``test_card_public_api.py``'s frozen
sets and ``test_read_capability.py``'s own marker block both follow.

``serialize_base_model`` stamps ``f"{cls.__module__}.{cls.__name__}"`` on every
:class:`~akgentic.core.utils.SerializableBaseModel`, and ``BaseToolParam`` is
one, so every card written since the card decomposition with a write parameter
set explicitly carries this literal string. Reading one back is ``import_module``
plus ``getattr`` on exactly it
(:func:`akgentic.core.utils.deserializer.import_class`); a path that has gone
raises ``UnresolvableClassError``, which turns a stored team's tool card into a
bad record rather than a card.

So the six write parameters keep resolving *here* whatever module actually
defines them — and the specs below are written so that the same source asserts
the same thing before and after any such move.
"""

WRITE_PARAM_NAMES = [
    "WorkspaceWrite",
    "WorkspaceDelete",
    "WorkspaceEdit",
    "WorkspaceMultiEdit",
    "WorkspacePatch",
    "WorkspaceMkdir",
]
"""The six parameters of the write capability, by the name a marker can carry."""


@pytest.fixture
def explicitly_configured_card() -> WorkspaceTool:
    """A card carrying two write parameters the author set by hand.

    Two rather than one: a marker is stamped per nested model, so a single one
    could be preserved by an accident that a second would expose. ``expose`` is
    the only field these six carry, so it is what the non-default value is spent
    on — a round trip that quietly substituted a default would be caught by it.
    """
    return WorkspaceTool(
        workspace_id=WORKSPACE_NAME,
        workspace_write=WorkspaceWrite(),
        workspace_mkdir=WorkspaceMkdir(expose={TOOL_CALL, COMMAND}),
    )


class TestStoredWriteParamsStillResolve:
    """The persisted ``__model__`` markers, end to end through core's own path."""

    def test_every_write_param_resolves_through_the_stored_module_path(self) -> None:
        """``import_module`` + ``getattr``, exactly as ``import_class`` does it.

        ``hasattr`` on the package would not catch this: once these six are
        defined elsewhere, what keeps them resolving is a **re-export**, which
        only an import of this precise module path exercises.
        """
        module = importlib.import_module(STORED_MARKER_MODULE)
        for name in WRITE_PARAM_NAMES:
            assert getattr(module, name, None) is not None, (
                f"{STORED_MARKER_MODULE}.{name} no longer resolves — every card "
                f"persisted with that parameter set explicitly carries that literal string"
            )

    def test_the_stored_path_serves_the_same_classes_the_card_uses(self) -> None:
        """A second definition would deserialise into a class nothing else uses.

        A re-export satisfies this; a copy of the class body, which is the
        tempting way to "keep the path working", does not.
        """
        stored = importlib.import_module(STORED_MARKER_MODULE)
        facade = importlib.import_module("akgentic.tool.workspace")
        for name in WRITE_PARAM_NAMES:
            assert getattr(stored, name) is getattr(facade, name)

    def test_a_freshly_dumped_card_carries_markers_that_resolve(
        self, explicitly_configured_card: WorkspaceTool
    ) -> None:
        """Whatever module a parameter lives in, the path a dump stamps must import back.

        This is the row that legitimately *changes* with a move — a dump names
        wherever the class is defined — so it is written as the invariant rather
        than as a literal: the stamped path resolves, and to the very class the
        card is holding.
        """
        dumped = serialize(explicitly_configured_card)
        assert isinstance(dumped, dict)

        for field in ("workspace_write", "workspace_mkdir"):
            marker = dumped[field]["__model__"]
            assert import_class(marker) is type(getattr(explicitly_configured_card, field))

    def test_a_record_written_before_the_move_still_validates_into_an_equal_card(
        self, explicitly_configured_card: WorkspaceTool
    ) -> None:
        """The end-to-end property, on the literal a deployment's database holds.

        The markers are rewritten to :data:`STORED_MARKER_MODULE` rather than
        left as the dump produced them, so this spec asserts the **same thing
        before and after** the parameters move: on the un-moved tree the rewrite
        is a no-op and the record is exactly what a deployment stored; afterwards
        it is the pre-move record, which is the one that has to keep loading.

        Its non-vacuity is the two specs above: they prove the path is real.
        """
        stored = serialize(explicitly_configured_card)
        assert isinstance(stored, dict)
        stored["workspace_write"]["__model__"] = f"{STORED_MARKER_MODULE}.WorkspaceWrite"
        stored["workspace_mkdir"]["__model__"] = f"{STORED_MARKER_MODULE}.WorkspaceMkdir"

        restored = deserialize_object(stored)

        assert isinstance(restored, WorkspaceTool)
        assert restored.model_dump() == explicitly_configured_card.model_dump()
