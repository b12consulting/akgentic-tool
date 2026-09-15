"""The write capability of :class:`WorkspaceTool` — write, delete, edit, patch, mkdir.

It holds one capability whole: the six LLM-facing closures here, the mutation
gate in ``write/gate.py`` — with the observation map and the refusal vocabulary
that belong to it — and the capability's own parameters in ``write/params.py``.
It may import the package **spine** — ``workspace.py``, ``models.py``,
``edit.py``, ``event.py``, ``errors.py`` — and ``akgentic.tool.core``. It must
import **no other capability**, and nothing under ``card/`` at all: importing
``card.anything`` executes ``card/__init__.py``, which pulls in the journal, the
exec machinery, the RAG mixin, the vector-store registry and the actor. The rule
is enforced by ``tests/workspace/test_capability_import_closure.py``, which takes
the transitive closure of this package's own imports and checks it against an
allow-list — not by convention, and not by this paragraph. Four modules that
become capabilities of their own later are on that allow-list with their reasons;
the gate's docstring says what each is for.

**The two modules are two because the grouping reads well** (ADR-053 Decision 1):
the six closures are what an LLM sees, and the gate is what they converge on.
Merging them would make one 1500-line module out of a split that already tells a
reader which question they are asking.

A mutation runs **on the card**, through
:class:`~akgentic.tool.workspace.write.gate.CardGate`: it checks the live file
against what this agent last observed and performs the write itself, inside one
held ``fcntl.flock`` on the path (ADR-051 Decision 3). Nothing about the gate is
visible in an LLM-facing signature.

**Each closure captures the card, and that is allowed.** These closures used to
capture a proxy and an id precisely so that they held no edge back to the owning
agent (ADR-030); capturing ``self`` does not reintroduce one, because a
``ToolCard`` holds its observer **weakly** through the base setter. A closure
rooting the card therefore roots no stopped agent, and
``tests/test_toolcard_weak_observer.py`` is the guard that would catch it if that
ever stopped being true. A ``weakref`` dance around ``self`` here would be
ceremony against a retention that does not exist.

:class:`WriteFactories` is a **mixin**: it declares no Pydantic field, and the
names it consumes off ``self`` are declared under ``if TYPE_CHECKING:`` so they
reach mypy without ever reaching Pydantic's field collection (ADR-045 §1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from akgentic.tool.errors import RetriableError

# Runtime import, not a ``TYPE_CHECKING`` one: ``EditItem`` appears in the
# signature of ``workspace_multi_edit``, and pydantic-ai resolves a tool's
# annotations with ``get_type_hints`` against the defining module's globals when
# it builds the JSON schema.
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.models import MutationOutcome, MutationStatus
from akgentic.tool.workspace.write.params import (
    WorkspaceDelete,
    WorkspaceEdit,
    WorkspaceMkdir,
    WorkspaceMultiEdit,
    WorkspacePatch,
    WorkspaceWrite,
)


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


class WriteFactories:
    """The mutation factory bodies of :class:`WorkspaceTool`.

    Declares **no Pydantic field**: the annotations below are inside
    ``if TYPE_CHECKING:``, so they are never executed and never reach
    ``__annotations__``, which is where Pydantic v2 collects fields from across
    the MRO. mypy reads them normally.
    """

    if TYPE_CHECKING:
        # Supplied by ``CardGate``; the MRO binds them at runtime. Declared here
        # so mypy can check the calls, never defined.
        def apply_write(self, path: str, content: str) -> MutationOutcome: ...

        def apply_delete(self, path: str) -> MutationOutcome: ...

        def apply_edit(
            self, path: str, old_string: str, new_string: str, replace_all: bool = False
        ) -> MutationOutcome: ...

        def apply_multi_edit(self, edits: list[EditItem]) -> MutationOutcome: ...

        def apply_patch(self, patch_text: str) -> MutationOutcome: ...

        def apply_mkdir(self, path: str) -> MutationOutcome: ...

    def _write_factory(self, params: WorkspaceWrite) -> Callable[..., Any]:
        """Create the ``workspace_write`` tool callable.

        Args:
            params: Write capability configuration.

        Returns:
            Callable that writes content to a workspace file, through the actor.
        """
        card = self

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
            return _resolve_outcome(card.apply_write(path, content))

        workspace_write.__doc__ = params.format_docstring(workspace_write.__doc__)
        return workspace_write

    def _delete_factory(self, params: WorkspaceDelete) -> Callable[..., Any]:
        """Create the ``workspace_delete`` tool callable.

        Args:
            params: Delete capability configuration.

        Returns:
            Callable that deletes a file from the workspace, through the actor.
        """
        card = self

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
            return _resolve_outcome(card.apply_delete(path))

        workspace_delete.__doc__ = params.format_docstring(workspace_delete.__doc__)
        return workspace_delete

    def _edit_factory(self, params: WorkspaceEdit) -> Callable[..., Any]:
        """Create the ``workspace_edit`` tool callable.

        Args:
            params: Edit capability configuration.

        Returns:
            Callable that applies a surgical find-and-replace edit, through the actor.
        """
        card = self

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
            return _resolve_outcome(card.apply_edit(path, old_string, new_string, replace_all))

        workspace_edit.__doc__ = params.format_docstring(workspace_edit.__doc__)
        return workspace_edit

    def _multi_edit_factory(self, params: WorkspaceMultiEdit) -> Callable[..., Any]:
        """Create the ``workspace_multi_edit`` tool callable.

        Args:
            params: Multi-edit capability configuration.

        Returns:
            Callable that applies a batch of find-and-replace edits, through the actor.
        """
        card = self

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
            return _resolve_outcome(card.apply_multi_edit(edits))

        workspace_multi_edit.__doc__ = params.format_docstring(workspace_multi_edit.__doc__)
        return workspace_multi_edit

    def _patch_factory(self, params: WorkspacePatch) -> Callable[..., Any]:
        """Create the ``workspace_patch`` tool callable.

        Args:
            params: Patch capability configuration.

        Returns:
            Callable that applies a unified diff patch, through the actor.
        """
        card = self

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
            return _resolve_outcome(card.apply_patch(patch_text))

        workspace_patch.__doc__ = params.format_docstring(workspace_patch.__doc__)
        return workspace_patch

    def _mkdir_factory(self, params: WorkspaceMkdir) -> Callable[..., Any]:
        """Create the ``workspace_mkdir`` tool callable.

        Args:
            params: Mkdir capability configuration.

        Returns:
            Callable that creates a directory in the workspace, through the actor.
        """
        card = self

        def workspace_mkdir(path: str) -> str:
            """Create a directory and all missing parents in the team workspace.

            Args:
                path: Relative directory path from workspace root (e.g. "src/utils").

            Returns:
                Confirmation string "Created: <path>".

            Raises:
                RetriableError: If the path escapes the workspace root.
            """
            return _resolve_outcome(card.apply_mkdir(path))

        workspace_mkdir.__doc__ = params.format_docstring(workspace_mkdir.__doc__)
        return workspace_mkdir
