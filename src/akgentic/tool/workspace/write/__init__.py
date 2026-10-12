"""The write capability of :class:`WorkspaceTool` — write, delete, edit, patch, mkdir, rmdir.

It holds one capability whole: the seven LLM-facing closures here, the mutation
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
the seven closures are what an LLM sees, and the gate is what they converge on.
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

# Runtime import, not a ``TYPE_CHECKING`` one, for the same reason as ``EditItem``
# below: the command registry resolves ``/workspace_write``'s annotations with
# ``get_type_hints`` against this module's globals.
from akgentic.tool.core.commands import RestOfLine
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
    WorkspaceRmdir,
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

        def apply_rmdir(
            self,
            path: str,
            recursive: bool = False,
            force: bool = False,
            *,
            empty_only: bool = False,
        ) -> MutationOutcome: ...

        def human_write(self, path: str, content: str) -> MutationOutcome: ...

        def human_delete(self, path: str) -> MutationOutcome: ...

        def human_mkdir(self, path: str) -> MutationOutcome: ...

        def human_rmdir(self, path: str, recursive: bool) -> MutationOutcome: ...

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

            No prior read is required. If you have read the file, the delete
            is refused when it changed or was deleted since you read it, or
            when you read only part of it — deleting a file someone else has
            just rewritten destroys work you never saw.

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

    def _rmdir_factory(self, params: WorkspaceRmdir) -> Callable[..., Any]:
        """Create the ``workspace_rmdir`` tool callable.

        **Two closures, one returned**, chosen by ``allow_recursive``. pydantic-ai
        builds the schema from the signature, so a conditional default would
        still put ``recursive`` and ``force`` in front of the model on a
        deployment that has turned them off.

        Args:
            params: Rmdir capability configuration.

        Returns:
            Callable that removes a directory from the workspace, through the gate.
        """
        card = self

        def workspace_rmdir(path: str, recursive: bool = False, force: bool = False) -> str:
            """Remove a directory from the team workspace.

            Without recursive, only an empty directory is removed. Set
            recursive=true ONLY when the user has explicitly asked to remove the
            directory together with its contents.

            Refused if a file you have read under the directory changed since
            you read it — no file is removed. Set force=true ONLY when the user
            has explicitly asked to discard changes made since your last read.
            Never set it to get past a refusal on your own initiative.

            Args:
                path: Relative directory path from workspace root (e.g. "build").
                recursive: Remove the directory and everything under it.
                force: Remove it even if files under it changed since you read
                    them. Requires recursive=true.

            Returns:
                Confirmation string "Removed: <path>".

            Raises:
                RetriableError: If the removal is refused, the path is not a
                    directory, does not exist, or escapes the workspace root.
            """
            return _resolve_outcome(card.apply_rmdir(path, recursive, force))

        def workspace_rmdir_empty_only(path: str) -> str:
            """Remove an empty directory from the team workspace.

            A directory that still contains anything is refused.

            Args:
                path: Relative directory path from workspace root (e.g. "build").

            Returns:
                Confirmation string "Removed: <path>".

            Raises:
                RetriableError: If the removal is refused, the path is not an
                    empty directory, does not exist, or escapes the workspace root.
            """
            return _resolve_outcome(card.apply_rmdir(path, empty_only=True))

        chosen: Callable[..., str]
        if params.allow_recursive:
            chosen = workspace_rmdir
        else:
            chosen = workspace_rmdir_empty_only
            chosen.__name__ = chosen.__qualname__ = "workspace_rmdir"
        chosen.__doc__ = params.format_docstring(chosen.__doc__)
        return chosen

    ##
    ## The slash commands — the human's hand on the tree. Each closure is named
    ## exactly after its command, because the registry keys on ``__name__``, and
    ## each goes through the gate's human mode: no view check, no observation
    ## recorded, the commit attributed to ``human``. A refusal is returned as the
    ## result text rather than raised — the human reads it, and so does the agent.
    ##
    def _write_command_factory(self, params: WorkspaceWrite) -> Callable[..., Any]:
        """Create the ``/workspace_write`` command.

        Args:
            params: Write capability configuration.

        Returns:
            Callable named ``workspace_write`` that writes the rest of the line.
        """
        card = self

        def workspace_write(filepath: str, content: RestOfLine) -> str:
            """Create or overwrite a file with everything after its path, as UTF-8.

            The text is written exactly as typed, quotes and line breaks included,
            and any extension is accepted. Missing directories are created.

            Example: /workspace_write notes.md Remember to call the supplier.
            """
            return card.human_write(filepath, content).message

        return workspace_write

    def _delete_command_factory(self, params: WorkspaceDelete) -> Callable[..., Any]:
        """Create the ``/workspace_delete`` command.

        Args:
            params: Delete capability configuration.

        Returns:
            Callable named ``workspace_delete`` that deletes one file.
        """
        card = self

        def workspace_delete(filepath: str) -> str:
            """Delete one file from the team workspace.

            Example: /workspace_delete drafts/old.md
            """
            return card.human_delete(filepath).message

        return workspace_delete

    def _mkdir_command_factory(self, params: WorkspaceMkdir) -> Callable[..., Any]:
        """Create the ``/workspace_mkdir`` command.

        Args:
            params: Mkdir capability configuration.

        Returns:
            Callable named ``workspace_mkdir`` that creates a directory.
        """
        card = self

        def workspace_mkdir(path: str) -> str:
            """Create a directory, and any missing parents, in the team workspace.

            Example: /workspace_mkdir reports/2026
            """
            return card.human_mkdir(path).message

        return workspace_mkdir

    def _rmdir_command_factory(self, params: WorkspaceRmdir) -> Callable[..., Any]:
        """Create the ``/workspace_rmdir`` command.

        The human typing it is the explicit request the tool's docstring asks the
        model to wait for, so it removes the whole tree whenever the deployment
        allows recursive removal, and empty directories only when it does not.

        Args:
            params: Rmdir capability configuration.

        Returns:
            Callable named ``workspace_rmdir`` that removes a directory.
        """
        card = self
        recursive = params.allow_recursive

        def workspace_rmdir(path: str) -> str:
            """Remove a directory and everything under it from the team workspace.

            Where the workspace removes empty directories only, a directory that
            still holds anything is refused.

            Example: /workspace_rmdir build
            """
            return card.human_rmdir(path, recursive).message

        return workspace_rmdir
