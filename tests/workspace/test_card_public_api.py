"""Import-path guard for the workspace card decomposition.

Written against the **un-moved** tree and confirmed green there, so what it
asserts is what the move preserved rather than what the move produced.

Two different resolution mechanisms have to keep working, and they fail
differently:

- ``akgentic.tool.workspace.<Name>`` — the façade the catalog names in
  ``model_type``. Covered by the ``__all__`` assertions.
- ``akgentic.tool.workspace.tool.<Name>`` — the literal string
  ``serialize_type()`` stamped into every persisted ``__model__`` marker, which
  the deserializer resolves with ``import_module`` plus ``getattr``. Rows
  written before the decomposition still carry it, in deployments this
  repository cannot see, so the module has to keep resolving even though no
  source file imports it any more. ``hasattr`` on the package would not catch
  its loss — hence the explicit ``import_module``.

The frozen sets below were captured from the working tree, not transcribed from
a design document.
"""

from __future__ import annotations

import importlib

import akgentic.tool
import akgentic.tool.workspace as ws

# Captured verbatim from ``akgentic/tool/workspace/__init__.py``'s ``__all__``
# before the move: 102 names, plus the four story 45-3 added and the six the
# workspace path resolver adds (ADR-048 Decision 5). The package ships
# on PyPI, so every one of them is a public import path somebody may already
# depend on.
_WORKSPACE_ALL: frozenset[str] = frozenset(
    {
        "ANONYMOUS",
        "BlockSplitter",
        "DEFAULT_EXEC_POLL_ATTEMPTS",
        "DEFAULT_EXEC_POLL_DELAY_S",
        "DEFAULT_EXEC_TIMEOUT_S",
        "DEFAULT_GIT_TIMEOUT_S",
        "DEFAULT_MAX_DOCUMENTS",
        "DEFAULT_MAX_DOCUMENT_CHARS",
        "DEFAULT_MAX_OBSERVATIONS_PER_AGENT",
        "DEFAULT_MAX_TRACKED_WRITERS",
        "DocumentExtract",
        "DocumentReader",
        "EMBEDDING_STALE_AFTER_S",
        "EXEC_REPORT_MARGIN_S",
        "EXTRACTOR_VERSION",
        "EditItem",
        "EditMatcher",
        "ExecConfig",
        "ExecOutcome",
        "ExecStart",
        "ExecState",
        "ExecStatus",
        "ExpandMediaRefs",
        "FileEntry",
        "FilePatch",
        "FileTypeReader",
        "Filesystem",
        "GITIGNORE_NAME",
        "GIT_DIR_SUFFIX",
        "GitJournal",
        "Hunk",
        "HunkContextError",
        "IDENTITY_DOMAIN",
        "IDENTITY_FALLBACK",
        "IN_MEMORY_MAX_DOCUMENTS",
        "IN_MEMORY_MAX_DOCUMENT_CHARS",
        "Identity",
        "LEASE_GRACE_S",
        "LastWrite",
        "MAX_COMMIT_BODY_CHARS",
        "MAX_EXEC_BUDGET_S",
        "MAX_QUEUED_RUNS",
        "MAX_REJECTION_DIFF_LINES",
        "MAX_TRACKED_RUNS",
        "MatchResult",
        "MediaContent",
        "MutationOutcome",
        "MutationStatus",
        "NewFileMessage",
        "OUT_OF_BAND_AUTHOR",
        "Observation",
        "PERM_ERR_MSG",
        "PUBLISH_LOST_MSG",
        "PathEscapeError",
        "Precondition",
        "QueuedExec",
        "RAG_COLLECTION",
        "RUN_ID_CHARS",
        "RagChunk",
        "RagFile",
        "RagFileRow",
        "RagIndexState",
        "RagStatus",
        "Resource",
        "ResourceType",
        "RunningExec",
        "STAGING_SWEEP_GRACE_S",
        "Span",
        "TEXT_EXTENSIONS",
        "TIMED_OUT_EXIT_CODE",
        "TextSplitter",
        "WORKSPACE_ACTOR_NAME",
        "WORKSPACE_ACTOR_ROLE",
        "WRITE_DENIED_MSG",
        "Workspace",
        "WorkspaceActor",
        "WorkspaceAttached",
        "WorkspaceConfig",
        "WorkspaceDelete",
        "WorkspaceEdit",
        "WorkspaceExec",
        "WorkspaceGlob",
        "WorkspaceGrep",
        "WorkspaceHost",
        "WorkspaceList",
        "WorkspaceMkdir",
        "WorkspaceMultiEdit",
        "WorkspacePatch",
        "WorkspaceRagIndex",
        "WorkspaceRagList",
        "WorkspaceRagSearch",
        "WorkspaceRead",
        "WorkspaceState",
        "WorkspaceTool",
        "WorkspaceView",
        "WorkspaceWrite",
        "WriteEntry",
        "apply_file_patch",
        "content_sha",
        "deleted_paths",
        "derived_document_caps",
        "detect_line_ending",
        "effective_budget",
        "format_outcome",
        "format_status",
        "METADATA_SCOPE",
        "RESERVED_SCOPES",
        "get_workspace",
        "leaf_segment",
        "resolve_workspace_path",
        "user_segment",
        "git_dir_for",
        "gitignore_seed",
        "hunk_header",
        "in_progress",
        "is_pure_add",
        "is_staging_name",
        "new_run_id",
        "normalise_endings",
        "parse_patch",
        "patch_label",
        "poll_attempts_within",
        "queue_full",
        "queued",
        "render_file_patch",
        "resolve_mode",
        "sanitise_command",
        "sanitise_email_local",
        "sanitise_name",
        "substitute_edit",
        "timed_out",
        "unified",
        "wait_out_the_turn",
        "workspace_actor_name",
        "write_and_diff",
    }
)

# Every public name ``akgentic.tool.workspace.tool`` defined before the move.
# These are the models a stored ``__model__`` marker can name: sixteen, not the
# fifteen the story's AC4 enumerates — ``WorkspaceWrite`` is a
# ``BaseToolParam`` like its five siblings and is persisted the same way.
_SHIM_NAMES: frozenset[str] = frozenset(
    {
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
    }
)

# Captured from ``WorkspaceTool.model_fields`` on the working tree. The mixins
# introduced by the decomposition declare no Pydantic fields, so this set must
# not grow by a single name.
_WORKSPACE_TOOL_FIELDS: frozenset[str] = frozenset(
    {
        "workspace_id",
        "workspace_metadata_keys",
        "workspace_read",
        "workspace_view",
        "workspace_list",
        "workspace_glob",
        "workspace_grep",
        "expand_media_refs",
        "read_only",
        "workspace_write",
        "workspace_delete",
        "workspace_edit",
        "workspace_multi_edit",
        "workspace_patch",
        "workspace_mkdir",
        "git_journal",
        "workspace_exec",
        "resources",
        # Added by story 45-7, deliberately: the two retrieval capabilities, the
        # collection they share, and the two explicit overrides of the caps that
        # otherwise follow the vector backend. Seventeen names became
        # twenty-two; the set exists so a field is never added by accident, not
        # so it can never grow.
        "workspace_rag_index",
        "workspace_rag_list",
        "vector_store",
        "max_documents",
        "max_document_chars",
        # Added by story 45-8, deliberately: the search capability. Twenty-two
        # names became twenty-three.
        "workspace_rag_search",
    }
)

_TOOL_MODULE = "akgentic.tool.workspace.tool"


class TestWorkspacePublicSurface:
    """``akgentic.tool.workspace`` keeps exporting exactly what it exported."""

    def test_every_name_in_all_resolves_on_the_package(self) -> None:
        """A name in ``__all__`` that no longer resolves is a broken import path."""
        for name in ws.__all__:
            assert hasattr(ws, name), f"{name} listed in __all__ but not importable"

    def test_all_is_the_frozen_set(self) -> None:
        """Neither a name added nor a name lost — the surface is the contract."""
        assert set(ws.__all__) == set(_WORKSPACE_ALL)

    def test_all_has_no_duplicates(self) -> None:
        """The frozen-set comparison above cannot see a name listed twice."""
        assert len(ws.__all__) == len(set(ws.__all__))

    def test_facade_and_package_expose_the_same_class(self) -> None:
        """``akgentic.tool.WorkspaceTool`` and the package's must be one object."""
        assert akgentic.tool.WorkspaceTool is ws.WorkspaceTool


class TestStoredModelMarkersStillResolve:
    """The deserializer's exact mechanism, on the exact recorded path."""

    def test_tool_module_still_imports(self) -> None:
        """``import_module`` on the pre-move path must not raise."""
        assert importlib.import_module(_TOOL_MODULE) is not None

    def test_every_persisted_model_name_resolves_through_the_tool_module(self) -> None:
        """``import_module`` + ``getattr``, as ``deserializer.import_class`` does it."""
        module = importlib.import_module(_TOOL_MODULE)
        for name in sorted(_SHIM_NAMES):
            assert getattr(module, name, None) is not None, (
                f"{_TOOL_MODULE}.{name} no longer resolves — every row persisted "
                f"before the decomposition carries that literal string"
            )

    def test_the_tool_module_serves_the_same_objects_as_the_package(self) -> None:
        """A second definition would deserialise into a class nothing else uses."""
        module = importlib.import_module(_TOOL_MODULE)
        for name in sorted(_SHIM_NAMES):
            assert getattr(module, name) is getattr(ws, name)


class TestWorkspaceToolFieldSet:
    """The card's Pydantic field set is frozen — mixins contribute none."""

    def test_field_names_are_exactly_the_frozen_set(self) -> None:
        """A field appearing on a mixin would show up here as an extra name."""
        assert set(ws.WorkspaceTool.model_fields) == set(_WORKSPACE_TOOL_FIELDS)

    def test_schema_lists_no_other_property(self) -> None:
        """The JSON schema is what a catalog payload is validated against."""
        schema = ws.WorkspaceTool.model_json_schema()
        assert set(schema["properties"]) == set(_WORKSPACE_TOOL_FIELDS)

    def test_catalog_payload_shape_round_trips(self) -> None:
        """A card built from a catalog payload survives dump and re-validation."""
        payload = {
            "workspace_id": "acme",
            "read_only": False,
            "workspace_read": {"default_limit": 500},
            "workspace_exec": {"timeout_s": 12.0, "poll_attempts": 3},
            "resources": [{"file_name": "README.md", "content": "hello"}],
        }
        card = ws.WorkspaceTool.model_validate(payload)
        again = ws.WorkspaceTool.model_validate(card.model_dump())
        assert again.model_dump() == card.model_dump()

    def test_empty_catalog_payload_round_trips(self) -> None:
        """``id_workspace.yaml`` ships ``payload: {}`` — defaults must suffice."""
        card = ws.WorkspaceTool.model_validate({})
        assert set(card.model_dump()) >= set(_WORKSPACE_TOOL_FIELDS)
