"""``receiveMsg_NewFileMessage``: the handler an upload addresses, and its seven rules.

This handler is reachable from **outside** the framework, which is what makes
most of this file about what it refuses to do. An exception on this mailbox turn
would kill the actor that owns the write gate for the whole team, so every
never-raises spec here is followed by a real mutation through the gate: an actor
that survived the message but lost the gate would otherwise pass.

The spawn harness is 45-7's, reused rather than reimplemented — this story writes
no second spawn path, so its specs should be able to watch the first one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.workspace.actor import WorkspaceActor
from akgentic.tool.workspace.documents.models import (
    NewFileMessage,
    RagChunk,
    RagFile,
    RagStatus,
)
from akgentic.tool.workspace.models import MutationStatus
from akgentic.tool.workspace.workspace import PathEscapeError

from tests.workspace.conftest import WORKSPACE_NAME
from tests.workspace.test_rag_pipeline import RagHarness, write
from tests.workspace.test_rag_search import build_actor

_MESSAGE_MODULE = "akgentic.tool.workspace.documents.models"


@pytest.fixture
def upload(workspace_tree: Path, monkeypatch: pytest.MonkeyPatch) -> RagHarness:
    """A harness with retrieval enabled — the case an upload normally lands in."""
    built = RagHarness(build_actor())
    built.install(monkeypatch)
    built.enable()
    return built


@pytest.fixture
def upload_without_retrieval(workspace_tree: Path, monkeypatch: pytest.MonkeyPatch) -> RagHarness:
    """The same harness with no ``enable_rag`` — retrieval was never turned on."""
    built = RagHarness(build_actor())
    built.install(monkeypatch)
    return built


def gate_still_works(actor: WorkspaceActor, probe: str) -> bool:
    """Whether a real mutation still commits through this actor.

    Not decoration: the failure this exists to catch is an actor that survived a
    malformed message and lost the thing it exists for.
    """
    outcome = actor.apply_write("gate-probe", probe, "still here\n")
    return outcome.status is MutationStatus.ACCEPTED


class TestTheMessageModel:
    """AC13 — where it lives, what it carries, and what it deliberately is not."""

    def test_its_module_path_is_the_one_persisted_payloads_will_name(self) -> None:
        """``serialize()`` stamps ``__model__`` from here; nothing moves afterwards."""
        assert NewFileMessage.__module__ == _MESSAGE_MODULE

    def test_it_is_a_serializable_model_and_not_a_framework_message(self) -> None:
        """``Akgent.on_receive`` emits its telemetry sandwich only for ``Message``."""
        assert issubclass(NewFileMessage, SerializableBaseModel)
        assert "Message" not in {base.__name__ for base in NewFileMessage.__mro__[1:]}

    def test_it_carries_exactly_three_fields_with_the_documented_defaults(self) -> None:
        assert set(NewFileMessage.model_fields) == {"paths", "source", "force"}
        built = NewFileMessage(paths=["a.md"])
        assert (built.source, built.force) == ("upload", False)

    def test_it_round_trips_through_pydantic(self) -> None:
        """Golden Rule #1b — every field is serialisable, so a payload survives."""
        original = NewFileMessage(paths=["a.md", "b.txt"], source="frontend", force=True)

        assert NewFileMessage.model_validate(original.model_dump()) == original


class TestItIsATellThatNeverRaises:
    """AC14, AC15, AC16 — the four-row matrix, each followed by a live gate."""

    def test_the_handler_returns_none(self, upload: RagHarness, workspace_tree: Path) -> None:
        """The frontend's upload must not wait on a 500-page extraction."""
        write(workspace_tree, "a.md")

        assert upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"])) is None

    def test_an_escaping_path_is_skipped_and_the_gate_survives(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """``PathEscapeError`` is a ``PermissionError``, which ``_digest`` absorbs."""
        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["../../etc/passwd.md"]))

        assert upload.actor.state.rag_index == {}
        assert upload.requests == []
        assert gate_still_works(upload.actor, "after-escape.md")

    def test_a_missing_path_is_skipped_and_the_gate_survives(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """The message races the upload's own write: missing means "not yet"."""
        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["never-written.md"]))

        assert upload.actor.state.rag_index == {}
        assert gate_still_works(upload.actor, "after-missing.md")

    def test_an_unsupported_type_is_skipped_and_the_gate_survives(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """The candidate set is the read path's own line, not a second list."""
        (workspace_tree / "archive.zip").write_bytes(b"PK\x03\x04")
        (workspace_tree / "photo.png").write_bytes(b"\x89PNG")

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["archive.zip", "photo.png"]))

        assert upload.actor.state.rag_index == {}
        assert gate_still_works(upload.actor, "after-unsupported.md")

    @pytest.mark.parametrize(
        "payload",
        [
            {"paths": []},
            {"paths": None},
            {"paths": ["a.md", 3, None]},
            {"paths": "a.md"},
        ],
    )
    def test_a_malformed_payload_is_absorbed_and_the_gate_survives(
        self, upload: RagHarness, workspace_tree: Path, payload: dict[str, Any]
    ) -> None:
        """``model_construct`` is how a payload from outside arrives unvalidated."""
        upload.actor.receiveMsg_NewFileMessage(NewFileMessage.model_construct(**payload))

        assert gate_still_works(upload.actor, f"after-{len(str(payload))}.md")

    def test_one_unusable_member_does_not_abort_the_rest_of_the_batch(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """Per-path absorption, so an upload of ten files is not lost to one of them."""
        write(workspace_tree, "good.md")

        upload.actor.receiveMsg_NewFileMessage(
            NewFileMessage.model_construct(paths=[3, "good.md"], source="upload", force=False)
        )

        assert "good.md" in upload.actor.state.rag_index

    def test_a_broken_spawn_path_never_reaches_the_caller(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """``_spawn`` wraps its own failure; this asserts the handler agrees."""
        write(workspace_tree, "a.md")
        upload.spawn_error = RuntimeError("the actor system is gone")

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))  # must not raise

        assert gate_still_works(upload.actor, "after-spawn-failure.md")

    def test_a_failure_in_the_queueing_half_never_reaches_the_caller(
        self, upload: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wrapper covers the whole body, not only the path validation.

        The queueing half has no ``try`` of its own — ``_enqueue`` and
        ``_is_accounted_for`` are plain dict work — so this is the handler's own
        wrapper or nothing.
        """
        write(workspace_tree, "a.md")

        def boom(*args: Any, **kwargs: Any) -> bool:
            raise RuntimeError("the index moved underneath us")

        monkeypatch.setattr(upload.actor, "_is_accounted_for", boom)

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))  # must not raise

        assert gate_still_works(upload.actor, "after-queueing-failure.md")


class TestEveryPathIsValidated:
    """AC17 — through ``Filesystem``, and never ``backend._root / path``."""

    def test_a_real_file_outside_the_root_is_never_indexed(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """The bypass rows 35 / 47(c) record, on the most escape-prone new surface.

        The file genuinely exists and is genuinely of an indexable type, so nothing
        but the validation stands between it and the corpus. Joining onto the
        backend's private root instead would index it.
        """
        outside = workspace_tree.parent / "outside.md"
        outside.write_text("secrets\n", encoding="utf-8")

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["../outside.md"]))

        assert upload.actor.state.rag_index == {}
        assert upload.requests == []

    def test_the_validation_raises_a_permission_error_the_digest_absorbs(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """Confirmed rather than assumed: the whole skip depends on this subclassing."""
        assert issubclass(PathEscapeError, PermissionError)
        assert issubclass(PathEscapeError, OSError)

        with pytest.raises(PathEscapeError):
            upload.actor._workspace.read("../outside.md")


class TestIdempotence:
    """AC18 — a repeat at the live digest is a no-op, and ``force`` is the override."""

    def test_the_same_message_twice_is_one_index_run(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        write(workspace_tree, "a.md")
        message = NewFileMessage(paths=["a.md"])

        upload.actor.receiveMsg_NewFileMessage(message)
        upload.actor.receiveMsg_NewFileMessage(message)

        assert len(upload.requests) == 1

    def test_a_repeat_at_different_bytes_does_index_again(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """It is the digest that makes it idempotent, not the path."""
        write(workspace_tree, "a.md", "first\n")
        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        upload.report("a.md")
        write(workspace_tree, "a.md", "second\n")

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        assert len(upload.requests) == 2

    def test_force_re_indexes_a_file_that_is_already_current(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        write(workspace_tree, "a.md")
        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        upload.report("a.md")

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"], force=True))

        assert len(upload.requests) == 2


class TestTheAsymmetryWithTheGate:
    """AC19 — an upload indexes where a gate write only marks ``STALE``.

    Asserted in one place on purpose: the two halves are a deliberate decision
    rather than an omission, and a future editor collapsing them would otherwise
    turn one spec green while breaking the other.
    """

    def test_an_upload_queues_and_a_gate_write_only_marks_stale(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        write(workspace_tree, "uploaded.md")
        write(workspace_tree, "written.md")
        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["written.md"]))
        upload.report("written.md")
        upload.complete("written.md")
        assert upload.actor.state.rag_index["written.md"].status is RagStatus.EMBEDDED

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["uploaded.md"]))
        upload.actor.mark_paths_stale(["written.md"])

        assert upload.actor.state.rag_index["uploaded.md"].status is not RagStatus.STALE
        assert upload.actor.state.rag_index["written.md"].status is RagStatus.STALE

    def test_a_gate_write_spawns_no_worker_of_its_own(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """An agent mid-task rewrites the same file repeatedly; each save is free."""
        write(workspace_tree, "written.md")
        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["written.md"]))
        upload.report("written.md")
        upload.complete("written.md")
        before = len(upload.requests)

        upload.actor.mark_paths_stale(["written.md"])

        assert len(upload.requests) == before


class TestTheSpawnPathIsReused:
    """AC20 — validate, hash, set ``PENDING``, spawn, return, and nothing else."""

    def test_the_concurrency_cap_is_the_one_45_7_shipped(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """Files beyond the cap stay ``PENDING`` — which is what ``PENDING`` means."""
        from akgentic.tool.workspace.documents.worker import MAX_CONCURRENT_INDEX_WORKERS

        names = [f"f{index}.md" for index in range(MAX_CONCURRENT_INDEX_WORKERS + 2)]
        for name in names:
            write(workspace_tree, name, f"# {name}\n")

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=names))

        assert len(upload.requests) == MAX_CONCURRENT_INDEX_WORKERS
        pending = [
            path
            for path, entry in upload.actor.state.rag_index.items()
            if entry.status is RagStatus.PENDING
        ]
        assert len(pending) == 2

    def test_the_worker_is_handed_the_workspace_as_its_scope(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        write(workspace_tree, "a.md")

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        assert upload.requests[0].scope == WORKSPACE_NAME

    def test_queueing_notifies_once_and_a_message_that_queued_nothing_notifies_not_at_all(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        """The notify follows the queueing, so an unusable notification is free."""
        write(workspace_tree, "a.md")
        spy = upload.watch()

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["nope.zip"]))
        assert spy.notifications == []

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        assert len(spy.notifications) == 1


class TestTheCapabilityRefusal:
    """AC21 — with retrieval off it records and spends nothing."""

    def test_it_records_pending_rows_and_spawns_no_worker(
        self, upload_without_retrieval: RagHarness, workspace_tree: Path
    ) -> None:
        """An upload must not spend embedding credits in a team that never opted in."""
        write(workspace_tree, "a.md")

        upload_without_retrieval.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        assert upload_without_retrieval.actor.state.rag_index["a.md"].status is RagStatus.PENDING
        assert upload_without_retrieval.requests == []
        assert upload_without_retrieval.vs.calls == []

    def test_enabling_retrieval_afterwards_picks_the_recorded_files_up(
        self, upload_without_retrieval: RagHarness, workspace_tree: Path
    ) -> None:
        """The whole sequence, end to end — which is what makes ``PENDING`` the right row."""
        write(workspace_tree, "a.md")
        upload_without_retrieval.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        upload_without_retrieval.enable()
        upload_without_retrieval.actor.index_paths("")

        assert [request.path for request in upload_without_retrieval.requests] == ["a.md"]

    def test_a_half_enabled_tree_records_rather_than_spawning(
        self, upload_without_retrieval: RagHarness, workspace_tree: Path
    ) -> None:
        """The reachable half-enabled state, and the one this branch really buys.

        ``enable_rag`` sets the chunking parameters and *then* tries to acquire the
        proxy, so a workspace whose ``#VectorStore`` is missing or whose
        ``create_collection`` failed ends up with parameters and no proxy — the
        state ``test_a_missing_vector_store_degrades_rather_than_raising`` pins.
        Treating that as "retrieval is on" would spawn workers whose ``add()``
        calls go nowhere, leaving every file at ``EMBEDDING`` until the reaper
        queues it again — and again, every ten minutes, for ever. Both halves are
        required, which is why the branch tests both.
        """
        half = upload_without_retrieval
        half.vs_address = None
        half.enable()
        assert half.actor._rag_params is not None
        assert half.actor._vs_proxy is None
        write(workspace_tree, "a.md")

        half.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        assert half.actor.state.rag_index["a.md"].status is RagStatus.PENDING
        assert half.requests == []
        assert half.worker_names == []

    def test_the_recorded_row_carries_the_digest_it_was_recorded_at(
        self, upload_without_retrieval: RagHarness, workspace_tree: Path
    ) -> None:
        """Without it, enabling later could not tell the file from an unread one."""
        sha = write(workspace_tree, "a.md")

        upload_without_retrieval.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        assert upload_without_retrieval.actor.state.rag_index["a.md"].indexed_sha == sha

    def test_a_row_already_current_survives_a_second_notification(
        self, upload_without_retrieval: RagHarness, workspace_tree: Path
    ) -> None:
        """AC18's idempotence covers this path too, and here it protects data.

        ``WorkspaceState`` is persisted, so a resume whose ``#VectorStore`` is
        missing restores ``EMBEDDED`` rows onto a tree with no proxy. Re-queueing
        one would move its chunk ids into ``superseded_chunk_ids`` that no proxy
        will ever remove, drop the heading paths a search renders, and buy a
        re-embedding of content that was already embedded.
        """
        harness = upload_without_retrieval
        sha = write(workspace_tree, "a.md")
        harness.actor.state.rag_index["a.md"] = _embedded_row("a.md", sha)

        harness.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        row = harness.actor.state.rag_index["a.md"]
        assert row.status is RagStatus.EMBEDDED
        assert [chunk.chunk_id for chunk in row.chunks] == ["chunk-0"]
        assert row.superseded_chunk_ids == []

    def test_force_re_records_a_row_that_is_already_current(
        self, upload_without_retrieval: RagHarness, workspace_tree: Path
    ) -> None:
        """``force`` means the same thing on both sides of the capability check."""
        harness = upload_without_retrieval
        sha = write(workspace_tree, "a.md")
        harness.actor.state.rag_index["a.md"] = _embedded_row("a.md", sha)

        harness.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"], force=True))

        row = harness.actor.state.rag_index["a.md"]
        assert row.status is RagStatus.PENDING
        assert row.superseded_chunk_ids == ["chunk-0"]


def _embedded_row(path: str, sha: str) -> RagFile:
    """A row as a resume restores it: ``EMBEDDED``, at *sha*, carrying its chunks."""
    return RagFile(
        path=path,
        status=RagStatus.EMBEDDED,
        indexed_sha=sha,
        chunks=[RagChunk(chunk_id="chunk-0", ordinal=0, start=0, end=8, heading_path=["Title"])],
        chunk_count=1,
        updated_at=datetime.now(UTC),
    )


class _RagFileWithExtraField(RagFile):
    """A ``RagFile`` carrying a field this story's write paths have never heard of.

    45-7's guard, extended to the transitions this story adds. The subclass is
    the whole of the mechanism: a whole-model comparison can only compare fields
    that exist today, so an enumerated rebuild naming every one of them passes it
    green and destroys the field added tomorrow in silence. An enumerated
    ``RagFile(...)`` returns a plain ``RagFile`` and cannot carry this at all.
    """

    extra_field: str = "sentinel"


class TestUploadTransitionsAreCopies:
    """AC31 / Golden Rule #12, on the two write paths this story adds."""

    def _seed(self, harness: RagHarness, path: str) -> None:
        harness.actor.state.rag_index[path] = _RagFileWithExtraField(
            path=path,
            status=RagStatus.EMBEDDED,
            indexed_sha="an-older-digest",
            updated_at=datetime.now(UTC),
        )

    def test_the_upload_queue_preserves_an_unknown_field(
        self, upload: RagHarness, workspace_tree: Path
    ) -> None:
        sha = write(workspace_tree, "a.md")
        self._seed(upload, "a.md")

        upload.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        result = upload.actor.state.rag_index["a.md"]
        assert result.indexed_sha == sha
        assert isinstance(result, _RagFileWithExtraField)
        assert result.extra_field == "sentinel"

    def test_the_capability_refusal_preserves_an_unknown_field(
        self, upload_without_retrieval: RagHarness, workspace_tree: Path
    ) -> None:
        """The second write path, which reaches ``_enqueue`` by a different route."""
        sha = write(workspace_tree, "a.md")
        self._seed(upload_without_retrieval, "a.md")

        upload_without_retrieval.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))

        result = upload_without_retrieval.actor.state.rag_index["a.md"]
        assert result.status is RagStatus.PENDING
        assert isinstance(result, _RagFileWithExtraField)
        assert result.extra_field == "sentinel"

    def test_the_guard_is_not_vacuous(self) -> None:
        assert "extra_field" in _RagFileWithExtraField.model_fields
        assert "extra_field" not in RagFile.model_fields
