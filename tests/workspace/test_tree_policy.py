"""Story 55-5: the tree owns its retrieval policy, and a second binder is held to it.

Chunking parameters and the two document caps were per-actor state governing a
tree two teams share. Two teams get two actors, one ``<meta>/index/`` and one
``scope=``, and ``chunk_id(scope, path, source_sha, ordinal)`` carries no team —
so both chunkings minted rows into one collection and ``_render_hit``'s
``chunks[ordinal]`` lookup resolved a hit minted under one chunking against
offsets stored under the other. ADR-051 Decision 1 and the workspace shard both
claimed the actor "carries no state two instances could disagree about"; this is
what makes that true.

**Which specs spawn a second interpreter, and which honestly do not.** The
refusals are cross-process, and they have to be: the publishing process **exits**
before the second binds, so the record on disk is the only channel that could
carry the decision. A single-process spec would stay green over a module-level
dict. The record's shape, the placement, the bad-parse refusal and the
does-it-create-anything questions are one-process — spawning a subprocess to
prove a filter or a file name would be ceremony.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from akgentic.tool.workspace.rag import (
    LOCKS_DIR_NAME,
    POLICY_FILE_NAME,
    POLICY_LOCK_NAME,
    WORKSPACE_POLICY_REFUSED,
    WORKSPACE_POLICY_UNREADABLE,
    TreePolicy,
    agreed_tree_policy,
    policy_file_for,
    read_tree_policy,
)
from akgentic.tool.workspace.rag.params import WorkspaceRagIndex
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import meta_dir_for
from tests.workspace.conftest import (
    CHILD_TIMEOUT_S,
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    run_child,
    start_child,
    write_script,
)


def bind(orchestrator_proxy: FakeOrchestratorProxy, **card_kwargs: object) -> WorkspaceTool:
    """Wire a card onto the test workspace, exactly as an agent's bind does."""
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, **card_kwargs)  # type: ignore[arg-type]
    card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
    return card


def policy_on_disk() -> TreePolicy | None:
    """Read the published record back, through the shipped reader."""
    return read_tree_policy(WORKSPACE_PATH, "TestReader")


##
## AC 1 — the record is one file under ``<meta>``, and the chunking is stored whole
##


class TestTheRecordIsOneFileBesideTheTree:
    """Where it lives, what it carries, and what it deliberately does not."""

    def test_it_carries_the_chunking_whole_and_the_two_caps(self) -> None:
        """Three sections. A fourth would be a model this story invented."""
        assert set(TreePolicy.model_fields) == {
            "chunking",
            "max_documents",
            "max_document_chars",
        }

    def test_every_section_is_absent_by_default(self) -> None:
        """*Absent* and *declared at today's default* have to stay distinguishable
        — the distinction ``VectorStoreParam.backend_declared`` exists to keep."""
        policy = TreePolicy()
        assert policy.chunking is None
        assert policy.max_documents is None
        assert policy.max_document_chars is None

    def test_the_chunking_section_is_the_shipped_parameter_itself(self) -> None:
        """Never its fields enumerated: a chunking field added tomorrow has to
        take part in the comparison by construction (Golden Rule 12's shape)."""
        policy = TreePolicy(chunking=WorkspaceRagIndex(chunk_chars=999))
        assert isinstance(policy.chunking, WorkspaceRagIndex)

        restored = TreePolicy.model_validate(policy.model_dump(mode="json"))

        assert restored.chunking is not None
        assert restored.chunking.chunk_chars == 999

    def test_it_is_yaml_at_the_top_level_of_the_metadata_directory(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Beside ``exec.lock`` — not in ``rag/`` (one file per *document*) and not
        in ``locks/`` (lock files only). YAML because a human debugging a refused
        bind reads it."""
        bind(orchestrator_proxy, workspace_rag_index=True)

        record = meta_dir_for(WORKSPACE_PATH) / POLICY_FILE_NAME
        assert record.parent == meta_dir_for(WORKSPACE_PATH)
        assert record == policy_file_for(WORKSPACE_PATH)
        loaded = yaml.safe_load(record.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        assert set(loaded) == set(TreePolicy.model_fields) | {"__model__"}

    def test_the_lock_lives_in_the_existing_locks_family(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """One more file under ``<meta>/locks/``, not a new directory."""
        bind(orchestrator_proxy, workspace_rag_index=True)

        assert (meta_dir_for(WORKSPACE_PATH) / LOCKS_DIR_NAME / POLICY_LOCK_NAME).is_file()

    def test_neither_file_is_reachable_from_inside_the_tree(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Placement is a containment rule: a policy inside the tree would be
        readable by a read capability and deletable by a sandboxed ``rm -rf``."""
        bind(orchestrator_proxy, workspace_rag_index=True)

        assert not policy_file_for(WORKSPACE_PATH).is_relative_to(workspace_tree.resolve())
        assert list(workspace_tree.iterdir()) == []


##
## AC 2 — agreement, and the refusal that names both values
##


class TestAgreementSectionBySection:
    """The three answers per section, on the pure function the bind gate calls."""

    def test_an_empty_tree_takes_what_the_binder_declares(self) -> None:
        declared = TreePolicy(chunking=WorkspaceRagIndex(), max_documents=7)

        assert agreed_tree_policy(None, declared, WORKSPACE_PATH, "WorkspaceTool") == declared

    def test_an_equal_section_is_silent(self) -> None:
        published = TreePolicy(chunking=WorkspaceRagIndex())

        agreed = agreed_tree_policy(published, published, WORKSPACE_PATH, "WorkspaceTool")

        assert agreed == published

    def test_an_absent_section_is_nothing_to_disagree_with_and_is_filled(self) -> None:
        """A tree that has never heard of a cap is not in conflict with one."""
        published = TreePolicy(chunking=WorkspaceRagIndex())
        declared = TreePolicy(chunking=WorkspaceRagIndex(), max_documents=7)

        agreed = agreed_tree_policy(published, declared, WORKSPACE_PATH, "WorkspaceTool")

        assert agreed.max_documents == 7
        assert agreed.chunking == published.chunking

    def test_a_section_this_card_leaves_absent_asks_for_nothing(self) -> None:
        """A retrieval-off card declares no chunking, so it can never contest one."""
        published = TreePolicy(chunking=WorkspaceRagIndex(chunk_chars=999), max_documents=7)

        agreed = agreed_tree_policy(published, TreePolicy(), WORKSPACE_PATH, "WorkspaceTool")

        assert agreed == published

    def test_a_differing_chunking_field_the_comparison_never_names_still_refuses(self) -> None:
        """**The whole-model comparison is the point.** ``chunk_overlap_chars`` is
        named nowhere in the policy code; it takes part because the section is one
        model compared whole, which is what makes a field added tomorrow safe."""
        published = TreePolicy(chunking=WorkspaceRagIndex())
        declared = TreePolicy(chunking=WorkspaceRagIndex(chunk_overlap_chars=17))

        with pytest.raises(ValueError, match="chunking"):
            agreed_tree_policy(published, declared, WORKSPACE_PATH, "WorkspaceTool")

    def test_a_differing_cap_refuses_too(self) -> None:
        """One record, one lock, one refusal — the caps are not advisory."""
        published = TreePolicy(max_documents=8)
        declared = TreePolicy(max_documents=32)

        with pytest.raises(ValueError, match="max_documents"):
            agreed_tree_policy(published, declared, WORKSPACE_PATH, "WorkspaceTool")

    def test_the_refusal_names_both_values_and_is_composed_from_the_constant(self) -> None:
        """Asserted through the module constant, never a hand-typed sentence."""
        published = TreePolicy(max_documents=8)
        declared = TreePolicy(max_documents=32)

        with pytest.raises(ValueError) as refusal:
            agreed_tree_policy(published, declared, WORKSPACE_PATH, "WorkspaceTool")

        assert str(refusal.value) == WORKSPACE_POLICY_REFUSED.format(
            card="WorkspaceTool",
            path=WORKSPACE_PATH,
            section="max_documents",
            card_value=32,
            tree_value=8,
            file=policy_file_for(WORKSPACE_PATH),
        )
        assert "32" in str(refusal.value)
        assert "8" in str(refusal.value)


class TestASecondProcessIsHeldToWhatTheFirstPublished:
    """**Two interpreters, because the publisher exits before the second binds.**

    A single-process formulation of these would stay green over a module-level
    dict, an actor attribute or a class variable — none of which is what the epic
    is about. The publishing child is gone by the time the parent binds, so the
    record on disk is the only channel that could be carrying the decision.
    """

    _PUBLISH = """
        from akgentic.tool.workspace.rag.params import WorkspaceRagIndex

        card = bind("first", workspace_rag_index=WorkspaceRagIndex(chunk_chars=777))
        print("published", flush=True)
        """

    def _publish_from_a_child(self, tmp_path: Path, workspaces_root: Path) -> None:
        """Bind a retrieval card in a second interpreter and let it exit."""
        script = write_script(tmp_path, "publisher.py", self._PUBLISH)
        report = run_child(script, workspaces_root)
        assert report.code == 0, report
        assert report.out == "published", report

    def test_the_record_outlives_the_process_that_published_it(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """The positive control, and it is mandatory: a child that published
        nothing would make every refusal below pass for the wrong reason."""
        self._publish_from_a_child(tmp_path, workspaces_root)

        published = policy_on_disk()

        assert published is not None
        assert published.chunking == WorkspaceRagIndex(chunk_chars=777)

    def test_a_second_binder_with_different_chunking_is_refused(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """B5, stated as the thing it is. Silently winning, silently losing and
        merging are three spellings of one defect; this is the fourth answer."""
        self._publish_from_a_child(tmp_path, workspaces_root)

        with pytest.raises(ValueError) as refusal:
            bind(orchestrator_proxy, workspace_rag_index=WorkspaceRagIndex(chunk_chars=1200))

        assert "chunking" in str(refusal.value)
        assert "777" in str(refusal.value)
        assert "1200" in str(refusal.value)

    def test_a_second_binder_agreeing_binds_and_writes_nothing(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """The other positive control: a gate that refused *everything* would pass
        the two refusals above while making the tree unbindable."""
        self._publish_from_a_child(tmp_path, workspaces_root)
        before = policy_file_for(WORKSPACE_PATH).stat().st_mtime_ns

        bind(orchestrator_proxy, workspace_rag_index=WorkspaceRagIndex(chunk_chars=777))

        assert policy_file_for(WORKSPACE_PATH).stat().st_mtime_ns == before

    def test_a_second_binder_disagreeing_about_a_cap_is_refused(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """The caps share the record, the lock and the refusal. "Merge", "max
        wins" and "advisory" are all the silently-losing case."""
        script = write_script(
            tmp_path,
            "cap_publisher.py",
            """
            card = bind("first", workspace_rag_index=True, max_documents=8)
            print("published", flush=True)
            """,
        )
        assert run_child(script, workspaces_root).code == 0

        with pytest.raises(ValueError) as refusal:
            bind(orchestrator_proxy, workspace_rag_index=True, max_documents=64)

        assert "max_documents" in str(refusal.value)
        assert "64" in str(refusal.value)

    def test_a_retrieval_off_card_is_held_to_the_caps_too(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """It publishes nothing and still reads: the caps govern the extraction
        cache, which the **read** path fills with no retrieval anywhere."""
        script = write_script(
            tmp_path,
            "cap_publisher.py",
            """
            card = bind("first", workspace_rag_index=True, max_document_chars=1000)
            print("published", flush=True)
            """,
        )
        assert run_child(script, workspaces_root).code == 0

        with pytest.raises(ValueError, match="max_document_chars"):
            bind(orchestrator_proxy, max_document_chars=2000)

    def test_a_refused_bind_creates_no_actor_and_emits_no_event(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """The gate runs before ``_seed_resources`` and ``_bind_workspace_actor``,
        the ordering discipline the sharing gate already states."""
        self._publish_from_a_child(tmp_path, workspaces_root)
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME, workspace_rag_index=WorkspaceRagIndex(chunk_chars=1200)
        )
        observer = FakeActorToolObserver(orchestrator_proxy, name="alice")

        with pytest.raises(ValueError):
            card.observer(observer)

        assert orchestrator_proxy.create_calls == []
        assert observer.events == []


##
## AC 3 — only a retrieval binder publishes; every binder reads
##


class TestOnlyABinderWithSomethingToDeclarePublishes:
    """A bind with nothing to publish must leave the disk exactly as it found it."""

    def test_a_default_bind_creates_no_metadata_directory_at_all(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The same claim ``test_the_tree_and_its_metadata_sibling_hold_exactly_this``
        makes, restated here beside the code that could break it. If it reddens, a
        bind with nothing to declare is publishing — a defect, never a row to
        widen."""
        bind(orchestrator_proxy)

        assert not meta_dir_for(WORKSPACE_PATH).exists()

    def test_a_read_only_bind_creates_nothing_either(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Reading an absent record is one ``stat`` — no ``mkdir``, no touch."""
        bind(orchestrator_proxy, workspace_read=True, workspace_write=False)

        assert not meta_dir_for(WORKSPACE_PATH).exists()

    def test_a_retrieval_off_card_declaring_caps_publishes_nothing(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """**The stated residual hole, guarded so it stays the shape it was
        argued for.** Publishing here would create ``<meta>`` for every tree that
        ever binds a card and reverse the laziness ``FileLockBackend.acquire`` and
        ``YamlDocumentStore.put_document`` both argue for at length. Two
        retrieval-off cards with different declared caps therefore still evict
        each other's extractions; nothing is corrupted, and the moment either
        enables retrieval the record is published and the disagreement refused.
        """
        bind(orchestrator_proxy, max_documents=4)

        assert not policy_file_for(WORKSPACE_PATH).exists()

    def test_a_retrieval_binder_publishes_its_chunking_and_its_declared_caps(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        bind(
            orchestrator_proxy,
            workspace_rag_index=WorkspaceRagIndex(chunk_chars=640),
            max_documents=9,
        )

        published = policy_on_disk()

        assert published is not None
        assert published.chunking == WorkspaceRagIndex(chunk_chars=640)
        assert published.max_documents == 9
        # Not declared, so not published — *absent* stays distinguishable from
        # *declared at today's default*.
        assert published.max_document_chars is None


##
## AC 5 — a policy file that does not parse refuses the bind, and is left in place
##


class TestAPolicyThatDoesNotParseRefusesTheBind:
    """Deliberately the **opposite** call from the two forgiving readers beside it.

    ``YamlDocumentStore._read`` treats a bad parse as a miss because a record is
    derivable and disposable; ``FileLockBackend.release`` reclaims a marker by
    staleness. A policy has neither property — proceeding would let this binder
    impose its own policy on a tree that already had one, which is the
    silently-winning case, and unlinking it would be worse.
    """

    def _corrupt(self, body: str) -> Path:
        record = policy_file_for(WORKSPACE_PATH)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(body, encoding="utf-8")
        return record

    def test_unparseable_yaml_refuses_and_leaves_the_file(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        record = self._corrupt("{[ not yaml at all")

        with pytest.raises(ValueError) as refusal:
            bind(orchestrator_proxy, workspace_rag_index=True)

        assert str(record) in str(refusal.value)
        assert record.read_text(encoding="utf-8") == "{[ not yaml at all"

    def test_a_mapping_the_model_rejects_refuses_too(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Parsing as YAML is not enough — it has to be a ``TreePolicy``."""
        self._corrupt("max_documents: not-an-int\n")

        with pytest.raises(ValueError, match=POLICY_FILE_NAME):
            bind(orchestrator_proxy, workspace_rag_index=True)

    def test_a_retrieval_off_card_is_refused_by_it_as_well(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Every binder reads, so every binder meets the same refusal."""
        self._corrupt("- a list, not a mapping\n")

        with pytest.raises(ValueError, match=POLICY_FILE_NAME):
            bind(orchestrator_proxy)

    def test_the_refusal_is_composed_from_the_constant(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        self._corrupt("{[ not yaml at all")

        with pytest.raises(ValueError) as refusal:
            bind(orchestrator_proxy)

        prefix = WORKSPACE_POLICY_UNREADABLE.split("{reason}")[0].format(
            card="WorkspaceTool", path=WORKSPACE_PATH, file=policy_file_for(WORKSPACE_PATH)
        )
        assert str(refusal.value).startswith(prefix)

    def test_an_absent_file_is_not_a_bad_parse(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The negative control: a reader that refused on *every* miss would make
        the three specs above pass while breaking every first bind there is."""
        assert read_tree_policy(WORKSPACE_PATH, "WorkspaceTool") is None


##
## AC 4 — the publish is a read-modify-write under the lock
##


class TestTwoProcessesPublishingAtOnce:
    """One published policy, and — where they disagree — exactly one refusal.

    **Two interpreters, and they are released together by a filesystem barrier.**
    Without the lock both children read an absent record, both decide they are
    first, and both write: the loser's policy silently replaces the winner's and
    *no refusal is produced at all*. That is the lost update, and it is invisible
    to every single-process formulation because the GIL already serialises the
    pair.
    """

    def _race(self, tmp_path: Path, workspaces_root: Path, body: str) -> list[str]:
        meta = meta_dir_for(WORKSPACE_PATH)
        meta.mkdir(parents=True, exist_ok=True)
        script = write_script(tmp_path, "policy_racer.py", body)
        children = [start_child(script, workspaces_root, tag, str(meta)) for tag in ("a", "b")]
        reports = [(c.wait(CHILD_TIMEOUT_S), *c.communicate()) for c in children]
        assert [code for code, _out, _err in reports] == [0, 0], reports
        return [out.strip() for _code, out, _err in reports]

    def test_two_disagreeing_binds_produce_one_policy_and_one_refusal(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        messages = self._race(
            tmp_path,
            workspaces_root,
            """
            from akgentic.tool.workspace.rag.params import WorkspaceRagIndex

            tag = sys.argv[1]
            sizes = {"a": 700, "b": 900}
            barrier(sys.argv[2], tag, 2)
            try:
                bind(tag, workspace_rag_index=WorkspaceRagIndex(chunk_chars=sizes[tag]))
                print("BOUND " + tag, flush=True)
            except ValueError as exc:
                print("REFUSED " + tag, flush=True)
            """,
        )

        assert len([line for line in messages if line.startswith("BOUND")]) == 1, messages
        assert len([line for line in messages if line.startswith("REFUSED")]) == 1, messages
        published = policy_on_disk()
        assert published is not None
        assert published.chunking is not None
        assert published.chunking.chunk_chars in (700, 900)

    def test_two_agreeing_binds_both_succeed(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """The positive control: a publish that refused the second writer
        unconditionally would pass the spec above for the wrong reason."""
        messages = self._race(
            tmp_path,
            workspaces_root,
            """
            tag = sys.argv[1]
            barrier(sys.argv[2], tag, 2)
            try:
                bind(tag, workspace_rag_index=True)
                print("BOUND " + tag, flush=True)
            except ValueError as exc:
                print("REFUSED " + repr(exc), flush=True)
            """,
        )

        assert [line for line in messages if line.startswith("BOUND")] == messages, messages


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the directory mode this rests on")
class TestALockThatCannotBeTakenDegradesRatherThanRaising:
    """``CardGate._hold``'s stated choice, copied rather than re-decided.

    What a ``<meta>`` gone read-only costs is the *ordering* between two
    simultaneous first binds, never the bind itself — and the comparison inside
    still runs against whatever is on disk. Failing closed would wedge every bind
    on the tree.
    """

    def test_a_bind_still_lands_when_the_locks_directory_is_unwritable(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        meta = meta_dir_for(WORKSPACE_PATH)
        (meta / LOCKS_DIR_NAME).mkdir(parents=True, exist_ok=True)
        (meta / LOCKS_DIR_NAME).chmod(0o500)
        try:
            with caplog.at_level("WARNING"):
                bind(orchestrator_proxy, workspace_rag_index=True)
        finally:
            (meta / LOCKS_DIR_NAME).chmod(0o700)

        assert any("publishing unserialised" in record.message for record in caplog.records)
        assert policy_on_disk() is not None
