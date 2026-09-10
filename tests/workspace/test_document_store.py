"""The document store in isolation — no actor, no card, two objects over one tree.

Every spec here builds the store directly and drives it through *tree_key*, which
is the whole point of the design under test: the store holds no per-tree state, so
two instances are exactly as separate as two processes over one mount.

**Every environment variable goes through ``monkeypatch.context()``.** The
metadata root, the workspaces root and the store selector are all read at call
time, so a value that leaks out of one spec silently changes another module's
tests — and a spec that asserts against the developer's real checkout passes in
isolation and only fails on the next full-suite run.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from akgentic.tool.workspace import (
    DOCUMENT_STORE_CLASSES,
    DocumentEntry,
    DocumentExtract,
    DocumentStore,
    RagFile,
    RagStatus,
    YamlDocumentStore,
    get_workspace,
    meta_dir_for,
    resolve_document_store,
)

TREE_KEY = "scope-a/notes"


@pytest.fixture
def roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point both roots at *tmp_path* for the duration of one spec.

    Both, not one: ``meta_dir_for`` falls back to the workspaces root when no
    metadata root is set, and a spec that pinned only one of them would resolve
    the other against the process's working directory — the developer's checkout.
    """
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path / "trees"))
    monkeypatch.setenv("AKGENTIC_WORKSPACE_META_ROOT", str(tmp_path / "meta"))
    return tmp_path


def _extract(path: str, sha: str = "sha-1", markdown: str | None = "# body") -> DocumentExtract:
    """A filled extract for *path* — every field set, so nothing defaults silently."""
    return DocumentExtract(
        path=path,
        source_sha=sha,
        extractor_version=1,
        markdown=markdown,
        char_count=len(markdown) if markdown is not None else 0,
        extracted_at=datetime.now(UTC),
    )


def _row(path: str, status: RagStatus = RagStatus.EMBEDDED, sha: str = "sha-1") -> RagFile:
    """An index row for *path* at *status*."""
    return RagFile(path=path, status=status, indexed_sha=sha, updated_at=datetime.now(UTC))


def _rag_dir(tree_key: str = TREE_KEY) -> Path:
    """Where the store's files land for *tree_key*."""
    return meta_dir_for(tree_key) / "rag"


class TestTheRecordComposesTheShippedModels:
    """AC 1 — ``DocumentEntry`` is the key plus the two shipped halves, and no more."""

    def test_it_has_exactly_the_three_fields(self) -> None:
        """A fourth field would be a model this story invented, which it must not."""
        assert set(DocumentEntry.model_fields) == {"path", "extract", "row"}

    def test_both_halves_default_to_absent(self) -> None:
        """A queued-but-unread file and a read-but-unqueued one are both ordinary."""
        entry = DocumentEntry(path="a.md")
        assert entry.extract is None
        assert entry.row is None

    def test_it_round_trips_through_serialisation(self) -> None:
        """It is persisted, so both halves must survive a dump and a re-validate."""
        entry = DocumentEntry(path="a.md", extract=_extract("a.md"), row=_row("a.md"))
        restored = DocumentEntry.model_validate(entry.model_dump(mode="json"))
        assert restored.extract is not None
        assert restored.extract.markdown == "# body"
        assert restored.row is not None
        assert restored.row.status is RagStatus.EMBEDDED

    def test_the_yaml_store_satisfies_the_protocol(self) -> None:
        """The registry builds any entry with no type switch, so conformance is checked."""
        assert isinstance(YamlDocumentStore(), DocumentStore)

    def test_every_registered_backend_is_built_with_no_arguments(self) -> None:
        """The no-argument constructor is what lets one instance serve every tree."""
        for backend in DOCUMENT_STORE_CLASSES.values():
            assert isinstance(backend(), DocumentStore)


class TestTheFileIsYamlUnderTheMetadataDirectory:
    """AC 2 and AC 3 — where the file lands, what format it is, and what it carries."""

    def test_the_written_file_is_a_yaml_mapping(self, roots: Path) -> None:
        """Guard for AC 2: JSON that happens to parse would pass a weaker check."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md")))
        written = next(iter(_rag_dir().glob("*.yaml")))
        loaded = yaml.safe_load(written.read_text())
        assert isinstance(loaded, dict)
        assert not written.read_text().lstrip().startswith("{")

    def test_the_file_lands_under_the_metadata_rag_directory(self, roots: Path) -> None:
        """One file per source document, named by a digest of the path."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="notes/a.md"))
        assert len(list(_rag_dir().glob("*.yaml"))) == 1

    def test_the_directory_is_created_lazily(self, roots: Path) -> None:
        """A workspace that never reads a document must provision nothing."""
        YamlDocumentStore()
        assert not _rag_dir().exists()

    def test_no_file_is_written_inside_the_tree(self, roots: Path) -> None:
        """The metadata directory is a sibling, so no read capability can name a record."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md")))
        tree_root = Path(get_workspace(TREE_KEY)._root)
        written = next(iter(_rag_dir().glob("*.yaml")))
        assert not written.is_relative_to(tree_root)
        assert not list(tree_root.rglob("*.yaml"))

    def test_the_body_carries_the_path_in_plain_text(self, roots: Path) -> None:
        """The name is a digest, so the directory stays greppable only via the body."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="notes/deep/a.md"))
        written = next(iter(_rag_dir().glob("*.yaml")))
        assert "notes/deep/a.md" in written.read_text()

    def test_two_trees_do_not_share_a_directory(self, roots: Path) -> None:
        """*tree_key* is what separates them, since the store holds no per-tree state."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="a.md"))
        store.put_document("scope-b/notes", DocumentEntry(path="a.md"))
        assert len(list(_rag_dir().glob("*.yaml"))) == 1
        assert len(list(_rag_dir("scope-b/notes").glob("*.yaml"))) == 1
        assert store.get_document("scope-b/notes", "b.md") is None


class TestTheStoreIsKeyedByPathNotByContent:
    """AC 4 — a re-put replaces; it does not accumulate a file per version."""

    def test_a_re_put_at_new_bytes_leaves_exactly_one_file(self, roots: Path) -> None:
        """A content-keyed name would leak one file per version with nothing to evict them."""
        store = YamlDocumentStore()
        store.put_document(
            TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md", sha="digest-A"))
        )
        store.put_document(
            TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md", sha="digest-B"))
        )
        assert len(list(_rag_dir().glob("*.yaml"))) == 1

    def test_the_second_put_is_what_is_read_back(self, roots: Path) -> None:
        """Last writer wins, whole — never a blend of the two."""
        store = YamlDocumentStore()
        store.put_document(
            TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md", sha="digest-A"))
        )
        store.put_document(
            TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md", sha="digest-B"))
        )
        entry = store.get_document(TREE_KEY, "a.md")
        assert entry is not None
        assert entry.extract is not None
        assert entry.extract.source_sha == "digest-B"

    def test_a_row_is_found_at_the_path_after_its_bytes_changed(self, roots: Path) -> None:
        """``mark_paths_stale`` runs *because* the bytes changed and holds only the path."""
        store = YamlDocumentStore()
        store.put_document(
            TREE_KEY, DocumentEntry(path="a.md", row=_row("a.md", sha="the-old-digest"))
        )
        entry = store.get_document(TREE_KEY, "a.md")
        assert entry is not None
        assert entry.row is not None
        assert entry.row.indexed_sha == "the-old-digest"


class TestOneProcessWritesAndAnotherReads:
    """AC 5 — the cache is on disk, so nothing in memory can be doing the sharing."""

    def test_a_second_store_object_sees_the_first_ones_write(self, roots: Path) -> None:
        """Two objects, one tree_key: what A put, B gets — both halves of it."""
        writer = YamlDocumentStore()
        reader = YamlDocumentStore()
        writer.put_document(
            TREE_KEY,
            DocumentEntry(
                path="a.md",
                extract=_extract("a.md", markdown="# written by A"),
                row=_row("a.md", status=RagStatus.EMBEDDED),
            ),
        )
        entry = reader.get_document(TREE_KEY, "a.md")
        assert entry is not None
        assert entry.extract is not None
        assert entry.extract.markdown == "# written by A"
        assert entry.row is not None
        assert entry.row.status is RagStatus.EMBEDDED

    def test_a_second_store_object_lists_the_first_ones_writes(self, roots: Path) -> None:
        """The listing reads the directory, not an in-memory index."""
        writer = YamlDocumentStore()
        reader = YamlDocumentStore()
        for name in ("a.md", "b.md", "c.md"):
            writer.put_document(TREE_KEY, DocumentEntry(path=name, extract=_extract(name)))
        assert {entry.path for entry in reader.list_documents(TREE_KEY)} == {
            "a.md",
            "b.md",
            "c.md",
        }


class TestATornWriteLeavesThePreviousFile:
    """AC 6 — temp-then-replace, and no debris when the write dies part-way."""

    def test_the_previous_file_survives_a_failed_write(
        self, roots: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A serialisation that raises must not destroy a good record."""
        store = YamlDocumentStore()
        store.put_document(
            TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md", markdown="# good"))
        )

        def _explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("the dump failed half way through")

        with monkeypatch.context() as patch:
            patch.setattr(yaml, "dump", _explode)
            with pytest.raises(RuntimeError):
                store.put_document(
                    TREE_KEY,
                    DocumentEntry(path="a.md", extract=_extract("a.md", markdown="# doomed")),
                )
        entry = store.get_document(TREE_KEY, "a.md")
        assert entry is not None
        assert entry.extract is not None
        assert entry.extract.markdown == "# good"

    def test_no_temporary_file_is_left_behind(
        self, roots: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Debris would make ``list_documents`` trip over a half-written record."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md")))

        def _explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("the dump failed half way through")

        with monkeypatch.context() as patch:
            patch.setattr(yaml, "dump", _explode)
            with pytest.raises(RuntimeError):
                store.put_document(TREE_KEY, DocumentEntry(path="a.md"))
        assert list(_rag_dir().glob("*.tmp")) == []


class TestConcurrentPutsEndWithOneWholeFile:
    """AC 7 — no lock; atomic replacement is what the concurrent case requires."""

    def test_two_threads_leave_one_whole_entry(self, roots: Path) -> None:
        """A lost update costs one recompute; a blended or truncated file would not."""
        store = YamlDocumentStore()
        # The directory is created up front so the two threads race the *write*
        # rather than the mkdir, which is what the atomicity claim is about.
        store.put_document(TREE_KEY, DocumentEntry(path="a.md"))
        barrier = threading.Barrier(2)
        failures: list[BaseException] = []

        def _put(marker: str) -> None:
            try:
                barrier.wait(timeout=5)
                store.put_document(
                    TREE_KEY,
                    DocumentEntry(
                        path="a.md",
                        extract=_extract("a.md", sha=marker, markdown=f"# {marker}"),
                    ),
                )
            except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
                failures.append(exc)

        threads = [threading.Thread(target=_put, args=(marker,)) for marker in ("A", "B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert failures == []
        assert len(list(_rag_dir().glob("*.yaml"))) == 1
        assert list(_rag_dir().glob("*.tmp")) == []
        entry = store.get_document(TREE_KEY, "a.md")
        assert entry is not None
        assert entry.extract is not None
        # Whole, not a blend: the digest and the body must come from one writer.
        assert (entry.extract.source_sha, entry.extract.markdown) in {
            ("A", "# A"),
            ("B", "# B"),
        }


class TestEvictionAndUnreadableFiles:
    """AC 8 — eviction removes, a miss is a miss, and a bad parse is never a delete."""

    def test_evict_removes_the_file_and_the_next_get_misses(self, roots: Path) -> None:
        """The row cap's remedy is removal, and it must actually reach the disk."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md")))
        store.evict(TREE_KEY, "a.md")
        assert store.get_document(TREE_KEY, "a.md") is None
        assert list(_rag_dir().glob("*.yaml")) == []

    def test_evicting_an_unknown_path_does_not_raise(self, roots: Path) -> None:
        """Eviction runs over a cap on a tree that may never have been written to."""
        store = YamlDocumentStore()
        store.evict(TREE_KEY, "never-stored.md")

    def test_a_file_that_does_not_parse_is_a_miss_and_is_left_in_place(
        self, roots: Path
    ) -> None:
        """Removing a file we cannot read is how a cache turns a bad parse into data loss."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="a.md", extract=_extract("a.md")))
        corrupt = next(iter(_rag_dir().glob("*.yaml")))
        corrupt.write_text("{ this is not: valid: yaml: at all")
        assert store.get_document(TREE_KEY, "a.md") is None
        assert corrupt.exists()

    def test_a_file_the_model_rejects_is_a_miss_and_is_left_in_place(self, roots: Path) -> None:
        """Valid YAML carrying the wrong shape is the same class of problem."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="a.md"))
        wrong = next(iter(_rag_dir().glob("*.yaml")))
        wrong.write_text(yaml.dump({"not_a_path": 17}))
        assert store.get_document(TREE_KEY, "a.md") is None
        assert wrong.exists()

    def test_an_unreadable_file_is_skipped_by_the_listing(self, roots: Path) -> None:
        """One corrupt record costs one re-extraction, never the whole capability."""
        store = YamlDocumentStore()
        store.put_document(TREE_KEY, DocumentEntry(path="good.md", extract=_extract("good.md")))
        store.put_document(TREE_KEY, DocumentEntry(path="bad.md", extract=_extract("bad.md")))
        bad = _rag_dir() / f"{hashlib.sha256(b'bad.md').hexdigest()}.yaml"
        bad.write_text("{ nope")
        listed = store.list_documents(TREE_KEY)
        assert [entry.path for entry in listed] == ["good.md"]

    def test_listing_a_tree_with_no_directory_is_empty(self, roots: Path) -> None:
        """The ordinary state of a workspace that has never read a document."""
        assert YamlDocumentStore().list_documents(TREE_KEY) == []


class TestBackendSelectionIsSelfResolved:
    """AC 9 — the resolver mirrors ``resolve_lock_backend``, including its failure."""

    def test_an_unset_variable_resolves_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default is what every deployment that sets nothing gets."""
        with monkeypatch.context() as patch:
            patch.delenv("AKGENTIC_DOCUMENT_STORE", raising=False)
            assert isinstance(resolve_document_store(), YamlDocumentStore)

    def test_naming_yaml_resolves_the_same(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Naming the default explicitly must not take a different path."""
        with monkeypatch.context() as patch:
            patch.setenv("AKGENTIC_DOCUMENT_STORE", "yaml")
            assert isinstance(resolve_document_store(), YamlDocumentStore)

    def test_an_empty_value_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A compose file interpolating an unset variable arrives here as ``""``."""
        with monkeypatch.context() as patch:
            patch.setenv("AKGENTIC_DOCUMENT_STORE", "")
            assert isinstance(resolve_document_store(), YamlDocumentStore)

    def test_an_unregistered_name_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configuration error, deliberately at start-up rather than at first read."""
        with monkeypatch.context() as patch:
            patch.setenv("AKGENTIC_DOCUMENT_STORE", "nope")
            with pytest.raises(KeyError):
                resolve_document_store()

    def test_an_entry_registered_after_import_is_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Call-time lookup is what lets a deployment register its own store."""
        with monkeypatch.context() as patch:
            patch.setitem(DOCUMENT_STORE_CLASSES, "late", YamlDocumentStore)
            patch.setenv("AKGENTIC_DOCUMENT_STORE", "late")
            assert isinstance(resolve_document_store(), YamlDocumentStore)
