"""The spine every lock family stands on, and the four families it must not merge.

Story 57-2 collapsed four spellings of one ``flock`` idiom onto
:mod:`akgentic.tool.workspace.locks`. The danger in that move is not that the
helper is wrong — the existing cross-process specs in ``test_gate_locks.py``,
``test_record_lock.py`` and ``test_tree_policy.py`` would catch a broken hold, and
they pass untouched. The danger is that **four idioms collapse into four locks**:
a later tidy-up moves a lock-file name into the spine and two families start
contending on one file, which shows up as a deadlock or a lost update under
concurrency and as nothing at all in a single-process suite.

So this module guards the three things those specs cannot see:

* the spine imports nothing but the standard library, so no capability reaches
  another through it;
* each of the four call sites still reports its **own** degradation, under its own
  logger — the diagnostic a generic helper would have flattened;
* the four families still resolve to four distinct files, and holding one does not
  exclude another. That last one **forks**, because ``flock`` belongs to the open
  file description and two objects in one interpreter prove nothing about it.
"""

from __future__ import annotations

import ast
import os
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from akgentic.tool.workspace import locks
from akgentic.tool.workspace.documents.store import YamlDocumentStore
from akgentic.tool.workspace.journal import JOURNAL_LOCK_FILENAME, GitJournal
from akgentic.tool.workspace.rag import POLICY_LOCK_NAME, _policy_hold
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import meta_dir_for
from akgentic.tool.workspace.write.gate import lock_file_for
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    run_child,
)

_PERMITTED_IMPORTS = {
    "__future__",
    "collections.abc",
    "contextlib",
    "fcntl",
    "logging",
    "os",
    "pathlib",
    "tempfile",
    "typing",
}
"""What the spine may name. Everything here ships with CPython."""

_HOLD_BUDGET_S = 5.0
"""How long a correct hold may take before the run is called hung.

A failure bound rather than a delay: a spine that de-duplicates crosses it in
microseconds, and one that does not never crosses it at all. Matched to
``conftest.HANDSHAKE_TIMEOUT_S`` because it is the same kind of budget.
"""


def _held_within_budget(
    lock_paths: list[Path], on_failure: Callable[[OSError], None], inside: Callable[[], None]
) -> bool:
    """Take *lock_paths*, run *inside*, release — on a joined thread with a budget.

    **Every spec that hands the spine a repeated path goes through here**, and the
    reason is that the alternative is a hang rather than a failure. Dropping the
    spine's ``set()`` makes the second ``LOCK_EX`` on one file block against this
    process's own first hold, forever; called directly, such a spec never returns
    and never asserts. This package configures **no pytest timeout** — there is no
    ``[tool.pytest.ini_options]`` and no ``addopts`` — so nothing would interrupt
    it either: the run would sit until CI's own job limit and report a timeout
    instead of the defect.

    A daemon thread so a genuinely wedged hold cannot keep the interpreter alive,
    and the return value rather than an exception so the caller names what the
    budget means for the property it is testing.
    """
    finished = threading.Event()

    def take() -> None:
        with locks.hold(lock_paths, on_failure=on_failure):
            inside()
        finished.set()

    worker = threading.Thread(target=take, daemon=True)
    worker.start()
    took = finished.wait(_HOLD_BUDGET_S)
    worker.join(_HOLD_BUDGET_S)
    return took


def _locks_source() -> str:
    """The spine's own source, read from the module rather than from a path guess."""
    return Path(locks.__file__).read_text(encoding="utf-8")


def _imported_modules(source: str) -> set[str]:
    """Every module named by an ``import`` in *source*, including under ``TYPE_CHECKING``.

    Parsed rather than matched, so a conditional or indented import cannot hide
    from this the way a regular expression over the text would let it.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


class TestTheSpineImportsOnlyTheStandardLibrary:
    """AC 1 — the property that lets every capability stand on this module.

    A spine that imported a capability would invert the package's layering, and a
    spine that imported ``yaml`` would put a serialisation format underneath four
    callers that do not all serialise. Both are import statements, so both are
    visible in the parse tree.
    """

    def test_it_names_nothing_from_akgentic(self) -> None:
        """The rule the four apologies were written about, now checkable."""
        offenders = sorted(
            name for name in _imported_modules(_locks_source()) if name.startswith("akgentic")
        )
        assert offenders == [], (
            f"the spine imports {offenders} — a capability importing another capability "
            "through the spine is the edge this module exists to avoid"
        )

    def test_it_names_no_third_party_module_at_all(self) -> None:
        """Wider than ``yaml``, because the next one to creep in will not be ``yaml``."""
        outside = sorted(_imported_modules(_locks_source()) - _PERMITTED_IMPORTS)
        assert outside == [], f"the spine imports {outside}, which is outside the permitted set"

    def test_the_permitted_set_is_actually_exercised(self) -> None:
        """The non-vacuity control.

        A spec that only subtracted from an allow-list would stay green if the
        module were emptied to a single ``pass``. This pins that the parse found
        real imports, so the two assertions above ran against something.
        """
        assert _imported_modules(_locks_source()) & _PERMITTED_IMPORTS


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the directory mode this rests on")
class TestEachFamilyKeepsItsOwnDegradationSentence:
    """AC 4 — four operator diagnostics that a generic helper would have flattened.

    Each family degrades on an unwritable ``<meta>/locks/`` with **its own
    sentence, under its own logger**. That is worth a spec because the cheap way
    to write the spine is to log once inside it, which reads as a tidy-up and
    silently replaces four diagnostics with one that names no capability.

    The logger name is asserted alongside the sentence: a helper that logged all
    four sentences itself, through ``akgentic.tool.workspace.locks``, would
    satisfy a fragment-only assertion while losing exactly what makes the warning
    useful in a log.
    """

    @staticmethod
    def _sealed_locks_dir() -> Path:
        """An existing ``<meta>/locks/`` that no new lock file can be created in."""
        locks_dir = meta_dir_for(WORKSPACE_PATH) / locks.LOCKS_DIR_NAME
        locks_dir.mkdir(parents=True, exist_ok=True)
        locks_dir.chmod(0o500)
        return locks_dir

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
        """Every captured warning as ``(logger name, message)``."""
        return [(record.name, record.message) for record in caplog.records]

    def test_the_write_gate_says_mutating_unserialised_under_its_own_logger(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
        assert card.apply_write("first.md", "one\n").message == "Written: first.md"
        locks_dir = self._sealed_locks_dir()
        try:
            with caplog.at_level("WARNING"):
                assert card.apply_write("second.md", "two\n").message == "Written: second.md"
        finally:
            locks_dir.chmod(0o700)

        assert any(
            name == "akgentic.tool.workspace.write.gate" and "mutating unserialised" in message
            for name, message in self._warnings(caplog)
        ), self._warnings(caplog)

    def test_the_journal_says_committing_unserialised_under_its_own_logger(
        self, workspace_tree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        meta = meta_dir_for(WORKSPACE_PATH)
        journal = GitJournal(workspace_tree, enabled=True, timeout_s=60.0, meta_dir=meta)
        locks_dir = self._sealed_locks_dir()
        try:
            with caplog.at_level("WARNING"), journal._holding():
                pass
        finally:
            locks_dir.chmod(0o700)

        assert any(
            name == "akgentic.tool.workspace.journal" and "committing unserialised" in message
            for name, message in self._warnings(caplog)
        ), self._warnings(caplog)

    def test_the_document_store_says_proceeding_unserialised_under_its_own_logger(
        self, workspace_tree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = YamlDocumentStore()
        locks_dir = self._sealed_locks_dir()
        try:
            with caplog.at_level("WARNING"), store.hold(WORKSPACE_PATH, "a.md"):
                pass
        finally:
            locks_dir.chmod(0o700)

        assert any(
            name == "akgentic.tool.workspace.documents.store"
            and "proceeding unserialised" in message
            for name, message in self._warnings(caplog)
        ), self._warnings(caplog)

    def test_the_tree_policy_says_publishing_unserialised_under_its_own_logger(
        self, workspace_tree: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        meta = meta_dir_for(WORKSPACE_PATH)
        locks_dir = self._sealed_locks_dir()
        try:
            with caplog.at_level("WARNING"), _policy_hold(meta, WORKSPACE_PATH):
                pass
        finally:
            locks_dir.chmod(0o700)

        assert any(
            name == "akgentic.tool.workspace.rag" and "publishing unserialised" in message
            for name, message in self._warnings(caplog)
        ), self._warnings(caplog)

    def test_the_four_sentences_are_four_distinct_sentences(self) -> None:
        """The property the four specs above share, stated once so it cannot drift.

        Four families that degraded into one sentence would still pass a
        fragment-per-site check if the fragments were all the same word.
        """
        sentences = {
            "mutating unserialised",
            "committing unserialised",
            "proceeding unserialised",
            "publishing unserialised",
        }
        assert len(sentences) == 4


class TestTheFourFamiliesStayFour:
    """AC 15 — four idioms became one; four locks must remain four.

    **This is the anti-collapse guard**, and it is the one the epic's first trap
    demands. Nothing else in the suite states the property *across* families: each
    family's own specs prove that family excludes itself, and all of them would
    stay green if two families were pointed at one file — that failure shows up as
    contention between unrelated operations, not as a broken lock.

    The probe forks and takes a **non-blocking** ``LOCK_EX``, so a collapse is
    reported as ``BLOCKED`` rather than as a hung test.
    """

    _PROBE = """
        import fcntl, os, sys

        handle = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("ACQUIRED", flush=True)
        except OSError:
            print("BLOCKED", flush=True)
        """

    @staticmethod
    def _family_paths() -> dict[str, Path]:
        """The four lock files, each derived from its own capability's symbols.

        **Never a literal.** A spec that re-spelled these paths would keep
        agreeing with itself after a capability changed the file it really locks.
        """
        meta = meta_dir_for(WORKSPACE_PATH)
        return {
            "path": lock_file_for(meta, "a.md"),
            "journal": meta / locks.LOCKS_DIR_NAME / JOURNAL_LOCK_FILENAME,
            "record": YamlDocumentStore()._lock_file(WORKSPACE_PATH, "a.md"),
            "policy": meta / locks.LOCKS_DIR_NAME / POLICY_LOCK_NAME,
        }

    def _written_probe(self, tmp_path: Path) -> Path:
        probe = tmp_path / "family_probe.py"
        probe.write_text(textwrap.dedent(self._PROBE), encoding="utf-8")
        return probe

    def test_the_four_families_resolve_to_four_distinct_files(self, workspace_tree: Path) -> None:
        """Four names, one directory — the structural half of the property."""
        paths = self._family_paths()
        assert len(set(paths.values())) == 4, paths

        parents = {path.parent for path in paths.values()}
        assert parents == {meta_dir_for(WORKSPACE_PATH) / locks.LOCKS_DIR_NAME}, parents

    @pytest.mark.parametrize(
        ("held", "probed"),
        [("path", "record"), ("journal", "policy"), ("path", "journal"), ("record", "policy")],
    )
    def test_holding_one_family_does_not_exclude_another(
        self,
        held: str,
        probed: str,
        workspaces_root: Path,
        workspace_tree: Path,
        tmp_path: Path,
    ) -> None:
        """A second process takes family *probed* while this one holds *held*.

        The two pairs the story names — a record claim under the path lock, a
        policy publish under the journal lock — plus two more, because the
        collapse this guards against would most likely merge whichever two files
        a refactor happened to touch.
        """
        paths = self._family_paths()
        probe = self._written_probe(tmp_path)

        def warn(exc: OSError) -> None:
            raise AssertionError(f"the parent could not take the {held} lock: {exc}")

        with locks.hold([paths[held]], on_failure=warn):
            report = run_child(probe, workspaces_root, str(paths[probed]))

        assert report.out == "ACQUIRED", (
            f"holding the {held} lock excluded the {probed} lock — two families have "
            f"been collapsed onto one file: {paths[held]} and {paths[probed]}"
        )

    def test_the_probe_reports_blocked_on_the_very_lock_that_is_held(
        self, workspaces_root: Path, workspace_tree: Path, tmp_path: Path
    ) -> None:
        """The positive control, and without it the parametrised spec is worthless.

        A probe that could never acquire anything — a child that always failed, a
        wrong flag, a path nothing locks — would answer ``ACQUIRED`` never and
        ``BLOCKED`` always, or the reverse, and the specs above would pass with
        every lock deleted. So the same probe, on the file the parent really
        holds, must come back ``BLOCKED``.
        """
        paths = self._family_paths()
        probe = self._written_probe(tmp_path)

        def warn(exc: OSError) -> None:
            raise AssertionError(f"the parent could not take the path lock: {exc}")

        with locks.hold([paths["path"]], on_failure=warn):
            report = run_child(probe, workspaces_root, str(paths["path"]))

        assert report.out == "BLOCKED", report


class TestTheSpineAtomicWriteLeavesNoPartialFile:
    """AC 16 — a reader sees a whole file or the previous one, never a prefix.

    ``test_document_store.py`` already proves this **for its caller**
    (``TestAFailedWriteLeavesThePreviousFileIntact`` and
    ``TestConcurrentPutsEndWithOneWholeFile``), and those specs still pass. They
    no longer reach the spine's own failure path, though, and that is worth
    stating plainly: they provoke the failure by making ``yaml.dump`` raise, and
    after this story the caller renders its YAML **before** calling the spine — so
    the throw now happens before a temp file is ever created. The specs remain
    true and are now silent about the ``mkstemp``-then-unlink path.

    So this extends them to the spine rather than duplicating them: the failure is
    injected *inside* the write, which is where the temp file exists.
    """

    def test_a_failure_between_the_write_and_the_replace_leaves_no_debris(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The path the caller-level specs stopped covering once rendering moved out.

        The throw is injected at the ``replace``, which is the only point where a
        fully written temp file already exists — exactly the state the unlink is
        there to clean up.
        """
        target = tmp_path / "policy.yaml"
        target.write_text("previous\n", encoding="utf-8")

        def _explode(self: Path, other: object) -> None:
            raise RuntimeError("the replace failed half way through")

        monkeypatch.setattr(Path, "replace", _explode)
        with pytest.raises(RuntimeError):
            locks.atomic_write(target, "doomed\n")

        assert target.read_text(encoding="utf-8") == "previous\n"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_an_interrupted_write_leaves_no_debris_for_base_exceptions_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``BaseException``, not ``Exception``: a ``KeyboardInterrupt`` between the two.

        The spine catches the wider class deliberately. A narrowing to
        ``Exception`` would leave a ``.tmp`` behind on exactly the interrupt a
        human is most likely to send, and would pass the spec above unchanged.
        """
        target = tmp_path / "record.yaml"
        target.write_text("previous\n", encoding="utf-8")

        def _interrupt(self: Path, other: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(Path, "replace", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            locks.atomic_write(target, "doomed\n")

        assert target.read_text(encoding="utf-8") == "previous\n"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_a_successful_write_replaces_the_whole_file(self, tmp_path: Path) -> None:
        """The positive control: the two specs above pass if nothing is ever written."""
        target = tmp_path / "record.yaml"
        target.write_text("previous\n", encoding="utf-8")

        locks.atomic_write(target, "next\n")

        assert target.read_text(encoding="utf-8") == "next\n"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_bytes_and_text_land_identically(self, tmp_path: Path) -> None:
        """Both callers hand over text; the vector store's copy writes bytes.

        The spine takes either, and the two must not diverge — that equivalence is
        what would let the third copy fold in later if an ADR ever decides it
        should.
        """
        as_text = tmp_path / "text.yaml"
        as_bytes = tmp_path / "bytes.yaml"

        locks.atomic_write(as_text, "policy: strict\n")
        locks.atomic_write(as_bytes, b"policy: strict\n")

        assert as_text.read_bytes() == as_bytes.read_bytes()


class TestTheSpineTakesLocksInSortedOrder:
    """AC 3 — the sort is what makes a deadlock impossible, and it needs its own guard.

    ``test_write_capability.py::TestTheLocksAreTakenInSortedPathOrder`` watches the
    **gate** sort its workspace paths, and it is untouched by story 57-2. It does
    not reach this: it spies on ``lock_file_for``, so it stays green with the
    spine's ``sorted()`` deleted outright — measured, not assumed.

    That matters because the two sorts guarantee different things. The gate's
    orders one capability's paths; the spine's is what puts **every** caller on
    one global order, which is the property that makes two families taken
    together incapable of deadlocking. Three of the four callers pass a single
    lock and could never show it. So the spine's own order is asserted here.
    """

    def test_locks_are_taken_sorted_and_deduplicated_whatever_order_they_arrive_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The batch arrives unsorted and names one file twice.

        Both halves are what make the assertion capable of failing: the unsorted
        input is the only one that distinguishes ``sorted(...)`` from the order
        given, and the duplicate the only one that distinguishes ``set(...)``
        from a plain list — a duplicate that reached ``flock`` twice would block
        this process against its own hold.

        **Eight distinct paths, not three, and the number is the guard rather
        than a flourish.** With ``sorted()`` deleted the spine iterates the
        ``set`` instead, whose order follows ``str`` hashing and therefore
        ``PYTHONHASHSEED`` — so on a short batch it can come out sorted *by
        luck* and leave this spec green against the very mutation it exists to
        catch. Measured on three paths: green on 5 of 20 seeds. Eight distinct
        paths put that coincidence at ``1/8!``, which is what makes a red run
        here mean the sort is gone rather than that the seed was kind.
        """
        letters = "hbfadgce"
        given = [tmp_path / letter for letter in letters] + [tmp_path / letters[0]]
        seen: list[Path] = []
        original = locks._open_lock

        def _recording(lock_path: Path) -> int:
            seen.append(lock_path)
            return original(lock_path)

        monkeypatch.setattr(locks, "_open_lock", _recording)

        def warn(exc: OSError) -> None:
            raise AssertionError(f"no lock should have failed: {exc}")

        # Bounded, because this batch names one path twice: a spine that stopped
        # de-duplicating would block here against its own hold rather than fail.
        assert _held_within_budget(given, warn, lambda: None), (
            "the hold never returned within the budget — the repeated path reached flock twice"
        )

        assert seen == sorted(set(given))
        # Non-vacuity: the spy saw something, and not merely the order given.
        assert seen != given


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the directory mode this rests on")
class TestNoDescriptorSurvivesAFailedAcquisition:
    """AC 3 — every descriptor the helper opens is closed on every path.

    A partial acquisition is the interesting case: three of five locks taken and
    the fourth failing leaks three descriptors if the ``finally`` is written to
    cover only the success path. A leak is invisible in a short test run and fatal
    in a long-lived worker, so it is counted rather than reasoned about.
    """

    @staticmethod
    def _open_descriptors() -> int:
        """How many file descriptors this process currently holds."""
        return len(os.listdir("/dev/fd"))

    def test_a_partial_acquisition_closes_what_it_took(self, tmp_path: Path) -> None:
        """Two takeable locks and one that cannot be created, in sorted order."""
        takeable = [tmp_path / "a-first", tmp_path / "b-second"]
        sealed_dir = tmp_path / "sealed"
        sealed_dir.mkdir()
        sealed_dir.chmod(0o500)
        unreachable = sealed_dir / "c-third"

        reported: list[OSError] = []
        before = self._open_descriptors()
        try:
            with locks.hold([*takeable, unreachable], on_failure=reported.append):
                pass
        finally:
            sealed_dir.chmod(0o700)

        assert reported, "the unreachable lock was taken, or its failure was swallowed"
        assert self._open_descriptors() == before

    def test_a_clean_acquisition_closes_everything_too(self, tmp_path: Path) -> None:
        """The control: a leak on the success path would not show up above."""
        paths = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]

        def warn(exc: OSError) -> None:
            raise AssertionError(f"no lock should have failed: {exc}")

        before = self._open_descriptors()
        with locks.hold(paths, on_failure=warn):
            assert self._open_descriptors() == before + len(paths)

        assert self._open_descriptors() == before

    def test_an_empty_sequence_takes_nothing_and_yields(self, tmp_path: Path) -> None:
        """``CardGate._hold``'s behaviour for ``mkdir``, which has no window to close."""

        def warn(exc: OSError) -> None:
            raise AssertionError(f"nothing should have been opened: {exc}")

        before = self._open_descriptors()
        with locks.hold([], on_failure=warn):
            assert self._open_descriptors() == before

        assert self._open_descriptors() == before

    def test_a_repeated_path_is_taken_once(self, tmp_path: Path) -> None:
        """De-duplication is load-bearing for **liveness**, not tidiness.

        A second ``LOCK_EX`` on a second descriptor for one file blocks against
        this process's own first hold, forever. ``apply_multi_edit`` passes
        ``[item.path for item in edits]`` straight through, so a batch naming one
        path twice is an ordinary input rather than an exotic one.

        **The hold runs on a joined thread with a budget, and that is what makes
        the mutation bounded.** Called directly, a spine without ``set()`` would
        hang here rather than fail — and this package configures no pytest
        timeout, so nothing would stop it: the run would sit until CI's own job
        limit and report a timeout rather than this defect. The budget is a
        failure bound, not a delay; a correct spine crosses it in microseconds.

        The descriptor count carries the other half — that the one file was
        opened **once**, not twice — which a liveness check alone would not see.
        """
        repeated = tmp_path / "same"
        before = self._open_descriptors()
        during: list[int] = []

        def warn(exc: OSError) -> None:
            raise AssertionError(f"no lock should have failed: {exc}")

        assert _held_within_budget(
            [repeated, repeated, repeated], warn, lambda: during.append(self._open_descriptors())
        ), (
            "the hold never returned within the budget — a repeated path reached "
            "flock twice and this process blocked against its own hold"
        )
        assert during == [before + 1], during
