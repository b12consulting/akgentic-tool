"""The one flock idiom every workspace capability stands on.

Four capabilities take a cross-process ``flock`` under ``<meta>/locks/`` — the
per-path write lock, the journal's commit lock, a document record's lock and the
tree policy's. They contend on four **different** files and they stay four
distinct families; what they share is *how* a hold is taken, and that is what
lives here.

**This module imports nothing but the standard library**, and that is load-bearing
rather than tidy. A capability importing another capability's constant — the
retrieval capability reaching into ``write/gate.py`` for ``LOCKS_DIR_NAME`` — is a
runtime edge between two peers, and is worse than the copy it replaces. A module
that names only ``os``, ``fcntl``, ``pathlib`` and ``tempfile`` is importable by
every capability exactly as :mod:`akgentic.tool.workspace.workspace` is, and
creates no such edge. In particular it holds **no** ``yaml``: a caller that
serialises renders its own bytes and hands them over, so no serialisation format
reaches the spine.

**It holds no lock-file name and no prefix.** ``PATH_LOCK_PREFIX``,
``RECORD_LOCK_PREFIX``, ``JOURNAL_LOCK_FILENAME`` and ``POLICY_LOCK_NAME`` stay on
their four capabilities, because they are what make the four families four.
Moving one here would collapse four locks into fewer, which is the opposite of
this module's purpose: **four idioms become one, four locks stay four.**
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

LOCKS_DIR_NAME = "locks"
"""The directory under ``<meta>`` holding one lock file per contended thing.

One directory for every family, so nothing has to remember a second place to
look. What distinguishes the families is the **file name** inside it, which each
capability owns.
"""

LOCK_FILE_MODE = 0o600
"""Mode a lock file is created with — nothing outside the owner needs it."""


@contextlib.contextmanager
def hold(
    lock_paths: Sequence[Path],
    *,
    on_failure: Callable[[OSError], None],
) -> Iterator[None]:
    """Hold every lock in *lock_paths* exclusively for the duration of the block.

    An exclusive ``flock`` on each lazily created file, unlocked and closed in
    ``finally``, and **never unlinked** — unlinking would let a second process
    create a fresh inode and take a hold that excludes nobody, which is the
    classic way a file lock stops locking.

    **Acquired in sorted order, and that is what makes a deadlock impossible.**
    Two callers that both touch two files take them in the same order whatever
    order they were passed in, so neither can hold one while waiting for the
    other's. Duplicates are taken once: a second ``LOCK_EX`` on a second
    descriptor for one file blocks against the holder's own first hold, so
    de-duplicating is what stops a caller deadlocking against itself.

    **A lock that cannot be taken degrades rather than refusing.** *on_failure*
    is called with the ``OSError`` and the block runs unserialised. Failing
    closed instead would wedge every caller on a tree whose ``<meta>`` went
    read-only mid-session, and the bytes a caller writes are safe either way —
    only the ordering between two writers is lost, which is what was on offer
    before the lock existed at all. The caller supplies the reporting because the
    four families degrade into four different operator sentences, under four
    logger names; one generic sentence from here would destroy four diagnostics
    to save four lines.

    Every descriptor this opens is closed on every path, including a partial
    acquisition that failed part-way through the sequence.

    Args:
        lock_paths: The lock files to hold. An empty sequence takes nothing and
            yields — a caller with no contended thing has no check-then-write
            window to close.
        on_failure: Called with the ``OSError`` when a lock cannot be taken, to
            report the degradation in the caller's own words and under its own
            logger.
    """
    handles: list[int] = []
    try:
        try:
            for lock_path in sorted(set(lock_paths)):
                handles.append(_open_lock(lock_path))
        except OSError as exc:
            on_failure(exc)
        yield
    finally:
        for handle in reversed(handles):
            # Suppressed around the unlock alone: a failing ``LOCK_UN`` must not
            # skip the close and leak the descriptor, and closing releases the
            # hold regardless.
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)


def _open_lock(lock_path: Path) -> int:
    """Create *lock_path* if absent, take its exclusive ``flock``, return the fd."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, LOCK_FILE_MODE)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
    except BaseException:
        os.close(handle)
        raise
    return handle


def atomic_write(target: Path, payload: str | bytes) -> None:
    """Write *payload* to a temp file beside *target*, then replace it in one step.

    The temp file is created **in the destination directory** so the replace is a
    same-filesystem rename, which is what makes it atomic: a reader concurrent
    with the replacement sees either the whole previous file or the whole new
    one, never a prefix of either. Any ``BaseException`` — a full disk, a
    ``KeyboardInterrupt`` between the two — unlinks the temp file before
    re-raising, so a failed write leaves the previous file intact and no debris
    behind.

    **This does not create *target*'s parent**, because its callers disagree
    about that and were right to: one mkdirs at its own call site well before it
    writes, the other has nowhere else to do it. Adding the mkdir here would make
    one of them do it twice and hide where the directory comes from.

    Args:
        target: The file to replace.
        payload: Text, written as text; or bytes, written binary. A caller that
            serialises renders its own payload first — no serialisation format
            belongs in the spine.
    """
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        if isinstance(payload, bytes):
            with os.fdopen(fd, "wb") as binary:
                binary.write(payload)
        else:
            with os.fdopen(fd, "w") as text:
                text.write(payload)
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
