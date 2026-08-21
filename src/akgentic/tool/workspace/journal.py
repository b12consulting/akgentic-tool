"""The git journal: linear history, out-of-band commits, graceful absence.

Everything about git lives here. ``#Workspace`` calls this module and knows
nothing about arguments or exit codes — the actor is already close to its
complexity budget, and a journal spread through it would be a journal nobody can
turn off (ADR-036 §4).

**Three properties are load-bearing, and each one has a failure it prevents:**

1. **The repository is a sibling of the tree, never inside it.**
   ``Filesystem._validate_path`` refuses anything that is not
   ``is_relative_to(_root)``, so a ``.git`` *inside* the root would be listable
   by ``workspace_list``, matchable by ``workspace_glob``, greppable, readable —
   and mountable into a sandbox where ``git reset --hard`` destroys the journal.
   At ``<root>.git`` the path simply does not resolve. ``--separate-git-dir`` is
   forbidden for the same reason: it leaves a ``.git`` *file* in the root.

2. **History is linear.** No branch is created, no merge is attempted, no
   rebase, no reset. A three-way merge of two agents' concurrent edits resolves
   textually while leaving the file semantically contradictory, and it fires
   exactly when the system is contended. A rejection is strictly better, because
   the agent holds the context needed to redo the work and a merge algorithm
   does not.

3. **Git is optional; the gate is not.** With no ``git`` on ``PATH``, or
   ``workspace_git=False``, the journal degrades off after **one** warning and
   every gate behaviour stays identical. Losing git loses history, attribution
   and out-of-band *detection*; it does not lose correctness, because the gate
   detects out-of-band *writes* by hashing. No failure in this module can fail a
   mutation — the bytes are already on disk by the time a commit is attempted.

**Four things go wrong on somebody else's machine**, and each is handled at the
point it would bite:

- a git directory whose name is not ``.git``, initialised without a work tree,
  becomes **bare** — and every later ``add`` fails with *"this operation must be
  run in a work tree"*, at runtime, not in anyone's test. ``--work-tree`` is
  passed at init and ``core.bare`` is set to false explicitly;
- ``--author`` sets the *author*; the **committer** still comes from
  configuration, and a machine with no ``user.email`` fails the commit outright.
  All four identity variables travel in the child environment, which settles
  both at once and means no agent-derived text is ever parsed as a
  ``Name <email>`` string;
- ambient configuration can **hang** the fork: ``commit.gpgsign`` blocks on a
  passphrase prompt and ``init.templateDir`` installs hooks. Both turn a ~10 ms
  fork into an indefinite one, on the single thread every mutation in the team
  shares. ``-c commit.gpgsign=false`` and ``--no-verify`` remove both;
- an inherited ``GIT_*`` environment — a developer running the suite inside a
  repository, or CI — reaches the child, and so does that machine's
  ``~/.gitconfig``. Explicit flags beat ``GIT_DIR``, but the identity variables
  have no flag equivalent, so the child environment is scrubbed of every
  ``GIT_`` key before ours are set, and both configuration files are switched
  off with it — ``core.excludesFile`` and ``core.autocrlf`` change what a commit
  *contains*, not how it looks.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from akgentic.tool.workspace.models import (
    GIT_DIR_SUFFIX,
    GITIGNORE_NAME,
    OUT_OF_BAND_AUTHOR,
    gitignore_seed,
)

logger = logging.getLogger(__name__)

MAX_COMMIT_BODY_CHARS = 500
"""Cap on the agent-supplied text a commit body may carry.

The command string is the one place untrusted input reaches the journal. A
control character would end the subject line early and an unbounded string would
put a whole heredoc into the log, so it is stripped and clipped — and the message
travels through ``-F <file>``, never interpolated into an argument.
"""

IDENTITY_FALLBACK = "unknown-agent"
"""Stands in for an identity that sanitises to nothing.

Neither git identity field may be empty, and an agent whose whole name is
control characters would otherwise produce one.
"""

IDENTITY_DOMAIN = "akgentic"
"""Domain of the synthetic author email. The local part is the agent's id."""

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_ANGLE_RE = re.compile(r"[<>]")
_EMAIL_LOCAL_RE = re.compile(r"[^A-Za-z0-9._-]")


def _scrubbed_env() -> dict[str, str]:
    """This process's environment with every inherited ``GIT_`` key removed.

    A developer running the suite inside a repository, or CI, may export
    ``GIT_DIR``, ``GIT_INDEX_FILE`` or an identity. Explicit flags beat
    ``GIT_DIR``, but the identity variables have no flag equivalent and
    ``GIT_INDEX_FILE`` would silently point the staging area somewhere else.

    Both configuration files go with them. The system one is the rarer of the
    two; the **global** one is the file every developer and every CI image
    actually has, and two of its ordinary settings change what the journal
    records rather than merely how it looks: ``core.excludesFile`` drops an
    agent's own file out of that agent's commit (and logs a warning for every
    mutation thereafter), and ``core.autocrlf`` rewrites the bytes the commit
    holds. Neither is a failure anybody would trace back to ``~/.gitconfig``.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _detail(result: subprocess.CompletedProcess[str]) -> str:
    """The first few hundred characters git had to say about a failure."""
    return (result.stderr or result.stdout).strip()[:400]


def _subject(capability: str, paths: Sequence[str]) -> str:
    """Compose a commit subject — one convention, whether the paths were declared or found.

    Args:
        capability: ``write``, ``edit``, ``exec`` … — the first field.
        paths: The write set. Non-empty; a caller with nothing to record makes
            no commit at all.

    Returns:
        ``<capability>: <path>`` for a single file, else ``<capability>: <n> files``.
    """
    return f"{capability}: {paths[0]}" if len(paths) == 1 else f"{capability}: {len(paths)} files"


def _porcelain_path(line: str) -> str:
    """Extract the path from one ``git status --porcelain`` line.

    The first two characters are the status codes and the third is a space, so
    the path starts at index 3. A rename is reported as ``old -> new``; the new
    name is the one the tree now has.
    """
    path = line[3:] if len(line) > 3 else line.strip()
    _, separator, renamed = path.partition(" -> ")
    return renamed if separator else path


@contextlib.contextmanager
def _message_file(subject: str, body: str) -> Iterator[Path]:
    """Write the commit message to a temporary file and yield its path.

    ``-F <file>`` rather than ``-m <text>`` because the body carries
    agent-supplied text, and a message that travels as an argument is a message
    that can be made to look like something else. The file lives in the system
    temp directory, never in the workspace — one written inside the tree would
    dirty it and land in the very commit it describes.
    """
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 — closed below, unlinked in finally
        mode="w", encoding="utf-8", suffix=".msg", delete=False
    )
    try:
        handle.write(f"{subject}\n\n{body}\n" if body else f"{subject}\n")
        handle.close()
        yield Path(handle.name)
    finally:
        handle.close()
        with contextlib.suppress(OSError):
            os.unlink(handle.name)


def git_dir_for(root: Path) -> Path:
    """Return the repository directory for the workspace tree at *root*.

    Derived in exactly one place, because two derivations that drift give two
    repositories over one tree.

    Args:
        root: The workspace root, already resolved absolute.

    Returns:
        The sibling ``<root>.git`` — outside the tree by construction.
    """
    return root.parent / f"{root.name}{GIT_DIR_SUFFIX}"


def sanitise_name(value: str) -> str:
    """Return *value* usable as a git identity **name**.

    Control characters and angle brackets are removed: a newline would end the
    identity line and ``<`` would open the email field, so either lets an agent
    id say something other than a name.

    Args:
        value: An agent's display name or id — untrusted text.

    Returns:
        The cleaned name, or :data:`IDENTITY_FALLBACK` when nothing survives.
    """
    cleaned = _ANGLE_RE.sub("", _CONTROL_RE.sub("", value)).strip()
    return cleaned or IDENTITY_FALLBACK


def sanitise_command(value: str) -> str:
    """Return *value* usable as commit-body text.

    Args:
        value: An agent-supplied command string — untrusted text.

    Returns:
        The command with control characters collapsed to spaces and the whole
        clipped to :data:`MAX_COMMIT_BODY_CHARS`, or ``""`` when nothing
        survives. A body is optional, so an empty result is simply no body.
    """
    cleaned = " ".join(_CONTROL_RE.sub(" ", value).split())
    if len(cleaned) > MAX_COMMIT_BODY_CHARS:
        cleaned = cleaned[:MAX_COMMIT_BODY_CHARS] + " …"
    return cleaned


def sanitise_email_local(value: str) -> str:
    """Return *value* usable as the local part of a git identity **email**.

    Args:
        value: An agent's id — untrusted text.

    Returns:
        The cleaned local part, or :data:`IDENTITY_FALLBACK` when nothing
        survives.
    """
    cleaned = _EMAIL_LOCAL_RE.sub("-", value).strip("-")
    return cleaned or IDENTITY_FALLBACK


class Identity:
    """Who a commit is attributed to, as two already-sanitised fields.

    Built from an agent's registered display name and its id: the name is what a
    human reads in the log, the id is what makes two agents sharing a name
    distinguishable. An unregistered agent falls back to its id as the name —
    degraded, never broken.
    """

    __slots__ = ("email", "name")

    def __init__(self, name: str, email_local: str) -> None:
        self.name = sanitise_name(name)
        self.email = f"{sanitise_email_local(email_local)}@{IDENTITY_DOMAIN}"

    @classmethod
    def out_of_band(cls) -> Identity:
        """The identity for changes no agent in this team made."""
        return cls(OUT_OF_BAND_AUTHOR, OUT_OF_BAND_AUTHOR)


class GitJournal:
    """A linear git history of one workspace tree, or nothing at all.

    Every method is a no-op once the journal is off, so the actor never has to
    ask. Nothing here raises: a failure is logged and the mutation — whose bytes
    are already on disk — stands.
    """

    def __init__(self, root: Path, *, enabled: bool, timeout_s: float) -> None:
        """Prepare a journal over *root*; nothing runs until :meth:`initialise`.

        Args:
            root: The workspace root, resolved absolute.
            enabled: The card's ``workspace_git``. False turns the journal off
                before ``git`` is even looked for.
            timeout_s: Wall-clock budget for a single ``git`` invocation.
        """
        self._root = root
        self._git_dir = git_dir_for(root)
        self._enabled = enabled
        self._timeout_s = timeout_s
        self._git: str | None = None

    @property
    def enabled(self) -> bool:
        """Whether commits are being recorded at all."""
        return self._enabled and self._git is not None

    @property
    def git_dir(self) -> Path:
        """The sibling repository directory this journal writes to."""
        return self._git_dir

    def initialise(self) -> bool:
        """Resolve ``git``, create the repository if absent, and report readiness.

        Exactly one warning is emitted on every path that turns the journal off,
        which is what keeps a git-less host from logging once per mutation.

        Returns:
            True when the journal is live, False when it degraded off.
        """
        if not self._enabled:
            logger.warning(
                "Workspace %s: git journal disabled by configuration — mutations are "
                "still gated, but nothing is recorded",
                self._root.name,
            )
            return False
        if self._root.name.endswith(GIT_DIR_SUFFIX):
            # A workspace literally named "<name>.git" has the same directory as
            # workspace "<name>"'s journal. Operator-set, not agent-set — but the
            # failure is destructive and confusing, so refuse rather than explain.
            logger.warning(
                "Workspace %s: name collides with a sibling journal directory — git "
                "journal disabled. Rename the workspace to record history.",
                self._root.name,
            )
            self._enabled = False
            return False
        if self._git_dir.exists() and not (self._git_dir / "HEAD").is_file():
            # The same collision from the other side, and this is its destructive
            # half: workspace "foo" journals to "foo.git", which is *workspace*
            # "foo.git"'s tree. Initialising there would scatter HEAD, config,
            # objects/ and refs/ through another team's workspace — listable,
            # readable and writable by its agents, one of whom overwrites config
            # and takes this journal with it. Refusing costs the history of one
            # misnamed workspace; not refusing costs another team's tree.
            logger.warning(
                "Workspace %s: %s exists and is not a repository — git journal disabled "
                "rather than initialised over it.",
                self._root.name,
                self._git_dir,
            )
            self._enabled = False
            return False
        resolved = shutil.which("git")
        if resolved is None:
            logger.warning(
                "Workspace %s: git is not on PATH — mutations are still gated, but "
                "nothing is recorded",
                self._root.name,
            )
            self._enabled = False
            return False
        self._git = resolved
        return self._create_repository()

    def seed_gitignore(self, write: Callable[[str, bytes], None]) -> None:
        """Write the ignore file through the actor's own backend, only if absent.

        An existing ``.gitignore`` is **never** overwritten: an agent may have
        written its own, and a seeded default is not worth destroying it for.

        The file is a real file in the tree — listable, readable, and gated like
        any other. That is accepted rather than worked around; the alternative is
        a second hidden path into the workspace.

        Args:
            write: The actor's ``Filesystem.write``, so the seed is staged and
                published exactly as any other file is.
        """
        if not self.enabled or (self._root / GITIGNORE_NAME).exists():
            return
        write(GITIGNORE_NAME, gitignore_seed().encode("utf-8"))

    def is_dirty(self) -> bool:
        """Whether the tree holds changes no commit has taken yet."""
        return bool(self.changed_paths())

    def changed_paths(self) -> list[str]:
        """Every path the tree has changed since the last commit.

        ``-uall`` is not optional, and this is the call story 29-5's mutation
        test guards. Bare ``--porcelain`` collapses an untracked *directory* to a
        single entry, so a build that creates ``dist/`` with forty files inside
        is reported as ``dist/`` — one path where there are forty. That is wrong
        for exactly the case exec exists for, because exec mostly *creates*
        files. Do not simplify the flag away.

        Returns:
            The changed paths, or an empty list when the tree is clean, the
            journal is off, or git could not answer.
        """
        if not self.enabled:
            return []
        result = self._run(["status", "--porcelain", "-uall"])
        if result is None:
            return []
        if result.returncode != 0:
            self._warn("status", result)
            return []
        return [_porcelain_path(line) for line in result.stdout.splitlines() if line.strip()]

    def commit_out_of_band(self) -> None:
        """Commit whatever is in the tree as ``out-of-band``, if anything is.

        Called before a mutation touches disk, so an upload, a previous timed-out
        exec or a second team's write is never folded into the next agent's
        commit. The dirt exists whether or not the mutation is then accepted, and
        it belongs to nobody either way.
        """
        if not self.enabled or not self.is_dirty():
            return
        self._commit(["-A"], Identity.out_of_band(), "out-of-band: changes from outside the tools")

    def commit_paths(self, paths: Sequence[str], identity: Identity, capability: str) -> None:
        """Commit exactly *paths* as *identity*'s one mutation.

        Staging is by explicit pathspec, never a bare ``git add -A``: a gated
        mutation knows precisely which paths it wrote, so a write landing between
        publication and commit cannot be swept into an agent's commit. (Exec, in
        29-5, has to *discover* its write set; that is the difference.)

        Args:
            paths: The mutation's own write set, workspace-relative.
            identity: Who to attribute it to.
            capability: ``write``, ``edit``, ``patch`` … — the commit subject's
                first field.
        """
        if not self.enabled or not paths:
            return
        self._commit(["-A", "--", *paths], identity, _subject(capability, paths))

    def commit_discovered(self, identity: Identity, capability: str, detail: str = "") -> None:
        """Commit whatever the tree now shows, as *identity*'s one run.

        The counterpart of :meth:`commit_paths`, and the difference is the whole
        point of it. A gated mutation **declares** its write set, so it stages by
        explicit pathspec and a write landing in between cannot be swept into an
        agent's commit. Exec cannot declare anything, so its write set has to be
        **discovered** — which is the one thing git is genuinely needed for here:
        not branching, not merging, but post-hoc discovery of mutations nobody
        named.

        A run that changed nothing adds no commit and is not a failure.

        Args:
            identity: Who to attribute the run to.
            capability: The commit subject's first field.
            detail: Agent-supplied text for the commit **body** — the command
                string. Sanitised here, and passed through a message file rather
                than an argument. It never reaches the subject: a subject is not
                the place for untrusted text.
        """
        if not self.enabled:
            return
        paths = self.changed_paths()
        if not paths:
            return
        self._commit(["-A"], identity, _subject(capability, paths), body=sanitise_command(detail))

    ##
    ## Everything below runs git
    ##
    def _create_repository(self) -> bool:
        """Create or reuse the sibling repository, and prove it is not bare."""
        created = self._run(["init", "-b", "master"])
        if created is None:
            return False
        if created.returncode != 0:
            # An older git has no `init -b`; that is a degradation, not a crash.
            self._disable(f"the repository could not be initialised: {_detail(created)}")
            return False
        # Set explicitly rather than trusting flag order: a git directory whose
        # name is not ".git" is exactly the shape that comes out bare, and a bare
        # repository fails every later `add` instead of failing here.
        configured = self._run(["config", "core.bare", "false"])
        if configured is None:
            return False
        if configured.returncode != 0:
            self._disable(f"the repository has no work tree: {_detail(configured)}")
            return False
        return True

    def _commit(
        self, add_args: list[str], identity: Identity, subject: str, body: str = ""
    ) -> None:
        """Stage, then commit. A commit with nothing staged is a no-op, not a failure."""
        staged = self._run(["add", *add_args])
        if staged is None:
            return
        if staged.returncode != 0:
            self._warn("add", staged)
            return
        with _message_file(subject, body) as message_path:
            committed = self._run(
                ["commit", "--no-verify", "-F", str(message_path)],
                env=self._child_env(identity),
            )
        if committed is None or committed.returncode == 0:
            return
        if committed.returncode == 1:
            # git exits 1 with nothing to commit — a mkdir git cannot record, or
            # an edit that changed no bytes. Say nothing.
            logger.debug("Workspace %s: nothing to commit for %r", self._root.name, subject)
            return
        self._warn("commit", committed)

    def _base(self) -> list[str]:
        """The invariant prefix of every invocation."""
        assert self._git is not None
        return [
            self._git,
            "--git-dir",
            str(self._git_dir),
            "--work-tree",
            str(self._root),
            "-c",
            "commit.gpgsign=false",
        ]

    def _run(
        self, args: list[str], env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str] | None:
        """Run one git invocation under an explicit timeout.

        Returns:
            The completed process, or ``None`` when the journal has just been
            disabled — the caller stops rather than reading a result it has not
            got.
        """
        try:
            return subprocess.run(
                self._base() + args,
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                env=env if env is not None else _scrubbed_env(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            # A hung git that stayed enabled would cost every subsequent mutation
            # the same wall clock, on the one thread the whole team's mutations
            # share. One timeout is a lost commit; a timeout per mutation is a
            # wedged team.
            self._disable(f"git exceeded its {self._timeout_s}s budget")
            return None
        except OSError as exc:
            self._disable(f"git could not be spawned: {exc}")
            return None

    @staticmethod
    def _child_env(identity: Identity) -> dict[str, str]:
        """The environment a commit runs in: ours scrubbed, then our identity.

        Identity travels here rather than in a formatted ``--author=`` string
        because an agent id is untrusted text and a formatted field is where it
        would get to be something other than a name. All four variables are set,
        not just the author pair: ``--author`` leaves the *committer* to come
        from configuration, and a machine with no ``user.email`` fails the commit
        outright.
        """
        env = _scrubbed_env()
        env.update(
            {
                "GIT_AUTHOR_NAME": identity.name,
                "GIT_AUTHOR_EMAIL": identity.email,
                "GIT_COMMITTER_NAME": identity.name,
                "GIT_COMMITTER_EMAIL": identity.email,
            }
        )
        return env

    def _warn(self, command: str, result: subprocess.CompletedProcess[str]) -> None:
        """Report a non-zero exit. The journal stays on: one bad commit is not a broken git."""
        logger.warning(
            "Workspace %s: git %s exited %d — %s",
            self._root.name,
            command,
            result.returncode,
            _detail(result),
        )

    def _disable(self, reason: str) -> None:
        """Turn the journal off for the life of the actor, with one warning."""
        if self.enabled:
            logger.warning(
                "Workspace %s: git journal disabled — %s. Mutations are still gated.",
                self._root.name,
                reason,
            )
        self._enabled = False
