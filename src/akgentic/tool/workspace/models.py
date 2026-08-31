"""Domain models for the ``#Workspace`` tool actor, and the one content digest.

Everything that crosses the actor boundary here is a Pydantic model.

:func:`content_sha` is deliberately the **only** definition of the digest in this
package. The write gate hashes the live file from disk and compares it against
what a read recorded through :class:`Observation`; two independently written
digest expressions — one over raw bytes, one over decoded and line-ending
normalised text — would make every comparison fail closed, which looks like a
working gate right up until nobody can overwrite anything (ADR-036 §3).
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.utils.serializer import SerializableBaseModel

DEFAULT_MAX_OBSERVATIONS_PER_AGENT = 256
"""Per-agent bound on the observation map.

Bounds the **path** dimension only — see ``WorkspaceActor.record_observation``
for why the agent dimension is deliberately left unbounded.
"""

DEFAULT_MAX_TRACKED_WRITERS = 512
"""Bound on the last-writer map, which is keyed by **path** across all agents.

Deliberately a separate constant from the observation cap: that one bounds one
agent's paths, this one bounds the whole tree's, so a single number would be
wrong at one end or the other. Both exist for the same reason — an uncapped map
on a team singleton leaks for the life of the team.
"""

MAX_REJECTION_DIFF_LINES = 200
"""Cap on the unified diff a refusal carries back to the model.

A refusal is a ``RetriableError``, so its whole text lands in the agent's next
turn. Uncapped, a stale write to a large file would put that file's entire diff
into the context window — the refusal, not the write, would then be what breaks
the turn. Two hundred lines is well past what an agent needs to see that its
change collided, and the notice below the cut says what was elided.
"""

PERM_ERR_MSG = "Path escapes workspace root — use a path relative to the workspace"
"""Refusal text for a path that resolves outside the workspace root.

Shared by the read closures and the actor's mutation methods so the two cannot
drift; agents see one wording whichever side rejects them.
"""

WRITE_DENIED_MSG = (
    "The change was not published: the operating system refused this process permission to "
    "replace the file. The path is correct and inside the workspace — it did not escape it. "
    "A file created by a sandboxed run under a different user does this. Retry, or write to a "
    "different path."
)
"""Refusal text for an OS-level permission denial while publishing.

Deliberately **not** :data:`PERM_ERR_MSG`. The two arrive as the same
``PermissionError`` and mean opposite things: one says the path is illegal, the
other says the path is fine and the file is not replaceable. Told the first when
the second is true, an agent rewrites a correct path indefinitely — the one
refusal in the gate with no recoverable next step.

The distinction became reachable with ``workspace_exec``: publication is by
rename, so a file a container created as root is a file the host process may not
replace on the next write.
"""

PUBLISH_LOST_MSG = (
    "The change was not published: a staged file vanished before it could be put in place. "
    "Nothing about the file changed and nobody else wrote it — retry exactly the same change."
)
"""Refusal text for a file lost between staging and ``os.replace``.

This is the sweep race, and it is real: ``WorkspaceTool(workspace_id="shared")``
is a supported configuration, a tool actor's unicity domain is the *team*, so two
teams over one tree means two actors each sweeping it at start. One can unlink
the other's staged file in the sub-millisecond window inside ``Filesystem.write``.

The wording deliberately does **not** reuse a staleness reason. Nothing about the
file changed, and telling the agent it did would send it re-reading a file that
is exactly as it left it, then redoing work that was already correct.
"""

DEFAULT_GIT_TIMEOUT_S = 15.0
"""Wall-clock budget for a single ``git`` invocation.

Comfortably below the orchestrator's 30 s stop backstop, because every
invocation runs on the actor's single thread — the one every mutation in the
team shares. A budget above the backstop would let one hung fork outlive the
teardown that is trying to reclaim it.
"""

STAGING_SWEEP_GRACE_S = 30.0
"""How recently a staging file may have been touched to survive the startup sweep.

A staging file this young is being written **now** by somebody, and with a
``workspace_id`` shared across two teams that somebody may not be us. Sweeping it
would make the other team's ``os.replace`` raise, turning a healthy write into a
refusal. Orphans, by contrast, are minutes or restarts old — no real value of
this constant separates the two badly.
"""

GIT_DIR_SUFFIX = ".git"
"""Suffix of the sibling repository directory: workspace ``foo`` journals to ``foo.git``."""

GITIGNORE_NAME = ".gitignore"

OUT_OF_BAND_AUTHOR = "out-of-band"
"""Author of every commit no agent in this team is responsible for."""

_EXEC_DEBRIS = ("__pycache__/", "*.pyc", ".venv/", "node_modules/")


def gitignore_seed() -> str:
    """Return the ignore file written once at journal init.

    Derived from what this package actually writes, not from a generic template.
    Every pattern is anchor-free so it matches at **every** depth: sidecars are
    written beside their source file, wherever that is.

    Without this the tree is dirty continuously — read paths write sidecars, so
    a document read or an image view dirties the tree, and every agent's commit
    would then be preceded by an ``out-of-band`` commit of regenerable noise.

    Returns:
        The file's full text, ending in a newline.
    """
    from akgentic.tool.workspace.readers import _MIME_MAP  # noqa: PLC0415 — avoids a cycle

    lines = [
        "# Seeded once by the workspace journal, and never rewritten.",
        "# Edit or delete it freely — an existing .gitignore is left alone.",
        "",
        "# Atomic-write staging files: .<name>.<32 hex>.tmp",
        ".*.tmp",
        "",
        "# Extracted-document sidecars: .<name>.md",
        ".*.md",
        "",
        "# Resized-image sidecars: .<stem>.<ext>.<max_dim>.<ext>",
        *(f".*{suffix}" for suffix in sorted(_MIME_MAP)),
        "",
        "# Exec debris",
        *_EXEC_DEBRIS,
        "",
    ]
    return "\n".join(lines)


Precondition = str | Literal["absent"]
"""What must hold of a file before an agent may replace it wholesale.

Either a digest the live file must still match, or ``"absent"`` — the file must
not exist, which is what "you have not read this" means for a whole-file write.

There is deliberately **no** third value meaning "no check". One value standing
for both *must-not-exist* and *no-precondition* turns a forgotten argument into
a silent ungated clobber, and a bypass an LLM can reach for destroys the
mechanism the first time a rejection is not understood (ADR-036 §3).
"""


def content_sha(data: bytes) -> str:
    """Return the digest of *data*, over the raw bytes and nothing else.

    Args:
        data: The exact bytes the backend returned for the file.

    Returns:
        Hex SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()


class Observation(SerializableBaseModel):
    """What one agent last saw of one file.

    Attributes:
        sha: Digest of the file's raw bytes, from :func:`content_sha`. It
            describes the **file**, never the window a paginated read displayed.
        full: True only when the read covered the whole file. A page is not a
            precondition for a whole-file overwrite, so the flag has to travel
            with the digest rather than be inferred from it.
    """

    sha: str
    full: bool


class MutationStatus(StrEnum):
    """How a mutation ended, and therefore what the tool callable does with it.

    The three values map onto the package's existing error contract exactly, and
    that mapping is the whole point of the enum — the actor decides, the closure
    only translates:

    - :attr:`ACCEPTED` — the closure **returns** the message.
    - :attr:`REJECTED` — the closure **raises** ``RetriableError(message)``. The
      gate's refusals live here, which is what carries the diff into the model's
      next turn; so do the conditions that already raised, such as a path that
      escapes the root or a file that is not there.
    - :attr:`FAILED` — the closure **returns** the message even though nothing
      was written. This is what preserves ``[ERROR] old_string not found in …``
      and ``[ERROR] <path>: <exc>`` as *returned* strings rather than raises.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class MutationOutcome(SerializableBaseModel):
    """What the actor did with one mutation, and what to tell the agent.

    Attributes:
        status: See :class:`MutationStatus` — decides return-versus-raise.
        message: The exact text the agent receives. On the accept path it is the
            unchanged confirmation string or diff; on a refusal it is the
            rejection the actor composed.
    """

    status: MutationStatus
    message: str


class LastWrite(SerializableBaseModel):
    """The agent behind the most recent accepted mutation of one path.

    Attributes:
        agent_id: Identity of the writing agent, as a string.
        sha: Digest of the bytes that agent wrote.

    The digest is what keeps attribution honest. A refusal may name this agent
    only while the live file still hashes to ``sha``; once anything else has
    touched the path, the last accepted writer is no longer the author of what
    is on disk, and naming them would pin an out-of-band change — an upload, a
    sandbox run, another team — on whichever agent happened to write last.
    """

    agent_id: str
    sha: str


class WorkspaceConfig(BaseConfig):
    """Configuration of the ``#Workspace-<workspace_name>`` singleton.

    Attributes:
        workspace_name: The resolved workspace, i.e. the card's ``workspace_id``
            or the team id. It names the tree the actor owns, and it is also the
            suffix of the actor's name — ``getChildrenOrCreate`` keys on that
            name, so the two must be derived from one value or two cards with
            different workspaces would collapse onto one actor owning one tree.
        max_observations_per_agent: Cap on the per-agent observation map.
        max_tracked_writers: Cap on the path-keyed last-writer map, which the
            gate consults only to name the other writer in a refusal.
        git_journal: Whether to keep a git journal of accepted mutations. The
            gate is unaffected either way — it is pure Python and independent.
        git_timeout_s: Wall-clock budget for one ``git`` invocation.
    """

    workspace_name: str
    max_observations_per_agent: int = DEFAULT_MAX_OBSERVATIONS_PER_AGENT
    max_tracked_writers: int = DEFAULT_MAX_TRACKED_WRITERS
    git_journal: bool = False
    git_timeout_s: float = DEFAULT_GIT_TIMEOUT_S


class WorkspaceState(BaseState):
    """Persisted actor state — deliberately empty of observation data.

    ``Akgent`` is generic over a state type, so the actor needs one. What it must
    not carry is the observation map: reads are the majority of workspace
    traffic, and a snapshot per recorded read would put an event-store write on
    the read path that ADR-036's NFR1 exists to keep free. Observations live as a
    plain actor instance attribute and do not survive a resume, which degrades
    towards *refusing* a later write rather than accepting a stale one.
    """
