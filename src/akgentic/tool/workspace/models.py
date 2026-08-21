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

from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.utils.serializer import SerializableBaseModel

DEFAULT_MAX_OBSERVATIONS_PER_AGENT = 256
"""Per-agent bound on the observation map.

Bounds the **path** dimension only — see ``WorkspaceActor.record_observation``
for why the agent dimension is deliberately left unbounded.
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


class WorkspaceConfig(BaseConfig):
    """Configuration of the ``#Workspace-<workspace_name>`` singleton.

    Attributes:
        workspace_name: The resolved workspace, i.e. the card's ``workspace_id``
            or the team id. It names the tree the actor owns, and it is also the
            suffix of the actor's name — ``getChildrenOrCreate`` keys on that
            name, so the two must be derived from one value or two cards with
            different workspaces would collapse onto one actor owning one tree.
        max_observations_per_agent: Cap on the per-agent observation map.
    """

    workspace_name: str
    max_observations_per_agent: int = DEFAULT_MAX_OBSERVATIONS_PER_AGENT


class WorkspaceState(BaseState):
    """Persisted actor state — deliberately empty of observation data.

    ``Akgent`` is generic over a state type, so the actor needs one. What it must
    not carry is the observation map: reads are the majority of workspace
    traffic, and a snapshot per recorded read would put an event-store write on
    the read path that ADR-036's NFR1 exists to keep free. Observations live as a
    plain actor instance attribute and do not survive a resume, which degrades
    towards *refusing* a later write rather than accepting a stale one.
    """
