"""Capability parameters and the seeded-resource models for :class:`WorkspaceTool`.

Configuration only. Every class here is a :class:`~akgentic.tool.core.BaseToolParam`
subclass — or, for :class:`Resource`, a plain
:class:`~akgentic.core.utils.SerializableBaseModel` — carrying what a capability
is configured *with*, never what its callable is *called* with (ADR-020).

No helper and no factory lives in this module. It was once the leaf of ``card/``'s
import graph, importing no sibling at all; it is not that any more and the change
is worth stating rather than leaving a reader to discover it. Re-exporting a
capability's parameters means **importing that capability**, so importing
``card.params`` now executes ``read/``, ``write/`` and ``rag/`` — and, through
``rag/``, the splitter, the index worker and the policy record. It is still the
leaf of the *card's* own graph in the sense that matters: it imports no other
module under ``card/``, so a sibling may take its parameters from here without
depending on the façade.

**Four capabilities' parameters are re-exported, not defined here.** The read
capability's six live in :mod:`akgentic.tool.workspace.read.params`, the write
capability's six in :mod:`akgentic.tool.workspace.write.params`, the retrieval
capability's three in :mod:`akgentic.tool.workspace.rag.params` and the exec
capability's one in :mod:`akgentic.tool.workspace.execution.params`, each beside
the closures they configure, because importing anything under ``card/`` executes
``card/__init__.py`` and therefore every other capability — so a capability
module that took its parameters from here would depend on all of them. For exec
that is not merely a dependency but a **cycle**: ``card/__init__.py`` imports
``ExecFactories`` from ``execution/card.py``.

The re-export is what keeps stored records readable. ``serialize_type`` stamps
``f"{cls.__module__}.{cls.__name__}"`` into ``__model__`` on every
``SerializableBaseModel``, and ``BaseToolParam`` is one, so every card a
deployment persisted with a read, write or retrieval parameter set explicitly
carries ``akgentic.tool.workspace.card.params.<Name>``. Reading it back is
``import_module`` plus ``getattr`` on exactly that path
(:func:`akgentic.core.utils.deserializer.import_class`); a path that has gone
raises ``UnresolvableClassError``, which is how a stored team's tool card becomes
a bad record rather than a card. ``workspace/tool.py`` is on disk for the same
reason after an earlier decomposition.

``Resource`` and ``ResourceType`` stay defined here: they are the card's own
seeding vocabulary and belong to no capability.
"""

from __future__ import annotations

import base64
from enum import StrEnum

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.workspace.execution.params import WorkspaceExec
from akgentic.tool.workspace.rag.params import (
    WorkspaceRagIndex,
    WorkspaceRagList,
    WorkspaceRagSearch,
)
from akgentic.tool.workspace.read.params import (
    ExpandMediaRefs,
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceRead,
    WorkspaceView,
)
from akgentic.tool.workspace.write.params import (
    WorkspaceDelete,
    WorkspaceEdit,
    WorkspaceMkdir,
    WorkspaceMultiEdit,
    WorkspacePatch,
    WorkspaceWrite,
)

# mypy strict implies ``no_implicit_reexport``, so the sixteen re-exported names
# need an explicit export to keep serving the module path stored records name.
# ``__all__`` does it in sixteen lines where ``X as X`` aliases would cost an
# import statement each — the spelling ``workspace/tool.py`` already uses, and for
# the same reason.
#
# **Nothing in this package imports the sixteen from here.** ``card/__init__.py``
# and every capability module take them from ``read/params.py``,
# ``write/params.py``, ``rag/params.py`` and ``execution/params.py``, where they
# are defined, so this re-export exists for exactly one purpose — the stored
# ``__model__`` markers described above — and ``test_read_capability.py``,
# ``test_write_capability.py``, ``test_rag_card.py`` and ``test_exec.py`` are what
# hold it in place. Were production to import them from here instead, deleting the
# re-export would break the package loudly and the compatibility path would have
# no guard of its own.
__all__ = [
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
    "WorkspaceRagIndex",
    "WorkspaceRagList",
    "WorkspaceRagSearch",
    "WorkspaceRead",
    "WorkspaceView",
    "WorkspaceWrite",
]


class ResourceType(StrEnum):
    """Encoding of a seeded resource's ``content`` field.

    Acts as the explicit encoding discriminator for a :class:`Resource`: it
    decides how ``content`` is decoded into bytes (see :meth:`Resource.to_bytes`).
    Encoding is always explicit — never inferred from the filename extension.
    """

    TEXT = "text"  # content is UTF-8 text, written verbatim
    IMAGE = "image"  # content is base64-encoded binary, decoded before write


class Resource(SerializableBaseModel):
    """A file seeded into the team workspace at team-creation time.

    Fully Pydantic-serializable: primitive fields plus a :class:`ResourceType`
    ``StrEnum`` only, so it round-trips cleanly through ``model_dump`` /
    ``model_validate``. The file extension lives in ``file_name`` (e.g.
    ``logo.png``); ``file_type`` carries the encoding discriminator, not a MIME
    type.
    """

    file_name: str
    file_type: ResourceType = ResourceType.TEXT
    content: str

    def to_bytes(self) -> bytes:
        """Decode ``content`` into the bytes to write to the workspace.

        Returns:
            ``base64.b64decode(content)`` when ``file_type`` is
            :attr:`ResourceType.IMAGE`, else ``content.encode("utf-8")``.

        Raises:
            binascii.Error: If ``file_type`` is :attr:`ResourceType.IMAGE` and
                ``content`` is not valid base64.
        """
        if self.file_type is ResourceType.IMAGE:
            return base64.b64decode(self.content)
        return self.content.encode("utf-8")
