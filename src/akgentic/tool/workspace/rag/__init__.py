"""The retrieval capability of :class:`WorkspaceTool` — index, list, search.

It holds one capability whole: the three factories, its parameters
(``rag/params.py``), the splitter (``rag/splitter.py``), the context state and
its render (``rag/context.py``), the search itself (``rag/search.py``), the index
worker (``rag/worker.py``), the actor-side indexing pipeline (``rag/actor.py``)
and the tree-policy record. It may import the package **spine** —
``workspace.py``, ``models.py``, ``readers.py``, ``event.py`` — plus
``akgentic.tool.core`` and ``akgentic.tool.vector_store``.

**Two of the three callables reach no actor at all.** A search and a listing read
the document records through the
:class:`~akgentic.tool.workspace.documents.cache.DocumentCache` this card built
in ``observer()``, on the calling agent's own thread; only
``workspace_rag_index`` is an ask, because queueing a file for indexing is
dispatch and the pipeline's children need a mailbox to report to (ADR-053
Decision 6).

**``documents/`` is not spine and is named here on this capability's own row.**
It stopped being spine in story 55-8, when the two document caps left
``WorkspaceConfig`` and ``workspace/models.py`` stopped naming the package at all.
What this capability reaches is stated per module rather than as a directory:
``documents.models`` for the records and the id rule, and — since the pipeline
moved in — ``documents.store`` for a ``DocumentEntry`` and ``documents.cache``
for the object the card announces. Reaching any *other* module of that package
would be a new entry on the row and the guard would say so.

It must import nothing under ``card/`` at all: importing ``card.anything``
executes ``card/__init__.py``, which pulls in the journal, the exec machinery, the
vector-store registry and the actor. The rule is enforced by
``tests/workspace/test_capability_import_closure.py``, which takes the transitive
closure of this package's own imports and checks it against an allow-list — not
by convention, and not by this paragraph.

**It names one other capability, and that is admitted rather than hidden.**
:meth:`RagFactories._rag_reader` resolves the card's extraction configuration,
which is nested inside ``WorkspaceRead.document_reader``, so ``rag/`` imports
``read/params.py``. It is benign in a way ``write/``'s old dependency on
``journal/`` was not: ``read/`` is the *always-available* capability, present in
every bind, so depending on it is depending on machinery that is there anyway.
The structural answer — an extraction-configuration vocabulary at the spine — is
a cross-capability decision for the ADR, not a story's to take.

So the stated limit is this: ``rag/`` is **deletable** without touching another
capability, and it is **not standalone** — it names ``read/``, which is always
present.

**``actor/__init__.py`` imports ``rag/actor.py``, and the deletability sentence
survives it.** The actor is the *assembly point*, not a capability — the role
``card/__init__.py`` plays for :class:`RagFactories` — so its import is the same
edge in the other direction and neither one is a capability depending on another.
Deleting this directory means the assembly point stops composing a mixin it no
longer has, which is what "deletable" has always meant here.

**The tree owns its retrieval policy, and that record lives here.**
:class:`TreePolicy` is written to ``<meta>/policy.yaml`` — a top-level file
beside ``exec.lock``, not inside ``rag/`` (one file per *document*) and not
inside ``locks/`` (lock files only). It carries the chunking parameter the whole
tree is chunked with and the two document caps the whole tree is held to, which
were per-actor state until this module got them: two teams over one tree get two
actors, one ``<meta>/index/`` and one ``scope=``, and
``chunk_id(scope, path, source_sha, ordinal)`` carries no team — so two chunkings
minted rows into one collection and a hit resolved against the wrong offsets.

**A disagreement raises; it never merges.** Silently winning, silently losing and
"merging" are three spellings of the same defect, so a second binder whose
chunking or caps differ from what the tree publishes fails its bind, naming both
values (:data:`WORKSPACE_POLICY_REFUSED`). That raise fails card binding and
therefore team creation, deliberately, exactly as the sharing gate's does; it
must never be caught and turned into a fallback. A section the record does not
carry is nothing to disagree with, so an absent one is filled rather than
contested.

**No migration is owed and none is written.** The three-segment layout shipped to
nobody and no deployment holds workspace data worth keeping (ADR-052
§Consequences). A ``<meta>`` with no ``policy.yaml`` is simply a tree whose policy
has not been published yet; the next retrieval binder publishes it. There is no
version field and no compatibility branch.

**Only a binder that has retrieval on publishes; every binder is held to what is
published.** The document caps govern the extraction cache, and that cache is
filled from the **read** path, which needs no retrieval — so a tree can hold a
cached extraction under a policy nobody published. Publishing at every bind would
close that and would create ``<meta>`` for every tree that ever binds a card,
reversing the laziness ``FileLockBackend.acquire`` and
``YamlDocumentStore.put_document`` both argue for at length. Reversing two stated
decisions to close a hole that only opens when an author hand-declares a cap is
the wrong trade, so the limit is stated instead:

    Two retrieval-off cards over one tree with *different declared*
    ``max_documents`` still evict each other's extractions. Nothing is corrupted —
    the cache is derivable and disposable (ADR-045 §C5) — and no index exists to
    hold two chunkings. The moment either card enables retrieval, the record is
    published and the disagreement is refused.

**Why the record lives in this module rather than in a spine one.** ADR-053
Decision 1 calls policy spine material, and the obstacle to that has not
dissolved — it has **reversed direction**, and is now stronger than it was.
``TreePolicy``'s chunking field is typed :class:`WorkspaceRagIndex`. While that
class lived in ``card/params.py``, a spine ``workspace/policy.py`` importing it
would have closed a cycle with the very module that imports the spine. Now that
it lives in ``rag/params.py``, no cycle closes — instead the **spine would import
a capability**, which puts ``rag`` into every capability's transitive closure and
therefore into the allow-list every capability stands on. That is exactly the
rot this story removed by taking the splitter and the context state out of it.
The other ways out are unchanged and still worse: typing the section
``dict[str, Any]`` is Golden Rule 1, and enumerating the chunking fields on
:class:`TreePolicy` is the field-drift defect Golden Rule 12 exists for.

**One consequence, stated rather than left to be discovered.**
:meth:`RagFactories._require_tree_policy` runs on **every** bind, retrieval-off
included — every binder is held to what the tree publishes — so a retrieval-off
bind executes a method defined under ``rag/``. That is not a regression of
"enabling nothing costs nothing": :class:`RagFactories` is a mixin on
:class:`~akgentic.tool.workspace.card.WorkspaceTool` whatever the fields say, the
card is the assembler and is deliberately not attributed by the closure guard,
and the cost is the one ``stat`` on an absent file that
:func:`read_tree_policy` already argues for.

:class:`RagFactories` declares **no Pydantic field**. Every field stays on
:class:`~akgentic.tool.workspace.card.WorkspaceTool` in ``card/__init__.py``,
which is what keeps that card's frozen field set meaningful; the annotations
below are inside ``if TYPE_CHECKING:``, so they never reach ``__annotations__``
and Pydantic never collects them.

Putting the wiring here rather than in the façade is deliberate rather than
tidy-minded: ``card/__init__.py`` is already the longest module in the package,
and the exec pair is the precedent for what a capability's wiring looks like —
not a rule that it must live in the façade.

**The import edge runs one way.** ``card/params.py`` re-exports this capability's
three parameters from ``rag/params.py``; nothing here imports back through that
re-export, or through anything else under ``card/``.

:class:`WorkspaceRead` comes from ``read/params.py``, where it is **defined**,
rather than through ``card/params.py``'s re-export of it. That re-export exists
for one purpose — keeping the module path stored ``__model__`` markers name
resolving — and a production importer leaning on it would leave that
compatibility path with no guard of its own.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.core import ContextState, _resolve
from akgentic.tool.vector_store.protocol import (
    PATH_PREFIX_REJECTED,
    PATH_PREFIX_WILDCARDS,
    VectorStoreParam,
)
from akgentic.tool.workspace.rag.context import render_index_state
from akgentic.tool.workspace.rag.params import (
    WorkspaceRagIndex,
    WorkspaceRagList,
    WorkspaceRagSearch,
)
from akgentic.tool.workspace.rag.search import search_documents
from akgentic.tool.workspace.read.params import WorkspaceRead
from akgentic.tool.workspace.readers import DocumentReader
from akgentic.tool.workspace.workspace import meta_dir_for

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from akgentic.tool.vector_store.protocol import VectorStoreService
    from akgentic.tool.workspace.actor import WorkspaceActor
    from akgentic.tool.workspace.documents.cache import DocumentCache

logger = logging.getLogger(__name__)

_UNAVAILABLE = "Retrieval indexing is not available for this workspace."
"""What all three callables answer when this card has no retrieval to offer.

Deliberately one sentence for every cause: an agent should not have to tell "no
vector store is wired" apart from "this card never bound a tree" or "the actor is
gone", because its next step is the same in all three. ``rag/actor.py`` returns
the identical sentence from ``index_paths``, which is the one callable still
behind a mailbox.
"""

_REJECTED_PREFIX = PATH_PREFIX_REJECTED
"""What a search answers for a ``path_prefix`` carrying ``*`` or ``?``.

The characters and the sentence both live in ``vector_store/protocol.py``, beside
the two backends that read a prefix differently, so this capability and the guard
underneath it cannot drift apart. This layer **returns** the sentence rather than
raising it: a wildcard is a mistake the agent can correct from the answer alone.

**One copy, since story 57-1.** ``rag/actor.py`` aliased the same constant, as
defence in depth for an ask with a public-shaped signature; no such ask exists any
more, and a second copy would be a second sentence waiting to drift.
"""

IN_ACTOR_BACKEND = "inmemory"
"""The backend that keeps its index inside the store actor's serialisable state."""

WORKSPACE_LOCAL_BACKEND = "local"
"""The backend a workspace's index lives in — files under the tree's ``<meta>``."""

WORKSPACE_IN_MEMORY_REFUSED = (
    "{card} declares vector_store.backend='{in_actor}', which is not a workspace "
    "backend: an in-memory index is lost with the process, so every row this "
    "workspace persists would claim embeddings that are no longer there. Use "
    "backend='{local}', which indexes into files under the tree's metadata "
    "directory, or name a cluster backend. Declaring no backend at all already "
    "resolves to '{local}' for a workspace."
)
"""Why a workspace may not run on the in-actor index, said at wiring time.

The knowledge graph and the plan keep their **rows** in actor state alongside
their index, so the two are lost and restored together and never disagree. A
workspace's rows are files under ``<meta>`` and outlive every process, so an
in-memory index leaves a persisted ``EMBEDDED`` row over an empty engine after
every restart — the mismatch the deleted re-mark rule existed to repair
(ADR-051 Decision 11).
"""


POLICY_FILE_NAME = "policy.yaml"
"""The tree's published retrieval policy, directly under ``<meta>``.

**Top level, beside ``exec.lock``** — not under ``rag/``, which holds one file per
*document*, and not under ``locks/``, which holds lock files. YAML, like the
document records and for their reason: a human debugging a refused bind reads
this file.
"""

POLICY_LOCK_NAME = "policy"
"""The lock file publishers contend on, inside :data:`LOCKS_DIR_NAME`."""

LOCKS_DIR_NAME = "locks"
"""The directory under ``<meta>`` holding one lock file per contended thing.

**Spelled here rather than imported**, and it is the third such spelling —
``write/gate.py`` and ``vector_store/backends/local.py`` hold the other two.
Importing the gate's would make the retrieval capability import the write
capability at runtime, which is worse than a third copy. One spine helper for the
family is a cross-capability decision, recorded in the story's Open Questions
rather than taken here.
"""

WORKSPACE_POLICY_REFUSED = (
    "{card} asks workspace {path} to use {section}={card_value!r}, but the tree's "
    "published retrieval policy holds {section}={tree_value!r}. Two teams over one "
    "tree share one index and one scope, so a second policy would chunk one file "
    "two ways into one collection or evict the other's cached extractions — there "
    "is no merge that is not one of those. Give this card the tree's value, or "
    "point it at a different workspace_id. The published policy is {file}."
)
"""Why a second binder's policy is refused rather than merged or ignored.

Composed from a module constant with placeholders, the shape
:data:`WORKSPACE_IN_MEMORY_REFUSED` and ``_require_sharing_permitted``'s message
already use, so a spec asserts through the constant and never against a
hand-typed sentence.
"""

WORKSPACE_POLICY_UNREADABLE = (
    "{card} cannot read workspace {path}'s published retrieval policy at {file}: "
    "{reason}. The bind is refused and the file is left in place. A policy is "
    "neither derivable from the tree nor reclaimable by staleness, so proceeding "
    "would let this card impose its own policy on a tree that already had one. "
    "Repair or remove the file, whichever its content says is right."
)
"""Why a policy file that does not parse refuses the bind.

Deliberately the **opposite** call from :meth:`YamlDocumentStore._read`, which
treats a bad parse as a miss because a record is derivable and disposable, and
from :meth:`FileLockBackend.release`, which reclaims a marker by staleness. A
policy has neither property.
"""


class TreePolicy(SerializableBaseModel):
    """What every card binding one tree has to agree about.

    **The chunking half is the shipped parameter stored whole, never its fields
    enumerated.** A chunking field added tomorrow takes part in the comparison by
    construction — Golden Rule 12's reasoning, applied to a comparison rather than
    to a copy. The comparison itself iterates ``model_fields`` for the same
    reason, so a *section* added tomorrow is compared too.

    The two caps are ``int`` because that is what
    :class:`~akgentic.tool.workspace.models.WorkspaceConfig` holds, and ``| None``
    so that *absent* and *declared at today's default* stay distinguishable — the
    distinction ``VectorStoreParam.backend_declared`` exists to keep.

    Attributes:
        chunking: The chunking parameter every card on this tree indexes with, or
            ``None`` when no retrieval binder has published one yet.
        max_documents: The extraction cache's row cap as an author declared it, or
            ``None`` when nobody declared one.
        max_document_chars: The same for the character cap.
    """

    chunking: WorkspaceRagIndex | None = None
    max_documents: int | None = None
    max_document_chars: int | None = None


def policy_file_for(workspace_path: str) -> Path:
    """The file *workspace_path*'s published policy lives in.

    Derived through :func:`~akgentic.tool.workspace.workspace.meta_dir_for` so the
    record lands beside the tree the gate and the journal are guarding, and so no
    read capability can name it.
    """
    return meta_dir_for(workspace_path) / POLICY_FILE_NAME


def read_tree_policy(workspace_path: str, card_name: str) -> TreePolicy | None:
    """Return the policy *workspace_path* publishes, or ``None`` when it publishes none.

    **Creates nothing**: no ``mkdir``, no touch, no ``<meta>``. A tree that has
    never had a retrieval binder costs one ``stat`` on a file that does not exist.

    Args:
        workspace_path: The resolved three-segment path.
        card_name: Card class name, for the error message.

    Returns:
        The published policy, or ``None`` when the file is absent.

    Raises:
        ValueError: When the file exists and cannot be read or does not parse.
            See :data:`WORKSPACE_POLICY_UNREADABLE`.
    """
    file = policy_file_for(workspace_path)
    try:
        raw = file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(_unreadable(card_name, workspace_path, file, exc)) from exc
    try:
        return TreePolicy.model_validate(yaml.safe_load(raw))
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        raise ValueError(_unreadable(card_name, workspace_path, file, exc)) from exc


def _unreadable(card_name: str, workspace_path: str, file: Path, exc: BaseException) -> str:
    """Compose the refusal for a policy file that cannot be read."""
    return WORKSPACE_POLICY_UNREADABLE.format(
        card=card_name, path=workspace_path, file=file, reason=exc
    )


def agreed_tree_policy(
    published: TreePolicy | None, declared: TreePolicy, workspace_path: str, card_name: str
) -> TreePolicy:
    """Return what the tree should hold once *declared* has been admitted, or refuse.

    Three answers per section, and the sections are taken from ``model_fields`` so
    one added tomorrow is compared without this function being edited:

    - the tree carries nothing there → *declared*'s value fills it;
    - the tree's value equals *declared*'s → nothing happens, silently;
    - they differ → :class:`ValueError`, naming both.

    **Disagreement is whole-section inequality.** The chunking section is one
    model compared whole, so a card differing in any field of it — including one
    added after this was written — is refused. A section *this card* leaves
    ``None`` asks for nothing and can disagree with nothing.

    Args:
        published: What the tree already holds, or ``None`` when it holds nothing.
        declared: What this card asks the tree to be.
        workspace_path: The resolved three-segment path, for the error message.
        card_name: Card class name, for the error message.

    Returns:
        The policy the tree should hold — *published* with the absent sections
        this card fills, derived by ``model_copy(update=...)`` so a field added
        tomorrow survives (Golden Rule 12).

    Raises:
        ValueError: On any disagreeing section. See :data:`WORKSPACE_POLICY_REFUSED`.
    """
    if published is None:
        return declared
    updates: dict[str, Any] = {}
    for section in TreePolicy.model_fields:
        mine = getattr(declared, section)
        theirs = getattr(published, section)
        if mine is None or mine == theirs:
            continue
        if theirs is None:
            updates[section] = mine
            continue
        raise ValueError(
            WORKSPACE_POLICY_REFUSED.format(
                card=card_name,
                path=workspace_path,
                section=section,
                card_value=mine,
                tree_value=theirs,
                file=policy_file_for(workspace_path),
            )
        )
    return published.model_copy(update=updates)


def publish_tree_policy(workspace_path: str, declared: TreePolicy, card_name: str) -> None:
    """Admit *declared* into *workspace_path*'s published policy, under the lock.

    A read-modify-write: the record is re-read **inside** the hold and written
    temp-then-``replace()``, so two processes binding one tree at the same instant
    produce one published policy and, where they disagree, exactly one refusal.
    The re-read inside the hold is the load-bearing half — a hold taken around a
    decision made on a read from *before* it serialises nothing.

    Args:
        workspace_path: The resolved three-segment path.
        declared: What this card asks the tree to be.
        card_name: Card class name, for the error message.

    Raises:
        ValueError: When the file does not parse, or when a section disagrees.
    """
    meta_dir = meta_dir_for(workspace_path)
    with _policy_hold(meta_dir, workspace_path):
        published = read_tree_policy(workspace_path, card_name)
        agreed = agreed_tree_policy(published, declared, workspace_path, card_name)
        if agreed != published:
            _write_tree_policy(meta_dir / POLICY_FILE_NAME, agreed)


@contextlib.contextmanager
def _policy_hold(meta_dir: Path, workspace_path: str) -> Iterator[None]:
    """Hold ``<meta>/locks/policy`` for the duration of the block.

    The idiom is :meth:`~akgentic.tool.workspace.card.CardGate._hold`'s and
    :meth:`~akgentic.tool.vector_store.backends.local.LocalBackend._hold`'s: an
    exclusive ``flock`` on a lazily created file, unlocked and closed in
    ``finally``, **never unlinked** — unlinking would let a second process create a
    fresh inode and take a hold that excludes nobody.

    A ``<meta>`` whose locks directory cannot be created logs one WARNING and
    proceeds unserialised, which is ``CardGate._hold``'s stated choice copied
    rather than re-decided: what is lost is the ordering between two simultaneous
    first binds, and the comparison inside still runs against whatever is on disk.
    """
    handle: int | None = None
    try:
        try:
            lock_path = meta_dir / LOCKS_DIR_NAME / POLICY_LOCK_NAME
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            logger.warning(
                "Workspace %s: could not take the policy lock under %s — publishing unserialised",
                workspace_path,
                meta_dir / LOCKS_DIR_NAME,
                exc_info=True,
            )
        yield
    finally:
        if handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)


def _write_tree_policy(target: Path, policy: TreePolicy) -> None:
    """Dump *policy* to a temp file beside *target*, then replace it in one step.

    ``YamlDocumentStore._atomic_write``'s shape — a copy rather than an import, so
    this capability does not reach into the document store for eight lines. The
    temp file is created **in the destination directory** so the replace is a
    same-filesystem rename, which is what makes it atomic; any ``BaseException``
    unlinks it before re-raising, so a failed write leaves the previous file
    untouched and no debris behind.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.dump(policy.model_dump(mode="json"), handle, default_flow_style=False)
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def workspace_backend(param: VectorStoreParam) -> str:
    """The backend this workspace's collection actually uses.

    A card that **named** a backend gets exactly that one; a card that named none
    and resolved to the in-actor fallback gets the local one instead, because the
    workspace is the one consumer with a filesystem to hang an index off. The two
    cases must not be answered the same way: substituting under a declaration
    would silently ignore what an author wrote, and refusing a value nobody wrote
    would fail every default ``WorkspaceTool``.

    **The distinction is ``backend_declared``, and deliberately not
    ``model_fields_set``.** A whole-model serializer emits every field, so one
    ``model_dump()`` / ``model_validate()`` round trip — which the agent-card
    store performs on every team resume — inflates ``model_fields_set`` until
    every field looks authored. Reading it here would refuse a card that named
    nothing the moment its team was resumed.

    An explicit :data:`IN_ACTOR_BACKEND` is neither substituted nor accepted; it
    is refused by :func:`require_workspace_backend` before this is ever reached.

    Args:
        param: The card's ``vector_store`` field, exactly as its author left it.

    Returns:
        The backend name the resolved param carries.
    """
    if param.backend_declared:
        return param.backend
    return WORKSPACE_LOCAL_BACKEND if param.backend == IN_ACTOR_BACKEND else param.backend


def require_workspace_backend(param: VectorStoreParam, card_name: str) -> None:
    """Raise when *param* explicitly names a backend a workspace cannot use.

    Called at ``observer()`` time beside ``require_backend_configured``, and for
    its reason: a configuration that cannot work must fail the team's build in
    front of the admin who wrote it rather than degrade at the first index.

    **Reads ``backend_declared``, never ``model_fields_set``** — see
    :func:`workspace_backend`. A card that named no backend must keep binding
    after it has been stored and read back, and ``model_fields_set`` cannot tell
    the two apart once it has.

    Args:
        param: The card's ``vector_store`` field.
        card_name: Card class name, for the error message.

    Raises:
        ValueError: When the author declared the in-actor backend.
    """
    if param.backend_declared and param.backend == IN_ACTOR_BACKEND:
        raise ValueError(
            WORKSPACE_IN_MEMORY_REFUSED.format(
                card=card_name, in_actor=IN_ACTOR_BACKEND, local=WORKSPACE_LOCAL_BACKEND
            )
        )


class RagFactories:
    """The three retrieval factories and their binding.

    Declares no Pydantic field: the names below are what the card supplies,
    declared under ``if TYPE_CHECKING:`` so mypy sees them and Pydantic does not.
    """

    if TYPE_CHECKING:
        workspace_rag_index: WorkspaceRagIndex | bool
        workspace_rag_list: WorkspaceRagList | bool
        workspace_rag_search: WorkspaceRagSearch | bool
        workspace_read: WorkspaceRead | bool
        vector_store: VectorStoreParam
        max_documents: int | None
        max_document_chars: int | None

        _workspace_proxy: WorkspaceActor | None
        _workspace_tell: WorkspaceActor | None
        _agent_id: str
        _resolved_store: VectorStoreParam | None
        # The three the search closure's vector leg reads. All resolved in
        # ``observer()``, which runs before ``get_tools()``.
        _vector_store: VectorStoreService | None
        _workspace_path: str
        # The records a search and a listing read, built by the card on **every**
        # bind — actor or no actor — which is what lets both answer without one.
        # The import above it is ``TYPE_CHECKING``-only: it is a parameter type
        # here and is never constructed by this module.
        _document_cache: DocumentCache | None

    ##
    ## Enablement — one predicate, because three sites have to agree on it
    ##
    def _rag_enabled(self) -> bool:
        """Whether this card carries any retrieval capability at all.

        Read by four sites that must never disagree: the backend-derived document
        caps, the backend configuration check, the store actor's creation, and the
        bind-time announcement. A card with retrieval off must create no
        collection, create no store actor, impose no backend requirement, and
        shrink no cache.

        **All three capabilities are terms, and the third is the one that is easy
        to forget.** A card enabling only ``workspace_rag_search`` would otherwise
        send no ``enable_rag``, so the actor would acquire no proxy, create no
        collection and hold no chunking parameters — and every search would answer
        that retrieval is unavailable with nothing anywhere explaining why.
        """
        return (
            _resolve(self.workspace_rag_index, WorkspaceRagIndex) is not None
            or _resolve(self.workspace_rag_list, WorkspaceRagList) is not None
            or _resolve(self.workspace_rag_search, WorkspaceRagSearch) is not None
        )

    def _retrieval_bound(self) -> bool:
        """Whether this bind actually produced something to search.

        **The card-side reproduction of the gate the actor used to hold**, which
        was ``self._vs_proxy is None or self._rag_params is None`` — and a search
        that had to ask for it was the mailbox dependency story 57-1 removed. Read
        at ``get_tools()`` time, which is after ``observer()`` has run, exactly as
        the three values beside it are.

        Two terms, because the actor's were two. ``_resolved_store`` is derived
        exactly when :meth:`_rag_enabled` holds and is the param ``_announce_rag``
        sends, so it stands for "no card ever announced ``enable_rag``";
        ``_vector_store`` is the engine ``_bind_vector_store`` resolved, so it
        stands for "no store was announced". Either one ``None`` means the actor
        would have degraded, and one term would not reproduce both.

        **The actor's gate had a third term this cannot see, and that is an
        accepted behaviour change.** ``_vs_proxy`` was also ``None`` when
        ``create_collection`` *raised*, and no card-side signal reports that: the
        only cheap source of one is an ask before every search. So a tree whose
        collection failed to be created now answers from the keyword leg over the
        records on disk — which exist regardless of the vector collection — or
        that nothing matched, rather than that retrieval is unavailable. Neither
        answer is wrong and the trade is deliberate: the alternative is a mailbox
        round trip on every search, carrying one boolean. **That state also still
        spends** one embed and one failing ``search`` per query, exactly as it did
        before the move; what changed is that the result is used instead of being
        discarded behind the actor's sentence.

        **A second state answers differently, and it is reachable.**
        :meth:`~akgentic.tool.workspace.card.WorkspaceTool._bind_vector_store`
        catches every resolution failure and leaves ``_vector_store`` at ``None``
        with one WARNING saying *retrieval stays off for this card*. In a team
        whose other card resolved a store and announced it, such a card used to
        reach the actor's keyword leg through that sibling's announcement; it now
        answers the sentence. That is the warning's own words made true rather
        than a regression — but it is a second behaviour change, so it is named
        here instead of being left to be discovered.
        """
        return self._vector_store is not None and self._resolved_store is not None

    def _rag_params(self) -> WorkspaceRagIndex:
        """The chunking configuration this card contributes.

        A card that enables only ``workspace_rag_list`` still contributes one: the
        actor needs the splitter's parameters whatever made retrieval turn on, and
        the defaults are what ``WorkspaceRagIndex()`` already means.
        """
        return _resolve(self.workspace_rag_index, WorkspaceRagIndex) or WorkspaceRagIndex()

    def _rag_reader(self) -> DocumentReader:
        """The extraction configuration this card contributes.

        It has to travel to the actor, because the worker extracts and the
        extraction configuration lives on the **card**: it is nested inside
        ``workspace_read``, and a card whose ``document_reader`` is ``False`` or
        absent contributes a plain :class:`DocumentReader` rather than nothing —
        an indexer with no extractor could not index a PDF at all.
        """
        read = _resolve(self.workspace_read, WorkspaceRead)
        configured = read.document_reader if read is not None else True
        return configured if isinstance(configured, DocumentReader) else DocumentReader()

    ##
    ## Bind time — the tree's policy, then one fire-and-forget announcement
    ##
    def _declared_policy(self) -> TreePolicy:
        """What this card asks the tree's retrieval policy to be.

        The chunking section is this card's contribution **only when retrieval is
        on**: a card with no retrieval capability chunks nothing and has no
        parameters to offer, so it leaves that section ``None`` and can neither
        publish it nor disagree about it.

        The two caps are the **author's declarations**, not the derived values. A
        derived cap is a property of this bind — the resolved backend, on this
        host — and publishing it would put a value nobody wrote on the tree, where
        the next host's derivation would then be refused against it.
        """
        return TreePolicy(
            chunking=self._rag_params() if self._rag_enabled() else None,
            max_documents=self.max_documents,
            max_document_chars=self.max_document_chars,
        )

    def _require_tree_policy(self, workspace_path: str, card_name: str) -> None:
        """Agree with the tree's published policy, publishing it if retrieval is on.

        Runs in ``observer()`` **after** the store param is resolved and **before**
        anything with a side effect, so a refused bind seeds nothing, creates no
        actor and emits no ``WorkspaceAttached`` — the ordering discipline
        ``_require_sharing_permitted`` already states.

        **Only a retrieval binder publishes; every binder reads.** A card with
        retrieval off writes nothing and creates nothing — no ``mkdir``, no touch,
        no ``<meta>`` — and a ``read`` of an absent record creates nothing either.
        That is what keeps a default bind's disk footprint empty, which
        ``test_the_tree_and_its_metadata_sibling_hold_exactly_this`` pins. It is
        still *held to* what is published: its declared caps are compared, and a
        disagreement refuses it.

        Args:
            workspace_path: The resolved three-segment path.
            card_name: Card class name, for the error messages.

        Raises:
            ValueError: When the policy file does not parse, or when any section
                of it disagrees with this card. It fails card binding and
                therefore team creation, deliberately, and must never be caught
                and turned into a fallback.
        """
        declared = self._declared_policy()
        if self._rag_enabled():
            publish_tree_policy(workspace_path, declared, card_name)
            return
        published = read_tree_policy(workspace_path, card_name)
        if published is not None:
            agreed_tree_policy(published, declared, workspace_path, card_name)

    def _announce_rag(self) -> None:
        """Tell the actor to turn retrieval on for this tree — fire and forget.

        The same shape as ``_announce_exec``, and for the identical reason: the
        first bind fixes ``WorkspaceConfig`` for every card on the tree, and the
        card that binds a tree first is routinely one with no retrieval capability
        at all. **The actor does not inspect cards** — it cannot, it has no handle
        on one — so a card tells it.

        **It never raises.** A stand-in proxy that does not carry the method, or an
        actor that died between the get-or-create and this line, must not take the
        whole card binding down. The degradation is a workspace whose
        ``workspace_rag_index`` answers that retrieval is unavailable: visible, and
        recoverable by rebinding.

        **It sends the resolved param, not the author's field.** The backend and
        the root the store was actually built over are what the actor hands to
        ``create_collection``, and a collection created under the author's
        declaration would name a backend nothing was built for.
        """
        # ``_resolved_store`` is derived exactly when :meth:`_rag_enabled` holds, so
        # this **is** the enablement gate rather than a second one beside it — and
        # a param that is not ``None`` is the only thing there is to announce.
        collection = self._resolved_store
        if collection is None:
            return
        tell = self._workspace_tell
        if tell is None:
            return
        try:
            tell.enable_rag(self._agent_id, self._rag_params(), self._rag_reader(), collection)
        except Exception:
            logger.debug("Could not enable retrieval on #Workspace", exc_info=True)

    ##
    ## The three callables
    ##
    def _rag_search_factory(self, params: WorkspaceRagSearch) -> Callable[..., Any]:
        """Create the ``workspace_rag_search`` callable.

        **Both legs run here, on the calling agent's own thread, and no part of a
        search reaches a mailbox.** The *vector* leg's inputs are the card's — the
        engine this card resolved in ``observer()``, the collection param it
        derived, the workspace path it resolved once — and so are the *keyword*
        leg's: it reads the document records through the
        :class:`~akgentic.tool.workspace.documents.cache.DocumentCache` this card
        built, which is the same object the actor reads them through. The ask that
        used to carry the second half was justified as "their inputs are actor
        state", and that was false; what it cost was a search queueing behind
        ``request_exec``, ``exec_status`` and every index worker's report, for work
        that is file reading (ADR-053 Decision 6).

        **Why the caller's thread and not a child actor.** A search is an *ask*:
        the calling agent is blocked on its answer for the whole call, so a child
        would only move the block — the actor would wait for the child's report and
        the mailbox would be held exactly as long. The only shape that frees a
        mailbox for an ask is a deferred/poll protocol, and ``#Workspace``'s
        :class:`~akgentic.tool.core.deferred.DeferredResultActor` holds **exec**
        outcomes; putting a search through it would evict a running agent's exec
        result and mis-type the cache's value — the argument ``rag/worker.py`` and
        ``EmbeddingWorker`` both already make about that mechanism. The calling
        agent's thread is already blocked for the duration and blocks nobody else.
        Two agents of one team searching concurrently now genuinely search
        concurrently, instead of serialising on one mailbox.

        **The order of the three gates below is load-bearing and is pinned by a
        spec.** A ``None`` cache is "this card never bound a tree", which is the
        position the ``None`` proxy held before — ``_build_document_cache`` runs in
        ``observer()``, so an unbound card has neither. The wildcard check comes
        next, so a refused prefix costs nothing at all whatever the tree's state.
        The availability gate comes last of the three, and it now sits **ahead of
        the vector leg** where the actor's sat behind it. That changes no answer,
        and it saves the keyword leg's directory scan and parse on a degraded
        tree — but it saves **no round trip**, and a claim that it did stood here
        until review measured it: the gate's two terms are exactly what
        :func:`~akgentic.tool.workspace.rag.search._vector_hits` already
        short-circuits on, for free. The spend that docstring recorded as
        unclosable belongs to the *other* degraded state — ``create_collection``
        raised — which this gate does not cover and which still pays it; see
        :meth:`_retrieval_bound`. Swapping the middle two gates would silently
        flip what a degraded tree answers to a wildcard prefix.

        The values are read **at ``get_tools()`` time**, which is after
        ``observer()`` has run, so the cache, the engine, the collection param and
        the workspace path are all populated by the time the closure captures them.

        Args:
            params: The result budget and the two fusion knobs, captured here so
                a team's configured values travel with the call.

        Returns:
            The callable, which never raises.
        """
        cache = self._document_cache
        store = self._vector_store
        resolved = self._resolved_store
        scope = self._workspace_path
        available = self._retrieval_bound()
        top_k, alpha, threshold = params.top_k, params.alpha, params.score_threshold

        def workspace_rag_search(query: str, top_k: int = top_k, path_prefix: str = "") -> str:
            """Search the indexed workspace documents for passages about *query*.

            Combines meaning-based and word-based matching over the files that
            ``workspace_rag_index`` has indexed. Use it to find *where* something
            is said before reading a whole document.

            Args:
                query: What to look for, in natural language.
                path_prefix: Restrict the search to paths starting with this, e.g.
                    "reports/". Wildcards are not accepted. Defaults to the whole
                    workspace.
                top_k: How many passages to return.

            Returns:
                The matching passages with their file, heading path and score, or
                a sentence saying that nothing matched.
            """
            if cache is None:
                return _UNAVAILABLE  # harness shapes that wire a bare observer never bind one
            # Ahead of the embed and the search, so a refused prefix costs
            # nothing at all — no round trip and no embedding credit.
            if any(character in path_prefix for character in PATH_PREFIX_WILDCARDS):
                return _REJECTED_PREFIX
            if not available:
                return _UNAVAILABLE
            try:
                return search_documents(
                    cache,
                    store,
                    resolved,
                    query,
                    top_k=top_k,
                    scope=scope,
                    path_prefix=path_prefix,
                    alpha=alpha,
                    score_threshold=threshold,
                )
            except Exception:
                logger.debug("Could not search the retrieval index", exc_info=True)
                return _UNAVAILABLE

        workspace_rag_search.__doc__ = params.format_docstring(workspace_rag_search.__doc__)
        return workspace_rag_search

    def _rag_index_factory(self, params: WorkspaceRagIndex) -> Callable[..., Any]:
        """Create the ``workspace_rag_index`` callable.

        The closure captures the **ask** proxy, because the counts are the answer.
        What is behind that ask is bounded: a tree walk through the backend, one
        read per candidate to hash it, and up to four actor spawns. No extraction,
        no split and no embedding happens on the calling agent's thread or on the
        actor's.

        Args:
            params: The chunking configuration — captured for its docstring only;
                the actor already holds the parameters it will chunk with.

        Returns:
            The callable, which never raises and never blocks on the pipeline.
        """
        proxy = self._workspace_proxy

        def workspace_rag_index(path: str = "", force: bool = False) -> str:
            """Queue workspace files for retrieval indexing, and return immediately.

            Indexing runs in the background. Use ``workspace_rag_list`` to see
            where each file got to.

            Args:
                path: A file, a directory, or "" for the whole workspace.
                force: Re-index files that are already indexed at their current
                    content. Defaults to False.

            Returns:
                How many files were queued, were already current, and were of a
                type that cannot be indexed.
            """
            if proxy is None:
                return _UNAVAILABLE
            try:
                return str(proxy.index_paths(path, force))
            except Exception:
                logger.debug("Could not queue %r for indexing", path, exc_info=True)
                return _UNAVAILABLE

        workspace_rag_index.__doc__ = params.format_docstring(workspace_rag_index.__doc__)
        return workspace_rag_index

    def _rag_list_factory(self, params: WorkspaceRagList) -> Callable[..., Any]:
        """Create the ``workspace_rag_list`` command callable.

        COMMAND channel only. The same snapshot the context-state provider takes,
        rendered in full rather than as a delta — a person asking for the list
        wants the list, not what changed since last turn.

        **It renders with no actor at all**, over the records this card's own
        cache reads: a listing is a render, not dispatch, so a retrieval-off tree
        and a tree whose actor died both still answer (ADR-053 Decision 6). The
        degraded case is a ``None`` cache — this card never bound a tree — and its
        answer is the sentence, exactly where a ``None`` proxy used to give it.

        Args:
            params: The render cap.

        Returns:
            The callable, which never raises.
        """
        cache = self._document_cache
        cap = params.max_pending_shown

        def workspace_rag_list() -> str:
            """Show where every workspace file stands in the retrieval index.

            Returns:
                One line per file, with a tail counting the pending files the cap
                left out.
            """
            if cache is None:
                return _UNAVAILABLE  # harness shapes that wire a bare observer never bind one
            try:
                return str(render_index_state(cache, cap).render_full())
            except Exception:
                logger.debug("Could not render the retrieval index", exc_info=True)
                return _UNAVAILABLE

        workspace_rag_list.__doc__ = params.format_docstring(workspace_rag_list.__doc__)
        return workspace_rag_list

    def _rag_list_state_factory(
        self, params: WorkspaceRagList
    ) -> Callable[[], ContextState | None]:
        """Create the ``LLM_CONTEXT`` provider for the retrieval index.

        **No mailbox, and no tree walk.** This runs on every turn of every agent
        carrying the card, and it used to be an ask — so a display queued behind
        exec dispatch and every index worker's report. It is one directory scan of
        ``<meta>/rag/`` and one parse per record now, bounded by
        ``max_documents``, on the calling agent's own thread; no file **inside**
        the tree is opened and no ``stat`` per candidate is spent, which is the
        property that keeps the filesystem off the hot path.

        Args:
            params: The render cap, captured here at ``get_context_states`` time.

        Returns:
            A provider that returns ``None`` — never raises — when this card
            bound no tree, which is the ``ContextState`` contract.
        """
        cache = self._document_cache
        cap = params.max_pending_shown

        def provider() -> ContextState | None:
            if cache is None:
                return None  # harness shapes that wire a bare observer never bind one
            try:
                return render_index_state(cache, cap)
            except Exception:
                logger.debug("Could not read the retrieval index", exc_info=True)
                return None

        return provider
