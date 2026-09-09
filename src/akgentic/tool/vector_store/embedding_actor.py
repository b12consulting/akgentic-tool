"""``#embed-``: the off-thread half of embedding one batch, owned by the consumer.

The vector store is a storage engine and embeds nothing. The code that owns a
``VectorStoreParam`` — the workspace indexer, and through
:func:`build_embedding_service` the planning actor and the knowledge graph —
spawns this worker itself, hands it whole :class:`~akgentic.tool.vector_store.vector.VectorEntry`
objects with empty vectors, and gets the same entries back with their vectors
filled in. Nothing is reconstructed on the way, so the entries keep the ``scope``,
``path`` and ``ordinal`` that every scoped removal and every scoped search filters
on.

The worker is spawned with ``createActor`` under a ``#``-prefixed name, is handed
exactly one payload, reports to the **consumer's** two handlers, and stops itself
in a ``finally``.

**Reports travel by ``proxy_tell``, never by ``send``.** ``Akgent.send`` records
``SentMessage`` telemetry for ``Message`` instances and the persistence subscriber
writes that telemetry to the event store, so a ``Message`` payload here would
surface every transient worker as a busy team member and put every batch's vectors
— tens of kilobytes each — into persisted history. Every model below is a
``SerializableBaseModel``, and a ``proxy_tell`` is a method call on the actor ref
that enters no history.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import Field

from akgentic.core.agent import Akgent
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.core.deferred import DeferredPayload, DeferredWorker
from akgentic.tool.vector_store.vector import VectorEntry

if TYPE_CHECKING:
    from akgentic.tool.vector_store.vector import EmbeddingService

logger = logging.getLogger(__name__)

EMBED_WORKER_NAME_PREFIX: str = "#embed-"
"""Prefix of every embedding worker's actor name.

**Only the leading ``#`` is load-bearing** — it is what classifies the actor as a
tool actor during the orchestrator's two-phase stop, so a worker mid-call does not
hold every non-tool member's teardown behind it. The rest is a readability aid.
Deliberately not ``WORKER_NAME_PREFIX``: that one belongs to the cache actor's own
spawns, and these workers are spawned by consumers instead.
"""

_NAME_REQUEST_ID_CHARS: int = 12
"""How much of the request id rides in the worker name — readability only."""


def embedding_worker_name(collection: str, request_id: str) -> str:
    """Return the actor name of the worker embedding one batch for *collection*.

    Args:
        collection: Collection the batch will be written into.
        request_id: Identifier the consumer minted for this batch.

    Returns:
        ``#embed-<collection>-<first 12 characters of the request id>``.
    """
    return f"{EMBED_WORKER_NAME_PREFIX}{collection}-{request_id[:_NAME_REQUEST_ID_CHARS]}"


# ---------------------------------------------------------------------------
# Message models
# ---------------------------------------------------------------------------


class EmbeddingRequest(DeferredPayload):
    """One batch to embed, handed to an :class:`EmbeddingWorker`.

    Carries **whole entries**, not a projection of three of their fields. The
    worker fills each entry's ``vector`` and returns the entry itself, so the
    other six fields never leave the consumer's own object and there is nothing
    to restore on the way back.

    ``deferred_key`` narrows the base's ``Any`` to the request id the consumer
    minted, which is also the suffix of the worker's actor name.

    Attributes:
        deferred_key: The consumer's request id for this batch.
        collection: Collection the embedded batch will be written into.
        entries: The batch, with empty ``vector`` fields.
        request_ref: The consumer's own correlation key, echoed back unchanged —
            a file path, for the workspace indexer.
        embedding_model: Model the consumer's ``VectorStoreParam`` names.
        embedding_provider: Provider the consumer's ``VectorStoreParam`` names.
    """

    deferred_key: str = Field(
        ...,
        description="The consumer's request id for this batch; also the worker-name suffix.",
    )
    collection: str = Field(description="Target collection name")
    entries: list[VectorEntry] = Field(
        description="The batch to embed, with empty vector fields"
    )
    request_ref: str | None = Field(
        default=None,
        description="The consumer's correlation key, echoed back unchanged",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model identifier",
    )
    embedding_provider: Literal["openai", "azure"] = Field(
        default="openai",
        description="Embedding API provider",
    )


class EmbeddingResult(SerializableBaseModel):
    """One embedded batch, told back to the consumer that asked for it."""

    collection: str = Field(description="Target collection name")
    entries: list[VectorEntry] = Field(description="The request's entries, vectors populated")
    request_id: str = Field(description="Matching request identifier")
    request_ref: str | None = Field(
        default=None,
        description="The consumer's correlation key, returned exactly as it was given",
    )


class EmbeddingError(SerializableBaseModel):
    """One batch that could not be embedded, told back to its consumer."""

    collection: str = Field(description="Target collection name")
    error: str = Field(description="Error message from embedding failure")
    request_id: str = Field(description="Matching request identifier")
    request_ref: str | None = Field(
        default=None,
        description="The consumer's correlation key, returned exactly as it was given",
    )


class EmbeddingRequester(Protocol):
    """What a consumer must implement to be told about its own batches.

    The two handlers the worker reports through. It is a Protocol rather than a
    concrete type because three unrelated actors implement it, and the worker
    must not import any of them.
    """

    def receiveMsg_EmbeddingResult(self, msg: EmbeddingResult) -> None:  # noqa: N802
        """Take one embedded batch."""
        ...

    def receiveMsg_EmbeddingError(self, msg: EmbeddingError) -> None:  # noqa: N802
        """Take one batch that could not be embedded."""
        ...


# ---------------------------------------------------------------------------
# The embedding service, and the one place its budget is chosen
# ---------------------------------------------------------------------------


def build_embedding_service(
    model: str, provider: Literal["openai", "azure"]
) -> EmbeddingService:
    """Build an ``EmbeddingService`` carrying the worker's timeout budget.

    **This is where the number lives.** Four call sites construct the service with
    the same three arguments — the worker itself, and the query leg of each of the
    three consumers — and every one of them must carry a budget below the
    orchestrator's stop backstop, because each runs on a thread whose teardown
    someone else waits on.

    The ``vector`` import is deliberately inside the function: the module belongs
    to the ``[vector_search]`` extra, and a consumer without it must still be able
    to import this module.

    Args:
        model: Embedding model identifier.
        provider: Embedding API provider name.

    Returns:
        A service whose every request carries :attr:`EmbeddingWorker.timeout_s`.
    """
    from akgentic.tool.vector_store.vector import (  # noqa: PLC0415 — optional extra
        EmbeddingService,
    )

    return EmbeddingService(
        model=model, provider=provider, timeout_s=EmbeddingWorker.timeout_s
    )


# ---------------------------------------------------------------------------
# EmbeddingWorker
# ---------------------------------------------------------------------------


class EmbeddingWorker(DeferredWorker):
    """Embeds one batch on its own thread, tells its consumer, and stops.

    **A ``DeferredWorker``, and deliberately not spawned by a
    ``DeferredResultActor``** (ADR-049 Decision 4). ``core/deferred.py`` says a
    partial adoption is not an adoption, so the exemption is argued here rather
    than left for a reader to notice. Every one of the mechanism's seven rules
    protects a **polling** caller, and indexing has none: ``workspace_rag_index``
    returns immediately and completion is *pushed* to a requester that is already
    tracking the file. Adopting the cache actor would mean an LRU holding vectors
    — the memory the design exists to avoid — a negative TTL suppressing a failed
    batch for a minute when retry policy belongs to the indexer that knows which
    file is stuck, and in-flight de-duplication keyed on batches whose content is
    distinct by construction. Exec is the contrast: an exec tool call must hand the
    LLM a value inside its turn, so it polls, so ``#Workspace`` implements
    ``deliver`` and ``fail`` for it.

    **The rule for the next reader:** blocking work whose result is *pulled* by a
    waiting caller takes the cache actor and the worker; blocking work whose result
    is *pushed* to a caller that has already moved on takes the worker alone.

    What is inherited is the worker half in full: the timeout budget and its
    contract, ``WORKER_ROLE``, the single-shot ``on_start``, and one place in the
    tree where a reader can see every off-thread worker the package has.
    """

    def produce(self, payload: DeferredPayload) -> list[VectorEntry]:
        """Embed the batch and return its entries with their vectors filled in.

        :attr:`~akgentic.tool.core.deferred.DeferredWorker.timeout_s` reaches the
        HTTP call through :func:`build_embedding_service`. A budget that does not
        reach the client is decoration: a Python thread cannot be cancelled, and
        this worker holds its parent's ``stop_children(blocking=True)`` open until
        it returns.

        Args:
            payload: An :class:`EmbeddingRequest`.

        Returns:
            The request's own entries, each carrying its vector.

        Raises:
            TypeError: If handed a payload that is not an :class:`EmbeddingRequest`.
            ValueError: If the provider returned a different number of vectors than
                the batch had entries — ``strict=True`` below. A truncated,
                mis-attributed batch is worse than a failure the consumer can see.
        """
        if not isinstance(payload, EmbeddingRequest):
            raise TypeError(f"EmbeddingWorker requires an EmbeddingRequest, got {type(payload)}")

        service = build_embedding_service(payload.embedding_model, payload.embedding_provider)
        vectors = service.embed([entry.text for entry in payload.entries])
        # Golden Rule #12: the one field that was produced, on the entry that
        # already holds the other six.
        return [
            entry.model_copy(update={"vector": vector})
            for entry, vector in zip(payload.entries, vectors, strict=True)
        ]

    def receiveMsg_DeferredPayload(self, msg: DeferredPayload) -> None:
        """Produce, tell the consumer, and stop — always.

        **The base's shape with the report channel replaced.** ``DeferredWorker``
        reports through ``parent.deliver(key, value)`` / ``parent.fail(key, error)``,
        and on ``#Workspace`` those two belong to a ``DeferredResultActor`` holding
        **exec** outcomes: a batch of vectors delivered that way would evict a
        running agent's exec result and mis-type the cache's value. The consumer's
        own two handlers are the channel instead; ``_report_value`` and
        ``_report_failure`` are inherited and unused.

        ``self.stop()`` runs in a ``finally`` so the worker also goes away on the
        exception path; a worker that survives its unit of work is a leak and,
        while it lives, a member of the team roster.
        """
        if not isinstance(msg, EmbeddingRequest):
            # Nothing to report against: an alien payload names no collection and no
            # correlation key, so there is no consumer this could be attributed to.
            logger.warning(
                "[%s] Ignoring a payload that is not an EmbeddingRequest: %s",
                self.config.name,
                type(msg).__name__,
            )
            self.stop()
            return
        try:
            entries = self.produce(msg)
            self._tell_parent(
                EmbeddingResult(
                    collection=msg.collection,
                    entries=entries,
                    request_id=msg.deferred_key,
                    request_ref=msg.request_ref,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._tell_parent(
                EmbeddingError(
                    collection=msg.collection,
                    error=str(exc),
                    request_id=msg.deferred_key,
                    request_ref=msg.request_ref,
                )
            )
        finally:
            self.stop()

    def _tell_parent(self, report: EmbeddingResult | EmbeddingError) -> None:
        """Tell the consumer about this batch, and never raise doing it.

        A consumer that has since stopped must not take the worker's ``finally``
        down with it, so a delivery failure is logged and swallowed.
        """
        if self._parent is None:
            logger.warning("[%s] No parent address, cannot deliver report", self.config.name)
            return
        try:
            # ``proxy_tell``'s type parameter is bound to ``Akgent``; ``ProxyWrapper``
            # forwards any method name at runtime, so the consumer's own two handlers
            # are reachable through the cast to the Protocol they satisfy.
            requester = cast(EmbeddingRequester, self.proxy_tell(self._parent, Akgent))
            if isinstance(report, EmbeddingResult):
                requester.receiveMsg_EmbeddingResult(report)
            else:
                requester.receiveMsg_EmbeddingError(report)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to deliver an embedding report to the parent: %s",
                self.config.name,
                exc,
            )
