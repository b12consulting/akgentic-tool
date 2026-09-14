"""While a search's external call is in flight, ``#Workspace``'s mailbox answers.

**The property, stated precisely, because the loose form is untestable.** It is
not "the search runs on a particular thread" — a closure runs on whichever thread
calls it, and here that is the test's. It is: *while a search's external call is
in flight, the ``#Workspace`` actor's mailbox is not held*, so another message on
that mailbox is answered. That is what
:mod:`akgentic.tool.workspace.actor.__init__`'s "and never external" claims, and
what a query embed plus a ``store.search`` on the actor's own turn falsified.

**It lives in its own module rather than in ``test_rag_search.py``** for one
reason: that module drives the *actor* through a harness built for it
(``SearchHarness``, an inert actor, a store announced by hand), and the property
here is only observable against a **live** actor on a real thread, with a real
mailbox, a real exec lease and a real card closure. Nothing of either harness is
usable by the other.

**The seam the stand-in blocks at is ``VectorStoreService.search``**, and that is
the load-bearing choice. It is the one seam that exists in **both** shapes — on
the actor's turn before this story, on the caller's thread after — so the same
source asserts the same thing on either side of the move. A stand-in that stalled
the actor's embedder instead would be **vacuous afterwards**, because the actor no
longer embeds: it would pass having stalled nothing. The embed is stalled too, as
a second parameterised case, because the audit's finding names both calls; the
search seam is the one that must be there.

**Two sentinels stop this from passing vacuously**, and both are mandatory:

- ``entered`` is asserted to have been **set**. A stand-in the production path
  never reached would otherwise leave the probe unobstructed and the guard green
  with the defect in place.
- the probe is run once more with **no search in flight** and must answer. A
  mis-wired probe that never worked at all would otherwise make the guard green
  for the opposite reason.

**Red-before is a bounded failure, not a hang.** The probe runs on its own thread
and is joined with a timeout; the assertion is ``not probe.is_alive()`` plus the
answers it recorded. On the un-moved tree the probe thread is still alive at the
join because ``request_exec`` is queued behind the blocked mailbox.

Nothing here starts docker: ``sandbox_script`` installs
:class:`~tests.workspace.conftest.FakeBackend` at the ``local`` key, which is what
``test_exec.py`` does and says so in its own first paragraph. Nothing here reaches
a network either — the vector store is a stand-in built through the registry's
own factory seam, and ``build_embedding_service`` is replaced with one that
returns a fixed vector.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from akgentic.tool.vector_store.protocol import (
    CollectionStatus,
    SearchHit,
    SearchResult,
    VectorStoreParam,
)
from akgentic.tool.vector_store.vector import VectorEntry
from akgentic.tool.workspace.actor import workspace_actor_name
from akgentic.tool.workspace.documents.models import RAG_COLLECTION
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    SandboxScript,
    factory_for,
    tool_named,
)

PROBE_JOIN_S = HANDSHAKE_TIMEOUT_S
"""How long the probe thread is given to answer while a search is in flight.

A **budget**, never a delay: the probe joins the instant it is done, and the
timeout exists so that a held mailbox is reported as a failed assertion instead
of hanging the suite.
"""

STALL_BACKSTOP_S = 3 * HANDSHAKE_TIMEOUT_S
"""How long a parked stand-in waits to be released before it gives up.

**Strictly above :data:`PROBE_JOIN_S`, and that ordering is load-bearing.** At
equal budgets a red run races: the stand-in's own wait can expire first, which
releases the seam and turns "the mailbox was held" into an unrelated failure
inside the stand-in. The backstop exists only so a test that never reaches its
``finally`` cannot leak a blocked thread; it must never be the first thing to
fire.
"""

_HIT_TEXT = "Payment terms are net thirty."


class _Stall:
    """One seam a stand-in can be parked at, and the record that it was reached.

    ``armed`` is what makes the same two stand-ins serve both parameterised cases
    and the positive control: an unarmed stall is a straight pass-through, so the
    no-search-in-flight run and the un-stalled seam cost nothing at all.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self.armed = False

    def hold(self) -> None:
        """Announce arrival and wait to be let go — or pass straight through."""
        if not self.armed:
            return
        self.entered.set()
        assert self.released.wait(timeout=STALL_BACKSTOP_S), (
            "the stalled call was never released — the test leaked a blocked thread"
        )


class _StallingStore:
    """A cluster client with no actor behind it, stallable inside ``search``.

    Deliberately **not** a proxy over a ``#VectorStore`` child: ``_acquire_vs_proxy``'s
    own docstring records that the slot may hold either, and the client shape is
    the one where ``search`` is a second HTTP round trip on the same turn as the
    embed. A guard written against the actor-backed shape would prove the cheaper
    half.
    """

    def __init__(self, stall: _Stall) -> None:
        self.stall = stall
        self.searches: list[tuple[str, int, str | None, str | None]] = []
        self.collections: list[str] = []

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        self.collections.append(name)

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        raise AssertionError("a search must add nothing")

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        raise AssertionError("a search must remove nothing")

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
        query: Any = None,
    ) -> SearchResult:
        self.searches.append((collection, top_k, scope, path_prefix))
        self.stall.hold()
        return SearchResult(
            status=CollectionStatus.READY,
            hits=[
                SearchHit(
                    ref_type="workspace_chunk",
                    ref_id="chunk-0",
                    text=_HIT_TEXT,
                    score=0.9,
                    scope=scope,
                    path="invoice.md",
                    ordinal=0,
                )
            ],
        )


class _StallingEmbedder:
    """The query embedder, stallable — the audit's *other* call on that turn."""

    def __init__(self, stall: _Stall) -> None:
        self.stall = stall
        self.embeds: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embeds.append(list(texts))
        self.stall.hold()
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _Probe:
    """Two messages on ``#Workspace``'s mailbox, run on a thread of their own.

    Exactly the two calls the audit names — ``exec_status`` and ``request_exec``
    — reached through the card's own closures rather than by poking the actor, so
    what is proved is the path an agent actually takes.
    """

    def __init__(self, card: WorkspaceTool) -> None:
        self.card = card
        self.answers: list[str] = []
        self.error: BaseException | None = None

    def __call__(self) -> None:
        try:
            self.answers.append(str(tool_named(self.card, "workspace_exec_result")("no-such-run")))
            self.answers.append(str(tool_named(self.card, "workspace_exec")("echo hi")))
        except BaseException as exc:  # noqa: BLE001 — reported, never swallowed
            self.error = exc

    def run(self) -> threading.Thread:
        """Start the probe and return its thread, joined by the caller."""
        thread = threading.Thread(target=self, name="mailbox-probe", daemon=True)
        thread.start()
        return thread


class _Wired:
    """A live ``#Workspace``, a card with retrieval and exec on, and the two stalls."""

    def __init__(
        self,
        card: WorkspaceTool,
        observer: FakeActorToolObserver,
        store: _StallingStore,
        embedder: _StallingEmbedder,
    ) -> None:
        self.card = card
        self.observer = observer
        """Held because the card holds its observer weakly."""
        self.store = store
        self.embedder = embedder

    def search_on_its_own_thread(self) -> tuple[threading.Thread, list[str]]:
        """Call the card's real ``workspace_rag_search`` on a thread, capturing its answer."""
        answers: list[str] = []

        def run() -> None:
            answers.append(str(tool_named(self.card, "workspace_rag_search")("payment")))

        thread = threading.Thread(target=run, name="rag-search", daemon=True)
        thread.start()
        return thread, answers


@pytest.fixture
def stalls() -> dict[str, _Stall]:
    """One stall per seam, keyed by the name the parametrisation uses."""
    return {"search": _Stall(), "embed": _Stall()}


@pytest.fixture
def wired(
    threaded_orchestrator_proxy: FakeOrchestratorProxy,
    workspace_tree: Path,
    sandbox_script: SandboxScript,
    stalls: dict[str, _Stall],
    monkeypatch: pytest.MonkeyPatch,
) -> _Wired:
    """Bind a retrieval-and-exec card onto a **live** ``#Workspace``.

    The store arrives through the registry's own factory seam, which is the
    branch a cluster backend takes — no ``#VectorStore`` actor is created, so
    the object the card announces is the object both shapes of the search reach.

    ``build_embedding_service`` is replaced at its **source module**, which is
    where both the actor's function-level import and the capability's find it.
    """
    store = _StallingStore(stalls["search"])
    embedder = _StallingEmbedder(stalls["embed"])
    sandbox_script.gate.set()  # a run that simply completes; the probe is not about exec
    monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "https://cluster.example")
    monkeypatch.setattr(
        "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
        lambda model, provider: embedder,
    )
    observer = FakeActorToolObserver(threaded_orchestrator_proxy, name="alice")
    card = WorkspaceTool(
        workspace_id=WORKSPACE_NAME,
        workspace_rag_search=True,
        workspace_exec=WorkspaceExec(mode="local", poll_attempts=1, poll_delay_seconds=0.0),
        vector_store=VectorStoreParam(backend="weaviate"),
    )
    with factory_for("weaviate", lambda _context: store):
        card.observer(observer)
    # The two announcements are tells. One round trip on the same mailbox is what
    # proves they have been applied — ``enable_rag`` included, and with it the
    # ``create_collection`` that leaves the actor un-degraded.
    pykka_proxy = threaded_orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)][1]
    pykka_proxy.state.get(timeout=HANDSHAKE_TIMEOUT_S)
    assert store.collections == [RAG_COLLECTION], "retrieval never came up on the actor"
    return _Wired(card, observer, store, embedder)


class TestTheProbeItself:
    """The positive control. A probe that never worked would make the guard green."""

    def test_with_no_search_in_flight_both_calls_answer(self, wired: _Wired) -> None:
        """Neither seam is armed, so nothing stalls and the mailbox is plainly free."""
        probe = _Probe(wired.card)
        thread = probe.run()
        thread.join(timeout=PROBE_JOIN_S)

        assert probe.error is None
        assert not thread.is_alive()
        assert len(probe.answers) == 2


@pytest.mark.parametrize("seam", ["search", "embed"])
class TestTheMailboxAnswersWhileASearchIsInFlight:
    """The audit's two calls, against a search parked at an external seam."""

    def test_exec_status_and_request_exec_answer(
        self, wired: _Wired, stalls: dict[str, _Stall], seam: str
    ) -> None:
        """The whole property, in one sequence and with no timing in the assertion.

        The order is fixed: arm the seam, start the search, wait for the stand-in
        to announce arrival, probe, join with a budget, release, join the search.
        Every wait is on an event or a thread, never on a clock.
        """
        stall = stalls[seam]
        stall.armed = True
        search_thread, answers = wired.search_on_its_own_thread()
        try:
            assert stall.entered.wait(timeout=HANDSHAKE_TIMEOUT_S), (
                f"the {seam} stand-in was never reached — the guard would be vacuous"
            )
            assert not stall.released.is_set()

            probe = _Probe(wired.card)
            probe_thread = probe.run()
            probe_thread.join(timeout=PROBE_JOIN_S)
        finally:
            stall.released.set()

        assert probe.error is None
        assert not probe_thread.is_alive(), (
            "#Workspace did not answer while a search's external call was in flight — "
            "the mailbox is held for the duration of the round trip"
        )
        assert len(probe.answers) == 2

        search_thread.join(timeout=HANDSHAKE_TIMEOUT_S)
        assert not search_thread.is_alive()
        assert _HIT_TEXT in answers[0], "the search did not render the hits it was handed"

    def test_the_stand_in_was_actually_reached(
        self, wired: _Wired, stalls: dict[str, _Stall], seam: str
    ) -> None:
        """55-4's lesson, as its own row: ``entered`` set is the non-vacuity.

        Stated separately from the property above so that a stand-in pointed at a
        seam the production path never takes reddens *here*, naming the cause,
        rather than showing up as an unexplained timeout in the spec beside it.
        """
        stall = stalls[seam]
        stall.armed = True
        search_thread, _answers = wired.search_on_its_own_thread()
        try:
            entered = stall.entered.wait(timeout=HANDSHAKE_TIMEOUT_S)
        finally:
            stall.released.set()
        search_thread.join(timeout=HANDSHAKE_TIMEOUT_S)

        assert entered, f"the {seam} seam is not on the search's path"
        # Both seams, whichever was armed: the query is embedded once and the
        # collection is searched once, so neither stand-in can be parked
        # somewhere the production path merely happens to pass near.
        assert wired.embedder.embeds == [["payment"]]
        assert [collection for collection, *_rest in wired.store.searches] == [RAG_COLLECTION]
