"""What the observation record costs, measured under a real team of agents.

The design in ADR-036 rests on one number nobody had measured: the read path
gained a fire-and-forget ``tell`` per tool invocation, and the claim is that this
is free at ten agents. Story 29-3 turned that record from a blocking ``proxy_ask``
into a ``tell``, so the reader no longer waits on the actor at all. What is left
to measure is therefore **not** how long a reader blocks. It is:

1. the reader's own CPU cost — ``content_sha`` over the file's whole bytes, on
   the agent's own thread, including for a paginated read of a large file; and
2. what read traffic does to the singleton's **mailbox** — every read appends a
   message to the same queue the mutations use.

Four arms, all interleaved in one invocation, differing only by a patch applied
inside this process:

===============  ==============================================  ==============
arm              read path                                       journal
===============  ==============================================  ==============
``off``          no digest, no message                           off
``hash-only``    computes the digest, sends nothing              off
``on``           shipped behaviour — digest, then the ``tell``    off
``on+journal``   shipped behaviour                               on
===============  ==============================================  ==============

Two measurements come out of those four, and **they must not be mixed**, because
they adjudicate different claims:

* the **read-path** measurement compares ``off`` / ``hash-only`` / ``on``, all
  with the journal **off**. Git forks per mutation would contaminate numbers
  meant to judge a record on the read path.
* the **mutation-path** measurement compares ``on`` against ``on+journal``,
  because there the forks are exactly the thing being measured.

``hash-only`` is not a nicety. The fallback ADR-036 names — per-agent-turn
batching — removes *messages*, not *digests*: the hash is computed on the
reader's thread before anything is sent. Without the third arm this harness
cannot say whether that fallback would help at all.

**This is a script, not a test.** The package's CI runs ``pytest tests/`` with no
marker filter, so a marked benchmark would still run on every push; the package
has no marker vocabulary to select on; and the deterministic regression guard
already exists (29-3 pins the tell and the busy-mailbox property by handshake,
not by clock). ``python_files = ["test_*.py"]`` does not match this module, so
pytest never collects it. ``tests/workspace/test_read_cost_benchmark_smoke.py``
imports it at a tiny size and asserts its shape, so it cannot rot unnoticed.

Run it::

    PYTHONPATH=src python -m tests.benchmarks.workspace_read_cost --agents 10

Nothing here may change ``src/``. The two non-shipped arms are produced by
swapping :meth:`WorkspaceTool._observation_recorder` for the duration of a run,
in this process only. There is deliberately no production toggle: shipping the
switch would be shipping the fallback.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pykka
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils import SerializableBaseModel

from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.journal import git_dir_for
from akgentic.tool.workspace.models import (
    MutationOutcome,
    Observation,
    WorkspaceConfig,
    content_sha,
)
from akgentic.tool.workspace.readers import DocumentReader
from akgentic.tool.workspace.tool import WorkspaceRead, WorkspaceTool

##
## Arms
##

ARM_OFF = "off"
ARM_HASH_ONLY = "hash-only"
ARM_ON = "on"
ARM_ON_JOURNAL = "on+journal"

ALL_ARMS = (ARM_OFF, ARM_HASH_ONLY, ARM_ON, ARM_ON_JOURNAL)
"""Every arm, in the order the report prints them."""

READ_PATH_ARMS = (ARM_OFF, ARM_HASH_ONLY, ARM_ON)
"""The three the read-path verdict is computed from — journal off in all three."""

##
## Thresholds — fixed here, before any number was measured, and printed beside
## the measured values in the report.  Deciding what "material" means after
## seeing the numbers is the failure mode these constants exist to prevent.
##

T1_RATIO = 1.10
"""``on`` read p95 must exceed ``off`` p95 by this factor **and** by
:data:`T1_ABSOLUTE_MS` to trip. A relative rule alone flags noise on a
sub-millisecond operation; an absolute rule alone lets a real proportional
regression on a large file pass."""

T1_ABSOLUTE_MS = 2.0
"""The absolute half of T1, in milliseconds."""

T2_DELTA_MS = 25.0
"""``on`` mutation p95 minus ``off`` mutation p95, above which read traffic is
judged to be competing with mutations for the actor rather than slipping between
them — roughly one extra gate check on a large file."""

T3_DEPTH_FACTOR = 1.0
"""p95 sampled mailbox depth in ``on`` reaching ``team size × this`` while ``off``
stays below it: the actor is absorbing read traffic as a queue rather than as an
O(1) append, which is the claim the design rests on."""

##
## Operations.  Reported per operation and per size bucket, never as a mix-wide
## aggregate: only a *text* ``workspace_read`` records anything at all, so an
## average over the mix divides the real cost by the number of legs.
##

OP_GLOB = "glob"
OP_GREP = "grep"
OP_DOCUMENT = "document"
OP_OWNED = "read:owned"
OP_WRITE = "write"

RECORDING_FREE_OPS = (OP_GLOB, OP_GREP, OP_DOCUMENT)
"""The operations that record nothing at all, in every arm — the negative controls.

``workspace_glob`` never opens a file; ``workspace_grep`` reads content the agent
is never shown in full; and the document branch of ``workspace_read`` leaves the
observed bytes unset, on the extraction branch and on the sidecar cache hit alike.
The recorder is therefore not called for any of them, whichever arm is running, so
their ``off`` -> ``on`` p95 delta has **no true effect in it at all**: it is a
direct measurement of this harness's own noise on the statistic every verdict is
computed from.

That makes the band they span the honest scale to read a *read* delta against —
a stricter test than the run-to-run spread, which is a min-max over three runs and
routinely narrower than the controls show the real variance to be.
"""

DEFAULT_BUCKETS: tuple[tuple[str, int], ...] = (
    ("small", 2_048),
    ("medium", 200_000),
    ("large", 5_000_000),
)
"""Name and approximate size in bytes of each read bucket."""

QUICK_BUCKETS: tuple[tuple[str, int], ...] = (("small", 2_048), ("medium", 20_000))
"""What ``--quick`` and the smoke test use — no 5 MB leg in a 1 s budget."""

OWNED_BYTES = 2_048
"""Size of the file each agent owns and rewrites. Deliberately small: the
mutation number of interest is the actor round trip and the git forks, not the
cost of hashing a large file, and it is constant across the arms either way."""

FULL_READ_LIMIT = 10_000_000
"""A line budget no corpus file reaches, so the owned read is a *whole*-file read
and the whole-file write that follows it is accepted rather than refused."""

DEFAULT_READ_LIMIT = 200
"""Lines per page for the bucket reads — every bucket but ``small`` is therefore
a paginated read that still hashes the whole file."""

GATE_TIMEOUT_S = 120.0
"""Upper bound on the start handshake. Never a delay — only a failure budget."""

SHUTDOWN_TIMEOUT_S = 30
MAX_DEPTH_SAMPLES = 200_000
MAX_REPORTED_ERRORS = 5
MAX_ERROR_CHARS = 200

_PERCENTILE_P50 = 0.50
_PERCENTILE_P95 = 0.95


##
## Result records.  Pydantic rather than dicts because ``AgentResult`` and
## ``ActorSnapshot`` cross the actor boundary on the way back to the driver.
##


class OperationSamples(SerializableBaseModel):
    """Every measured duration for one operation, from one agent."""

    operation: str
    durations_ms: list[float]


class AgentResult(SerializableBaseModel):
    """What one benchmark agent measured on its own thread."""

    agent: str
    operations: list[OperationSamples]
    refusals: int
    errors: list[str]


class ActorSnapshot(SerializableBaseModel):
    """What the instrumented ``#Workspace`` saw, copied out through one ask.

    Attributes:
        depths: Mailbox depth sampled at the actor's own turn boundaries.
            ``queue.Queue.qsize`` is documented as unreliable under concurrency,
            so this series is **approximate**; T3's threshold is coarse enough to
            survive the imprecision, and the report says so rather than
            presenting the series as exact.

            A second imprecision, which the arms do not share equally: the sample
            is taken at a *turn boundary*, and the arms have different turns. With
            no observation arriving, ``off`` and ``hash-only`` sample only at the
            six mutation entry points, while ``on`` also samples at every read
            observation — several times as many samples, taken predominantly while
            read traffic is being served. ``on``'s series is therefore weighted
            towards the busy moments in a way the baseline's is not, so part of any
            ``off`` -> ``on`` gap is the difference in *what gets sampled* rather
            than in queue behaviour. There is no cheaper instrument that does not
            add cost to the path under measurement, so this is stated rather than
            corrected.
        read_observations: Recordings that arrived from a read closure.
        mutation_observations: Recordings an accepted mutation made for its own
            writer, which are not read-path traffic and are counted apart.
        journal_enabled: Whether the git journal was actually running.
    """

    depths: list[int]
    read_observations: int
    mutation_observations: int
    journal_enabled: bool


class Stat(SerializableBaseModel):
    """One operation's distribution within one run. Nearest-rank percentiles."""

    samples: int
    p50_ms: float
    p95_ms: float
    max_ms: float


class ArmRun(SerializableBaseModel):
    """One arm, one run: fresh actor system, fresh tree, warm-up discarded."""

    arm: str
    operations: dict[str, Stat]
    depth_p50: float
    depth_p95: float
    depth_max: int
    depth_samples: int
    read_observations: int
    mutation_observations: int
    journal_enabled: bool
    total_commits: int
    out_of_band_commits: int
    refusals: int
    errors: list[str]


class MetricSpread(SerializableBaseModel):
    """Median across runs of one arm, with the min and max it spanned.

    The spread is what lets a reader see whether the difference between two arms
    is larger than the difference between two repeats of the same arm.
    """

    median: float
    minimum: float
    maximum: float


class OperationSummary(SerializableBaseModel):
    """One operation's spread across the runs of one arm."""

    operation: str
    samples: int
    p50_ms: MetricSpread
    p95_ms: MetricSpread
    max_ms: MetricSpread


class ArmSummary(SerializableBaseModel):
    """One arm across all its runs."""

    arm: str
    runs: int
    journal_enabled: bool
    operations: dict[str, OperationSummary]
    depth_p50: MetricSpread
    depth_p95: MetricSpread
    depth_max: MetricSpread
    depth_samples: int
    read_observations: int
    mutation_observations: int
    total_commits: int
    out_of_band_commits: int
    refusals: int
    errors: list[str]


class RuleResult(SerializableBaseModel):
    """One pre-registered threshold, its measured value, and whether it tripped."""

    rule: str
    threshold: str
    measured: str
    tripped: bool


class Environment(SerializableBaseModel):
    """The machine, without which a benchmark number is not evidence."""

    cpu: str
    cores: int
    system: str
    python: str
    package_commit: str
    ripgrep: bool
    markitdown: bool
    git: bool


class BenchmarkResult(SerializableBaseModel):
    """Everything one invocation measured, and the verdict it implies."""

    environment: Environment
    command: str
    agents: int
    iterations: int
    warmup: int
    runs: int
    seed: int
    buckets: list[str]
    document_leg: str
    grep_engine: str
    arms: list[ArmSummary]
    rules: list[RuleResult]
    verdict: str
    notes: list[str]


##
## The run specification
##


@dataclass(frozen=True)
class RunSpec:
    """Everything one arm-run needs, and nothing that varies between arms."""

    agents: int
    iterations: int
    warmup: int
    seed: int
    read_limit: int = DEFAULT_READ_LIMIT
    document: bool = True
    buckets: tuple[tuple[str, int], ...] = DEFAULT_BUCKETS
    grep_files: int = 20


def read_key(bucket: str) -> str:
    """Return the operation key for a read in *bucket*."""
    return f"read:{bucket}"


def expected_operations(spec: RunSpec) -> list[str]:
    """Return every operation key a run of *spec* must produce, in report order."""
    keys = [read_key(name) for name, _ in spec.buckets]
    keys += [OP_GLOB, OP_GREP]
    if spec.document and _markitdown_available():
        keys.append(OP_DOCUMENT)
    keys += [OP_OWNED, OP_WRITE]
    return keys


##
## The start handshake.  A module-level rendezvous rather than a field on the
## agent's config, because a config is a serialisable Pydantic model and a
## barrier is not.  One benchmark process runs one arm at a time.
##


@dataclass(frozen=True)
class _Gate:
    """Ten agents warm up, meet at ``ready``, and start together on ``go``."""

    ready: threading.Barrier
    go: threading.Event


_GATE: _Gate | None = None


@contextmanager
def _gate_installed(gate: _Gate) -> Iterator[None]:
    """Publish *gate* to the agents for the duration of one run."""
    global _GATE
    _GATE = gate
    try:
        yield
    finally:
        _GATE = None


def _current_gate() -> _Gate:
    """Return the run's gate, or fail loudly rather than start unsynchronised."""
    if _GATE is None:
        raise RuntimeError("no start gate is installed for this run")
    return _GATE


##
## The arms.  Patching the recorder *after* ``get_tools()`` would do nothing:
## ``_observation_recorder()`` is called once inside ``_read_factory`` and the
## read closure captures the callable it returns.  The patch therefore lives
## around agent creation, and Task 4's assertion checks afterwards that the arm
## did what it claimed.
##

_Recorder = Callable[[str, bytes, bool], None]


def _off_recorder(_card: WorkspaceTool) -> _Recorder:
    """Build a recorder that does nothing at all — the baseline arm."""

    def record(path: str, data: bytes, full: bool) -> None:
        return None

    return record


def _hash_only_recorder(_card: WorkspaceTool) -> _Recorder:
    """Build a recorder that computes the digest and sends nothing.

    It stops at :func:`content_sha`, where the shipped recorder goes on to build an
    ``Observation`` and send it inside a ``try``/``except``. So the ``hash-only``
    -> ``on`` step is an **upper bound** on what per-turn batching removes rather
    than an estimate of it: batching drops the message, not the model behind it.
    The bound flatters batching, which is the safe direction for a conclusion that
    batching is not the instrument this cost calls for.
    """

    def record(path: str, data: bytes, full: bool) -> None:
        content_sha(data)

    return record


@contextmanager
def _arm_patch(arm: str) -> Iterator[None]:
    """Install the arm's recorder for the duration of one run.

    ``on`` and ``on+journal`` patch nothing whatsoever — they must exercise the
    shipped code path exactly, or the headline number is a measurement of this
    file rather than of the design.
    """
    replacement = {ARM_OFF: _off_recorder, ARM_HASH_ONLY: _hash_only_recorder}.get(arm)
    if replacement is None:
        yield
        return
    original = WorkspaceTool._observation_recorder
    setattr(WorkspaceTool, "_observation_recorder", replacement)
    try:
        yield
    finally:
        setattr(WorkspaceTool, "_observation_recorder", original)


##
## The instrumented singleton
##


class _SamplingWorkspaceActor(WorkspaceActor):
    """``#Workspace`` that samples **its own** mailbox depth at each turn boundary.

    Installed by creating ``#Workspace-<workspace_id>`` through the orchestrator
    *before* the first card wires. Every card then binds to it, because
    ``Orchestrator.getChildrenOrCreate`` resolves an existing live child by
    ``config.name`` and never by class. That is what keeps this benchmark clear
    of the actor-internals rule: no ``ActorAddressImpl._actor_ref`` cast, no
    patch of any production module — ``self.actor_inbox`` is this actor's own
    pykka attribute, read on its own thread.

    Recordings an accepted mutation makes for its own writer (``_accept`` calls
    ``record_observation`` in the same turn) are counted apart from read-path
    traffic, otherwise the ``off`` arm would appear to record.
    """

    def on_start(self) -> None:
        """Initialise the sampling state after the real actor's own start."""
        super().on_start()
        self._depths: list[int] = []
        self._read_observations = 0
        self._mutation_observations = 0
        self._in_mutation = False

    ##
    ## Sampling — always on this actor's own thread
    ##
    def _sample_depth(self) -> None:
        """Record how much work was waiting when this turn began.

        ``self.actor_inbox`` is **this** actor's own pykka attribute, read on its
        own thread — no cast through ``ActorAddressImpl._actor_ref``, no reach
        into a second actor. The cast below is only pykka's own typing: it
        declares the inbox as the ``ActorInbox`` protocol, which does not carry
        ``qsize``, while a ``ThreadingActor``'s inbox is a ``queue.Queue``.
        """
        if len(self._depths) < MAX_DEPTH_SAMPLES:
            inbox = cast("queue.Queue[object]", self.actor_inbox)
            self._depths.append(inbox.qsize())

    def _mutating(self, run: Callable[[], MutationOutcome]) -> MutationOutcome:
        """Sample, then run one mutation with writer-side recording marked."""
        self._sample_depth()
        self._in_mutation = True
        try:
            return run()
        finally:
            self._in_mutation = False

    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        """Count and sample a read-path recording, then record it for real."""
        if self._in_mutation:
            self._mutation_observations += 1
        else:
            self._sample_depth()
            self._read_observations += 1
        super().record_observation(agent_id, path, observation)

    ##
    ## The six mutations.  Each is a turn boundary, so each samples.
    ##
    def apply_write(self, agent_id: str, path: str, content: str) -> MutationOutcome:
        """Sample, then delegate."""
        return self._mutating(lambda: WorkspaceActor.apply_write(self, agent_id, path, content))

    def apply_delete(self, agent_id: str, path: str) -> MutationOutcome:
        """Sample, then delegate."""
        return self._mutating(lambda: WorkspaceActor.apply_delete(self, agent_id, path))

    def apply_edit(
        self,
        agent_id: str,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> MutationOutcome:
        """Sample, then delegate."""
        return self._mutating(
            lambda: WorkspaceActor.apply_edit(
                self, agent_id, path, old_string, new_string, replace_all
            )
        )

    def apply_multi_edit(self, agent_id: str, edits: list[EditItem]) -> MutationOutcome:
        """Sample, then delegate."""
        return self._mutating(lambda: WorkspaceActor.apply_multi_edit(self, agent_id, edits))

    def apply_patch(self, agent_id: str, patch_text: str) -> MutationOutcome:
        """Sample, then delegate."""
        return self._mutating(lambda: WorkspaceActor.apply_patch(self, agent_id, patch_text))

    def apply_mkdir(self, agent_id: str, path: str) -> MutationOutcome:
        """Sample, then delegate."""
        return self._mutating(lambda: WorkspaceActor.apply_mkdir(self, agent_id, path))

    ##
    ## The driver's two calls, both asks — so the second one is also the drain
    ##
    def bench_reset(self) -> None:
        """Drop everything the warm-up produced, on the actor's own thread."""
        self._depths = []
        self._read_observations = 0
        self._mutation_observations = 0

    def bench_snapshot(self) -> ActorSnapshot:
        """Copy the series out. An ask, so every earlier tell has been processed."""
        return ActorSnapshot(
            depths=list(self._depths),
            read_observations=self._read_observations,
            mutation_observations=self._mutation_observations,
            journal_enabled=self._journal.enabled,
        )


##
## The benchmark agent
##


class BenchAgentConfig(BaseConfig):
    """One agent's slice of the corpus and its turn budget.

    Serialisable throughout — the rendezvous the agents share is a module-level
    object, not a field, because a ``threading.Barrier`` has no place in a config.
    """

    workspace_id: str = ""
    journal: bool = True
    warmup: int = 0
    iterations: int = 0
    read_limit: int = DEFAULT_READ_LIMIT
    bucket_paths: dict[str, str] = {}
    owned_path: str = ""
    document_path: str = ""
    grep_root: str = ""
    grep_pattern: str = ""
    grep_include: str = ""
    glob_pattern: str = ""


class _BenchAgent(Akgent[BenchAgentConfig, BaseState]):
    """A real ``Akgent`` carrying a real ``WorkspaceTool``, wired the shipped way.

    An ``Akgent`` already **is** an ``ActorToolObserver``, so the card binds
    through ``observer()`` with no double anywhere in the measured path — which
    matters, because a double is precisely what would let a wrong number look
    right. Every tool call runs here, on this agent's own actor thread; the
    driver never calls a callable itself.
    """

    def on_start(self) -> None:
        """Wire the card and resolve its callables once, off the measured window."""
        card = WorkspaceTool(
            workspace_id=self.config.workspace_id,
            git_journal=self.config.journal,
            workspace_view=False,
            # ``llm_client=None`` pins pass 2 off. A benchmark that reaches for
            # OpenAI on a short extraction would measure the network.
            workspace_read=WorkspaceRead(document_reader=DocumentReader(llm_client=None)),
        )
        card.observer(self)
        self._card = card
        self._tools: dict[str, Callable[..., Any]] = {
            tool.__name__: tool for tool in card.get_tools()
        }
        self._samples: dict[str, list[float]] = {}
        self._refusals = 0
        self._errors: list[str] = []
        self._revision = 0

    ##
    ## The measured window
    ##
    def run_mix(self) -> None:
        """Warm up, meet the other agents, then run the measured turns.

        Told rather than asked: the whole point is that this runs on the agent's
        own thread while nine others do the same.
        """
        self._seed_owned_file()
        for _ in range(self.config.warmup):
            self._turn(record=False)
        gate = _current_gate()
        try:
            gate.ready.wait(timeout=GATE_TIMEOUT_S)
        except threading.BrokenBarrierError:
            self._note("start barrier broke — an agent never reached it")
            return
        gate.go.wait(timeout=GATE_TIMEOUT_S)
        for _ in range(self.config.iterations):
            self._turn(record=True)

    def _seed_owned_file(self) -> None:
        """Create the file this agent owns, untimed, before anything is measured.

        This is what makes the read-then-write leg comparable **across arms**.
        The gate's precondition for a whole-file overwrite is an observation, and
        in the ``off`` and ``hash-only`` arms no read ever records one — so a
        pre-created file would be refused on every turn of those arms, and the
        mutation numbers they contributed would be the cost of a *refusal*, which
        is a different and cheaper path.

        The one row of the gate table that needs no observation is *create*:
        nothing recorded, nothing on disk. The create is accepted, and the actor
        refreshes the writer's own observation in the same turn, so every later
        turn's write is accepted in every arm without any read having recorded.
        """
        path = self.config.owned_path
        try:
            self._tools["workspace_write"](path, owned_body(path, 0))
        except Exception as exc:
            self._note(f"seeding {path}: {type(exc).__name__}: {exc}")

    def results(self) -> AgentResult:
        """Hand the driver what this agent measured. Asked, so it also joins."""
        return AgentResult(
            agent=self.config.name,
            operations=[
                OperationSamples(operation=key, durations_ms=values)
                for key, values in sorted(self._samples.items())
            ],
            refusals=self._refusals,
            errors=self._errors,
        )

    ##
    ## One turn of the mix
    ##
    def _turn(self, record: bool) -> None:
        """Paginated reads across the buckets, a glob, a grep, a document, a write."""
        read = self._tools["workspace_read"]
        for bucket, path in self.config.bucket_paths.items():
            self._timed(read_key(bucket), record, read, path, 1, self.config.read_limit)
        self._timed(OP_GLOB, record, self._tools["workspace_glob"], self.config.glob_pattern)
        self._timed(
            OP_GREP,
            record,
            self._tools["workspace_grep"],
            self.config.grep_pattern,
            self.config.grep_root,
            self.config.grep_include,
        )
        if self.config.document_path:
            self._timed(OP_DOCUMENT, record, read, self.config.document_path)
        self._read_then_write(record)

    def _read_then_write(self, record: bool) -> None:
        """Read a file this agent owns in full, then rewrite it.

        Whole-file, because a page read is not a precondition for a whole-file
        overwrite — a paginated read here would produce a *refused* write, which
        is a different and cheaper path and would flatter the mutation numbers.
        """
        path = self.config.owned_path
        self._timed(OP_OWNED, record, self._tools["workspace_read"], path, 1, FULL_READ_LIMIT)
        self._revision += 1
        body = owned_body(path, self._revision)
        self._timed(OP_WRITE, record, self._tools["workspace_write"], path, body)

    def _timed(self, key: str, record: bool, call: Callable[..., Any], *args: Any) -> None:
        """Time one tool call with ``perf_counter``; drop it if it did not succeed."""
        started = time.perf_counter()
        try:
            call(*args)
        except RetriableError as exc:
            self._refusals += 1
            self._note(f"{key}: refused: {exc}")
            return
        except Exception as exc:  # a benchmark must report, never abort a run
            self._note(f"{key}: {type(exc).__name__}: {exc}")
            return
        if record:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._samples.setdefault(key, []).append(elapsed_ms)

    def _note(self, message: str) -> None:
        """Keep a bounded record of what went wrong, for the report.

        Bounded in both directions: a refusal carries a diff, and an unbounded
        one would bury the report it is meant to annotate.
        """
        if len(self._errors) < MAX_REPORTED_ERRORS:
            self._errors.append(message[:MAX_ERROR_CHARS].replace("\n", " / "))


##
## The corpus
##


@dataclass(frozen=True)
class Corpus:
    """Where each agent reads and writes. Identical bytes in every arm."""

    bucket_paths: list[dict[str, str]]
    owned_paths: list[str]
    document_paths: list[str]
    grep_root: str
    grep_pattern: str
    grep_include: str
    glob_pattern: str
    document_note: str


_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whisky"
).split()

_blob_cache: dict[tuple[int, int], str] = {}


def _blob(seed: int, size_bytes: int) -> str:
    """Return deterministic text of about *size_bytes*, generated once per size.

    Cached so that every arm and every run reads byte-identical content, and so
    that generating a 5 MB corpus twelve times costs one generation.
    """
    key = (seed, size_bytes)
    cached = _blob_cache.get(key)
    if cached is not None:
        return cached
    rng = _lcg(seed ^ size_bytes)
    lines: list[str] = []
    total = 0
    while total < size_bytes:
        line = " ".join(_WORDS[next(rng) % len(_WORDS)] for _ in range(12))
        lines.append(line)
        total += len(line) + 1
    text = "\n".join(lines) + "\n"
    _blob_cache[key] = text
    return text


def _lcg(seed: int) -> Iterator[int]:
    """A tiny deterministic generator — no dependency on ``random``'s internals."""
    state = (seed or 1) & 0xFFFFFFFF
    while True:
        state = (1_103_515_245 * state + 12_345) & 0x7FFFFFFF
        yield state


def owned_body(path: str, revision: int) -> str:
    """Return the content an agent writes to the file it owns."""
    return f"# {path} revision {revision}\n{_blob(0, OWNED_BYTES)}"


def bucket_count(agents: int, size_bytes: int) -> int:
    """How many files a bucket holds, so ten agents are not on one cache entry.

    Halved above a megabyte: ten 5 MB files per run, twelve runs deep, is disk
    churn that buys nothing the halved count does not already buy.
    """
    return max(2, agents if size_bytes < 1_000_000 else agents // 2)


def build_corpus(tree: Path, spec: RunSpec) -> Corpus:
    """Write a fresh tree and return where each agent should look.

    Fresh per run (AC6): reads dirty a tree — a document read writes a sidecar —
    so a second run of the same arm against the same tree would not be doing the
    same work.
    """
    for name, size in spec.buckets:
        _write_bucket(tree / name, name, bucket_count(spec.agents, size), size, spec.seed)
    _write_bucket(tree / "grep", "grep", spec.grep_files, 2_048, spec.seed)
    owned = [f"owned/agent-{slot}.txt" for slot in range(spec.agents)]
    documents, note = _write_documents(tree, spec)
    bucket_paths = [_bucket_paths_for(slot, spec) for slot in range(spec.agents)]
    return Corpus(
        bucket_paths=bucket_paths,
        owned_paths=owned,
        document_paths=documents,
        grep_root="grep",
        grep_pattern="quebec romeo",
        grep_include="*.txt",
        glob_pattern="**/*.txt",
        document_note=note,
    )


def _write_bucket(directory: Path, name: str, count: int, size: int, seed: int) -> None:
    """Write *count* files of about *size* bytes into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    body = _blob(seed, size)
    for index in range(count):
        (directory / f"{name}-{index}.txt").write_text(f"# {name} {index}\n{body}", "utf-8")


def _bucket_paths_for(slot: int, spec: RunSpec) -> dict[str, str]:
    """Spread agents across the files of each bucket, one page-cache entry apart."""
    paths: dict[str, str] = {}
    for name, size in spec.buckets:
        count = bucket_count(spec.agents, size)
        paths[name] = f"{name}/{name}-{slot % count}.txt"
    return paths


def _write_documents(tree: Path, spec: RunSpec) -> tuple[list[str], str]:
    """Write one small PDF per agent, or record why the leg was skipped.

    The document leg contributes CPU contention and agent-thread occupancy and
    **no observations at all**: neither the extraction branch nor the sidecar
    cache-hit branch ever sets the observed bytes, because in both cases the
    agent sees derived text rather than the source.
    """
    if not spec.document:
        return [""] * spec.agents, "skipped: not requested"
    if not _markitdown_available():
        return [""] * spec.agents, "skipped: markitdown is not installed"
    directory = tree / "docs"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for slot in range(spec.agents):
        relative = f"docs/doc-{slot}.pdf"
        (tree / relative).write_bytes(minimal_pdf(_document_text(slot)))
        paths.append(relative)
    return paths, "ran: a one-page PDF per agent, served from its sidecar after the first read"


def _document_text(slot: int) -> str:
    """Body of the generated PDF, deliberately well over fifty characters.

    ``DocumentReader`` falls back to a second, LLM-assisted extraction pass when
    the first yields fewer than fifty non-whitespace characters. A benchmark must
    never take that branch.
    """
    return (
        f"Workspace benchmark document number {slot} - "
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
        "mike november oscar papa quebec romeo sierra tango uniform victor whisky"
    )


def _markitdown_available() -> bool:
    """Whether the ``docs`` extra is installed in this interpreter."""
    try:
        import markitdown  # noqa: F401 — a probe, not a dependency
    except ImportError:
        return False
    return True


def minimal_pdf(text: str) -> bytes:
    """Return a one-page PDF carrying *text* — stdlib only, no new dependency."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(bodies) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(bodies) + 1,
        xref_at,
    )
    return bytes(out)


##
## Driving one arm-run
##


def run_arm(spec: RunSpec, arm: str, base: Path) -> ArmRun:
    """Run one arm once, on a fresh actor system and a fresh tree."""
    workspace_id = f"bench-{uuid.uuid4().hex[:8]}"
    root = base / workspace_id
    tree = root / workspace_id
    tree.mkdir(parents=True)
    previous_root = os.environ.get("AKGENTIC_WORKSPACES_ROOT")
    os.environ["AKGENTIC_WORKSPACES_ROOT"] = str(root)
    corpus = build_corpus(tree, spec)
    system = ActorSystem()
    gate = _Gate(threading.Barrier(spec.agents + 1), threading.Event())
    try:
        with _arm_patch(arm), _gate_installed(gate):
            return _drive(system, spec, arm, workspace_id, corpus, gate, tree)
    finally:
        gate.go.set()
        system.shutdown(timeout=SHUTDOWN_TIMEOUT_S)
        pykka.ActorRegistry.stop_all()
        _restore_env(previous_root)
        shutil.rmtree(root, ignore_errors=True)


def _restore_env(previous: str | None) -> None:
    """Put ``AKGENTIC_WORKSPACES_ROOT`` back the way this run found it."""
    if previous is None:
        os.environ.pop("AKGENTIC_WORKSPACES_ROOT", None)
    else:
        os.environ["AKGENTIC_WORKSPACES_ROOT"] = previous


def _drive(
    system: ActorSystem,
    spec: RunSpec,
    arm: str,
    workspace_id: str,
    corpus: Corpus,
    gate: _Gate,
    tree: Path,
) -> ArmRun:
    """Build the team, release it, collect from it, and check the arm behaved."""
    journal = arm == ARM_ON_JOURNAL
    orch_addr = system.createActor(
        Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
    )
    orch = system.proxy_ask(orch_addr, Orchestrator)
    workspace = _install_sampling_actor(system, orch, workspace_id, journal)
    members = [
        _spawn_agent(orch, spec, arm, workspace_id, corpus, slot) for slot in range(spec.agents)
    ]
    for address in members:
        system.proxy_tell(address, _BenchAgent).run_mix()
    gate.ready.wait(timeout=GATE_TIMEOUT_S)
    workspace.bench_reset()
    gate.go.set()
    results = [system.proxy_ask(address, _BenchAgent).results() for address in members]
    # The drain. Every agent has stopped sending (its ``results()`` ask returned),
    # and an ask sits behind every tell already on the mailbox — so when this
    # returns, no observation is still in flight and the ``on`` arm cannot
    # under-report its own traffic.
    snapshot = workspace.bench_snapshot()
    _assert_arm_behaved(arm, snapshot)
    assert_samples_complete(spec, results)
    return _summarise_run(arm, results, snapshot, tree, journal)


def _install_sampling_actor(
    system: ActorSystem, orch: Orchestrator, workspace_id: str, journal: bool
) -> _SamplingWorkspaceActor:
    """Create ``#Workspace-<id>`` as the instrumented subclass, before any card wires.

    Every card then binds to it by name. The proxy's ``bench_snapshot`` would not
    resolve at all against a plain ``WorkspaceActor``, so a silent failure to
    install would be an immediate error rather than a series of zeroes that
    reads like good news.
    """
    address = orch.createActor(
        _SamplingWorkspaceActor,
        config=WorkspaceConfig(
            name=workspace_actor_name(workspace_id),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_name=workspace_id,
            git_journal=journal,
        ),
    )
    workspace = system.proxy_ask(address, _SamplingWorkspaceActor)
    workspace.bench_reset()
    return workspace


def _spawn_agent(
    orch: Orchestrator,
    spec: RunSpec,
    arm: str,
    workspace_id: str,
    corpus: Corpus,
    slot: int,
) -> ActorAddress:
    """Create one real member through the orchestrator, the shipped way."""
    return orch.createActor(
        _BenchAgent,
        config=BenchAgentConfig(
            name=f"bench-{slot}",
            role="Benchmark",
            workspace_id=workspace_id,
            journal=arm == ARM_ON_JOURNAL,
            warmup=spec.warmup,
            iterations=spec.iterations,
            read_limit=spec.read_limit,
            bucket_paths=corpus.bucket_paths[slot],
            owned_path=corpus.owned_paths[slot],
            document_path=corpus.document_paths[slot],
            grep_root=corpus.grep_root,
            grep_pattern=corpus.grep_pattern,
            grep_include=corpus.grep_include,
            glob_pattern=corpus.glob_pattern,
        ),
    )


def _assert_arm_behaved(arm: str, snapshot: ActorSnapshot) -> None:
    """Fail the run rather than publish a number from an arm that did the wrong thing.

    An arm that silently did the wrong thing is the one failure that makes every
    number in the report a lie — a patch applied too late leaves an ``off`` arm
    that is quietly the ``on`` arm.
    """
    records = arm in (ARM_ON, ARM_ON_JOURNAL)
    if records and snapshot.read_observations == 0:
        raise RuntimeError(f"arm {arm!r} recorded nothing — the shipped path did not run")
    if not records and snapshot.read_observations:
        raise RuntimeError(
            f"arm {arm!r} recorded {snapshot.read_observations} observations —"
            " the recorder patch was installed too late"
        )
    if arm == ARM_ON_JOURNAL and not snapshot.journal_enabled:
        raise RuntimeError("the journal arm ran with the journal off")
    if arm != ARM_ON_JOURNAL and snapshot.journal_enabled:
        raise RuntimeError(f"arm {arm!r} ran with the journal ON — not a valid read-path result")


def assert_samples_complete(spec: RunSpec, results: list[AgentResult]) -> None:
    """Fail the run rather than publish a percentile over calls that never happened.

    :meth:`_BenchAgent._timed` drops the sample of any call that raised or was
    refused, and every statistic downstream is computed over whatever survived. So
    an arm can quietly do less work than the arm it is compared against, and the
    report shows only a smaller ``n`` in a column nobody is reading. Worse,
    :func:`_rule_t2` baselines a *missing* ``write`` operation at 0.0 ms: an arm
    whose writes were all refused yields a 25 ms threshold measured against
    nothing.

    That is not a hypothetical. It is what the first shake-out run of this harness
    did — with recording patched out, no read recorded, so every whole-file write
    was refused and the two baseline arms contributed the latency of a refusal. It
    was caught by reading the refusal counter by hand. The rest of the harness
    already holds the principle this restores: fail the run rather than publish a
    number from an arm that did the wrong thing.
    """
    wanted = spec.iterations
    expected = set(expected_operations(spec))
    for result in results:
        if result.refusals or result.errors:
            raise RuntimeError(
                f"agent {result.agent} reported {result.refusals} refusals and"
                f" {len(result.errors)} errors — its samples are not comparable:"
                f" {result.errors}"
            )
        missing = sorted(expected - {samples.operation for samples in result.operations})
        if missing:
            raise RuntimeError(f"agent {result.agent} produced no samples at all for {missing}")
        for samples in result.operations:
            if len(samples.durations_ms) != wanted:
                raise RuntimeError(
                    f"agent {result.agent} produced {len(samples.durations_ms)} samples for"
                    f" {samples.operation}, not the {wanted} measured turns it was asked for"
                )


##
## Statistics
##


def percentile(samples: list[float], fraction: float) -> float:
    """Nearest-rank percentile on the sorted samples, so the number reproduces.

    Deliberately unit-free: it serves both the latencies and the mailbox depths,
    and the depths are counts.
    """
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _stat(samples: list[float]) -> Stat:
    """Summarise one operation's samples within one run."""
    return Stat(
        samples=len(samples),
        p50_ms=percentile(samples, _PERCENTILE_P50),
        p95_ms=percentile(samples, _PERCENTILE_P95),
        max_ms=max(samples) if samples else 0.0,
    )


def _spread(values: list[float]) -> MetricSpread:
    """Median across runs, with the min and max — the noise floor, made visible."""
    if not values:
        return MetricSpread(median=0.0, minimum=0.0, maximum=0.0)
    return MetricSpread(median=statistics.median(values), minimum=min(values), maximum=max(values))


def _summarise_run(
    arm: str, results: list[AgentResult], snapshot: ActorSnapshot, tree: Path, journal: bool
) -> ArmRun:
    """Fold every agent's samples into one row for this arm-run."""
    pooled: dict[str, list[float]] = {}
    errors: list[str] = []
    refusals = 0
    for result in results:
        refusals += result.refusals
        errors += result.errors
        for operation in result.operations:
            pooled.setdefault(operation.operation, []).extend(operation.durations_ms)
    depths = [float(depth) for depth in snapshot.depths]
    total, out_of_band = _journal_counts(tree) if journal else (0, 0)
    return ArmRun(
        arm=arm,
        operations={key: _stat(values) for key, values in pooled.items()},
        depth_p50=percentile(depths, _PERCENTILE_P50),
        depth_p95=percentile(depths, _PERCENTILE_P95),
        depth_max=max(snapshot.depths) if snapshot.depths else 0,
        depth_samples=len(snapshot.depths),
        read_observations=snapshot.read_observations,
        mutation_observations=snapshot.mutation_observations,
        journal_enabled=snapshot.journal_enabled,
        total_commits=total,
        out_of_band_commits=out_of_band,
        refusals=refusals,
        errors=errors[:MAX_REPORTED_ERRORS],
    )


def _journal_counts(tree: Path) -> tuple[int, int]:
    """Return (commits, commits authored ``out-of-band``) from the run's journal.

    Read from the artefact the run left behind rather than from instrumentation:
    it answers how often the dirty-tree branch of the mutation path actually
    fired, which is what decides whether a mutation costs three git invocations
    or five.
    """
    try:
        raw = subprocess.run(
            ["git", "--git-dir", str(git_dir_for(tree)), "log", "--format=%an"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        return (0, 0)
    if raw.returncode != 0:
        return (0, 0)
    authors = raw.stdout.split("\n")
    authors = [author for author in authors if author.strip()]
    return (len(authors), sum(1 for author in authors if author == "out-of-band"))


def summarise_arm(arm: str, runs: list[ArmRun]) -> ArmSummary:
    """Fold an arm's runs into medians with their spread."""
    keys = sorted({key for run in runs for key in run.operations})
    operations = {key: _summarise_operation(key, runs) for key in keys}
    return ArmSummary(
        arm=arm,
        runs=len(runs),
        journal_enabled=any(run.journal_enabled for run in runs),
        operations=operations,
        depth_p50=_spread([run.depth_p50 for run in runs]),
        depth_p95=_spread([run.depth_p95 for run in runs]),
        depth_max=_spread([float(run.depth_max) for run in runs]),
        depth_samples=sum(run.depth_samples for run in runs),
        read_observations=sum(run.read_observations for run in runs),
        mutation_observations=sum(run.mutation_observations for run in runs),
        total_commits=sum(run.total_commits for run in runs),
        out_of_band_commits=sum(run.out_of_band_commits for run in runs),
        refusals=sum(run.refusals for run in runs),
        errors=[error for run in runs for error in run.errors][:MAX_REPORTED_ERRORS],
    )


def _summarise_operation(key: str, runs: list[ArmRun]) -> OperationSummary:
    """Fold one operation across the runs of one arm."""
    stats = [run.operations[key] for run in runs if key in run.operations]
    return OperationSummary(
        operation=key,
        samples=sum(stat.samples for stat in stats),
        p50_ms=_spread([stat.p50_ms for stat in stats]),
        p95_ms=_spread([stat.p95_ms for stat in stats]),
        max_ms=_spread([stat.max_ms for stat in stats]),
    )


##
## The verdict
##


def evaluate_rules(spec: RunSpec, arms: dict[str, ArmSummary]) -> list[RuleResult]:
    """Apply the four pre-registered rules to the measured summaries."""
    rules: list[RuleResult] = []
    off = arms.get(ARM_OFF)
    on = arms.get(ARM_ON)
    if off is None or on is None:
        return rules
    rules += _rule_t1(spec, off, on)
    rules.append(_rule_t2(off, on))
    rules.append(_rule_t3(spec, off, on))
    t4 = _rule_t4(on, arms.get(ARM_ON_JOURNAL))
    if t4 is not None:
        rules.append(t4)
    return rules


def _rule_t1(spec: RunSpec, off: ArmSummary, on: ArmSummary) -> list[RuleResult]:
    """T1 — read latency, per size bucket, both conditions required."""
    out: list[RuleResult] = []
    for name, _size in spec.buckets:
        key = read_key(name)
        if key not in off.operations or key not in on.operations:
            continue
        base = off.operations[key].p95_ms.median
        measured = on.operations[key].p95_ms.median
        tripped = measured > T1_RATIO * base and measured > base + T1_ABSOLUTE_MS
        out.append(
            RuleResult(
                rule=f"T1 read latency [{name}]",
                threshold=(
                    f"on p95 > {T1_RATIO:.2f} x off p95 ({T1_RATIO * base:.3f} ms)"
                    f" AND > off p95 + {T1_ABSOLUTE_MS:.1f} ms ({base + T1_ABSOLUTE_MS:.3f} ms)"
                ),
                measured=f"off p95 {base:.3f} ms, on p95 {measured:.3f} ms",
                tripped=tripped,
            )
        )
    return out


def _rule_t2(off: ArmSummary, on: ArmSummary) -> RuleResult:
    """T2 — mutation latency: is read traffic competing with mutations?"""
    base = off.operations[OP_WRITE].p95_ms.median if OP_WRITE in off.operations else 0.0
    measured = on.operations[OP_WRITE].p95_ms.median if OP_WRITE in on.operations else 0.0
    return RuleResult(
        rule="T2 mutation latency",
        threshold=(
            f"on write p95 > off write p95 + {T2_DELTA_MS:.0f} ms ({base + T2_DELTA_MS:.3f} ms)"
        ),
        measured=(
            f"off p95 {base:.3f} ms, on p95 {measured:.3f} ms (delta {measured - base:+.3f} ms)"
        ),
        tripped=measured > base + T2_DELTA_MS,
    )


def _rule_t3(spec: RunSpec, off: ArmSummary, on: ArmSummary) -> RuleResult:
    """T3 — sustained mailbox depth of one message per agent."""
    limit = spec.agents * T3_DEPTH_FACTOR
    return RuleResult(
        rule="T3 mailbox depth",
        threshold=f"on depth p95 >= {limit:.0f} (team size) while off stays below it",
        measured=(
            f"off p95 {off.depth_p95.median:.1f} (max {off.depth_max.maximum:.0f}), "
            f"on p95 {on.depth_p95.median:.1f} (max {on.depth_max.maximum:.0f})"
        ),
        tripped=on.depth_p95.median >= limit > off.depth_p95.median,
    )


def _rule_t4(on: ArmSummary, journalled: ArmSummary | None) -> RuleResult | None:
    """T4 — not a cost, a defect: does the read tail track concurrent mutations?

    The property under test is 29-3's: a read's observation is fire-and-forget,
    so a reader never waits on the actor however long a mutation takes. The way
    to falsify that is to **change how long a mutation takes** and see whether
    the read tail follows. Turning the journal on does exactly that — it adds git
    invocations to every mutation and lengthens it by two orders of magnitude —
    so the discriminating question is whether the read tail grows by as much.

    A comparison of two read maxima against a single arm's write p50 was the
    first formulation and is not used: at any realistic sample count it compares
    two extreme-value statistics and fires on noise. This form has a signal the
    size of the whole journal cost.

    Returns:
        The rule, or ``None`` when the journal arm was not run and the question
        therefore cannot be put.
    """
    if journalled is None or OP_WRITE not in on.operations:
        return None
    if OP_WRITE not in journalled.operations:
        return None
    mutation_growth = (
        journalled.operations[OP_WRITE].p50_ms.median - on.operations[OP_WRITE].p50_ms.median
    )
    growths = [
        (summary.max_ms.median - on.operations[key].max_ms.median, key)
        for key, summary in journalled.operations.items()
        if key.startswith("read:") and key in on.operations
    ]
    worst, where = max(growths) if growths else (0.0, "n/a")
    return RuleResult(
        rule="T4 read tail tracks mutations",
        threshold=(
            f"worst read max growth >= mutation p50 growth ({mutation_growth:.3f} ms)"
            " when the journal lengthens every mutation"
        ),
        measured=f"worst read max growth {worst:.3f} ms on {where}",
        tripped=mutation_growth > 0.0 and worst >= mutation_growth,
    )


def mutation_path_note(arms: dict[str, ArmSummary]) -> str:
    """Describe what the journal costs a mutation — the second measurement."""
    plain = arms.get(ARM_ON)
    journalled = arms.get(ARM_ON_JOURNAL)
    if plain is None or journalled is None:
        return "mutation-path arm not run"
    if OP_WRITE not in plain.operations or OP_WRITE not in journalled.operations:
        return "mutation-path arm produced no write samples"
    base = plain.operations[OP_WRITE]
    with_git = journalled.operations[OP_WRITE]
    return (
        f"write p50 {base.p50_ms.median:.3f} -> {with_git.p50_ms.median:.3f} ms, "
        f"p95 {base.p95_ms.median:.3f} -> {with_git.p95_ms.median:.3f} ms; "
        f"{journalled.total_commits} commits, {journalled.out_of_band_commits} authored out-of-band"
    )


def decomposition_notes(arms: dict[str, ArmSummary]) -> list[str]:
    """Split the read-path cost into the digest and the message, per operation.

    This is the number the fallback question turns on. Per-agent-turn batching
    removes *messages*; it cannot remove the digest, which is computed on the
    reader's own thread before anything is sent. So if the cost sits in the
    ``off`` -> ``hash-only`` step, batching is the wrong instrument however
    clearly the headline argues for a fallback.

    Two things to read carefully in the output. The rows for
    :data:`RECORDING_FREE_OPS` are marked as controls: they carry no digest and no
    message whatsoever, so whatever appears in their two columns is this harness's
    noise wearing the labels of a cost. And the ``message`` column is an **upper
    bound** on what per-turn batching would remove, not an estimate of it: the
    ``hash-only`` arm stops after :func:`content_sha`, so the step from it to
    ``on`` also carries the ``Observation`` model construction and the recorder's
    ``try``/``except``, and a batching scheme would still build an observation per
    path. The bound errs towards making batching look *better* than it is, which
    is the safe direction for a conclusion that batching is not worth building.
    """
    off = arms.get(ARM_OFF)
    hash_only = arms.get(ARM_HASH_ONLY)
    on = arms.get(ARM_ON)
    if off is None or hash_only is None or on is None:
        return []
    notes: list[str] = []
    for key in sorted(on.operations):
        if key not in off.operations or key not in hash_only.operations:
            continue
        base = off.operations[key].p95_ms.median
        digest = hash_only.operations[key].p95_ms.median - base
        message = on.operations[key].p95_ms.median - hash_only.operations[key].p95_ms.median
        is_control = key in RECORDING_FREE_OPS
        control = "  [control: records nothing — this is noise]" if is_control else ""
        notes.append(
            f"{key}: p95 {base:.3f} ms baseline, digest {digest:+.3f} ms,"
            f" message {message:+.3f} ms{control}"
        )
    return notes


def control_notes(arms: dict[str, ArmSummary]) -> list[str]:
    """Report the noise band measured on the operations that cannot carry a cost.

    :data:`RECORDING_FREE_OPS` never call the recorder, in any arm, so their
    ``off`` -> ``on`` delta is a *measurement of this harness* rather than of the
    design. Printing the band they span puts the honest scale beside every read
    delta: a read delta inside it is not distinguishable from the instrument,
    whatever :func:`noise_notes` concludes from a three-run min-max.

    It cuts both ways, and that is the point. In the run this story published, the
    controls span a few milliseconds while the 5 MB read delta is +5.97 ms — close
    enough that the run-to-run spread is the *weakest* argument available for it.
    The strong ones are elsewhere and belong in the reader's hands: the digest term
    reproduces across every team size measured, and it is what sha256 over five
    megabytes costs.
    """
    off = arms.get(ARM_OFF)
    on = arms.get(ARM_ON)
    if off is None or on is None:
        return []
    deltas = [
        (key, on.operations[key].p95_ms.median - off.operations[key].p95_ms.median)
        for key in RECORDING_FREE_OPS
        if key in off.operations and key in on.operations
    ]
    if not deltas:
        return []
    widest = max(abs(delta) for _key, delta in deltas)
    detail = ", ".join(f"{key} {delta:+.3f} ms" for key, delta in deltas)
    return [
        f"negative controls (record nothing in any arm, so off -> on is pure noise): {detail}",
        f"control band +/-{widest:.3f} ms — read the read-path deltas against this, not only"
        " against the run-to-run spread, which is a min-max over a handful of runs",
    ]


def noise_notes(arms: dict[str, ArmSummary]) -> list[str]:
    """Name every inter-arm delta that is inside the noise floor of a single arm."""
    off = arms.get(ARM_OFF)
    on = arms.get(ARM_ON)
    if off is None or on is None:
        return []
    notes: list[str] = []
    for key, summary in sorted(on.operations.items()):
        if key not in off.operations:
            continue
        base = off.operations[key].p95_ms
        floor = max(base.maximum - base.minimum, summary.p95_ms.maximum - summary.p95_ms.minimum)
        delta = summary.p95_ms.median - base.median
        if abs(delta) <= floor:
            notes.append(
                f"{key}: p95 delta {delta:+.3f} ms is inside the run-to-run spread"
                f" ({floor:.3f} ms) — no measurable difference"
            )
    return notes


##
## Environment
##


def _cpu_model() -> str:
    """Best-effort CPU name; the platform string when the probe is unavailable."""
    if sys.platform == "darwin":
        try:
            probe = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except OSError:
            return platform.processor() or platform.machine()
        if probe.returncode == 0 and probe.stdout.strip():
            return probe.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text("utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _package_commit() -> str:
    """Return the short commit of the package this harness lives in."""
    package_root = Path(__file__).resolve().parents[2]
    try:
        probe = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except OSError:
        return "unknown"
    return probe.stdout.strip() if probe.returncode == 0 else "unknown"


def describe_environment() -> Environment:
    """Stamp the machine. A benchmark number without its machine is not evidence."""
    return Environment(
        cpu=_cpu_model(),
        cores=os.cpu_count() or 0,
        system=platform.platform(),
        python=platform.python_version(),
        package_commit=_package_commit(),
        ripgrep=shutil.which("rg") is not None,
        markitdown=_markitdown_available(),
        git=shutil.which("git") is not None,
    )


##
## The benchmark
##


def run_benchmark(
    spec: RunSpec, arms: tuple[str, ...] = ALL_ARMS, runs: int = 3, base: Path | None = None
) -> BenchmarkResult:
    """Run every arm *runs* times, **interleaved**, and return the whole result.

    Interleaved rather than run in blocks so a machine that warms up or throttles
    during the session cannot be mistaken for a difference between arms.
    """
    collected: dict[str, list[ArmRun]] = {arm: [] for arm in arms}
    with tempfile.TemporaryDirectory(prefix="ws-bench-") as raw_base:
        root = base or Path(raw_base)
        for _ in range(runs):
            for arm in arms:
                collected[arm].append(run_arm(spec, arm, root))
    summaries = {arm: summarise_arm(arm, collected[arm]) for arm in arms}
    rules = evaluate_rules(spec, summaries)
    notes = decomposition_notes(summaries)
    notes += control_notes(summaries)
    notes += noise_notes(summaries)
    notes.append(f"mutation path (journal off -> on): {mutation_path_note(summaries)}")
    notes.append(
        "mailbox depth is sampled with queue.Queue.qsize(), which Python documents"
        " as unreliable under concurrency — read the depth series as approximate"
    )
    notes.append(
        "the arms do not sample depth at the same events: with no observation"
        " arriving, off and hash-only sample only at mutation turns while on also"
        " samples at every observation, so on's series is weighted towards the"
        " moments read traffic is being served — part of any off -> on gap is that"
        " difference in sampling rather than in queue behaviour"
    )
    return BenchmarkResult(
        environment=describe_environment(),
        command=" ".join([Path(sys.executable).name, "-m", __name__, *sys.argv[1:]]),
        agents=spec.agents,
        iterations=spec.iterations,
        warmup=spec.warmup,
        runs=runs,
        seed=spec.seed,
        buckets=[f"{name}:{size}" for name, size in spec.buckets],
        document_leg=_document_leg_note(spec),
        grep_engine="ripgrep" if shutil.which("rg") else "pure-Python fallback",
        arms=[summaries[arm] for arm in arms],
        rules=rules,
        verdict=_verdict(rules),
        notes=notes,
    )


def _document_leg_note(spec: RunSpec) -> str:
    """Say whether the document leg ran, and why not when it did not."""
    if not spec.document:
        return "skipped: not requested"
    if not _markitdown_available():
        return "skipped: markitdown is not installed"
    return "ran"


def _verdict(rules: list[RuleResult]) -> str:
    """Say which rules tripped, or that none did."""
    if not rules:
        return "INCONCLUSIVE — the read-path arms were not both run"
    tripped = [rule.rule for rule in rules if rule.tripped]
    if not tripped:
        return "NOT MATERIAL — no pre-registered rule tripped"
    return "MATERIAL — tripped: " + ", ".join(tripped)


##
## Reporting
##


def render(result: BenchmarkResult) -> str:
    """Render the whole result as text, thresholds beside the measurements."""
    lines = _render_header(result)
    for arm in result.arms:
        lines += _render_arm(arm)
    lines += ["", "Pre-registered thresholds (fixed before the run)", "-" * 78]
    for rule in result.rules:
        lines.append(f"  [{'TRIPPED' if rule.tripped else '  ok   '}] {rule.rule}")
        lines.append(f"            threshold: {rule.threshold}")
        lines.append(f"            measured:  {rule.measured}")
    lines += ["", "Notes", "-" * 78]
    lines += [f"  - {note}" for note in result.notes]
    lines += ["", f"VERDICT: {result.verdict}", ""]
    return "\n".join(lines)


def _render_header(result: BenchmarkResult) -> list[str]:
    """Render the environment stamp and the run parameters."""
    env = result.environment
    return [
        "=" * 78,
        "Workspace observation cost — read path and mutation path",
        "=" * 78,
        f"  command       {result.command}",
        f"  cpu           {env.cpu} ({env.cores} cores)",
        f"  system        {env.system}",
        f"  python        {env.python}",
        f"  package       {env.package_commit}",
        f"  agents        {result.agents}",
        f"  iterations    {result.iterations} measured, {result.warmup} warm-up, discarded",
        f"  runs per arm  {result.runs}, arms interleaved, fresh system and tree each time",
        f"  seed          {result.seed}",
        f"  buckets       {', '.join(result.buckets)}",
        f"  document leg  {result.document_leg}",
        f"  grep engine   {result.grep_engine}",
        "",
    ]


def _render_arm(arm: ArmSummary) -> list[str]:
    """Render one arm's per-operation table and its mailbox depth."""
    lines = [
        "-" * 78,
        f"arm {arm.arm}  (journal {'on' if arm.journal_enabled else 'off'},"
        f" {arm.runs} runs, {arm.read_observations} read observations,"
        f" {arm.mutation_observations} writer refreshes, {arm.refusals} refusals)",
        "-" * 78,
        f"  {'operation':<14}{'n':>7}{'p50 ms':>11}{'p95 ms':>11}"
        f"{'p95 min':>11}{'p95 max':>11}{'max ms':>11}",
    ]
    for key in sorted(arm.operations):
        summary = arm.operations[key]
        lines.append(
            f"  {key:<14}{summary.samples:>7}{summary.p50_ms.median:>11.3f}"
            f"{summary.p95_ms.median:>11.3f}{summary.p95_ms.minimum:>11.3f}"
            f"{summary.p95_ms.maximum:>11.3f}{summary.max_ms.median:>11.3f}"
        )
    lines.append(
        f"  mailbox depth (approx): p50 {arm.depth_p50.median:.1f},"
        f" p95 {arm.depth_p95.median:.1f}, max {arm.depth_max.maximum:.0f}"
        f" over {arm.depth_samples} samples"
    )
    if arm.journal_enabled:
        lines.append(
            f"  journal: {arm.total_commits} commits,"
            f" {arm.out_of_band_commits} authored out-of-band"
        )
    if arm.errors:
        lines += [f"  ! {error}" for error in arm.errors]
    return lines


##
## CLI
##


def build_parser() -> argparse.ArgumentParser:
    """Build the command line — every knob the report has to name."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--read-limit", type=int, default=DEFAULT_READ_LIMIT)
    parser.add_argument("--arms", default=",".join(ALL_ARMS))
    parser.add_argument("--no-document", action="store_true")
    parser.add_argument("--quick", action="store_true", help="tiny sizes, for the smoke test")
    parser.add_argument(
        "--sweep",
        default="",
        help="comma-separated agent counts to sweep, e.g. 2,5,10,20",
    )
    return parser


def spec_from_args(args: argparse.Namespace, agents: int) -> RunSpec:
    """Turn parsed arguments into one :class:`RunSpec` at *agents* team size."""
    return RunSpec(
        agents=agents,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        read_limit=args.read_limit,
        document=not args.no_document and not args.quick,
        buckets=QUICK_BUCKETS if args.quick else DEFAULT_BUCKETS,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark (or the sweep) and print the report."""
    args = build_parser().parse_args(argv)
    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    counts = [int(value) for value in args.sweep.split(",") if value.strip()] or [args.agents]
    for agents in counts:
        result = run_benchmark(spec_from_args(args, agents), arms=arms, runs=args.runs)
        print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
