"""One cheap, timing-free test that keeps the read-cost benchmark from rotting.

The benchmark itself is ``tests/benchmarks/workspace_read_cost.py``, which pytest
never collects: ``python_files`` matches ``test_*.py`` only, and CI runs
``pytest tests/`` with no marker filter, so a marked benchmark would run on every
push instead. This module is the harness's one caller.

**It asserts shape, never time.** A timing assertion tuned to pass on a shared
runner has stopped being able to fail, which is worse than no test: it reads as
protection and provides none. What it checks is that the harness still starts a
real team, still drives every leg of the mix, and still comes back with a
well-formed result — the failures that would otherwise sit undiscovered until
somebody needed the number.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from pathlib import Path

import pykka
import pytest

from tests.benchmarks.workspace_read_cost import (
    ARM_ON,
    QUICK_BUCKETS,
    BenchmarkResult,
    RunSpec,
    expected_operations,
    minimal_pdf,
    percentile_ms,
    render,
    run_benchmark,
)

SMOKE_SPEC = RunSpec(
    agents=2,
    iterations=3,
    warmup=1,
    seed=7,
    read_limit=50,
    document=False,
    buckets=QUICK_BUCKETS,
    grep_files=3,
)
"""Two agents, three measured turns, one arm — roughly a second of the suite."""


@pytest.fixture(autouse=True)
def _stop_leaked_actors() -> Generator[None, None, None]:
    """Never let a failed run leave a live actor system behind for the next test."""
    yield
    pykka.ActorRegistry.stop_all()


@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory: pytest.TempPathFactory) -> BenchmarkResult:
    """Run the harness once, tiny, and share the result across this module.

    Module-scoped deliberately: the assertions below inspect one result from
    several angles, and re-running a two-agent team per assertion would spend the
    package suite's budget five times over for nothing. ``run_arm`` sets and
    restores ``AKGENTIC_WORKSPACES_ROOT`` itself, so the tree stays inside the
    temporary directory handed here.
    """
    base: Path = tmp_path_factory.mktemp("read-cost-smoke")
    return run_benchmark(SMOKE_SPEC, arms=(ARM_ON,), runs=1, base=base)


def test_the_harness_completes_and_reports_every_operation(
    smoke_result: BenchmarkResult,
) -> None:
    """Every leg of the mix ran, and each came back with the samples it was asked for."""
    assert len(smoke_result.arms) == 1
    arm = smoke_result.arms[0]
    assert arm.arm == ARM_ON
    assert arm.runs == 1

    expected = expected_operations(SMOKE_SPEC)
    assert set(arm.operations) == set(expected)

    wanted = SMOKE_SPEC.agents * SMOKE_SPEC.iterations
    for key in expected:
        assert arm.operations[key].samples == wanted, key


def test_every_latency_is_finite_and_non_negative(smoke_result: BenchmarkResult) -> None:
    """Shape only — no assertion about how long anything took."""
    arm = smoke_result.arms[0]
    for key, summary in arm.operations.items():
        for spread in (summary.p50_ms, summary.p95_ms, summary.max_ms):
            for value in (spread.median, spread.minimum, spread.maximum):
                assert math.isfinite(value), key
                assert value >= 0.0, key
            assert spread.minimum <= spread.median <= spread.maximum, key


def test_the_shipped_arm_recorded_and_kept_the_journal_off(
    smoke_result: BenchmarkResult,
) -> None:
    """The ``on`` arm exercises the real recorder, with no git in the numbers."""
    arm = smoke_result.arms[0]
    assert arm.read_observations > 0
    assert not arm.journal_enabled
    assert arm.refusals == 0, arm.errors
    assert arm.errors == []


def test_the_result_carries_its_thresholds_and_its_environment(
    smoke_result: BenchmarkResult,
) -> None:
    """A number without its machine, or without the rule it was judged by, is not evidence."""
    assert smoke_result.environment.cores > 0
    assert smoke_result.environment.python
    assert smoke_result.agents == SMOKE_SPEC.agents
    assert smoke_result.verdict
    # One arm cannot decide the read-path rules — the harness must say so rather
    # than invent a verdict from a comparison it never made.
    assert smoke_result.rules == []
    assert "INCONCLUSIVE" in smoke_result.verdict
    assert smoke_result.notes


def test_the_report_renders(smoke_result: BenchmarkResult) -> None:
    """The report is what a reader actually sees; it must survive every arm shape."""
    text = render(smoke_result)
    assert "VERDICT" in text
    assert ARM_ON in text
    assert "mailbox depth (approx)" in text


@pytest.mark.parametrize(
    ("samples", "fraction", "expected"),
    [
        ([], 0.95, 0.0),
        ([5.0], 0.95, 5.0),
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.0),
        ([1.0, 2.0, 3.0, 4.0], 0.95, 4.0),
    ],
)
def test_nearest_rank_percentile(samples: list[float], fraction: float, expected: float) -> None:
    """Nearest rank on the sorted samples, so a published number reproduces exactly."""
    assert percentile_ms(samples, fraction) == expected


def test_the_generated_pdf_is_long_enough_to_avoid_the_llm_pass() -> None:
    """The document corpus must never take ``DocumentReader``'s second pass.

    That pass constructs an OpenAI client, and a benchmark that reaches the
    network is measuring the network.
    """
    pdf = minimal_pdf("x" * 200)
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
