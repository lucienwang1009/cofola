from __future__ import annotations

import pytest

from scripts.benchmarks.analyze import (
    aggregate_rows,
    paired_runtime_summary,
    par2_summary,
    validate_repetitions,
)


def _row(
    case_id: str,
    backend: str,
    elapsed: float,
    *,
    status: str = "solved",
    family: str = "family_a",
) -> dict[str, str]:
    return {
        "suite": "growing",
        "case_id": case_id,
        "backend": backend,
        "status": status,
        "result": "1" if status == "solved" else "",
        "expected": "1",
        "elapsed_sec": str(elapsed),
        "error_type": "" if status == "solved" else "TimeoutError",
        "error_message": "",
        "tags": f"growing;family={family};domain=5",
        "source": "unit test",
    }


def test_aggregate_rows_uses_median_and_requires_every_repetition_to_solve() -> None:
    rows = [
        _row("a", "wfomc", 1),
        _row("a", "wfomc", 3),
        _row("b", "wfomc", 2),
        _row("b", "wfomc", 100, status="timeout"),
    ]

    aggregated = {
        row["case_id"]: row
        for row in aggregate_rows(rows)
    }

    assert aggregated["a"]["status"] == "solved"
    assert aggregated["a"]["elapsed_sec"] == "2.000000"
    assert aggregated["a"]["repetitions"] == 2
    assert aggregated["a"]["solved_repetitions"] == 2
    assert aggregated["b"]["status"] == "timeout"
    assert aggregated["b"]["solved_repetitions"] == 1


def test_validate_repetitions_reports_incomplete_runs() -> None:
    aggregated = aggregate_rows([_row("a", "wfomc", 1)])

    with pytest.raises(ValueError, match="Expected 3 repetitions"):
        validate_repetitions(aggregated, 3)


def test_par2_penalizes_every_non_solved_observation_at_twice_timeout() -> None:
    rows = [
        _row("a", "wfomc", 1),
        _row("a", "wfomc", 3),
        _row("b", "wfomc", 100, status="timeout"),
        _row("b", "wfomc", 0, status="timeout"),
    ]

    summary = par2_summary(rows, timeout=100)
    suite = next(row for row in summary if row["scope"] == "suite")
    family = next(row for row in summary if row["scope"] == "growing_family")

    assert suite["cases"] == 2
    assert suite["observations"] == 4
    assert suite["solved_observations"] == 2
    assert suite["penalized_observations"] == 2
    assert suite["par2_sec"] == "101.000000"
    assert family["par2_sec"] == "101.000000"


def test_paired_runtime_uses_per_backend_medians_and_reports_wins() -> None:
    raw_rows = [
        _row("a", "wfomc", 1),
        _row("a", "wfomc", 3),
        _row("a", "asp", 3),
        _row("a", "asp", 5),
        _row("b", "wfomc", 3),
        _row("b", "wfomc", 5),
        _row("b", "asp", 1),
        _row("b", "asp", 3),
    ]

    summary = paired_runtime_summary(
        aggregate_rows(raw_rows),
        reference="wfomc",
        tie_ratio=1.05,
    )
    suite = next(row for row in summary if row["scope"] == "suite")

    assert suite["paired_solved_cases"] == 2
    assert suite["geomean_speedup"] == "1.000000"
    assert suite["median_speedup"] == "1.250000"
    assert suite["reference_wins"] == 1
    assert suite["ties"] == 0
    assert suite["backend_wins"] == 1
