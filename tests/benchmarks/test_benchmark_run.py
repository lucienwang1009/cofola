from __future__ import annotations

import argparse

from scripts.benchmarks.cases import BenchmarkCase
from scripts.benchmarks.run import (
    STATUS_SOLVED,
    STATUS_TIMEOUT,
    STATUS_UNSOLVED,
    STATUS_WRONG,
    _backend_worker_args,
    _growing_domain,
    _growing_skip_key,
    _row_from_payload,
    _skipped_timeout_row,
)


def _case(expected: int | None = 3) -> BenchmarkCase:
    return BenchmarkCase(
        suite="unit",
        case_id="case",
        program="A = set(a, b, c)\nB = choose(A)\n|B| == 2",
        expected=expected,
    )


def test_propositionalwfomc_uses_wfomc_backend_with_propositional_algo() -> None:
    backend, algo, linear_order_encoding = _backend_worker_args(
        "propositionalwfomc",
        argparse.Namespace(algo="fastv2", linear_order_encoding=None),
    )

    assert backend == "wfomc"
    assert algo == "propositional"
    assert linear_order_encoding is None


def test_misspelled_propostionalwfomc_alias_is_accepted() -> None:
    backend, algo, _ = _backend_worker_args(
        "propostionalwfomc",
        argparse.Namespace(algo="fastv2", linear_order_encoding=None),
    )

    assert backend == "wfomc"
    assert algo == "propositional"


def test_asp_wrong_answer_is_wrong() -> None:
    row = _row_from_payload(
        _case(expected=3),
        backend="asp",
        payload={"ok": True, "result": 2},
        elapsed=0.1,
    )

    assert row["status"] == STATUS_WRONG
    assert row["result"] == 2
    assert row["error_type"] == "WrongAnswer"


def test_external_baseline_wrong_answer_is_unsolved() -> None:
    row = _row_from_payload(
        _case(expected=3),
        backend="essence",
        payload={"ok": True, "result": 2},
        elapsed=0.1,
    )

    assert row["status"] == STATUS_UNSOLVED
    assert row["result"] == 2
    assert row["error_type"] == "WrongAnswer"


def test_external_baseline_error_is_unsolved() -> None:
    row = _row_from_payload(
        _case(expected=3),
        backend="essence",
        payload={
            "ok": False,
            "error_type": "CoSoEncodingError",
            "error_message": "unsupported",
        },
        elapsed=0.1,
    )

    assert row["status"] == STATUS_UNSOLVED
    assert row["result"] == ""
    assert row["error_type"] == "CoSoEncodingError"


def test_wfomc_correct_answer_still_solved() -> None:
    row = _row_from_payload(
        _case(expected=3),
        backend="wfomc",
        payload={"ok": True, "result": 3},
        elapsed=0.1,
    )

    assert row["status"] == STATUS_SOLVED


def test_growing_timeout_skip_row_records_timeout() -> None:
    case = BenchmarkCase(
        suite="growing",
        case_id="mathcounts_010",
        program="A = set(a0...10)\nB = choose(A, 2)",
        expected=45,
        tags=("mathcounts", "family=mathcounts", "domain=10"),
    )

    row = _skipped_timeout_row(case, backend="wfomc", timeout=100, cutoff=5)

    assert _growing_skip_key(case, "wfomc") == ("wfomc", "mathcounts")
    assert _growing_domain(case) == 10
    assert row["status"] == STATUS_TIMEOUT
    assert row["elapsed_sec"] == "0.000000"
    assert row["error_type"] == "SkippedAfterTimeout"
