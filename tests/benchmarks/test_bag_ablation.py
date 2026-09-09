from __future__ import annotations

import pytest

from cofola.solver import parse_and_solve
from scripts.benchmarks.bag_ablation import regression_reason
from scripts.benchmarks.bag_lifting_cases import make_case
from scripts.benchmarks.analyze_bag_ablation import aggregate


@pytest.mark.parametrize("shape,expected", [
    ("choose", 3), ("chain", 4), ("hidden", 10),
    ("intersection", 4), ("named", 1), ("fallback", 19),
])
def test_controlled_oracle_against_small_hand_counts(shape, expected):
    case = make_case(shape, 2, 2)
    assert case.expected == expected
    assert parse_and_solve(case.program, lifted_bags=False) == expected
    assert parse_and_solve(case.program, lifted_bags=True) == expected


@pytest.mark.parametrize("before,after,reason", [
    (1.0, 1.3, "slowdown"), (0.1, 0.15, ""), (2.0, 2.1, ""),
])
def test_recheck_requires_both_relative_and_absolute_slowdown(before, after, reason):
    off = dict(status="solved", result=3, elapsed_sec=before)
    on = dict(status="solved", result=3, elapsed_sec=after)
    assert regression_reason(off, on) == reason


def test_recheck_detects_lost_solution_and_answer_mismatch():
    off = dict(status="solved", result=3, elapsed_sec=1)
    assert regression_reason(off, dict(status="timeout")) == "lost-solution"
    assert regression_reason(off, dict(status="solved", result=2)) == "answer-mismatch"


def test_ablation_aggregation_uses_three_rechecks_not_initial_outlier():
    rows = []
    for flag, times in (("False", [1, 1, 1, 1]), ("True", [5, 1.05, 1.1, 1.15])):
        for repetition, elapsed in enumerate(times):
            rows.append(dict(suite="unit", case_id="case", lifted_bags=flag,
                             repetition=str(repetition), status="solved", result="3",
                             elapsed_sec=str(elapsed), error_type="", tags="",
                             recheck_reason="slowdown" if repetition else ""))
    pair, = aggregate(rows)
    assert pair["on_sec"] == 1.1
    assert pair["confirmed_regression"] is False

    with pytest.raises(ValueError, match="Incomplete pair"):
        aggregate(rows[:-1])
