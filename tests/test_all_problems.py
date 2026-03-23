"""Test all encodable problems from problems/all.json.

By default only "benchmark" problems are run (20 representative problems
covering all major feature areas).  Set the environment variable
COFOLA_ALL_TESTS=1 to run the full suite.
"""
from __future__ import annotations

import json
import os
import pytest
from pathlib import Path

from cofola.solver import parse_and_solve


# Load problems from JSON
PROBLEMS_FILE = Path(__file__).parent.parent / "problems" / "all.json"

# Run the full suite only when explicitly requested
_RUN_ALL = os.environ.get("COFOLA_ALL_TESTS", "0").strip() not in ("", "0", "false", "no")


def load_problems():
    """Load all problems from the JSON file."""
    with open(PROBLEMS_FILE, "r") as f:
        return json.load(f)


def get_encodable_problems():
    """Return problems to test.

    Default (COFOLA_ALL_TESTS unset): benchmark problems only.
    Full run (COFOLA_ALL_TESTS=1): all encodable problems
    (excludes timeout / unencodeable / not-combinatorics).
    """
    all_problems = load_problems()
    skip_tags = {"timeout", "unencodeable", "not combinatorics"}
    result = []

    for problem_id, problem_data in all_problems.items():
        tags = set(problem_data.get("tags", []))

        if tags & skip_tags:
            continue

        if not _RUN_ALL and "benchmark" not in tags:
            continue

        result.append((problem_id, problem_data))

    return result


@pytest.fixture(scope="module")
def all_problems():
    """Load all problems once for the test module."""
    return load_problems()


class TestAllEncodableProblems:
    """Test suite for all encodable problems from all.json."""

    @pytest.mark.parametrize("problem_id,problem_data", get_encodable_problems())
    def test_problem(self, problem_id, problem_data):
        """Test a single problem from the dataset."""
        program = problem_data["program"]
        expected_answer = int(problem_data["answer"])

        result = parse_and_solve(program)

        assert result == expected_answer, (
            f"Problem {problem_id}: expected {expected_answer}, got {result}\n"
            f"Problem: {problem_data.get('problem', 'N/A')}\n"
            f"Program: {program}"
        )


def test_problems_file_exists():
    """Verify the problems file exists."""
    assert PROBLEMS_FILE.exists(), f"Problems file not found at {PROBLEMS_FILE}"


def test_can_load_problems(all_problems):
    """Verify we can load and parse the problems JSON."""
    assert isinstance(all_problems, dict)
    assert len(all_problems) > 0

    # Check structure of first problem
    first_problem = next(iter(all_problems.values()))
    assert "program" in first_problem
    assert "answer" in first_problem
    assert "tags" in first_problem


if __name__ == "__main__":
    # Print summary of encodable problems
    encodable = get_encodable_problems()
    all_p = load_problems()
    benchmark = [p for p in all_p.values() if "benchmark" in p.get("tags", [])]
    print(f"\nMode: {'FULL' if _RUN_ALL else 'BENCHMARK'}")
    print(f"Running: {len(encodable)} problems")
    print(f"Total benchmark problems: {len(benchmark)}")
    print(f"Total problems in dataset: {len(all_p)}")
