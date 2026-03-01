"""Test all encodable problems from problems/all.json."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from wfomc import Algo

from cofola.parser.parser import parse
from cofola.solver import solve


# Load problems from JSON
PROBLEMS_FILE = Path(__file__).parent.parent / "problems" / "all.json"


def load_problems():
    """Load all problems from the JSON file."""
    with open(PROBLEMS_FILE, "r") as f:
        return json.load(f)


def get_encodable_problems():
    """Filter problems to only include encodable ones (no timeout/unencodeable tags)."""
    all_problems = load_problems()
    encodable = []

    unchecked_tags = {"timeout", "unencodeable"}

    for problem_id, problem_data in all_problems.items():
        tags = set(problem_data.get("tags", []))

        # Skip problems with timeout or unencodeable tags
        if tags & unchecked_tags:
            continue

        encodable.append((problem_id, problem_data))

    return encodable


@pytest.fixture(scope="module")
def all_problems():
    """Load all problems once for the test module."""
    return load_problems()


class TestAllEncodableProblems:
    """Test suite for all encodable problems from all.json."""

    @pytest.mark.parametrize("problem_id,problem_data", get_encodable_problems())
    def test_problem(self, problem_id, problem_data):
        """Test a single problem from the dataset."""
        # Parse the problem program
        program = problem_data["program"]
        expected_answer = int(problem_data["answer"])

        # Parse and solve
        cofola_problem = parse(program)
        result = solve(
            cofola_problem,
            Algo.FASTv2,
            use_partition_constraint=True,
            lifted=False
        )

        # Verify the answer
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
    print(f"\nTotal encodable problems: {len(encodable)}")
    print(f"Total problems in dataset: {len(load_problems())}")
    print(f"Skipped (timeout/unencodeable): {len(load_problems()) - len(encodable)}")