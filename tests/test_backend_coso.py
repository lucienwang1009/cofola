"""CoSo backend encoding and solver-boundary tests."""
from __future__ import annotations

import pytest

from cofola.backend.coso.backend import (
    COSO_GLOBAL_PASSES,
    COSO_LOCAL_PASSES,
    CoSoBackend,
)
from cofola.backend.coso.encoder import CoSoEncodingError, encode
from cofola.frontend.problem import Problem
from cofola.parser.parser import parse
from cofola.planing.analysis.entities import AnalysisResult
from cofola.planing.pipeline import PlaningPipeline
from cofola.solver import parse_and_solve


COSO_CORPUS_PROBLEM_IDS = (
    "0",
    "3",
    "5",
    "14",
    "18",
    "20",
    "25",
    "27",
    "32",
    "35",
    "36",
    "48",
    "49",
    "50",
    "59",
    "60",
    "63",
    "66",
    "69",
    "71",
    "75",
    "84",
    "85",
    "86",
    "91",
    "94",
    "98",
    "103",
    "105",
    "114",
    "116",
    "125",
    "126",
    "127",
    "128",
    "131",
    "134",
    "139",
    "141",
    "147",
    "157",
    "158",
    "162",
    "163",
    "166",
    "171",
    "176",
    "178",
    "182",
    "184",
    "185",
    "199",
    "204",
    "209",
    "211",
    "212",
    "223",
    "225",
    "230",
    "232",
    "233",
    "237",
    "238",
    "241",
    "245",
    "246",
    "256",
    "258",
    "261",
    "266",
    "271",
    "273",
    "274",
    "280",
    "287",
    "288",
    "290",
    "298",
    "375",
)


def _single_component(source: str) -> tuple[Problem, AnalysisResult]:
    schedule = PlaningPipeline(CoSoBackend().planning_profile()).process(parse(source))
    assert len(schedule.branches) == 1
    branch = schedule.branches[0]
    assert len(branch.components) == 1
    return branch.components[0]


def _lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}


def test_coso_backend_declares_planning_profile() -> None:
    profile = CoSoBackend().planning_profile()

    assert profile.global_passes == COSO_GLOBAL_PASSES
    assert profile.local_passes == COSO_LOCAL_PASSES


def test_encode_set_choose_as_coso_subset() -> None:
    problem, analysis = _single_component(
        """
A = set(a, b, c)
B = choose(A)
|B| == 2
"""
    )

    program = encode(problem, analysis)

    assert _lines(program.cola) == {
        "universe u={a,b,c};",
        "cfg in {u};",
        "#cfg=2;",
    }


def test_encode_choose_tuple_as_coso_permutation() -> None:
    problem, analysis = _single_component(
        """
A = set(a, b, c)
row = choose_tuple(A, 2)
"""
    )

    program = encode(problem, analysis)

    assert _lines(program.cola) == {
        "universe u={a,b,c};",
        "cfg in [u];",
        "#cfg=2;",
    }


def test_encode_target_superset_constraint_as_counting_constraint() -> None:
    problem, analysis = _single_component(
        """
A = set(a, b, c)
S = set(a, b)
B = choose(A)
|B| == 2
S subset B
"""
    )

    program = encode(problem, analysis)

    assert "property p_1={a,b};" in _lines(program.cola)
    assert "#cfg&p_1=2;" in _lines(program.cola)


def test_parse_and_solve_routes_to_coso_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_run_coso_program(cola: str, *, debug: bool = False) -> int:
        seen.append(cola)
        return 3

    import cofola.backend.coso.backend as backend_module

    monkeypatch.setattr(backend_module, "run_coso_program", fake_run_coso_program)

    result = parse_and_solve(
        """
A = set(a, b, c)
B = choose(A)
|B| == 2
""",
        backend="coso",
    )

    assert result == 3
    assert seen == ["universe u={a,b,c};\ncfg in {u};\n#cfg=2;\n"]


def test_coso_rejects_multiple_target_configurations() -> None:
    problem, analysis = _single_component(
        """
A = set(a, b, c)
B = choose(A)
C = choose(A)
|B| == 1
|C| == 1
B disjoint C
"""
    )

    with pytest.raises(CoSoEncodingError, match="exactly one target configuration"):
        encode(problem, analysis)


def test_coso_rejects_sequence_patterns() -> None:
    problem, analysis = _single_component(
        """
A = set(a, b, c)
row = sequence(A)
a < b in row
"""
    )

    with pytest.raises(CoSoEncodingError, match="sequence"):
        encode(problem, analysis)


def test_encode_tuple_index_and_count_constraints() -> None:
    problem, analysis = _single_component(
        """
A = set(a, b, c)
row = choose_tuple(A, 2)
row[0] == a
row.count(b) > 0
"""
    )

    program = encode(problem, analysis)

    assert _lines(program.cola) == {
        "universe u={a,b,c};",
        "cfg in [u];",
        "#cfg=2;",
        "cfg[1]=a;",
        "#cfg&b>0;",
    }


def test_encode_target_intersection_size_constraint() -> None:
    problem, analysis = _single_component(
        """
defective = set(d0...3)
working = set(w0...9)
purchase = choose(defective + working, 5)
|purchase & defective| >= 2
"""
    )

    program = encode(problem, analysis)

    assert "#cfg&p_0>=2;" in _lines(program.cola)


def test_encode_indexed_composition_part_sizes() -> None:
    problem, analysis = _single_component(
        """
workers = set(w0...14)
groups = compose(workers, 3)
|groups[0]| == 7
|groups[1]| == 5
|groups[2]| == 2
"""
    )

    program = encode(problem, analysis)

    assert "#( #part=7 )=1;" in _lines(program.cola)
    assert "#( #part=5 )=1;" in _lines(program.cola)
    assert "#( #part=2 )=1;" in _lines(program.cola)
    assert program.count_divisor == 6


@pytest.mark.parametrize("problem_id", COSO_CORPUS_PROBLEM_IDS)
def test_coso_backend_on_verified_corpus_problem(problem_id: str) -> None:
    pytest.importorskip("coso.launcher")

    import json
    from pathlib import Path

    problems_path = Path(__file__).parent.parent / "problems" / "real" / "corpus.json"
    problem_data = json.loads(problems_path.read_text())[problem_id]

    result = parse_and_solve(problem_data["program"], backend="coso")

    assert result == int(problem_data["answer"]), (
        f"Problem {problem_id}: expected {problem_data['answer']}, got {result}"
    )
