from __future__ import annotations

import json

from cofola.parser.parser import parse
from scripts.benchmarks.cases import (
    growing_domain_cases,
    load_real_world_cases,
    load_saved_cases,
    old_synthetic_cases,
    save_cases,
    synthetic_cases,
)


def test_growing_cases_include_mathcounts_and_fourletter_families() -> None:
    cases = {case.case_id: case for case in growing_domain_cases()}

    assert len(cases) == 100
    assert cases["mathcounts_005"].expected == 4
    assert cases["mathcounts_100"].expected == 2235025
    assert cases["fourletter_005"].expected == 18
    assert cases["fourletter_100"].expected == 28518
    assert "mathcounts" in cases["mathcounts_005"].tags
    assert "fourletter" in cases["fourletter_005"].tags
    assert "domain=5" in cases["mathcounts_005"].tags
    assert "family=mathcounts" in cases["mathcounts_005"].tags


def test_synthetic_cases_load_materialized_benchmark_manifest() -> None:
    cases = synthetic_cases()

    assert len(cases) == 82
    assert all(case.expected is not None for case in cases)
    assert all("entitycount=" in ";".join(case.tags) for case in cases)
    assert {case.source for case in cases} == {"problems/benchmarks/synthetic/manifest.json"}


def test_old_synthetic_cases_cover_extended_coso_supported_families() -> None:
    cases = old_synthetic_cases(seed=0)
    families = {case.case_id.split("_", maxsplit=1)[0] for case in cases}

    assert len(cases) == 72
    assert families == {"ss", "sr", "ms", "pm", "bm", "sq", "pt", "cp"}


def test_real_world_benchmark_cases_skip_unencodeable_programs() -> None:
    cases = load_real_world_cases()

    assert cases
    assert all("unencodeable" not in case.tags for case in cases)


def test_real_world_syntax_typo_cases_are_parseable() -> None:
    problems = json.loads(open("problems/real/corpus.json").read())

    for problem_id in ("9", "197", "228", "317"):
        parse(problems[problem_id]["program"])


def test_saved_benchmark_manifest_round_trips_cases(tmp_path) -> None:
    cases = old_synthetic_cases(seed=0)[:3]

    save_cases(cases, tmp_path)
    loaded = load_saved_cases(tmp_path / "manifest.json")

    assert loaded == cases
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format"] == "cofola-benchmark-manifest-v1"
    assert manifest["num_cases"] == 3
    for record in manifest["cases"]:
        program_path = tmp_path / record["program_path"]
        assert program_path.exists()
        assert record["program_sha256"]
    assert (tmp_path / "manifest.csv").exists()
