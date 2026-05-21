from __future__ import annotations

import json

from scripts.benchmarks.cases import (
    growing_domain_cases,
    load_saved_cases,
    save_cases,
    synthetic_cases,
)


def test_growing_cases_include_mathcounts_and_fourletter_families() -> None:
    cases = {case.case_id: case for case in growing_domain_cases()}

    assert len(cases) == 55
    assert cases["mathcounts_00"].expected == 100
    assert cases["mathcounts_10"].expected == 9400
    assert cases["fourletter_00"].expected == 36
    assert cases["fourletter_10"].expected == 546
    assert "mathcounts" in cases["mathcounts_00"].tags
    assert "fourletter" in cases["fourletter_00"].tags


def test_synthetic_cases_cover_extended_coso_supported_families() -> None:
    cases = synthetic_cases(seed=0)
    families = {case.case_id.split("_", maxsplit=1)[0] for case in cases}

    assert len(cases) == 72
    assert families == {"ss", "sr", "ms", "pm", "bm", "sq", "pt", "cp"}


def test_saved_benchmark_manifest_round_trips_cases(tmp_path) -> None:
    cases = synthetic_cases(seed=0)[:3]

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
