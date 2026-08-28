from __future__ import annotations

import json
from collections import Counter

from cofola.parser.parser import parse
from cofola.solver import parse_and_solve
from scripts.benchmarks.cases import (
    growing_domain_cases,
    load_real_world_cases,
    load_saved_cases,
    old_synthetic_cases,
    save_cases,
    synthetic_cases,
)
from scripts.benchmarks.synthetic import (
    DEFAULT_SYNTHETIC_SEED,
    SYNTHETIC_DOMAINS,
    SYNTHETIC_STRATA,
    SYNTHETIC_VARIANTS_PER_CELL,
    materialize_synthetic_suite,
    stratified_synthetic_cases,
)


def test_growing_cases_include_cola_and_extended_families() -> None:
    cases = {case.case_id: case for case in growing_domain_cases()}

    assert len(cases) == 160
    assert cases["mathcounts_005"].expected == 4
    assert cases["mathcounts_100"].expected == 2235025
    assert cases["fourletter_005"].expected == 18
    assert cases["fourletter_100"].expected == 28518
    assert "mathcounts" in cases["mathcounts_005"].tags
    assert "fourletter" in cases["fourletter_005"].tags
    assert "domain=5" in cases["mathcounts_005"].tags
    assert "family=mathcounts" in cases["mathcounts_005"].tags
    assert cases["nested_choice_005"].expected == 20
    assert cases["nested_choice_100"].expected == 15_684_900
    assert cases["selected_pairs_005"].expected == 15
    assert cases["selected_pairs_100"].expected == 11_763_675
    assert cases["selected_sequence_005"].expected == 48
    assert cases["selected_sequence_100"].expected == 7_300_608
    for family in ("nested_choice", "selected_pairs", "selected_sequence"):
        assert "outside_cola_fragment" in cases[f"{family}_005"].tags
        assert f"family={family}" in cases[f"{family}_005"].tags


def test_extended_growing_cases_parse() -> None:
    cases = growing_domain_cases(min_domain=5, max_domain=5)

    for case in cases:
        if "outside_cola_fragment" in case.tags:
            parse(case.program)


def test_synthetic_cases_load_materialized_benchmark_manifest() -> None:
    cases = synthetic_cases()

    assert len(cases) == 384
    assert all(case.expected is not None for case in cases)
    assert all("domain=" in ";".join(case.tags) for case in cases)
    assert {case.source for case in cases} == {"problems/benchmarks/synthetic/manifest.json"}


def test_stratified_synthetic_cases_are_balanced_and_nontrivial() -> None:
    cases = stratified_synthetic_cases()

    assert len(cases) == (
        len(SYNTHETIC_STRATA)
        * len(SYNTHETIC_DOMAINS)
        * SYNTHETIC_VARIANTS_PER_CELL
    ) == 384
    assert len({case.case_id for case in cases}) == len(cases)
    assert len({case.program for case in cases}) == len(cases)
    assert all(case.expected is not None and case.expected > 1 for case in cases)

    strata = Counter(tag for case in cases for tag in case.tags if tag.startswith("stratum="))
    difficulties = Counter(
        tag for case in cases for tag in case.tags if tag.startswith("difficulty=")
    )
    domains = Counter(tag for case in cases for tag in case.tags if tag.startswith("domain="))
    dependent = sum("dependent_configuration" in case.tags for case in cases)

    assert set(strata.values()) == {24}
    assert difficulties == {
        "difficulty=small": 128,
        "difficulty=medium": 128,
        "difficulty=large": 128,
    }
    assert domains == {"domain=8": 128, "domain=12": 128, "domain=16": 128}
    assert dependent == 192

    required_features = {
        "set",
        "bag",
        "tuple",
        "sequence",
        "partition",
        "composition",
    }
    observed_features = {tag for case in cases for tag in case.tags}
    assert required_features <= observed_features
    assert all(
        any(tag.startswith("constraintcount=") and tag != "constraintcount=0" for tag in case.tags)
        for case in cases
    )


def test_stratified_synthetic_generation_is_seeded() -> None:
    first = stratified_synthetic_cases(seed=DEFAULT_SYNTHETIC_SEED)
    repeated = stratified_synthetic_cases(seed=DEFAULT_SYNTHETIC_SEED)
    changed = stratified_synthetic_cases(seed=DEFAULT_SYNTHETIC_SEED + 1)

    assert first == repeated
    assert [case.program for case in first] != [case.program for case in changed]


def test_stratified_synthetic_cases_parse() -> None:
    for case in stratified_synthetic_cases():
        parse(case.program)


def test_small_stratified_synthetic_oracles_match_wfomc() -> None:
    cases = {
        next(tag.split("=", maxsplit=1)[1] for tag in case.tags if tag.startswith("stratum=")): case
        for case in stratified_synthetic_cases()
        if "difficulty=small" in case.tags and "variant=0" in case.tags
    }

    assert set(cases) == set(SYNTHETIC_STRATA)
    for stratum in SYNTHETIC_STRATA:
        case = cases[stratum]
        assert parse_and_solve(case.program) == case.expected, stratum


def test_materialized_synthetic_suite_replaces_only_manifest_cases(tmp_path) -> None:
    old_cases = old_synthetic_cases(seed=0)[:2]
    save_cases(old_cases, tmp_path, suite_subdirectories=False)
    old_paths = {tmp_path / f"{case.case_id}.cfl" for case in old_cases}
    unrelated = tmp_path / "keep-me.txt"
    unrelated.write_text("not generated by the synthetic manifest")

    new_cases = stratified_synthetic_cases()[:3]
    materialize_synthetic_suite(new_cases, tmp_path)

    assert all(not path.exists() for path in old_paths)
    assert unrelated.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["num_cases"] == 3
    assert {record["program_path"] for record in manifest["cases"]} == {
        f"{case.case_id}.cfl" for case in new_cases
    }
    assert load_saved_cases(tmp_path) == new_cases


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
