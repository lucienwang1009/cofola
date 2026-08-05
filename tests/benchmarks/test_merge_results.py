from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.benchmarks.cases import BenchmarkCase, save_cases
from scripts.benchmarks.merge_results import merge_rows, write_merged_run


BACKENDS = ["wfomc", "asp"]


def _row(case: BenchmarkCase, backend: str, result: int) -> dict[str, str]:
    expected = "" if case.expected is None else str(case.expected)
    return {
        "suite": case.suite,
        "case_id": case.case_id,
        "backend": backend,
        "status": "solved" if str(result) == expected else "wrong",
        "result": str(result),
        "expected": expected,
        "elapsed_sec": "0.100000",
        "error_type": "",
        "error_message": "",
        "tags": ";".join(case.tags),
        "source": case.source,
    }


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_merge_rows_validates_matrix_and_applies_replacement(tmp_path: Path) -> None:
    cases = [
        BenchmarkCase("real", "1", "A = set(a)", 1),
        BenchmarkCase("synthetic", "s1", "A = set(a, b)", 2),
    ]
    source = tmp_path / "source.csv"
    replacement = tmp_path / "replacement.csv"
    _write_rows(
        source,
        [
            _row(case, backend, case.expected or 0)
            for case in cases
            for backend in BACKENDS
        ],
    )
    _write_rows(replacement, [_row(cases[0], "wfomc", 1)])

    rows = merge_rows(
        cases=cases,
        backends=BACKENDS,
        sources=[(source, "all", "wfomc"), (source, "all", "asp")],
        replacements=[(replacement, "real", "wfomc")],
    )

    assert len(rows) == 4
    assert [(row["case_id"], row["backend"]) for row in rows] == [
        ("1", "wfomc"),
        ("1", "asp"),
        ("s1", "wfomc"),
        ("s1", "asp"),
    ]


def test_merge_rows_rejects_missing_matrix_entry(tmp_path: Path) -> None:
    case = BenchmarkCase("real", "1", "A = set(a)", 1)
    source = tmp_path / "source.csv"
    _write_rows(source, [_row(case, "wfomc", 1)])

    with pytest.raises(ValueError, match="Missing 1 result rows"):
        merge_rows(
            cases=[case],
            backends=BACKENDS,
            sources=[(source, "all", "wfomc")],
            replacements=[],
        )


def test_write_merged_run_writes_results_summary_and_metadata(
    tmp_path: Path,
) -> None:
    case = BenchmarkCase("real", "1", "A = set(a)", 1)
    manifest_dir = tmp_path / "manifest"
    output_dir = tmp_path / "merged"
    save_cases([case], manifest_dir)
    rows = [_row(case, "wfomc", 1)]

    write_merged_run(
        output_dir=output_dir,
        manifest=manifest_dir / "manifest.json",
        cases=[case],
        backends=["wfomc"],
        sources=[],
        replacements=[],
        rows=rows,
    )

    assert (output_dir / "results.csv").exists()
    summary = list(csv.DictReader((output_dir / "summary.csv").open()))
    assert summary[0]["solved"] == "1"
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["num_cases"] == 1
    assert metadata["args"]["backends"] == ["wfomc"]
