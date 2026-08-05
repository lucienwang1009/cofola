"""Merge disjoint benchmark shards into one validated result set.

Each ``--source`` selects one suite/backend pair from a runner-produced CSV.
Rows from ``--replacement`` files are then applied by
``(suite, case_id, backend)`` key.  The merged matrix must contain exactly one
row for every manifest case and requested backend.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from scripts.benchmarks.cases import BenchmarkCase, load_saved_cases
from scripts.benchmarks.run import summarize, write_csv


ResultRow = dict[str, str]
ResultKey = tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Canonical benchmark manifest used to validate case coverage.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        required=True,
        help="Backends expected for every manifest case, in output order.",
    )
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        required=True,
        metavar=("RESULTS", "SUITE", "BACKEND"),
        help=(
            "Select SUITE/BACKEND rows from RESULTS.csv. Use SUITE=all to "
            "select all suites for that backend. May be repeated."
        ),
    )
    parser.add_argument(
        "--replacement",
        nargs=3,
        action="append",
        default=[],
        metavar=("RESULTS", "SUITE", "BACKEND"),
        help=(
            "Select replacement rows from RESULTS.csv by SUITE/BACKEND. "
            "May be repeated."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[ResultRow]:
    with path.open(newline="") as handle:
        return [
            {key: value or "" for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def row_key(row: ResultRow) -> ResultKey:
    return row["suite"], row["case_id"], row["backend"]


def merge_rows(
    *,
    cases: list[BenchmarkCase],
    backends: list[str],
    sources: list[tuple[Path, str, str]],
    replacements: list[tuple[Path, str, str]],
) -> list[ResultRow]:
    case_order = {
        (case.suite, case.case_id): index for index, case in enumerate(cases)
    }
    backend_order = {backend: index for index, backend in enumerate(backends)}
    expected_keys = {
        (case.suite, case.case_id, backend)
        for case in cases
        for backend in backends
    }
    merged: dict[ResultKey, ResultRow] = {}

    for path, suite, backend in sources:
        for row in load_rows(path):
            if row["backend"] != backend:
                continue
            if suite != "all" and row["suite"] != suite:
                continue
            key = row_key(row)
            if key not in expected_keys:
                raise ValueError(f"Unexpected source result key {key!r} in {path}")
            if key in merged:
                raise ValueError(f"Duplicate source result key {key!r}")
            merged[key] = row

    for path, suite, backend in replacements:
        for row in load_rows(path):
            if row["backend"] != backend or row["suite"] != suite:
                continue
            key = row_key(row)
            if key not in expected_keys:
                raise ValueError(f"Unexpected replacement result key {key!r} in {path}")
            if key not in merged:
                raise ValueError(f"Replacement has no source row for key {key!r}")
            merged[key] = row

    missing = expected_keys - merged.keys()
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(f"Missing {len(missing)} result rows; first keys: {sample!r}")
    if len(merged) != len(expected_keys):
        raise ValueError(
            f"Expected {len(expected_keys)} unique rows, got {len(merged)}"
        )

    for case in cases:
        expected = "" if case.expected is None else str(case.expected)
        for backend in backends:
            key = (case.suite, case.case_id, backend)
            if merged[key]["expected"] != expected:
                raise ValueError(
                    f"Expected-answer mismatch for {key!r}: manifest={expected!r}, "
                    f"result={merged[key]['expected']!r}"
                )

    return sorted(
        merged.values(),
        key=lambda row: (
            case_order[(row["suite"], row["case_id"])],
            backend_order[row["backend"]],
        ),
    )


def write_merged_run(
    *,
    output_dir: Path,
    manifest: Path,
    cases: list[BenchmarkCase],
    backends: list[str],
    sources: list[tuple[Path, str, str]],
    replacements: list[tuple[Path, str, str]],
    rows: list[ResultRow],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "results.csv", rows)
    write_csv(output_dir / "summary.csv", summarize(rows))
    metadata = {
        "args": {
            "manifest": str(manifest),
            "backends": backends,
            "sources": [
                {"results": str(path), "suite": suite, "backend": backend}
                for path, suite, backend in sources
            ],
            "replacements": [
                {"results": str(path), "suite": suite, "backend": backend}
                for path, suite, backend in replacements
            ],
        },
        "num_cases": len(cases),
        "cases": [asdict(case) for case in cases],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    cases = load_saved_cases(args.manifest)
    sources = [
        (Path(path), suite, backend) for path, suite, backend in args.source
    ]
    replacements = [
        (Path(path), suite, backend)
        for path, suite, backend in args.replacement
    ]
    rows = merge_rows(
        cases=cases,
        backends=args.backends,
        sources=sources,
        replacements=replacements,
    )
    write_merged_run(
        output_dir=args.output_dir,
        manifest=args.manifest,
        cases=cases,
        backends=args.backends,
        sources=sources,
        replacements=replacements,
        rows=rows,
    )
    print(
        f"Wrote {len(rows)} validated result rows for {len(cases)} cases "
        f"and {len(args.backends)} backends to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
