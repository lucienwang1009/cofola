"""Analyze repeated Cofola benchmark runs without changing the timed runner.

The input files are independent ``results.csv`` files produced by
``scripts.benchmarks.run``.  This script writes median per-instance results,
PAR-2 summaries, and paired runtime comparisons against a reference backend.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STATUS_SOLVED = "solved"
AGGREGATE_FAILURE_ORDER = (
    "wrong",
    "error",
    "timeout",
    "unsolved",
    "solved_unchecked",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Independent results.csv files from repeated benchmark runs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        required=True,
        help="Per-instance timeout in seconds; PAR-2 penalizes failures at twice this value.",
    )
    parser.add_argument(
        "--reference",
        default="wfomc",
        help="Reference backend for paired runtime comparisons. Default: wfomc.",
    )
    parser.add_argument(
        "--tie-ratio",
        type=float,
        default=1.05,
        help="Ratios within this multiplicative factor count as ties. Default: 1.05.",
    )
    parser.add_argument(
        "--expected-repetitions",
        type=int,
        help="Fail if any case/backend has a different number of repetitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for median_results.csv, par2.csv, and paired_runtime.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.tie_ratio < 1:
        raise ValueError("--tie-ratio must be at least 1")
    if args.expected_repetitions is not None and args.expected_repetitions <= 0:
        raise ValueError("--expected-repetitions must be positive")

    rows = load_rows(args.results)
    median_rows = aggregate_rows(rows)
    if args.expected_repetitions is not None:
        validate_repetitions(median_rows, args.expected_repetitions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "median_results.csv", median_rows)
    par2_rows = par2_summary(rows, timeout=args.timeout)
    write_csv(args.output_dir / "par2.csv", par2_rows)
    paired_rows = paired_runtime_summary(
        median_rows,
        reference=args.reference,
        tie_ratio=args.tie_ratio,
    )
    write_csv(args.output_dir / "paired_runtime.csv", paired_rows)
    print(
        f"Analyzed {len(rows)} observations into {len(median_rows)} "
        f"case/backend medians under {args.output_dir}"
    )


def load_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        seen: set[tuple[str, str, str]] = set()
        with path.open(newline="") as file:
            for record in csv.DictReader(file):
                row = dict(record)
                key = _case_backend_key(row)
                if key in seen:
                    raise ValueError(f"Duplicate case/backend row in {path}: {key}")
                seen.add(key)
                row["_run"] = str(path)
                rows.append(row)
    if not rows:
        raise ValueError("No benchmark rows were loaded")
    return rows


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_case_backend_key(row)].append(row)

    aggregated: list[dict[str, Any]] = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        statuses = [str(row["status"]) for row in group]
        elapsed = [float(row["elapsed_sec"]) for row in group]
        solved_repetitions = statuses.count(STATUS_SOLVED)
        status = _aggregate_status(statuses)
        result_values = {str(row.get("result", "")) for row in group}
        row = {
            key: value
            for key, value in first.items()
            if key != "_run"
        }
        row.update(
            {
                "status": status,
                "result": result_values.pop() if len(result_values) == 1 else "",
                "elapsed_sec": f"{statistics.median(elapsed):.6f}",
                "error_type": "" if status == STATUS_SOLVED else _joined_values(group, "error_type"),
                "error_message": "" if status == STATUS_SOLVED else _joined_values(group, "error_message"),
                "repetitions": len(group),
                "solved_repetitions": solved_repetitions,
            }
        )
        aggregated.append(row)
    return aggregated


def validate_repetitions(rows: Iterable[dict[str, Any]], expected: int) -> None:
    mismatches = [
        _case_backend_key(row)
        for row in rows
        if int(row["repetitions"]) != expected
    ]
    if mismatches:
        preview = ", ".join(":".join(key) for key in mismatches[:5])
        raise ValueError(
            f"Expected {expected} repetitions for every case/backend; "
            f"found {len(mismatches)} mismatch(es), including {preview}"
        )


def par2_summary(
    rows: Iterable[dict[str, Any]],
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for scope, suite, family in _summary_groups(row):
            grouped[(scope, suite, family, str(row["backend"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (scope, suite, family, backend), group in sorted(grouped.items()):
        solved = [row for row in group if row["status"] == STATUS_SOLVED]
        score = sum(float(row["elapsed_sec"]) for row in solved)
        score += (len(group) - len(solved)) * 2 * timeout
        summary.append(
            {
                "scope": scope,
                "suite": suite,
                "family": family,
                "backend": backend,
                "cases": len({str(row["case_id"]) for row in group}),
                "observations": len(group),
                "solved_observations": len(solved),
                "penalized_observations": len(group) - len(solved),
                "par2_sec": f"{score / len(group):.6f}",
                "timeout_sec": f"{timeout:.6f}",
            }
        )
    return summary


def paired_runtime_summary(
    median_rows: Iterable[dict[str, Any]],
    *,
    reference: str = "wfomc",
    tie_ratio: float = 1.05,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in median_rows:
        for scope, suite, family in _summary_groups(row):
            grouped[(scope, suite, family)].append(row)

    summary: list[dict[str, Any]] = []
    for (scope, suite, family), group in sorted(grouped.items()):
        by_backend: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            by_backend[str(row["backend"])][str(row["case_id"])] = row
        reference_rows = by_backend.get(reference)
        if reference_rows is None:
            continue
        for backend in sorted(set(by_backend) - {reference}):
            backend_rows = by_backend[backend]
            common = sorted(set(reference_rows) & set(backend_rows))
            ratios: list[float] = []
            for case_id in common:
                reference_row = reference_rows[case_id]
                backend_row = backend_rows[case_id]
                if (
                    reference_row["status"] != STATUS_SOLVED
                    or backend_row["status"] != STATUS_SOLVED
                ):
                    continue
                reference_time = float(reference_row["elapsed_sec"])
                backend_time = float(backend_row["elapsed_sec"])
                if reference_time <= 0 or backend_time <= 0:
                    continue
                ratios.append(backend_time / reference_time)

            reference_wins = sum(ratio > tie_ratio for ratio in ratios)
            backend_wins = sum(ratio < 1 / tie_ratio for ratio in ratios)
            ties = len(ratios) - reference_wins - backend_wins
            summary.append(
                {
                    "scope": scope,
                    "suite": suite,
                    "family": family,
                    "reference": reference,
                    "backend": backend,
                    "common_cases": len(common),
                    "paired_solved_cases": len(ratios),
                    "geomean_speedup": _format_float(_geometric_mean(ratios)),
                    "median_speedup": _format_float(
                        statistics.median(ratios) if ratios else None
                    ),
                    "reference_wins": reference_wins,
                    "ties": ties,
                    "backend_wins": backend_wins,
                    "tie_ratio": f"{tie_ratio:.6f}",
                }
            )
    return summary


def _case_backend_key(row: dict[str, Any]) -> tuple[str, str, str]:
    try:
        return (str(row["suite"]), str(row["case_id"]), str(row["backend"]))
    except KeyError as exc:
        raise ValueError(f"Missing required benchmark column: {exc.args[0]}") from exc


def _aggregate_status(statuses: list[str]) -> str:
    if statuses and all(status == STATUS_SOLVED for status in statuses):
        return STATUS_SOLVED
    for status in AGGREGATE_FAILURE_ORDER:
        if status in statuses:
            return status
    return statuses[0] if statuses else "error"


def _joined_values(group: Iterable[dict[str, Any]], key: str) -> str:
    values = sorted({str(row.get(key, "")) for row in group if row.get(key, "")})
    return " | ".join(values)


def _summary_groups(row: dict[str, Any]) -> Iterable[tuple[str, str, str]]:
    suite = str(row["suite"])
    yield ("suite", suite, "")
    family = _tag_value(str(row.get("tags", "")), "family")
    if suite == "growing" and family is not None:
        yield ("growing_family", suite, family)


def _tag_value(tags: str, key: str) -> str | None:
    prefix = f"{key}="
    for tag in tags.split(";"):
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def _geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
