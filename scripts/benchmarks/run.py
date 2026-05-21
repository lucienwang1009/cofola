"""Run Cofola/CoSo benchmark experiments.

Examples:
    uv run python -m scripts.benchmarks.run --suite real --backends coso
    uv run python -m scripts.benchmarks.run --suite all --backends wfomc coso --timeout 300
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from queue import Empty
from typing import Any

from loguru import logger

from cofola.log import setup_logging
from cofola.solver import parse_and_solve
from scripts.benchmarks.cases import BenchmarkCase, load_saved_cases, save_cases, select_cases


STATUS_SOLVED = "solved"
STATUS_SOLVED_UNCHECKED = "solved_unchecked"
STATUS_WRONG = "wrong"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
DEFAULT_BENCHMARK_DIR = Path("problems/benchmarks")
DEFAULT_OUTPUT_DIR = Path("check-points/coso")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "real", "growing", "synthetic"),
        nargs="+",
        default=("all",),
        help="Benchmark suite(s) to run.",
    )
    parser.add_argument(
        "--backends",
        choices=("wfomc", "coso"),
        nargs="+",
        default=("coso",),
        help="Backends to compare. Use 'coso' for the CoSo-only run.",
    )
    parser.add_argument(
        "--real-path",
        type=Path,
        default=Path("problems/real/corpus.json"),
        help="Path to the real-world benchmark JSON.",
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        help=(
            "Load benchmark cases from a saved manifest.json or benchmark "
            "directory instead of regenerating them."
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        help=(
            "Directory where generated/loaded benchmarks are saved as .cfl "
            f"files plus manifest.json/manifest.csv. Default: {DEFAULT_BENCHMARK_DIR}."
        ),
    )
    parser.add_argument(
        "--no-save-benchmarks",
        action="store_true",
        help="Do not save reusable benchmark .cfl files and manifests.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save benchmark cases and exit without running any backend.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Optional case ids to run after suite generation.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per backend/case timeout in seconds.",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=0,
        help="Random seed for synthetic benchmark generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Cofola debug logging inside child processes.",
    )
    parser.add_argument(
        "--trust-unchecked-backends",
        nargs="+",
        default=(),
        choices=("wfomc", "coso"),
        help=(
            "Treat results from these backends as correct even when a case has "
            "no independent expected answer."
        ),
    )
    parser.add_argument(
        "--algo",
        choices=("fastv2", "incremental", "recursive", "propositional"),
        default="fastv2",
        help="WFOMC algorithm (only used by the wfomc backend). Default: fastv2.",
    )
    parser.add_argument(
        "--linear-order-encoding",
        choices=("pin", "axioms"),
        default=None,
        help="Encoding for order axioms when --algo=propositional. Default: pin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(False)
    logger.remove()

    ids = set(args.ids) if args.ids else None
    if args.benchmark_manifest is not None:
        cases = load_saved_cases(args.benchmark_manifest)
        if ids is not None:
            cases = [case for case in cases if case.case_id in ids]
    else:
        cases = select_cases(
            suites=args.suite,
            real_path=args.real_path,
            ids=ids,
            synthetic_seed=args.synthetic_seed,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = args.benchmark_dir or DEFAULT_BENCHMARK_DIR
    should_save_benchmarks = (
        not args.no_save_benchmarks
        and (
            args.benchmark_manifest is None
            or args.benchmark_dir is not None
            or args.save_only
        )
    )
    if should_save_benchmarks:
        save_cases(cases, benchmark_dir)
        if args.save_only:
            print(f"Saved {len(cases)} benchmark case(s) to {benchmark_dir}")
            return
    elif args.save_only:
        raise ValueError("--save-only cannot be combined with --no-save-benchmarks")

    metadata = {
        "args": {
            "suite": list(args.suite),
            "backends": list(args.backends),
            "real_path": str(args.real_path),
            "benchmark_manifest": (
                str(args.benchmark_manifest) if args.benchmark_manifest is not None else None
            ),
            "benchmark_dir": str(benchmark_dir) if should_save_benchmarks else None,
            "ids": sorted(ids) if ids else None,
            "timeout": args.timeout,
            "synthetic_seed": args.synthetic_seed,
            "algo": args.algo,
            "linear_order_encoding": args.linear_order_encoding,
        },
        "num_cases": len(cases),
        "cases": [asdict(case) for case in cases],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rows: list[dict[str, Any]] = []
    result_path = args.output_dir / "results.csv"
    for case in cases:
        for backend in args.backends:
            row = run_case(
                case,
                backend=backend,
                timeout=args.timeout,
                debug=args.debug,
                algo=args.algo,
                linear_order_encoding=args.linear_order_encoding,
            )
            if (
                case.expected is None
                and backend in set(args.trust_unchecked_backends)
                and row["status"] == STATUS_SOLVED_UNCHECKED
            ):
                row["status"] = STATUS_SOLVED
            rows.append(row)
            write_csv(result_path, rows)
            print(
                f"{row['suite']}:{row['case_id']} {backend} "
                f"{row['status']} result={row['result']} expected={row['expected']} "
                f"time={row['elapsed_sec']}"
            )

    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary)
    print(json.dumps(summary, indent=2))


def run_case(
    case: BenchmarkCase,
    *,
    backend: str,
    timeout: float,
    debug: bool,
    algo: str = "fastv2",
    linear_order_encoding: str | None = None,
) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_solve_worker,
        args=(queue, case.program, backend, debug, algo, linear_order_encoding),
    )
    started = time.perf_counter()
    process.start()
    process.join(timeout)
    elapsed = time.perf_counter() - started

    if process.is_alive():
        process.terminate()
        process.join(2)
        return _row(
            case,
            backend=backend,
            status=STATUS_TIMEOUT,
            result=None,
            elapsed=elapsed,
            error_type="TimeoutError",
            error_message=f"Timed out after {timeout:.3f}s",
        )

    try:
        payload = queue.get_nowait()
    except Empty:
        return _row(
            case,
            backend=backend,
            status=STATUS_ERROR,
            result=None,
            elapsed=elapsed,
            error_type="NoResult",
            error_message=f"Worker exited with code {process.exitcode} without a result.",
        )

    if payload["ok"]:
        result = int(payload["result"])
        if case.expected is None:
            status = STATUS_SOLVED_UNCHECKED
            error_type = ""
            error_message = ""
        elif result == case.expected:
            status = STATUS_SOLVED
            error_type = ""
            error_message = ""
        else:
            status = STATUS_WRONG
            error_type = "WrongAnswer"
            error_message = f"expected {case.expected}, got {result}"
        return _row(
            case,
            backend=backend,
            status=status,
            result=result,
            elapsed=elapsed,
            error_type=error_type,
            error_message=error_message,
        )

    return _row(
        case,
        backend=backend,
        status=STATUS_ERROR,
        result=None,
        elapsed=elapsed,
        error_type=payload["error_type"],
        error_message=payload["error_message"],
    )


def _solve_worker(
    queue: mp.Queue,
    program: str,
    backend: str,
    debug: bool,
    algo: str = "fastv2",
    linear_order_encoding: str | None = None,
) -> None:
    kwargs = dict(backend=backend, algo=algo, linear_order_encoding=linear_order_encoding)
    try:
        if debug:
            result = parse_and_solve(program, debug=True, **kwargs)
        else:
            logger.remove()
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    result = parse_and_solve(program, debug=False, **kwargs)
    except Exception as exc:  # noqa: BLE001 - benchmark harness must classify all failures.
        queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc).splitlines()[0] if str(exc) else repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return
    queue.put({"ok": True, "result": result})


def _row(
    case: BenchmarkCase,
    *,
    backend: str,
    status: str,
    result: int | None,
    elapsed: float,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "suite": case.suite,
        "case_id": case.case_id,
        "backend": backend,
        "status": status,
        "result": "" if result is None else result,
        "expected": "" if case.expected is None else case.expected,
        "elapsed_sec": f"{elapsed:.6f}",
        "error_type": error_type,
        "error_message": error_message,
        "tags": ";".join(case.tags),
        "source": case.source,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["suite"]), str(row["backend"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (suite, backend), group in sorted(grouped.items()):
        solved = [row for row in group if row["status"] == STATUS_SOLVED]
        unchecked = [
            row for row in group if row["status"] == STATUS_SOLVED_UNCHECKED
        ]
        elapsed_values = [float(row["elapsed_sec"]) for row in solved]
        solved_or_unchecked = solved + unchecked
        all_elapsed_values = [float(row["elapsed_sec"]) for row in solved_or_unchecked]
        summary.append(
            {
                "suite": suite,
                "backend": backend,
                "total": len(group),
                "expected_known": sum(row["expected"] != "" for row in group),
                "solved": len(solved),
                "solved_unchecked": len(unchecked),
                "wrong": sum(row["status"] == STATUS_WRONG for row in group),
                "errors": sum(row["status"] == STATUS_ERROR for row in group),
                "timeouts": sum(row["status"] == STATUS_TIMEOUT for row in group),
                "avg_solved_sec": (
                    f"{sum(elapsed_values) / len(elapsed_values):.6f}"
                    if elapsed_values else ""
                ),
                "max_solved_sec": f"{max(elapsed_values):.6f}" if elapsed_values else "",
                "avg_solved_or_unchecked_sec": (
                    f"{sum(all_elapsed_values) / len(all_elapsed_values):.6f}"
                    if all_elapsed_values else ""
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
