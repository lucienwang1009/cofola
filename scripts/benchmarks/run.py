"""Run Cofola benchmark experiments.

Examples:
    uv run python -m scripts.benchmarks.run --suite real --backends coso
    uv run python -m scripts.benchmarks.run --suite all --backends wfomc coso asp essence --timeout 300
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
from scripts.benchmarks.cases import (
    DEFAULT_GROWING_DOMAIN_STEP,
    DEFAULT_GROWING_MAX_DOMAIN,
    DEFAULT_GROWING_MIN_DOMAIN,
    BenchmarkCase,
    load_saved_cases,
    save_cases,
    select_cases,
)
# Essence tool locations are environment-specific and have no built-in default;
# pass ``--conjure-dir`` and ``--java-bin`` when running it.
DEFAULT_CONJURE_DIR: Path | None = None
DEFAULT_JAVA_BIN: Path | None = None


STATUS_SOLVED = "solved"
STATUS_SOLVED_UNCHECKED = "solved_unchecked"
STATUS_WRONG = "wrong"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
STATUS_UNSOLVED = "unsolved"
DEFAULT_BENCHMARK_DIR = Path("problems/benchmarks")
DEFAULT_OUTPUT_DIR = Path("check-points/coso")
BACKEND_CHOICES = (
    "wfomc",
    "propositionalwfomc",
    "propostionalwfomc",
    "coso",
    "asp",
    "essence",
)


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
        choices=BACKEND_CHOICES,
        nargs="+",
        default=("coso",),
        help=(
            "Backends to compare. 'propositionalwfomc' is WFOMC with "
            "algo=propositional; the misspelled 'propostionalwfomc' alias is "
            "also accepted."
        ),
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
        help=(
            "Deprecated; retained for CLI compatibility. The synthetic suite "
            "now loads the fixed materialized manifest under "
            "problems/benchmarks/synthetic/manifest.json."
        ),
    )
    parser.add_argument(
        "--growing-min-domain",
        type=int,
        default=DEFAULT_GROWING_MIN_DOMAIN,
        help=f"Smallest domain size for generated growing benchmarks. Default: {DEFAULT_GROWING_MIN_DOMAIN}.",
    )
    parser.add_argument(
        "--growing-max-domain",
        type=int,
        default=DEFAULT_GROWING_MAX_DOMAIN,
        help=f"Largest domain size for generated growing benchmarks. Default: {DEFAULT_GROWING_MAX_DOMAIN}.",
    )
    parser.add_argument(
        "--growing-domain-step",
        type=int,
        default=DEFAULT_GROWING_DOMAIN_STEP,
        help=f"Domain-size interval for generated growing benchmarks. Default: {DEFAULT_GROWING_DOMAIN_STEP}.",
    )
    parser.add_argument(
        "--no-skip-larger-growing-after-timeout",
        action="store_true",
        help=(
            "Disable monotone timeout propagation for growing benchmarks. By "
            "default, after one backend times out for a growing family at "
            "domain n, larger domains for the same backend/family are recorded "
            "as timeout without invoking the solver."
        ),
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
        choices=("wfomc", "coso", "asp", "essence"),
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
    parser.add_argument(
        "--conjure-dir",
        type=Path,
        default=DEFAULT_CONJURE_DIR,
        help=(
            "Directory containing the Conjure/Savile Row tools for the Essence "
            "backend. Required when Conjure is not on PATH (no default)."
        ),
    )
    parser.add_argument(
        "--java-bin",
        type=Path,
        default=DEFAULT_JAVA_BIN,
        help=(
            "Java executable used by the Essence backend. Its parent directory "
            "is prepended to PATH when it exists. No default; falls back to java "
            "on PATH."
        ),
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
            growing_min_domain=args.growing_min_domain,
            growing_max_domain=args.growing_max_domain,
            growing_domain_step=args.growing_domain_step,
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
            "growing_min_domain": args.growing_min_domain,
            "growing_max_domain": args.growing_max_domain,
            "growing_domain_step": args.growing_domain_step,
            "skip_larger_growing_after_timeout": (
                not args.no_skip_larger_growing_after_timeout
            ),
            "algo": args.algo,
            "linear_order_encoding": args.linear_order_encoding,
            "conjure_dir": str(args.conjure_dir) if args.conjure_dir is not None else None,
            "java_bin": str(args.java_bin) if args.java_bin is not None else None,
        },
        "num_cases": len(cases),
        "cases": [asdict(case) for case in cases],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rows: list[dict[str, Any]] = []
    result_path = args.output_dir / "results.csv"
    growing_timeout_cutoffs: dict[tuple[str, str], int] = {}
    for case in cases:
        for backend in args.backends:
            skip_key = _growing_skip_key(case, backend)
            domain = _growing_domain(case)
            cutoff = (
                growing_timeout_cutoffs.get(skip_key)
                if skip_key is not None
                else None
            )
            if (
                not args.no_skip_larger_growing_after_timeout
                and skip_key is not None
                and domain is not None
                and cutoff is not None
                and domain > cutoff
            ):
                row = _skipped_timeout_row(
                    case,
                    backend=backend,
                    timeout=args.timeout,
                    cutoff=cutoff,
                )
            else:
                row = run_case(
                    case,
                    backend=backend,
                    timeout=args.timeout,
                    debug=args.debug,
                    algo=args.algo,
                    linear_order_encoding=args.linear_order_encoding,
                    conjure_dir=args.conjure_dir,
                    java_bin=args.java_bin,
                )
            if (
                case.expected is None
                and backend in set(args.trust_unchecked_backends)
                and row["status"] == STATUS_SOLVED_UNCHECKED
            ):
                row["status"] = STATUS_SOLVED
            if (
                not args.no_skip_larger_growing_after_timeout
                and skip_key is not None
                and domain is not None
                and _is_timeout_row(row)
            ):
                previous = growing_timeout_cutoffs.get(skip_key)
                if previous is None or domain < previous:
                    growing_timeout_cutoffs[skip_key] = domain
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
    conjure_dir: Path | None = DEFAULT_CONJURE_DIR,
    java_bin: Path | None = DEFAULT_JAVA_BIN,
) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    worker_backend, worker_algo, worker_linear_order_encoding = _backend_worker_args(
        backend,
        argparse.Namespace(algo=algo, linear_order_encoding=linear_order_encoding),
    )
    worker_timeout = timeout
    join_timeout = timeout
    if backend == "essence":
        from cofola.backend.essence.solver import _TERMINATION_GRACE_SECONDS

        worker_timeout = max(0.1, timeout - _TERMINATION_GRACE_SECONDS)
        join_timeout = timeout + _TERMINATION_GRACE_SECONDS + 3.0
    process = mp.Process(
        target=_solve_worker,
        args=(
            queue,
            case.program,
            worker_backend,
            debug,
            worker_algo,
            worker_linear_order_encoding,
            conjure_dir,
            java_bin,
            worker_timeout,
        ),
    )
    started = time.perf_counter()
    process.start()
    process.join(join_timeout)
    elapsed = time.perf_counter() - started

    if process.is_alive():
        process.terminate()
        process.join(2)
        return _row_from_payload(
            case,
            backend=backend,
            payload={
                "ok": False,
                "error_type": "TimeoutError",
                "error_message": f"Timed out after {timeout:.3f}s",
            },
            elapsed=elapsed,
        )

    try:
        payload = queue.get_nowait()
    except Empty:
        return _row_from_payload(
            case,
            backend=backend,
            payload={
                "ok": False,
                "error_type": "NoResult",
                "error_message": (
                    f"Worker exited with code {process.exitcode} without a result."
                ),
            },
            elapsed=elapsed,
        )

    return _row_from_payload(case, backend=backend, payload=payload, elapsed=elapsed)


def _growing_skip_key(case: BenchmarkCase, backend: str) -> tuple[str, str] | None:
    if case.suite != "growing":
        return None
    family = _tag_value(case, "family")
    if family is None:
        family = case.case_id.rsplit("_", maxsplit=1)[0]
    return (backend, family)


def _growing_domain(case: BenchmarkCase) -> int | None:
    tagged = _tag_value(case, "domain")
    if tagged is not None:
        try:
            return int(tagged)
        except ValueError:
            return None
    try:
        return int(case.case_id.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return None


def _tag_value(case: BenchmarkCase, key: str) -> str | None:
    prefix = f"{key}="
    for tag in case.tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def _is_timeout_row(row: dict[str, Any]) -> bool:
    return row["status"] == STATUS_TIMEOUT or row["error_type"] == "TimeoutError"


def _skipped_timeout_row(
    case: BenchmarkCase,
    *,
    backend: str,
    timeout: float,
    cutoff: int,
) -> dict[str, Any]:
    return _row(
        case,
        backend=backend,
        status=STATUS_TIMEOUT,
        result=None,
        elapsed=0.0,
        error_type="SkippedAfterTimeout",
        error_message=(
            f"Skipped because {backend} already timed out for this growing "
            f"family at domain {cutoff}; timeout budget was {timeout:.3f}s."
        ),
    )


def _backend_worker_args(
    backend: str,
    args: argparse.Namespace,
) -> tuple[str, str, str | None]:
    if backend in {"propositionalwfomc", "propostionalwfomc"}:
        return "wfomc", "propositional", args.linear_order_encoding
    return backend, args.algo, args.linear_order_encoding


def _row_from_payload(
    case: BenchmarkCase,
    *,
    backend: str,
    payload: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
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

    status = STATUS_ERROR
    if payload["error_type"] == "TimeoutError":
        status = STATUS_TIMEOUT
    return _row(
        case,
        backend=backend,
        status=status,
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
    conjure_dir: Path | None = DEFAULT_CONJURE_DIR,
    java_bin: Path | None = DEFAULT_JAVA_BIN,
    timeout: float = 300.0,
) -> None:
    kwargs = dict(backend=backend, algo=algo, linear_order_encoding=linear_order_encoding)
    try:
        if backend == "essence":
            from cofola.backend.essence.backend import EssenceBackend

            kwargs["backend"] = EssenceBackend(
                conjure_dir=conjure_dir,
                java_bin=java_bin,
                timeout=timeout,
                debug=debug,
            )
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
                "unsolved": sum(row["status"] == STATUS_UNSOLVED for row in group),
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
