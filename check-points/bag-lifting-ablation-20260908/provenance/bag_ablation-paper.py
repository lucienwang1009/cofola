"""Paired, end-to-end lifted-bag ablation using the ordinary benchmark worker.

Each instance runs once in each mode, in alternating order. A slowdown of at
least 20% and 0.1 s, a lost solution, a wrong answer, or an unexpected error
triggers three additional paired repetitions. No encoding sizes are measured.
Run separate shards pinned to different physical cores on the same host.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from scripts.benchmarks.cases import load_saved_cases
from scripts.benchmarks.run import run_case, write_csv


def regression_reason(off: dict, on: dict) -> str:
    """Declare the recheck rule independently of the observed data."""
    if any(row["status"] in {"wrong", "error"} for row in (off, on)):
        return "wrong-or-error"
    solved = {"solved", "solved_unchecked"}
    if off["status"] in solved and on["status"] not in solved:
        return "lost-solution"
    if off["status"] in solved and on["status"] in solved:
        if str(off["result"]) != str(on["result"]):
            return "answer-mismatch"
        before, after = float(off["elapsed_sec"]), float(on["elapsed_sec"])
        if after >= 1.2 * before and after - before >= 0.1:
            return "slowdown"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("problems/benchmarks/manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("Require 0 <= shard < shards")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    cases = load_saved_cases(args.manifest)
    if args.ids:
        cases = [case for case in cases if case.case_id in set(args.ids)]
    indexed_cases = list(enumerate(cases))[args.shard::args.shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.csv"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}; choose a fresh output directory")
    versions = {}
    for package in ("wfomc", "sympy", "python-flint", "numpy", "scipy"):
        versions[package] = importlib.metadata.version(package)
    source_digest = hashlib.sha256()
    for directory in (Path("src"), Path("scripts/benchmarks")):
        for path in sorted(directory.rglob("*.py")):
            source_digest.update(str(path).encode() + b"\0" + path.read_bytes())
    metadata = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "hostname": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "versions": versions,
        "source_sha256": source_digest.hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "environment": {key: os.environ.get(key) for key in ("GANAK", "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "protocol": "Alternating paired off/on; one initial run; three additional pairs for >=20% AND >=0.1s slowdown, lost solution, wrong answer, mismatch, or error. No timeout propagation. No encoding-size instrumentation.",
        "cases": [asdict(case) for _, case in indexed_cases],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for index, case in indexed_cases:
        reason = ""
        for repetition in range(4):
            if repetition and not reason:
                break
            pair = {}
            order = (False, True) if (index + repetition) % 2 == 0 else (True, False)
            for lifted in order:
                row = run_case(case, backend="wfomc", timeout=args.timeout, debug=False, lifted_bags=lifted)
                row.update(lifted_bags=lifted, repetition=repetition, order=order.index(lifted), recheck_reason=reason)
                rows.append(row)
                pair[lifted] = row
                write_csv(result_path, rows)
                logger.info("{}/{} {}:{} lifting={} rep={} {} {:.3f}s", index + 1, len(cases), case.suite, case.case_id, lifted, repetition, row["status"], float(row["elapsed_sec"]))
            if repetition == 0:
                reason = regression_reason(pair[False], pair[True])
    summary = dict(Counter((row["status"]) for row in rows))
    (args.output_dir / "complete.json").write_text(json.dumps({"num_rows": len(rows), "statuses": summary}, indent=2))
    logger.info("Complete: {}", summary)


if __name__ == "__main__":
    main()
