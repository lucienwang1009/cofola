"""Summarize paired bag-lifting results and render the controlled-workload plots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

from scripts.benchmarks.run import write_csv


SOLVED = {"solved", "solved_unchecked"}


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["case_id"])].append(row)
    pairs = []
    for (suite, case_id), runs in sorted(grouped.items()):
        repeated = any(int(row["repetition"]) > 0 for row in runs)
        pair = {"suite": suite, "case_id": case_id, "rechecked": repeated}
        for mode, flag in (("off", "False"), ("on", "True")):
            selected = [row for row in runs if row["lifted_bags"] == flag and (int(row["repetition"]) > 0 if repeated else int(row["repetition"]) == 0)]
            expected_runs = 3 if repeated else 1
            if len(selected) != expected_runs:
                raise ValueError(f"Incomplete pair: {suite}:{case_id} {mode}, {len(selected)} of {expected_runs} runs")
            statuses = {row["status"] for row in selected}
            results = {row["result"] for row in selected if row["status"] in SOLVED}
            pair[f"{mode}_status"] = selected[0]["status"] if len(statuses) == 1 and len(results) <= 1 else "unstable"
            pair[f"{mode}_sec"] = statistics.median(float(row["elapsed_sec"]) for row in selected)
            pair[f"{mode}_error"] = ";".join(sorted({row["error_type"] for row in selected if row["error_type"]}))
            pair[f"{mode}_result"] = next(iter(results)) if len(results) == 1 else ""
        pair["speedup"] = pair["off_sec"] / pair["on_sec"] if pair["off_status"] in SOLVED and pair["on_status"] in SOLVED else ""
        pair["confirmed_regression"] = pair["off_status"] in SOLVED and (
            pair["on_status"] not in SOLVED or pair["off_result"] != pair["on_result"]
            or (pair["on_sec"] >= 1.2 * pair["off_sec"] and pair["on_sec"] - pair["off_sec"] >= 0.1)
        )
        pair["recheck_reason"] = ";".join(sorted({row["recheck_reason"] for row in runs if row["recheck_reason"]}))
        pair.update(item.split("=", 1) for item in runs[0]["tags"].split(";") if "=" in item)
        pairs.append(pair)
    return pairs


def summarize_pairs(pairs: list[dict], timeout: float) -> list[dict]:
    suites = defaultdict(list)
    for pair in pairs:
        suites[pair["suite"]].append(pair)
    summary = []
    for suite, group in suites.items():
        for mode in ("off", "on"):
            solved = [pair[f"{mode}_sec"] for pair in group if pair[f"{mode}_status"] in SOLVED]
            summary.append({
                "suite": suite, "mode": mode, "cases": len(group), "solved": len(solved),
                "statuses": dict(Counter(pair[f"{mode}_status"] for pair in group)),
                "mean_solved_sec": statistics.mean(solved) if solved else None,
                "max_solved_sec": max(solved) if solved else None,
                "par2_sec": (sum(solved) + 2 * timeout * (len(group) - len(solved))) / len(group),
            })
    return summary


def historical_check(input_dir: Path, baseline_path: Path, initial_rows: list[dict], output_dir: Path) -> None:
    """Keep historical regression checks separate from the on/off ablation."""
    with baseline_path.open() as stream:
        baseline_rows = [row for row in csv.DictReader(stream) if row["backend"] == "wfomc"]
    old = {(row["suite"], row["case_id"]): row for row in baseline_rows}
    initial = {(row["suite"], row["case_id"]): row for row in initial_rows
               if row["suite"] != "bag_lifting" and row["lifted_bags"] == "False" and row["repetition"] == "0"}
    if set(old) != set(initial):
        raise ValueError("Historical baseline does not match the original manifest")
    candidates = set()
    status_changes = []
    for key, row in initial.items():
        baseline = old[key]
        if row["status"] != baseline["status"]:
            status_changes.append(key)
        elif row["status"] in SOLVED:
            before, after = float(baseline["elapsed_sec"]), float(row["elapsed_sec"])
            if after >= before * 1.2 and after - before >= 0.1:
                candidates.add(key)
    rows = []
    for shard in sorted((input_dir / "historical").glob("shard-*")):
        if not shard.is_dir():
            continue
        if not (shard / "complete.json").exists():
            raise ValueError(f"Incomplete historical recheck: {shard}")
        with (shard / "results.csv").open() as stream:
            rows.extend(csv.DictReader(stream))
    pairs = aggregate(rows)
    if {(pair["suite"], pair["case_id"]) for pair in pairs} != candidates:
        raise ValueError("Historical rechecks must cover every and only the identified candidate")
    for pair in pairs:
        pair["historical_sec"] = float(old[(pair["suite"], pair["case_id"])]["elapsed_sec"])
        pair["new_over_historical"] = pair["off_sec"] / pair["historical_sec"]
        pair["persists"] = pair["off_status"] not in SOLVED or (
            pair["new_over_historical"] >= 1.2 and pair["off_sec"] - pair["historical_sec"] >= 0.1
        )
    write_csv(output_dir / "historical-runs.csv", rows)
    write_csv(output_dir / "historical-baseline.csv", baseline_rows)
    report = {
        "baseline_path": str(baseline_path),
        "baseline_file_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "initial_status_changes": status_changes,
        "candidate_count": len(candidates), "additional_runs": len(rows),
        "rechecks": pairs, "persistent_regressions": [pair for pair in pairs if pair["persists"]],
    }
    (output_dir / "historical-summary.json").write_text(json.dumps(report, indent=2))
    logger.info("Historical rechecks: {} candidates, {} persistent regressions", len(candidates), len(report["persistent_regressions"]))


def plot_controlled(pairs: list[dict], output_dir: Path, timeout: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    shapes = [("choose", "Single choice"), ("chain", "Observed chain"), ("hidden", "Unobserved\nintermediate"), ("intersection", "Intersection")]
    colors = {"off": "#526979", "on": "#b84a62"}
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.8), sharey=True, layout="constrained")
    for col, (shape, title) in enumerate(shapes):
        for row, (axis_key, fixed_key, fixed_value, ticks, xlabel) in enumerate([
            ("entities", "multiplicity", 2, [2, 4, 8, 16, 32], "Entities n (m = 2)"),
            ("multiplicity", "entities", 4, [2, 4, 8, 16], "Multiplicity m (n = 4)"),
        ]):
            ax = axes[row, col]
            selected = sorted([pair for pair in pairs if pair.get("shape") == shape and int(pair[fixed_key]) == fixed_value], key=lambda pair: int(pair[axis_key]))
            for mode, marker in (("off", "o"), ("on", "D")):
                x = [int(pair[axis_key]) for pair in selected]
                y = [pair[f"{mode}_sec"] if pair[f"{mode}_status"] in SOLVED else math.nan for pair in selected]
                ax.plot(x, y, color=colors[mode], marker=marker, markersize=4, linewidth=1.3)
                failed = [int(pair[axis_key]) for pair in selected if pair[f"{mode}_status"] not in SOLVED]
                # Separate coincident failure crosses slightly; ticks retain the actual n/m.
                offset = 0.98 if mode == "off" else 1.02
                ax.scatter([value * offset for value in failed], [timeout] * len(failed), color=colors[mode], marker="x", s=35, linewidths=1.4, zorder=5)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(ticks, labels=[str(tick) for tick in ticks])
            ax.set_yticks([0.5, 1, 10, 100], labels=["0.5", "1", "10", "100"])
            ax.set_ylim(0.3, 160)
            ax.grid(axis="y", which="major", alpha=0.22)
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel("Runtime (s)")
            if row == 0:
                ax.set_title(title, fontsize=10)
    fig.legend(handles=[
        Line2D([], [], color=colors["off"], marker="o", label="Lifting off"),
        Line2D([], [], color=colors["on"], marker="D", label="Lifting on"),
        Line2D([], [], color="black", marker="x", linestyle="none", label="Not solved (placed at 100 s)"),
    ], loc="outside upper center", ncol=3, frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"experiment_bag_lifting.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=100)
    parser.add_argument("--historical-baseline", type=Path)
    args = parser.parse_args()
    rows = []
    for group in ("paper", "controlled"):
        shards = sorted((args.input_dir / group).glob("shard-*"))
        shards = [path for path in shards if path.is_dir()]
        if len(shards) != 4 or any(not (path / "complete.json").exists() for path in shards):
            raise ValueError(f"Require four completed shards in {group}")
        for shard in shards:
            with (shard / "results.csv").open() as stream:
                rows.extend(csv.DictReader(stream))
    pairs = aggregate(rows)
    summary = summarize_pairs(pairs, args.timeout)
    out = args.input_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "all-runs.csv", rows)
    # Optional tags vary by suite; normalize before CSV serialization.
    fields = sorted({key for pair in pairs for key in pair})
    write_csv(out / "paired.csv", [{key: pair.get(key, "") for key in fields} for pair in pairs])
    speedups = {}
    for suite in sorted({pair["suite"] for pair in pairs}):
        values = [pair["speedup"] for pair in pairs if pair["suite"] == suite and pair["speedup"] != ""]
        speedups[suite] = {"common_solved": len(values), "geomean_off_over_on": math.exp(statistics.mean(math.log(value) for value in values)) if values else None}
    report = {
        "aggregation": "Initial runs, except rechecked cases use medians of three additional repetitions per mode. Unstable outcomes remain failures. No claim of statistical significance.",
        "num_cases": len(pairs), "num_runs": len(rows), "summary": summary,
        "speedups": speedups,
        "rechecked": [pair for pair in pairs if pair["rechecked"]],
        "confirmed_regressions": [pair for pair in pairs if pair["confirmed_regression"]],
        "wrong_or_unstable": [pair for pair in pairs if any(pair[f"{mode}_status"] in {"wrong", "unstable"} for mode in ("off", "on"))],
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    for row in summary:
        logger.info("{}", row)
    logger.info("Speedups: {}", speedups)
    logger.info("Confirmed regressions: {}", report["confirmed_regressions"])
    if args.figure_dir:
        plot_controlled(pairs, args.figure_dir, args.timeout)
    if args.historical_baseline:
        historical_check(args.input_dir, args.historical_baseline, rows, out)


if __name__ == "__main__":
    main()
