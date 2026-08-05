"""Create paper-ready PDF plots from benchmark CSV files.

Examples:
    uv run python -m scripts.benchmarks.plot \
        --results check-points/coso/results.csv check-points/wfomc/results.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


STATUS_ORDER = ("solved", "solved_unchecked", "unsolved", "wrong", "error", "timeout")
STATUS_LABELS = {
    "solved": "Correct",
    "solved_unchecked": "Solved (unchecked)",
    "unsolved": "Unsolved",
    "wrong": "Wrong",
    "error": "Error",
    "timeout": "Timeout",
}
STATUS_COLORS = {
    "solved": "#2f6f4e",
    "solved_unchecked": "#88b04b",
    "unsolved": "#9a8f6a",
    "wrong": "#c94c4c",
    "error": "#7a7a7a",
    "timeout": "#d08c2f",
}
BACKEND_COLORS = {
    "coso": "#2a6fbb",
    "wfomc": "#b84a62",
    "propositionalwfomc": "#8c5fbf",
    "propostionalwfomc": "#8c5fbf",
    "asp": "#4b8f8c",
    "essence": "#b8843f",
}
BACKEND_MARKERS = {
    "coso": "o",
    "wfomc": "s",
    "propositionalwfomc": "D",
    "propostionalwfomc": "D",
    "asp": "^",
    "essence": "v",
}
SUITE_ORDER = ("real", "growing", "synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=(
            Path("check-points/coso/results.csv"),
            Path("check-points/wfomc/results.csv"),
        ),
        help="One or more benchmark results.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("check-points/plots"),
        help="Directory where PDF plots are written.",
    )
    parser.add_argument(
        "--prefix",
        default="benchmark",
        help="Filename prefix for generated plots.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf",),
        choices=("pdf", "png", "svg"),
        help="Output figure formats. PDF is the default for papers.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Generate the paper figures/tables by delegating to "
            "scripts.benchmarks.plot_paper. This is useful for full-benchmark "
            "runs whose results.csv contains real, growing, and synthetic suites."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="metadata.json used by --paper to infer synthetic object types.",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path("problems/benchmarks/synthetic"),
        help="Fallback synthetic .cfl directory used by --paper without --metadata.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=100.0,
        help="Timeout rail used by --paper growing-domain plots.",
    )
    parser.add_argument(
        "--print-tables",
        action="store_true",
        help="With --paper, print LaTeX-ready per-object table rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.paper:
        write_paper_outputs(args, rows)
        return

    figures = {
        "outcomes": plot_outcomes(rows),
        "runtime_distribution": plot_runtime_distribution(rows),
        "growing_runtime": plot_growing_runtime(rows),
        "real_runtime_scatter": plot_real_runtime_scatter(rows),
    }
    for name, fig in figures.items():
        save_figure(fig, args.output_dir / f"{args.prefix}_{name}", args.formats)
        plt.close(fig)

    write_plot_summary(rows, args.output_dir / f"{args.prefix}_plot_summary.csv")
    print(f"Wrote {len(figures)} figure(s) to {args.output_dir}")


def write_paper_outputs(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    from scripts.benchmarks import plot_paper

    real_rows = [r for r in rows if r["suite"] == "real"]
    synth_rows = [r for r in rows if r["suite"] == "synthetic"]
    growing_rows = [r for r in rows if r["suite"] == "growing"]
    real_types = plot_paper.real_object_types(real_rows)
    if args.metadata is not None:
        synth_types = plot_paper.metadata_object_types(args.metadata, suite="synthetic")
    else:
        synth_types = plot_paper.synthetic_object_types(args.synthetic_dir)

    figures = {
        "real_cactus": plot_paper.plot_cactus(real_rows, suite_label="real"),
        "synthetic_cactus": plot_paper.plot_cactus(synth_rows, suite_label="synthetic"),
        "synthetic_object_summary": plot_paper.plot_object_summary(
            synth_rows, synth_types,
        ),
        "growing_family_runtime": plot_paper.plot_growing_runtime(
            growing_rows, timeout_sec=args.timeout_sec,
        ),
    }
    for name, fig in figures.items():
        save_figure(fig, args.output_dir / f"{args.prefix}_{name}", args.formats)
        plt.close(fig)
    print(f"Wrote {len(figures)} paper figure(s) to {args.output_dir}")

    if args.print_tables:
        print()
        plot_paper.print_per_object_table(real_rows, real_types, label="real")
        print()
        plot_paper.print_per_object_table(synth_rows, synth_types, label="synthetic")


def load_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )


def plot_outcomes(rows: list[dict[str, str]]) -> Figure:
    configure_matplotlib()
    backends = sorted({row["backend"] for row in rows})
    suites = [suite for suite in SUITE_ORDER if any(row["suite"] == suite for row in rows)]
    groups = [(suite, backend) for suite in suites for backend in backends]

    fig, ax = plt.subplots(figsize=(6.8, 2.75))
    x_positions = list(range(len(groups)))
    counters = {
        group: Counter(
            row["status"]
            for row in rows
            if row["suite"] == group[0] and row["backend"] == group[1]
        )
        for group in groups
    }

    bottoms = [0] * len(groups)
    for status in STATUS_ORDER:
        heights = [counters[group][status] for group in groups]
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            width=0.72,
            label=STATUS_LABELS[status],
            color=STATUS_COLORS[status],
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    ax.set_ylabel("Number of cases")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{suite}\n{backend}" for suite, backend in groups])
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    fig.tight_layout(pad=0.8)
    return fig


def plot_runtime_distribution(rows: list[dict[str, str]]) -> Figure:
    configure_matplotlib()
    suites = [suite for suite in SUITE_ORDER if any(row["suite"] == suite for row in rows)]
    fig, axes = plt.subplots(1, len(suites), figsize=(6.8, 2.35), sharey=True)
    if len(suites) == 1:
        axes = [axes]

    for ax, suite in zip(axes, suites):
        for backend in sorted({row["backend"] for row in rows if row["suite"] == suite}):
            values = sorted(
                value
                for value in (
                    parse_float(row["elapsed_sec"])
                    for row in rows
                    if row["suite"] == suite
                    and row["backend"] == backend
                    and row["status"] in {"solved", "solved_unchecked"}
                )
                if value is not None and value > 0
            )
            if not values:
                continue
            ax.plot(
                range(1, len(values) + 1),
                values,
                marker=BACKEND_MARKERS.get(backend, "o"),
                markersize=2.2,
                linewidth=1.2,
                label=backend,
                color=BACKEND_COLORS.get(backend),
            )
        ax.set_yscale("log")
        ax.set_title(suite)
        ax.set_xlabel("Solved cases sorted by time")
        ax.grid(True, which="both", axis="y", alpha=0.25)
    axes[0].set_ylabel("Runtime (s, log)")
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.8)
    return fig


def plot_growing_runtime(rows: list[dict[str, str]]) -> Figure:
    configure_matplotlib()
    families = growing_families(rows)
    if not families:
        fig, ax = plt.subplots(figsize=(6.8, 2.35))
        ax.text(0.5, 0.5, "No growing benchmark rows", ha="center", va="center")
        ax.axis("off")
        return fig

    ncols = min(3, len(families))
    nrows = math.ceil(len(families) / ncols)
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(6.8, 2.2 * nrows),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)

    for ax, family in zip(axes, families):
        family_rows = [
            row for row in rows
            if row["suite"] == "growing" and growing_family(row["case_id"]) == family
        ]
        for backend in sorted({row["backend"] for row in family_rows}):
            backend_rows = sorted(
                (row for row in family_rows if row["backend"] == backend),
                key=lambda row: growing_index(row["case_id"]),
            )
            xs = [growing_index(row["case_id"]) for row in backend_rows]
            ys = [
                parse_float(row["elapsed_sec"])
                if row["status"] in {"solved", "solved_unchecked"}
                else math.nan
                for row in backend_rows
            ]
            ax.plot(
                xs,
                ys,
                marker=BACKEND_MARKERS.get(backend, "o"),
                markersize=3.0,
                linewidth=1.2,
                label=backend,
                color=BACKEND_COLORS.get(backend),
            )
            failed_xs = [
                x for x, row in zip(xs, backend_rows)
                if row["status"] not in {"solved", "solved_unchecked"}
            ]
            if failed_xs:
                ax.scatter(
                    failed_xs,
                    [0.08] * len(failed_xs),
                    marker="x",
                    s=18,
                    color=BACKEND_COLORS.get(backend),
                    linewidths=0.9,
                )
        ax.set_title(family.replace("_", " ").upper())
        ax.set_xlabel("Variant index")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.05)
        ax.grid(True, which="both", axis="y", alpha=0.25)

    for ax in axes[len(families):]:
        ax.axis("off")
    axes[0].set_ylabel("Runtime (s, log)")
    axes[min(len(families), len(axes)) - 1].legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.8)
    return fig


def plot_real_runtime_scatter(rows: list[dict[str, str]]) -> Figure:
    configure_matplotlib()
    real_rows = [row for row in rows if row["suite"] == "real"]
    backends = sorted({row["backend"] for row in real_rows})
    fig, ax = plt.subplots(figsize=(6.8, 2.8))

    offsets = {
        backend: (idx - (len(backends) - 1) / 2) * 0.18
        for idx, backend in enumerate(backends)
    }
    for backend in backends:
        solved = [
            row for row in real_rows
            if row["backend"] == backend and row["status"] in {"solved", "solved_unchecked"}
        ]
        solved.sort(key=lambda row: numeric_case_id(row["case_id"]))
        xs = [numeric_case_id(row["case_id"]) + offsets[backend] for row in solved]
        ys = [parse_float(row["elapsed_sec"]) for row in solved]
        ax.scatter(
            xs,
            ys,
            s=9,
            alpha=0.75,
            label=backend,
            marker=BACKEND_MARKERS.get(backend, "o"),
            color=BACKEND_COLORS.get(backend),
            linewidths=0,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Real problem id")
    ax.set_ylabel("Runtime (s, log)")
    ax.set_title("Solved real-problem runtimes")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.8)
    return fig


def write_plot_summary(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row["suite"], row["backend"])][row["status"]] += 1

    fieldnames = ["suite", "backend", *STATUS_ORDER]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for suite, backend in sorted(grouped):
            counter = grouped[(suite, backend)]
            writer.writerow(
                {
                    "suite": suite,
                    "backend": backend,
                    **{status: counter[status] for status in STATUS_ORDER},
                }
            )


def save_figure(fig: Figure, stem: Path, formats: Iterable[str]) -> None:
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), bbox_inches="tight")


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def numeric_case_id(case_id: str) -> int:
    try:
        return int(case_id)
    except ValueError:
        match = re.search(r"(\d+)$", case_id)
        return int(match.group(1)) if match else 0


def growing_index(case_id: str) -> int:
    match = re.search(r"_(\d+)$", case_id)
    return int(match.group(1)) if match else numeric_case_id(case_id)


def growing_family(case_id: str) -> str:
    return re.sub(r"_\d+$", "", case_id)


def growing_families(rows: list[dict[str, str]]) -> list[str]:
    families = {
        growing_family(row["case_id"])
        for row in rows
        if row["suite"] == "growing"
    }
    return sorted(families)


if __name__ == "__main__":
    main()
