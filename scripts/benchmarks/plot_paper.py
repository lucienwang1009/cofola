"""Paper figures and tables for the Cofola experiments section.

This script generates the figures used in the *Experiments* section after the
v3 revision and prints the LaTeX-ready numbers that populate the per-object-type
result tables.

Figures (default prefix ``experiment``):

* ``experiment_real_cactus.pdf``     -- cactus plot on the real suite
  (``x``: runtime in seconds, log scale; ``y``: number of real instances
  solved).
* ``experiment_real_overview.pdf`` -- outcome composition and PAR-2
  comparison on the real suite.
* ``experiment_synthetic_cactus.pdf`` -- analogous cactus plot on the
  synthetic suite.
* ``experiment_synthetic_strata.pdf`` -- correctly solved synthetic
  instances for every backend and structural stratum.
* ``experiment_synthetic_overview.pdf`` -- outcome composition and PAR-2
  comparison on the synthetic robustness suite.
* ``experiment_synthetic_tiers.pdf`` -- correctly solved synthetic instances
  at the small, medium, and large size tiers.
* ``experiment_growing_family_runtime.pdf`` -- one-panel-per-family
  runtime plot using the extended ``domain in {5, 10, ..., 100}`` results.
  Instances on which a backend timed out, raised an error, or returned a wrong
  answer are drawn as crosses placed at the per-instance timeout (100 s) so
  that they appear on the same time scale as solved runs.

Tables: the script also prints the LaTeX rows for the per-object-type tables
of the real and synthetic suites (``--print-tables``).  These are pasted into
``experiments.tex``.

Conventions:

* "Instances" is the consistent unit name across plots and tables (the paper
  text uses the same word).
* A row is counted as *solved* only when ``status == "solved"``.  Wrong
  answers, encoding errors, runtime errors, and timeouts are all collapsed to
  *not solved* so the bars/columns give a single uniform coverage measure
  across the five backends.

Example::

    uv run python -m scripts.benchmarks.plot_paper \
        --results check-points/full-benchmarks/results.csv \
        --metadata check-points/full-benchmarks/metadata.json \
        --output-dir ../comb_aij/figs/experiments \
        --prefix experiment \
        --timeout-sec 100 \
        --exclude-real-tag circle \
        --print-tables
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


BACKEND_ORDER = ("wfomc", "propositionalwfomc", "coso", "asp", "essence")
BACKEND_LABELS = {
    "wfomc": "Cofola-WFOMC",
    "propositionalwfomc": "CNF-Ganak",
    "coso": "CoLa-CoSo",
    "asp": "ASP-Clingo",
    "essence": "ESSENCE-Conjure",
}
BACKEND_COLORS = {
    "wfomc": "#b84a62",
    "propositionalwfomc": "#8c5fbf",
    "coso": "#2a6fbb",
    "asp": "#4b8f8c",
    "essence": "#b8843f",
}
BACKEND_MARKERS = {
    "wfomc": "s",
    "propositionalwfomc": "D",
    "coso": "o",
    "asp": "^",
    "essence": "v",
}

SYNTHETIC_STRATA = (
    ("set_choice_intersection", "Set choice\nintersection"),
    ("bag_choice_multiplicity", "Bag choice\nmultiplicity"),
    ("set_tuple_position_count", "Set tuple\nposition/count"),
    ("bag_tuple_multiplicity", "Bag tuple\nmultiplicity"),
    ("set_sequence_patterns", "Set sequence\npatterns"),
    ("bag_sequence_adjacency", "Bag sequence\nadjacency"),
    ("set_partition_min_size", "Partition\nminimum size"),
    ("set_composition_sizes", "Composition\npart sizes"),
    ("dependent_choice_disjoint", "Dependent\nchoice"),
    ("dependent_tuple_position_count", "Dependent\ntuple"),
    ("dependent_sequence_patterns", "Dependent\nsequence"),
    ("dependent_partition_min_size", "Dependent\npartition"),
    ("dependent_composition_sizes", "Dependent\ncomposition"),
    ("dependent_bag_tuple", "Dependent\nbag tuple"),
    ("nested_choice_chain", "Nested\nchoice chain"),
    ("dependent_selected_subsets", "Selected\nsubsets"),
)

OBJECT_TYPES = (
    "set",
    "bag",
    "tuple",
    "sequence",
    "partition",
    "composition",
)

# Growing-domain families: name → human-readable subtitle anchored to the
# problem number in either the paper or the CoLa/CoSo paper.
GROWING_PANELS = (
    ("tvs", "P3 (CoSo) - TVs"),
    ("workers", "P4 (CoSo) - Workers"),
    ("banana", "P5 (CoSo) - BANANA"),
    ("mathcounts", "P2 (ours) - MATHCOUNTS"),
    ("fourletter", "P3 (ours) - Four-letter"),
    ("nested_choice", "Nested choice - Team captain"),
    ("selected_pairs", "Dependent partition - Selected pairs"),
    ("selected_sequence", "Dependent sequence - Adjacency"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        help="Single full-benchmark results.csv containing "
                             "real, growing, and synthetic suites.")
    parser.add_argument("--metadata", type=Path,
                        help="metadata.json produced by the benchmark runner; "
                             "used to infer object types for generated suites.")
    parser.add_argument("--real-results", type=Path,
                        help="Legacy: results.csv covering the real suite.")
    parser.add_argument("--synthetic-results", type=Path,
                        help="Legacy: results.csv covering the synthetic suite.")
    parser.add_argument("--growing-results", type=Path,
                        help="Legacy: results.csv covering the growing suite.")
    parser.add_argument("--synthetic-dir", type=Path,
                        help="Legacy fallback: directory containing synthetic "
                             ".cfl files used to infer object types per instance.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="experiment")
    parser.add_argument("--formats", nargs="+", default=("pdf",),
                        choices=("pdf", "png", "svg"))
    parser.add_argument("--timeout-sec", type=float, default=100.0,
                        help="Per-instance timeout in seconds; failed runs are "
                             "drawn at this y-value on the growing-domain plot.")
    parser.add_argument("--exclude-real-tag", action="append", default=[],
                        help="Exclude every real case carrying this exact tag; "
                             "repeat the option to exclude multiple tags.")
    parser.add_argument("--print-tables", action="store_true",
                        help="Print LaTeX-ready rows for the per-object-type "
                             "tables of the real and synthetic suites.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.results is not None:
        all_rows = load_rows(args.results)
        real_rows = [r for r in all_rows if r["suite"] == "real"]
        synth_rows = [r for r in all_rows if r["suite"] == "synthetic"]
        growing_rows = [r for r in all_rows if r["suite"] == "growing"]
    else:
        real_rows = [r for r in load_rows(args.real_results) if r["suite"] == "real"]
        synth_rows = [r for r in load_rows(args.synthetic_results) if r["suite"] == "synthetic"]
        growing_rows = [r for r in load_rows(args.growing_results) if r["suite"] == "growing"]

    real_rows = exclude_tagged_cases(real_rows, set(args.exclude_real_tag))
    real_types = real_object_types(real_rows)
    if args.metadata is not None:
        synth_types = metadata_object_types(args.metadata, suite="synthetic")
    else:
        synth_types = synthetic_object_types(args.synthetic_dir)

    figs = {
        "real_cactus": plot_cactus(real_rows, suite_label="real"),
        "real_overview": plot_outcome_overview(
            real_rows, timeout_sec=args.timeout_sec,
        ),
        "synthetic_cactus": plot_cactus(synth_rows, suite_label="synthetic"),
        "synthetic_strata": plot_synthetic_strata(synth_rows),
        "synthetic_overview": plot_outcome_overview(
            synth_rows, timeout_sec=args.timeout_sec,
        ),
        "synthetic_tiers": plot_synthetic_tiers(synth_rows),
        "growing_family_runtime": plot_growing_runtime(
            growing_rows, timeout_sec=args.timeout_sec,
        ),
    }
    for name, fig in figs.items():
        save_figure(fig, args.output_dir / f"{args.prefix}_{name}", args.formats)
        plt.close(fig)
    print(f"Wrote {len(figs)} figure(s) to {args.output_dir}")

    if args.print_tables:
        print()
        print_per_object_table(real_rows, real_types, label="real")
        print()
        print_per_object_table(synth_rows, synth_types, label="synthetic")


# ---------------------------------------------------------------------------
# IO and helpers
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def exclude_tagged_cases(
    rows: list[dict[str, str]],
    excluded_tags: set[str],
) -> list[dict[str, str]]:
    """Exclude complete cases carrying any tag in ``excluded_tags``."""

    if not excluded_tags:
        return rows
    excluded_case_ids = {
        row["case_id"]
        for row in rows
        if excluded_tags.intersection(
            tag.strip() for tag in row.get("tags", "").split(";") if tag.strip()
        )
    }
    return [row for row in rows if row["case_id"] not in excluded_case_ids]


def validate_args(args: argparse.Namespace) -> None:
    if args.results is not None:
        legacy = (args.real_results, args.synthetic_results, args.growing_results)
        if any(value is not None for value in legacy):
            raise SystemExit("--results cannot be combined with per-suite result flags")
    elif not (args.real_results and args.synthetic_results and args.growing_results):
        raise SystemExit(
            "Pass either --results FULL_RESULTS.csv or all of "
            "--real-results, --synthetic-results, and --growing-results."
        )
    if args.metadata is None and args.synthetic_dir is None:
        raise SystemExit(
            "Pass --metadata metadata.json or --synthetic-dir to infer synthetic object types."
        )


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


def save_figure(fig: Figure, stem: Path, formats: Iterable[str]) -> None:
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), bbox_inches="tight")


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def backends_present(rows: list[dict[str, str]]) -> list[str]:
    present = {row["backend"] for row in rows}
    ordered = [b for b in BACKEND_ORDER if b in present]
    ordered.extend(sorted(present - set(BACKEND_ORDER)))
    return ordered


# ---------------------------------------------------------------------------
# Object-type inference
# ---------------------------------------------------------------------------

OBJECT_TYPE_PATTERNS = {
    "set": (r"\bset\s*\(", r"\bsupp\s*\("),
    "bag": (r"\bbag\s*\(", r"\bchoose_replace\s*\(", r"\badd_union\s*\("),
    "tuple": (r"\btuple\s*\(", r"\bchoose_tuple\s*\(",
              r"\bchoose_replace_tuple\s*\("),
    "sequence": (r"\bsequence\s*\(", r"\bchoose_sequence\s*\(",
                 r"\bchoose_replace_sequence\s*\("),
    "partition": (r"\bpartition\s*\(",),
    "composition": (r"\bcompose\s*\(",),
}


def detect_object_types(program: str) -> set[str]:
    found: set[str] = set()
    for typ, patterns in OBJECT_TYPE_PATTERNS.items():
        if any(re.search(p, program) for p in patterns):
            found.add(typ)
    return found


def real_object_types(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    type_set = set(OBJECT_TYPES)
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        tags = {t.strip() for t in row["tags"].split(";") if t.strip()}
        out[row["case_id"]].update(tags & type_set)
    return out


def synthetic_object_types(synthetic_dir: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in sorted(synthetic_dir.glob("*.cfl")):
        out[path.stem] = detect_object_types(path.read_text())
    return out


def metadata_object_types(metadata_path: Path, suite: str) -> dict[str, set[str]]:
    data = json.loads(metadata_path.read_text())
    out: dict[str, set[str]] = {}
    for case in data.get("cases", []):
        if str(case.get("suite")) != suite:
            continue
        out[str(case["case_id"])] = detect_object_types(str(case.get("program", "")))
    return out


# ---------------------------------------------------------------------------
# Tables: per-backend, per-object-type solve count + average runtime
# ---------------------------------------------------------------------------

def per_object_stats(
    rows: list[dict[str, str]],
    case_types: dict[str, set[str]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return ``stats[backend][object_type] = {solved, total, avg_sec}``."""
    backends = backends_present(rows)
    stats: dict[str, dict[str, dict[str, float | int]]] = {
        b: {
            t: {"solved": 0, "runtime_total": 0.0, "runtime_n": 0}
            for t in OBJECT_TYPES
        }
        for b in backends
    }
    totals: dict[str, int] = {t: 0 for t in OBJECT_TYPES}
    seen: set[str] = set()
    for row in rows:
        case_id = row["case_id"]
        if case_id not in seen:
            for t in case_types.get(case_id, set()):
                if t in totals:
                    totals[t] += 1
            seen.add(case_id)
        if row["backend"] not in stats:
            continue
        if row["status"] != "solved":
            continue
        elapsed = parse_float(row.get("elapsed_sec")) or 0.0
        for t in case_types.get(case_id, set()):
            if t not in stats[row["backend"]]:
                continue
            stats[row["backend"]][t]["solved"] += 1
            if elapsed > 0:
                stats[row["backend"]][t]["runtime_total"] += elapsed
                stats[row["backend"]][t]["runtime_n"] += 1
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for b in backends:
        out[b] = {}
        for t in OBJECT_TYPES:
            s = stats[b][t]
            avg = s["runtime_total"] / s["runtime_n"] if s["runtime_n"] else float("nan")
            out[b][t] = {"solved": s["solved"], "total": totals[t], "avg_sec": avg}
    return out


def print_per_object_table(
    rows: list[dict[str, str]],
    case_types: dict[str, set[str]],
    label: str,
) -> None:
    """Print LaTeX rows that populate a per-object-type result table."""
    stats = per_object_stats(rows, case_types)
    backends = backends_present(rows)
    totals_row = "  & " + " & ".join(
        f"{t} ({stats[backends[0]][t]['total']})" for t in OBJECT_TYPES
    ) + " \\\\"
    print(f"% --- {label} per-object-type table rows ---")
    print(f"% Header (object types and their total counts):")
    print(totals_row)
    for b in backends:
        cells = []
        for t in OBJECT_TYPES:
            cell = stats[b][t]
            solved = cell["solved"]
            avg = cell["avg_sec"]
            if solved == 0:
                cells.append("--")
            elif math.isnan(avg):
                cells.append(f"{solved}")
            else:
                cells.append(f"{solved}\\,({avg:.2f}s)")
        print(f"{BACKEND_LABELS[b]} & " + " & ".join(cells) + " \\\\")


# ---------------------------------------------------------------------------
# Cactus plot (x = runtime, y = instances solved)
# ---------------------------------------------------------------------------

def plot_cactus(rows: list[dict[str, str]], suite_label: str) -> Figure:
    configure_matplotlib()
    backends = backends_present(rows)

    figsize = (4.0, 3.8) if suite_label == "real" else (3.5, 2.75)
    fig, ax = plt.subplots(figsize=figsize)
    for backend in backends:
        runtimes = sorted(
            v for v in (
                parse_float(row["elapsed_sec"])
                for row in rows
                if row["backend"] == backend and row["status"] == "solved"
            )
            if v is not None and v > 0
        )
        if not runtimes:
            # Show legend entry "(0)" so the reader knows the backend was run.
            ax.plot(
                [],
                [],
                marker=BACKEND_MARKERS.get(backend, "o"),
                linewidth=1.2,
                label=f"{BACKEND_LABELS.get(backend, backend)} (0)",
                color=BACKEND_COLORS.get(backend),
            )
            continue
        ax.step(
            runtimes,
            range(1, len(runtimes) + 1),
            where="post",
            linewidth=1.2,
            label=f"{BACKEND_LABELS.get(backend, backend)} ({len(runtimes)})",
            color=BACKEND_COLORS.get(backend),
        )
    ax.set_xscale("log")
    ax.axvline(100, color="#555555", linewidth=0.8, linestyle=":", label="100s timeout")
    ax.set_xlim(0.08, 120)
    ax.set_ylim(0, len({row["case_id"] for row in rows}) + (8 if suite_label == "real" else 4))
    ax.set_xlabel("Runtime on solved instances (s, log)")
    ax.set_ylabel(f"{suite_label.capitalize()} instances solved")
    ax.grid(True, which="both", axis="both", alpha=0.25)
    if suite_label == "real":
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            columnspacing=0.7,
            handlelength=1.6,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.17, right=0.975, bottom=0.14, top=0.73)
    else:
        ax.legend(
            frameon=False,
            loc="lower right",
            ncol=1,
            columnspacing=0.7,
            handlelength=1.6,
        )
        fig.tight_layout(pad=0.8)
    return fig


def plot_synthetic_strata(rows: list[dict[str, str]]) -> Figure:
    """Plot correctly solved counts for the balanced synthetic strata."""

    configure_matplotlib()
    backends = backends_present(rows)
    stratum_names = [name for name, _ in SYNTHETIC_STRATA]
    stratum_labels = [label for _, label in SYNTHETIC_STRATA]
    totals: dict[str, set[str]] = defaultdict(set)
    solved: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        stratum = next(
            (
                tag.split("=", maxsplit=1)[1]
                for tag in row.get("tags", "").split(";")
                if tag.startswith("stratum=")
            ),
            None,
        )
        if stratum not in stratum_names:
            continue
        totals[stratum].add(row["case_id"])
        if row["status"] == "solved":
            solved[(row["backend"], stratum)] += 1

    values = [
        [solved[(backend, stratum)] for stratum in stratum_names]
        for backend in backends
    ]
    max_total = max((len(case_ids) for case_ids in totals.values()), default=1)
    fig, ax = plt.subplots(figsize=(7.4, 2.65))
    heatmap = ax.imshow(
        values,
        cmap="YlGn",
        vmin=0,
        vmax=max_total,
        aspect="auto",
        interpolation="nearest",
    )

    for backend_index, backend_values in enumerate(values):
        for stratum_index, value in enumerate(backend_values):
            text_color = "white" if value >= max_total * 0.62 else "#222222"
            ax.text(
                stratum_index,
                backend_index,
                str(value),
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
            )

    ax.set_xticks(range(len(stratum_labels)), labels=stratum_labels)
    ax.set_yticks(
        range(len(backends)),
        labels=[BACKEND_LABELS.get(backend, backend) for backend in backends],
    )
    ax.tick_params(axis="x", labelrotation=45, length=0)
    ax.tick_params(axis="y", length=0)
    ax.axvline(7.5, color="white", linewidth=1.5)
    ax.text(3.5, -0.9, "Single-source strata", ha="center", va="bottom", fontsize=7)
    ax.text(11.5, -0.9, "Dependent strata", ha="center", va="bottom", fontsize=7)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label(f"Correctly solved (of {max_total})")
    colorbar.set_ticks((0, max_total // 2, max_total))
    ax.grid(False)
    fig.tight_layout(pad=0.7)
    return fig


def plot_outcome_overview(
    rows: list[dict[str, str]],
    timeout_sec: float = 100.0,
) -> Figure:
    """Plot outcome composition and PAR-2 for one benchmark suite."""

    configure_matplotlib()
    observed_backends = backends_present(rows)
    status_order = ("solved", "timeout", "wrong", "error")
    status_labels = {
        "solved": "Correct",
        "timeout": "Timeout",
        "wrong": "Wrong",
        "error": "Error",
    }
    status_colors = {
        "solved": "#4d9563",
        "timeout": "#e0a43a",
        "wrong": "#c45152",
        "error": "#777777",
    }

    counts: dict[str, dict[str, int]] = {}
    par2: dict[str, float] = {}
    for backend in observed_backends:
        backend_rows = [row for row in rows if row["backend"] == backend]
        counts[backend] = {
            status: sum(row["status"] == status for row in backend_rows)
            for status in status_order
        }
        costs = [
            (parse_float(row.get("elapsed_sec")) or 0.0)
            if row["status"] == "solved"
            else 2.0 * timeout_sec
            for row in backend_rows
        ]
        par2[backend] = sum(costs) / len(costs) if costs else math.nan

    backends = sorted(observed_backends, key=lambda backend: par2[backend])
    total_cases = max(
        (sum(counts[backend].values()) for backend in backends),
        default=0,
    )

    fig, (coverage_ax, par2_ax) = plt.subplots(
        2,
        1,
        figsize=(4.0, 3.8),
        gridspec_kw={"height_ratios": (1.0, 1.0)},
    )
    y_positions = list(range(len(backends)))
    left = [0] * len(backends)
    for status in status_order:
        values = [counts[backend][status] for backend in backends]
        coverage_ax.barh(
            y_positions,
            values,
            left=left,
            height=0.62,
            label=status_labels[status],
            color=status_colors[status],
            edgecolor="white",
            linewidth=0.35,
        )
        for y, value, offset in zip(y_positions, values, left):
            if value:
                if value >= max(10, total_cases * 0.04):
                    coverage_ax.text(
                        offset + value / 2,
                        y,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if status != "timeout" else "#222222",
                        fontweight="bold" if status == "solved" else "normal",
                    )
                else:
                    coverage_ax.annotate(
                        str(value),
                        (offset + value, y),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=7,
                        color=status_colors[status],
                        fontweight="bold",
                    )
        left = [offset + value for offset, value in zip(left, values)]

    coverage_ax.set_yticks(
        y_positions,
        labels=[BACKEND_LABELS.get(backend, backend) for backend in backends],
    )
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlim(0, total_cases * 1.1)
    coverage_ax.set_xlabel(f"Outcome count ({total_cases} instances)")
    coverage_ax.set_title("Coverage and failure modes", loc="left")
    coverage_ax.grid(True, axis="x", alpha=0.2)
    coverage_ax.grid(False, axis="y")
    legend_handles, legend_labels = coverage_ax.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        columnspacing=0.9,
        handlelength=1.2,
        borderaxespad=0.0,
    )

    baseline = par2.get("wfomc", math.nan)
    par2_values = [par2[backend] for backend in backends]
    for y, backend, value in zip(y_positions, backends, par2_values):
        par2_ax.hlines(
            y,
            1.0,
            value,
            color=BACKEND_COLORS.get(backend, "#555555"),
            linewidth=1.4,
        )
        par2_ax.plot(
            value,
            y,
            marker=BACKEND_MARKERS.get(backend, "o"),
            markersize=5,
            color=BACKEND_COLORS.get(backend, "#555555"),
        )
        ratio = value / baseline if baseline > 0 else math.nan
        par2_ax.annotate(
            f"{value:.2f}s ({ratio:.1f}x)",
            (value, y),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
        )
    par2_ax.set_xscale("log")
    par2_ax.set_xlim(0.9, max(par2_values, default=1.0) * 2.05)
    par2_ax.set_xlabel("PAR-2 (s, log)")
    par2_ax.set_title("Failure-aware runtime", loc="left")
    par2_ax.set_yticks(
        y_positions,
        labels=[BACKEND_LABELS.get(backend, backend) for backend in backends],
    )
    par2_ax.invert_yaxis()
    par2_ax.grid(True, axis="x", which="both", alpha=0.25)
    par2_ax.grid(False, axis="y")

    fig.subplots_adjust(
        left=0.31,
        right=0.975,
        bottom=0.11,
        top=0.86,
        hspace=0.68,
    )
    return fig


def plot_synthetic_tiers(rows: list[dict[str, str]]) -> Figure:
    """Plot solved-instance coverage as synthetic instances grow."""

    configure_matplotlib()
    tiers = ("small", "medium", "large")
    tier_labels = ("Small\n(n=8)", "Medium\n(n=12)", "Large\n(n=16)")
    backends = backends_present(rows)
    values: dict[str, list[int]] = {}
    totals: dict[str, set[str]] = {tier: set() for tier in tiers}

    for row in rows:
        difficulty = next(
            (
                tag.split("=", maxsplit=1)[1]
                for tag in row.get("tags", "").split(";")
                if tag.startswith("difficulty=")
            ),
            None,
        )
        if difficulty in totals:
            totals[difficulty].add(row["case_id"])

    for backend in backends:
        solved_by_tier = {tier: 0 for tier in tiers}
        for row in rows:
            if row["backend"] != backend or row["status"] != "solved":
                continue
            difficulty = next(
                (
                    tag.split("=", maxsplit=1)[1]
                    for tag in row.get("tags", "").split(";")
                    if tag.startswith("difficulty=")
                ),
                None,
            )
            if difficulty in solved_by_tier:
                solved_by_tier[difficulty] += 1
        values[backend] = [solved_by_tier[tier] for tier in tiers]

    max_total = max((len(case_ids) for case_ids in totals.values()), default=1)
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    x_positions = list(range(len(tiers)))
    for backend in backends:
        linewidth = 2.2 if backend == "wfomc" else 1.2
        zorder = 4 if backend == "wfomc" else 2
        ax.plot(
            x_positions,
            values[backend],
            marker=BACKEND_MARKERS.get(backend, "o"),
            markersize=4.5,
            linewidth=linewidth,
            color=BACKEND_COLORS.get(backend),
            label=BACKEND_LABELS.get(backend, backend),
            zorder=zorder,
        )
        large_value = values[backend][-1]
        label_offset = {
            "wfomc": 5,
            "asp": 5,
            "essence": 1,
            "propositionalwfomc": -8,
            "coso": 5,
        }.get(backend, 4)
        ax.annotate(
            str(large_value),
            (x_positions[-1], large_value),
            xytext=(0, label_offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if label_offset >= 0 else "top",
            fontsize=6.5,
            color=BACKEND_COLORS.get(backend),
        )

    ax.set_xticks(x_positions, labels=tier_labels)
    ax.set_xlim(-0.12, len(tiers) - 0.88)
    ax.set_ylim(0, max_total + 9)
    ax.set_ylabel(f"Correctly solved (of {max_total})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        columnspacing=0.8,
        handlelength=1.7,
    )
    fig.tight_layout(pad=0.7)
    return fig


# ---------------------------------------------------------------------------
# Growing-domain runtime: one row, one panel per family, x = #entities
# ---------------------------------------------------------------------------

def plot_growing_runtime(
    rows: list[dict[str, str]],
    timeout_sec: float = 100.0,
) -> Figure:
    configure_matplotlib()
    n_panels = len(GROWING_PANELS)
    n_columns = 3
    n_rows = math.ceil(n_panels / n_columns)
    fig, axes_arr = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.4, 2.2 * n_rows),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_arr.flat)

    legend_handles: dict[str, plt.Line2D] = {}

    for ax, (family, subtitle) in zip(axes, GROWING_PANELS):
        fam_rows = [r for r in rows if r["case_id"].startswith(f"{family}_")]
        backends_here = sorted(
            {row["backend"] for row in fam_rows},
            key=lambda b: BACKEND_ORDER.index(b) if b in BACKEND_ORDER else 99,
        )
        for backend in backends_here:
            backend_rows = sorted(
                (r for r in fam_rows if r["backend"] == backend),
                key=lambda r: int(r["case_id"].split("_")[-1]),
            )
            xs: list[int] = []
            ys: list[float] = []
            fail_xs: list[int] = []
            for r in backend_rows:
                domain = int(r["case_id"].split("_")[-1])
                if r["status"] == "solved":
                    et = parse_float(r["elapsed_sec"])
                    if et is not None and et > 0:
                        xs.append(domain)
                        ys.append(et)
                else:
                    fail_xs.append(domain)
            (line,) = ax.plot(
                xs,
                ys,
                marker=BACKEND_MARKERS.get(backend, "o"),
                markersize=3.5,
                linewidth=1.2,
                color=BACKEND_COLORS.get(backend),
                label=BACKEND_LABELS.get(backend, backend),
            )
            legend_handles.setdefault(backend, line)
            if fail_xs:
                # Place crosses at the timeout value so they sit at the
                # ``upper rail'' of the runtime axis rather than on the floor.
                ax.scatter(
                    fail_xs,
                    [timeout_sec] * len(fail_xs),
                    marker="x",
                    s=18,
                    color=BACKEND_COLORS.get(backend),
                    linewidths=0.9,
                )
        ax.set_title(subtitle)
        ax.set_xlabel("Domain size n")
        ax.set_yscale("log")
        ax.set_ylim(top=timeout_sec * 1.6, bottom=0.05)
        ax.set_xlim(3, 102)
        ax.set_xticks([5, 25, 50, 75, 100])
        # Mark the timeout with a faint dashed horizontal rail.
        ax.axhline(
            y=timeout_sec, linestyle=":", linewidth=0.7, color="black", alpha=0.4,
        )
        ax.grid(True, which="both", axis="both", alpha=0.2)

    for ax in axes[n_panels:]:
        ax.axis("off")
    axes[0].set_ylabel("Runtime (s, log)")
    fig.legend(
        legend_handles.values(),
        [h.get_label() for h in legend_handles.values()],
        loc="lower right",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.985, 0.08),
    )
    fig.tight_layout(pad=0.8, rect=(0, 0.02, 1, 1))
    return fig


if __name__ == "__main__":
    main()
