"""Paper figures and tables for the Cofola experiments section.

This script generates the figures used in the *Experiments* section after the
v3 revision and prints the LaTeX-ready numbers that populate the per-object-type
result tables.

Figures (default prefix ``experiment``):

* ``experiment_real_cactus.pdf``     -- cactus plot on the real suite
  (``x``: runtime in seconds, log scale; ``y``: number of real instances
  solved).
* ``experiment_synthetic_cactus.pdf`` -- analogous cactus plot on the
  synthetic suite.
* ``experiment_growing_family_runtime.pdf`` -- single-row, one-panel-per-family
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
        --real-results check-points/paper-real-growing-timeout100/results.csv \
        --synthetic-results check-points/paper-synthetic-timeout100/results.csv \
        --growing-results check-points/paper-growing-domain5-100-timeout100/results.csv \
        --synthetic-dir problems/benchmarks/synthetic \
        --output-dir ../comb_aij/figs/experiments \
        --prefix experiment \
        --timeout-sec 100 \
        --print-tables
"""
from __future__ import annotations

import argparse
import csv
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

OBJECT_TYPES = (
    "set",
    "bag",
    "tuple",
    "sequence",
    "circle",
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-results", type=Path, required=True,
                        help="results.csv covering the real + growing suite.")
    parser.add_argument("--synthetic-results", type=Path, required=True,
                        help="results.csv covering the synthetic suite.")
    parser.add_argument("--growing-results", type=Path, required=True,
                        help="results.csv covering the 5..100 growing suite.")
    parser.add_argument("--synthetic-dir", type=Path, required=True,
                        help="Directory containing synthetic .cfl files used "
                             "to infer object types per instance.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="experiment")
    parser.add_argument("--formats", nargs="+", default=("pdf",),
                        choices=("pdf", "png", "svg"))
    parser.add_argument("--timeout-sec", type=float, default=100.0,
                        help="Per-instance timeout in seconds; failed runs are "
                             "drawn at this y-value on the growing-domain plot.")
    parser.add_argument("--print-tables", action="store_true",
                        help="Print LaTeX-ready rows for the per-object-type "
                             "tables of the real and synthetic suites.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_rows = [r for r in load_rows(args.real_results) if r["suite"] == "real"]
    synth_rows = [r for r in load_rows(args.synthetic_results) if r["suite"] == "synthetic"]
    growing_rows = [r for r in load_rows(args.growing_results) if r["suite"] == "growing"]

    real_types = real_object_types(real_rows)
    synth_types = synthetic_object_types(args.synthetic_dir)

    figs = {
        "real_cactus": plot_cactus(real_rows, suite_label="real"),
        "synthetic_cactus": plot_cactus(synth_rows, suite_label="synthetic"),
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
    "circle": (r"\bcircle\s*\(", r"\bchoose_circle\s*\(",
               r"\bchoose_replace_circle\s*\("),
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

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
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
        ax.plot(
            runtimes,
            range(1, len(runtimes) + 1),
            marker=BACKEND_MARKERS.get(backend, "o"),
            markersize=3.0,
            linewidth=1.2,
            label=f"{BACKEND_LABELS.get(backend, backend)} ({len(runtimes)})",
            color=BACKEND_COLORS.get(backend),
        )
    ax.set_xscale("log")
    ax.set_xlabel(f"Runtime on solved {suite_label} instances (s, log)")
    ax.set_ylabel(f"{suite_label.capitalize()} instances solved")
    ax.grid(True, which="both", axis="x", alpha=0.25)
    # Cactus curves rise monotonically from the bottom-left, so the upper-left
    # corner is the empty quadrant; put the legend there to avoid occluding
    # the data.
    ax.legend(frameon=False, loc="upper left", ncol=1)
    fig.tight_layout(pad=0.8)
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
    fig, axes_arr = plt.subplots(
        1, n_panels, figsize=(11.5, 2.4), sharey=True, squeeze=False,
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
        ax.set_xlabel("# entities")
        ax.set_yscale("log")
        ax.set_ylim(top=timeout_sec * 1.6, bottom=0.05)
        # Mark the timeout with a faint dashed horizontal rail.
        ax.axhline(
            y=timeout_sec, linestyle=":", linewidth=0.7, color="black", alpha=0.4,
        )
        ax.grid(True, which="both", axis="y", alpha=0.25)

    axes[0].set_ylabel("Runtime (s, log)")
    fig.legend(
        legend_handles.values(),
        [h.get_label() for h in legend_handles.values()],
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(pad=0.8, rect=(0, 0.04, 1, 1))
    return fig


if __name__ == "__main__":
    main()
