# Cofola Benchmark and Baseline Runbook

This document is the operational guide for rerunning the Cofola experiments on
the server. It covers the benchmark suites, the baseline meanings, required
tools, entry points, recommended commands, result files, and common failure
diagnostics.

All commands below assume:

```bash
cd /home/sunshixin/lucien/cofola
```

## 1. Repository and Scripts

The benchmark code is concentrated under `scripts/benchmarks/`:

- `cases.py` builds benchmark cases.
- `run.py` is the single entry point for all benchmark backends.
- `coso_baselines.py` implements the ASP/Essence baselines locally. It uses the
  installed `coso` Python package for CoLa parsing and data structures, but it
  does not import or execute `coso/src/tester.py` from a CoSo source checkout.
- `plot.py` reads one or more `results.csv` files and generates figures.

The real benchmark data lives in:

- `problems/real/corpus.json`

Reusable generated benchmark files and manifests are written to:

- `problems/benchmarks/manifest.json`
- `problems/benchmarks/manifest.csv`
- `problems/benchmarks/<suite>/<case_id>.cfl`

Run outputs are written to an output directory, with these files:

- `metadata.json`: command settings and full case metadata.
- `results.csv`: one row per case/backend.
- `summary.csv`: aggregate counts per suite/backend.

## 2. Backends

The unified command is:

```bash
uv run python -m scripts.benchmarks.run ...
```

It supports these backend names:

- `wfomc`: Cofola compiled to lifted WFOMC, using `--algo fastv2` by default.
- `propositionalwfomc`: Cofola compiled to WFOMC, then grounded/counts via the
  WFOMC propositional backend and Ganak. This is the CNF-Ganak baseline.
- `propostionalwfomc`: accepted typo alias for `propositionalwfomc`; prefer the
  correct spelling in paper runs.
- `coso`: Cofola translated to CoLa and solved by CoSo.
- `asp`: Cofola translated through CoLa-style encodings to ASP and counted by
  Clingo.
- `essence`: Cofola translated through CoLa-style encodings to Essence and
  counted by Conjure/Savile Row/Minion.

There is no `sharpsat` backend in the paper run. The propositional baseline is
`propositionalwfomc`, i.e. CNF-Ganak through the WFOMC library.

## 3. Status Semantics

`results.csv` uses these status values:

- `solved`: the backend returned the expected answer.
- `solved_unchecked`: the backend returned an answer for a case with no
  independent expected answer.
- `unsolved`: used for ASP/Essence when the baseline returns a wrong answer,
  raises an encoding/runtime error, or times out. These baselines are treated as
  successful only when they solve a case correctly.
- `wrong`: used for `wfomc`, `propositionalwfomc`, and `coso` when the backend
  returns a value different from the expected answer.
- `error`: used for `wfomc`, `propositionalwfomc`, and `coso` when the backend
  raises an error.
- `timeout`: used for `wfomc`, `propositionalwfomc`, and `coso` when the
  per-case timeout expires.

For ASP/Essence, wrong answers and errors are intentionally collapsed to
`unsolved` because the paper should report those baselines as unable to solve
the instance rather than as producing trusted negative results.

## 4. Tool Setup

### Python Environment

The project uses `uv`. To restore the environment:

```bash
uv sync --extra coso --group dev
```

Quick import check:

```bash
uv run python - <<'PY'
import cofola, coso, clingo, wfomc
print("imports ok")
PY
```

### Java for Essence-Conjure

Essence uses Conjure and Savile Row, which require Java. The current server has
OpenLogic OpenJDK JRE 11 extracted at:

```bash
/home/sunshixin/lucien/tools/java/bin/java
```

Check it:

```bash
/home/sunshixin/lucien/tools/java/bin/java -version
```

Expected version family:

```text
openjdk version "11.0.30"
```

The runner prepends the parent directory of `--java-bin` to `PATH`, so you do
not need to export it manually if you pass or use the default:

```bash
--java-bin /home/sunshixin/lucien/tools/java/bin/java
```

### Conjure/Savile Row for Essence

The current Conjure bundle is installed at:

```bash
/home/sunshixin/lucien/CoSo/tools/conjure
```

Check it:

```bash
/home/sunshixin/lucien/CoSo/tools/conjure/conjure --version
```

The default benchmark runner path is already:

```bash
--conjure-dir /home/sunshixin/lucien/CoSo/tools/conjure
```

### Ganak for Propositional WFOMC

The CNF-Ganak baseline needs a `ganak` executable on `PATH`. Known locations on
this server include:

```bash
/home/sunshixin/software/ganak/ganak
/home/sunshixin/lucien/ganak/build/ganak
```

Before running `propositionalwfomc`, put one of them on `PATH`:

```bash
export PATH=/home/sunshixin/software/ganak:$PATH
which ganak
ganak --help | head
```

If `propositionalwfomc` reports `GanakError`, first verify that `ganak` is the
expected binary and that it runs outside Python.

## 5. Benchmark Suites

### Real Benchmark

The real suite is loaded from `problems/real/corpus.json`.

The dataset construction follows the MATH-dataset filtering process used for
the paper:

1. Start from the MATH `counting_and_statistics` category: 1245 problems.
2. Keep problems containing the "how many" keyword: 505 problems.
3. Remove problems that depend on figures, tables, diagrams, or other
   non-natural-language information: 454 problems remain.
4. Remove problems that are not combinatorial counting, such as direct numeric
   computation, Pascal-triangle questions, and geometry questions.
5. Encode the remaining suitable problems in Cofola where possible.
6. The loader skips any case tagged `unencodeable`, even when a `program` field
   is present.

Current loader count:

```bash
uv run python - <<'PY'
from scripts.benchmarks.cases import load_real_world_cases
print(len(load_real_world_cases()))
PY
```

At the time of writing, this prints `272`.

### Growing-Domain Benchmark

The growing suite is generated in `cases.py` and has five families. By default,
each family is regenerated at domain sizes `5, 10, ..., 100`, so the suite has
100 cases:

- `mathcounts_005` to `mathcounts_100`: bag subset/counting variant inspired by
  the MATHCOUNTS magnet problem.
- `fourletter_005` to `fourletter_100`: tuple/position/counting variant.
- `tvs_005` to `tvs_100`: defective TV subset problem, matching the CoLa/CoSo
  paper's growing-domain P3 pattern.
- `workers_005` to `workers_100`: worker composition problem, matching the
  CoLa/CoSo paper's growing-domain P4 pattern.
- `banana_005` to `banana_100`: bounded multiset word/tuple problem, matching
  the CoLa/CoSo paper's growing-domain P5 pattern.

The domain interval and upper bound are configurable:

```bash
--growing-min-domain 5 --growing-max-domain 100 --growing-domain-step 5
```

The runner uses monotone timeout propagation for growing-domain cases by
default. Once a backend times out for a family at domain size `n`, all larger
domains in that same backend/family are recorded as `timeout` without invoking
the solver. Pass `--no-skip-larger-growing-after-timeout` to disable this.

### Synthetic Benchmark

The `synthetic` suite loads 100 fixed materialized `.cfl` problems from:

```bash
problems/benchmarks/synthetic/manifest.json
```

Each case id includes the generation family, such as
`entitycount-12-operatorcount-3-constraintcount-2`, and tags record entity,
operator, constraint, and depth metadata. The older random CoLa-style generator
is still available in code as `old_synthetic_cases()`, but it is not used by
the evaluation entry point.

## 6. Recommended Run Commands

### Smoke Test

Run this first after logging into the server:

```bash
export PATH=/home/sunshixin/software/ganak:$PATH

uv run python -m scripts.benchmarks.run \
  --suite real \
  --ids 0 2 \
  --backends wfomc coso asp essence \
  --timeout 30 \
  --output-dir check-points/smoke-real \
  --no-save-benchmarks
```

Expected pattern:

- case `0` should solve for the main backends.
- case `2` may be `unsolved` for ASP/Essence.
- ASP/Essence wrong/error cases should appear as `unsolved`, not `wrong` or
  `error`.

Then smoke-test CNF-Ganak separately:

```bash
uv run python -m scripts.benchmarks.run \
  --suite real \
  --ids 0 \
  --backends propositionalwfomc \
  --timeout 100 \
  --output-dir check-points/smoke-propositionalwfomc \
  --no-save-benchmarks
```

If this fails with `GanakError`, fix the Ganak setup before starting the full
run.

### Save the Benchmark Manifest

Save the exact current real+growing benchmark set:

```bash
uv run python -m scripts.benchmarks.run \
  --suite real growing \
  --growing-min-domain 5 \
  --growing-max-domain 100 \
  --growing-domain-step 5 \
  --save-only \
  --benchmark-dir problems/benchmarks
```

This writes `problems/benchmarks/manifest.json`. Use this manifest for all
backend runs so every backend sees the exact same cases.

### Full Real+Growing Run

Recommended one-shot command:

```bash
export PATH=/home/sunshixin/software/ganak:$PATH

uv run python -m scripts.benchmarks.run \
  --benchmark-manifest problems/benchmarks/manifest.json \
  --backends wfomc propositionalwfomc coso asp essence \
  --timeout 100 \
  --output-dir check-points/paper-real-growing-timeout100
```

The runner writes `results.csv` after every case/backend row, so it is safe to
inspect progress while the run is active:

```bash
tail -n 20 check-points/paper-real-growing-timeout100/results.csv
cat check-points/paper-real-growing-timeout100/summary.csv
```

### Run Backends Separately

If one backend is slow or flaky, run separate output directories and merge only
at plotting time.

WFOMC/CoSo:

```bash
uv run python -m scripts.benchmarks.run \
  --benchmark-manifest problems/benchmarks/manifest.json \
  --backends wfomc coso \
  --timeout 100 \
  --output-dir check-points/paper-core-timeout100
```

CNF-Ganak:

```bash
export PATH=/home/sunshixin/software/ganak:$PATH

uv run python -m scripts.benchmarks.run \
  --benchmark-manifest problems/benchmarks/manifest.json \
  --backends propositionalwfomc \
  --timeout 100 \
  --output-dir check-points/paper-propositionalwfomc-timeout100
```

ASP/Essence:

```bash
uv run python -m scripts.benchmarks.run \
  --benchmark-manifest problems/benchmarks/manifest.json \
  --backends asp essence \
  --timeout 100 \
  --output-dir check-points/paper-coso-baselines-timeout100
```

Strict CoLa/CoSo growing-domain subset:

```bash
uv run python -m scripts.benchmarks.run \
  --suite growing \
  --ids tvs_005 tvs_010 tvs_015 tvs_020 tvs_025 tvs_030 tvs_035 tvs_040 tvs_045 tvs_050 \
        workers_005 workers_010 workers_015 workers_020 workers_025 workers_030 workers_035 workers_040 workers_045 workers_050 \
        banana_005 banana_010 banana_015 banana_020 banana_025 banana_030 banana_035 banana_040 banana_045 banana_050 \
  --backends wfomc propositionalwfomc coso asp essence \
  --timeout 100 \
  --output-dir check-points/paper-growing-coso-style-timeout100 \
  --no-save-benchmarks
```

## 7. Plotting

If all backends were run together:

```bash
uv run python -m scripts.benchmarks.plot \
  --results check-points/paper-real-growing-timeout100/results.csv \
  --output-dir check-points/paper-real-growing-timeout100/plots \
  --prefix paper \
  --formats pdf png
```

If backends were run separately:

```bash
uv run python -m scripts.benchmarks.plot \
  --results \
    check-points/paper-core-timeout100/results.csv \
    check-points/paper-propositionalwfomc-timeout100/results.csv \
    check-points/paper-coso-baselines-timeout100/results.csv \
  --output-dir check-points/paper-combined-plots-timeout100 \
  --prefix paper \
  --formats pdf png
```

Generated figures:

- `paper_outcomes.pdf/png`: stacked outcome counts by suite/backend.
- `paper_runtime_distribution.pdf/png`: solved-runtime distributions.
- `paper_growing_runtime.pdf/png`: growing-domain runtime curves.
- `paper_real_runtime_scatter.pdf/png`: real-problem solved runtimes by id.
- `paper_plot_summary.csv`: status counts used by the plots.

## 8. Reading Results

Useful summaries:

```bash
cat check-points/paper-real-growing-timeout100/summary.csv
```

Count statuses by backend:

```bash
uv run python - <<'PY'
import csv, collections
p = "check-points/paper-real-growing-timeout100/results.csv"
rows = list(csv.DictReader(open(p)))
for backend in sorted({r["backend"] for r in rows}):
    c = collections.Counter(r["status"] for r in rows if r["backend"] == backend)
    print(backend, dict(c))
PY
```

List unsolved ASP/Essence cases:

```bash
uv run python - <<'PY'
import csv
p = "check-points/paper-real-growing-timeout100/results.csv"
for r in csv.DictReader(open(p)):
    if r["backend"] in {"asp", "essence"} and r["status"] == "unsolved":
        print(r["suite"], r["case_id"], r["backend"], r["error_type"], r["error_message"])
PY
```

List wrong/error cases for the primary backends:

```bash
uv run python - <<'PY'
import csv
p = "check-points/paper-real-growing-timeout100/results.csv"
for r in csv.DictReader(open(p)):
    if r["backend"] in {"wfomc", "propositionalwfomc", "coso"} and r["status"] in {"wrong", "error", "timeout"}:
        print(r["suite"], r["case_id"], r["backend"], r["status"], r["error_type"], r["error_message"])
PY
```

## 9. Common Issues

### `ModuleNotFoundError`

Run:

```bash
uv sync --extra coso --group dev
```

Then rerun through `uv run ...`.

### `GanakError`

Check:

```bash
export PATH=/home/sunshixin/software/ganak:$PATH
which ganak
ganak --help | head
```

If Ganak exists but still fails, run a small `propositionalwfomc` smoke case and
inspect `results.csv` for the exact error message.

### Essence reports missing Java

Check:

```bash
/home/sunshixin/lucien/tools/java/bin/java -version
```

Then pass:

```bash
--java-bin /home/sunshixin/lucien/tools/java/bin/java
```

### Leftover Conjure/Java/Minion Processes

The runner tries to terminate timed-out child processes. If a solver survives a
timeout, inspect:

```bash
pgrep -af "conjure|savilerow|java -ea|minion"
```

Terminate only the stale experiment processes if needed:

```bash
pkill -f "/home/sunshixin/lucien/CoSo/tools/conjure"
pkill -f "savilerow.jar"
```

### Synthetic Results Accidentally Included

For the current paper, use:

```bash
--suite real growing
```

or load a manifest created from `--suite real growing`. Avoid `--suite all`
until the synthetic benchmark is finalized.

## 10. Minimal Reproducibility Checklist

Before starting a full run:

```bash
cd /home/sunshixin/lucien/cofola
uv sync --extra coso --group dev
export PATH=/home/sunshixin/software/ganak:$PATH
/home/sunshixin/lucien/tools/java/bin/java -version
which ganak
uv run pytest tests/test_benchmark_run.py tests/test_benchmark_cases.py -q
```

Then run:

```bash
uv run python -m scripts.benchmarks.run \
  --suite real growing \
  --growing-min-domain 5 \
  --growing-max-domain 100 \
  --growing-domain-step 5 \
  --backends wfomc propositionalwfomc coso asp essence \
  --timeout 100 \
  --output-dir check-points/paper-real-growing-timeout100
```

Use the corresponding `results.csv`, `summary.csv`, and generated plot files in
the paper.
