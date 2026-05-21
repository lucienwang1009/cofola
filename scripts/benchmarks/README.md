# Benchmark Scripts

Reproducible runners comparing Cofola's WFOMC backend with the CoSo backend.

## Suites

- `real` — every problem in `problems/real/corpus.json`.
- `growing` — parameterised scaling families (`mathcounts`, `fourletter`,
  `tvs`, `workers`, `banana`), each with variants of increasing domain size.
- `synthetic` — deterministic random CoLa-style cases covering plain subsets
  (`ss`), set choose-with-replacement (`sr`), bag multisubsets (`ms`), set
  permutations (`pm`), small bounded bag permutations (`bm`), repeated
  sequences (`sq`), partitions (`pt`), and compositions (`cp`).
- `all` — all of the above.

## Run

CoSo only:

```bash
uv run python -m scripts.benchmarks.run --suite all --backends coso --timeout 300
```

Compare both backends:

```bash
uv run python -m scripts.benchmarks.run --suite all --backends wfomc coso --timeout 300
```

Results are written to `check-points/coso/` by default:

- `metadata.json` — generated benchmark cases and CLI settings.
- `results.csv` — one row per case/backend: solved, wrong, error, or timeout.
- `summary.csv` — aggregate counts and solved-runtime statistics.

The runner records unsupported encodings and backend errors rather than
filtering them out, so the supported fragment stays visible.

## Benchmark persistence

Each run writes its cases as `.cfl` files under `problems/benchmarks/` plus
`manifest.json`/`manifest.csv` (case id, suite, expected answer, tags, source,
relative program path, program SHA-256). Rerun the exact saved set later by
loading the manifest:

```bash
uv run python -m scripts.benchmarks.run \
  --benchmark-manifest problems/benchmarks/manifest.json \
  --backends wfomc coso --output-dir check-points/rerun
```

Materialize the benchmark set without running any solver:

```bash
uv run python -m scripts.benchmarks.run --suite all --save-only
```

Use `--no-save-benchmarks` when you only want result CSVs.

The synthetic suite has no independent expected answers. To treat a backend as
the reference solver:

```bash
uv run python -m scripts.benchmarks.run \
  --suite synthetic --backends coso --trust-unchecked-backends coso
```

## Plots

```bash
uv run python -m scripts.benchmarks.plot \
  --results check-points/coso/results.csv check-points/wfomc/results.csv \
  --output-dir check-points/plots
```

The plotting script writes:

- `benchmark_outcomes.pdf` — stacked outcome counts by suite/backend.
- `benchmark_runtime_distribution.pdf` — solved-runtime distributions (log).
- `benchmark_growing_runtime.pdf` — growing-domain runtimes.
- `benchmark_real_runtime_scatter.pdf` — solved real-problem runtimes by id.

All figures are vector PDFs with embedded TrueType fonts (`pdf.fonttype = 42`)
for inclusion in LaTeX manuscripts.
