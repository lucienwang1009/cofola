# Benchmark Scripts

Reproducible runners comparing Cofola's WFOMC backend, propositional WFOMC,
CoSo, and CoSo's ASP/Essence baseline encodings.

## Setup on a clean server

How to provision a fresh Linux server with nothing preinstalled so it can run
the full benchmark suite, including the optional CoSo and Essence baselines.
Every tool location is configurable — there are no hardcoded paths.

### 1. System packages

WFOMC has native dependencies (FLINT/GMP/MPFR for `python-flint`, plus a C/C++
toolchain and CMake for Ganak and `pynauty`). On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl wget unzip build-essential cmake \
  libgmp-dev libmpfr-dev libflint-dev
```

On other distros install the equivalents: git, curl, wget, unzip, a C/C++
compiler, make, cmake, and the GMP/MPFR/FLINT development headers.

### 2. uv and the Python environment

```bash
# install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL"                      # reload PATH so `uv` is found

git clone <cofola-repo-url> cofola && cd cofola

# core environment (WFOMC + propositional backend):
uv sync

# full environment incl. the CoSo baselines (clingo/portion/coso) and dev tools:
uv sync --extra coso --group dev
```

`uv` reads `.python-version` (3.11) and provisions the interpreter itself.
Verify the imports:

```bash
uv run python -c "import cofola, wfomc; print('core ok')"
uv run python -c "import coso, clingo, portion; print('coso extra ok')"  # after --extra coso
```

### 3. Ganak (the `propositionalwfomc` backend)

`propositionalwfomc` (WFOMC with `--algo=propositional`) shells out to the
`ganak` model counter. Build and install the pinned Ganak with WFOMC's helper
(exposed on the venv `PATH`):

```bash
uv run wfomc-install-ganak                  # installs into .venv/bin
# or pick a location: uv run wfomc-install-ganak --install-dir "$HOME/bin"
```

The helper needs the toolchain from step 1 (git, cmake, C++, gmp, mpfr, flint).
Confirm it is found at runtime:

```bash
uv run which ganak && uv run ganak --help | head -1
# if installed outside the venv: export PATH="$HOME/bin:$PATH"
```

Skip this step if you only run `--backends wfomc coso`.

### 4. CoSo baselines: `asp` and `essence` (optional)

Both baselines come with `uv sync --extra coso` (step 2), which installs the
`coso` package together with `clingo` and `portion`.

- **`asp`** needs nothing further: Cofola translates each problem to ASP and
  counts models with the Python `clingo` package.
- **`essence`** additionally runs Conjure plus Savile Row/Minion, so it needs a
  Java 11 runtime plus the Conjure tools.

Install Java and Conjure for `essence`. The versions below match CoSo's
[`install_tools.sh`](https://github.com/PietroTotis/CoSo/blob/master/install_tools.sh),
which Cofola's Essence translator targets:

```bash
# any JDK/JRE 11 works; e.g. the distro package:
sudo apt-get install -y openjdk-11-jre-headless
java -version                                # expect 11.x

# Conjure v2.3.0 plus bundled solvers (Savile Row + Minion).
# The linux-solvers zip contains Savile Row/Minion but not the `conjure`
# executable, so install both archives and expose them through one directory.
mkdir -p "$HOME/tools" && cd "$HOME/tools"
wget -q https://github.com/conjure-cp/conjure/releases/download/v2.3.0/conjure-v2.3.0-linux.zip
wget -q https://github.com/conjure-cp/conjure/releases/download/v2.3.0/conjure-v2.3.0-linux-solvers.zip
unzip -q conjure-v2.3.0-linux.zip            # -> conjure-v2.3.0-linux/
unzip -q conjure-v2.3.0-linux-solvers.zip    # -> conjure-v2.3.0-linux-solvers/

mkdir -p conjure-v2.3.0-combined
ln -sfn "$HOME/tools/conjure-v2.3.0-linux/conjure" conjure-v2.3.0-combined/conjure
for f in savilerow savilerow.jar minion glucose glucose-syrup \
         bc_minisat_all_release nbc_minisat_all_release lingeling \
         fzn-chuffed fzn-gecode open-wbo lib; do
  ln -sfn "$HOME/tools/conjure-v2.3.0-linux-solvers/$f" "conjure-v2.3.0-combined/$f"
done

"$HOME/tools/conjure-v2.3.0-combined/conjure" --version
cd -
```

`--conjure-dir` must point at the directory holding the `conjure` binary (it is
prepended to `PATH` so the Savile Row and Minion binaries next to it are found):

```bash
--conjure-dir "$HOME/tools/conjure-v2.3.0-combined"   # required for essence
--java-bin /usr/bin/java                              # optional; otherwise java from PATH
```

> CoSo's `install_tools.sh` also installs sharpSAT, gringo, and the lp2*
> normalisers for CoSo's own pipeline; Cofola's `asp`/`essence` baselines do not
> use those — only `clingo` (for `asp`) and Conjure + Java (for `essence`).

### 5. Smoke test

```bash
uv run pytest tests/benchmarks/test_benchmark_cases.py tests/benchmarks/test_benchmark_run.py -q
uv run python -m scripts.benchmarks.run --suite real --backends wfomc --timeout 30
```

Add `propositionalwfomc` once Ganak is on `PATH`, and `coso asp essence` once
the CoSo extra (and, for `essence`, Java + Conjure) are installed.

### Troubleshooting

- `ModuleNotFoundError` for `coso`/`clingo`/`portion`: rerun
  `uv sync --extra coso --group dev` and invoke through `uv run`.
- `GanakError`: ensure `ganak` is on `PATH` (`uv run which ganak`) and runs
  standalone (`ganak --help`).
- Essence reports missing Java: pass `--java-bin /path/to/java`, or put `java`
  on `PATH`; pass `--conjure-dir /path/to/conjure`.

## Suites

- `real` — every problem in `problems/real/corpus.json`.
- `growing` — parameterised scaling families (`mathcounts`, `fourletter`,
  `tvs`, `workers`, `banana`), generated by default at domain sizes
  `5, 10, ..., 100`.
- `synthetic` — 100 fixed synthetic `.cfl` problems saved under
  `problems/benchmarks/synthetic/`. Case ids include the generation
  family, for example
  `entitycount-12-operatorcount-3-constraintcount-2-...`.
- `all` — all of the above.

## Run

CoSo only:

```bash
uv run python -m scripts.benchmarks.run --suite real growing --backends coso --timeout 100
```

Compare the main backends:

```bash
uv run python -m scripts.benchmarks.run \
  --suite real growing \
  --backends wfomc propositionalwfomc coso asp essence \
  --timeout 100
```

`propositionalwfomc` runs the WFOMC backend with `--algo=propositional`.
The external CoSo baselines are `asp` and `essence`; SharpSAT is not part of
this runner. ASP/Essence wrong answers and backend errors are reported as
`unsolved`, since these baselines should only count cases they solve correctly.
Essence needs Conjure/Savile Row and Java; their locations are
environment-specific and have no built-in default, so point the runner at them
with `--conjure-dir` (required for the `essence` baseline) and `--java-bin`
(optional; otherwise `java` is taken from `PATH`).

For growing-domain cases, the runner uses monotone timeout propagation by
default: after a backend times out for one family at domain size `n`, larger
domains for the same backend/family are recorded as `timeout` without running
the solver. Use `--no-skip-larger-growing-after-timeout` to disable this.

Results are written to `check-points/coso/` by default:

- `metadata.json` — generated benchmark cases and CLI settings.
- `results.csv` — one row per case/backend: solved, solved_unchecked, unsolved,
  wrong, error, or timeout.
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
  --backends wfomc propositionalwfomc coso asp essence \
  --output-dir check-points/rerun
```

Materialize the benchmark set without running any solver:

```bash
uv run python -m scripts.benchmarks.run \
  --suite real growing \
  --growing-min-domain 5 \
  --growing-max-domain 100 \
  --growing-domain-step 5 \
  --save-only
```

Use `--no-save-benchmarks` when you only want result CSVs.

The synthetic benchmark set is already materialized under
`problems/benchmarks/synthetic/`. Rerun the exact saved synthetic set with:

```bash
uv run python -m scripts.benchmarks.run \
  --benchmark-manifest problems/benchmarks/synthetic/manifest.json \
  --backends wfomc propositionalwfomc coso asp essence \
  --timeout 100 \
  --output-dir check-points/paper-synthetic-timeout100
```

The previous random CoLa-style synthetic generator is retained in
`old_synthetic_cases()` for future use, but `--suite synthetic` now loads the
fixed materialized synthetic manifest.

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
