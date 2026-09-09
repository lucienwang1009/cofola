# Bag-lifting ablation — 2026-09-08

## Final results

All 831 instances completed their off/on comparison: 1,776 solver invocations
including the protocol's rechecks. A separate historical comparison added
174 invocations (29 candidates × 3 repetitions × 2 modes), for 1,950 total.

| Suite | Off correct | On correct | Off mean on solved (s) | On mean on solved (s) |
| --- | ---: | ---: | ---: | ---: |
| Real | 246/247 | 246/247 | 0.911946 | 0.914784 |
| Synthetic | 384/384 | 384/384 | 1.183455 | 1.187507 |
| Growing | 160/160 | 160/160 | 0.894033 | 0.895852 |
| Controlled bags | 23/40 | 38/40 | 11.467366 | 0.607510 |

The controlled-suite means are on different solved subsets; the correct
paired speedup is the geometric mean on the 23 common solved cases: **3.20×**.
Controlled PAR-2 falls from 91.59 to 10.58 s. All 32 supported-family cases
solve with lifting in at most 1.303256 s, versus 18/32 without lifting.
The two remaining enabled failures are the large unsupported-union controls
(n=8 and n=16); both modes fail there. Failure types are MemoryError or native
FLINT allocation failures (worker exit -6), not wrong answers or timeouts.
No precise speedup is assigned to a failed run.

Examples on jointly solved cases:

- Observed chain n=8, m=2: 33.830407 → 0.471091 s (71.81×).
- Observed chain n=4, m=8: 95.639895 → 0.500687 s (191.02×).
- Unobserved intermediate n=4, m=8: 86.470641 → 0.416032 s (207.85×).

No wrong answer, unstable repeated outcome, or persistent on/off regression
was found. Real case 308 is the unchanged timeout. Real cases 142 and 265
triggered initial slowdown checks, but their three-pair medians showed no
slowdown. The 17 controlled failure cases were also repeated three times.

Compared separately with the 2026-09-01 same-server results, the original
791 cases have identical answer/solved status. All 29 first-run timing
candidates were repeated three times with both modes; none remained slower
under the declared threshold (the worst new lifting-off median was only 1.0243× its
historical runtime). These later measurements do not replace values in the
primary ablation table. See `analysis/historical-summary.json`.

Verification: 525 tests passed, 1 skipped; Pyright 0 errors, 110 existing
warnings. Only experiment scripts/data/tests changed in Cofola; clean-main
and all solver sources remain untouched.

## Scope and provenance

The merged bag-lifting implementation (clean-main `507b7f4`) is tested through
experiments `b7a59f59bf0ddbf64cbaa63ad910a3bf5ef33192`. No solver source was
modified for this study. The Git tree of `src` is
`b8b11e5072675ac7bc3a073d58baa323858cbd1a`.
Changes are experiment-only: forward `lifted_bags` through the normal runner,
clean up Ganak subprocesses on POSIX timeouts, run paired trials, generate a
controlled suite with independent answers, and analyze/plot the measurements.

All timing measurements were taken on `school_server` / `server29`, in the
isolated snapshot `/home/sunshixin/lucien/cofola-bag-ablation-20260908`.
The old `/home/sunshixin/lucien/cofola` checkout and its environment were not
modified. The existing environment at
`/home/sunshixin/lucien/cofola-lifted-bags-test-20260903/.venv` was reused, with
`PYTHONPATH=src` pointing at the new snapshot.

- Hardware: Intel Xeon Gold 5218 at 2.30 GHz; 4 sockets, 64 physical cores,
  128 logical CPUs, 503 GiB host memory.
- Python 3.11.14; WFOMC `e736680b5d7afe073cd10838a666184902e368dc`;
  Ganak `82a1d1fb6f0d6fb4a46b825f84b29567728ae483`.
- SymPy 1.14.0, python-flint 0.8.0, NumPy 2.4.3, SciPy 1.17.1.
- Ganak binary SHA-256:
  `9634bf1f3ded8d35c2e6454624a828fdd55c5e0147613190eca143088229e23e`.

Each shard also records versions and source/manifest hashes in `metadata.json`.

## Protocol

The original manifest contains 791 instances: 247 real, 384 synthetic, 160
growing. The additional manifest has 40 controlled bag instances and does not
replace any original cases. All expected answers for the new cases are
computed by integer DP over entity multiplicity vectors, independent of the
solver. Six tiny cases are also checked against hand counts and both modes.

Each instance initially runs once per mode (off/on). Mode order alternates
between cases and between repetitions. All runs use the normal `run_case`
worker, default outer algorithm `fastv2` (including its normal automatic
handling of ordered problems), and a 100 s wall-clock limit. Larger cases
are never skipped after a smaller timeout. Timing includes the full solve
pipeline, including local Ganak counting on applicable lifted cases. It does
not include generating the independent reference answers. No encoding sizes
are collected.

Recheck rule declared in the driver before launch: a slowdown of at least
20% **and** 0.1 s, lost solution, answer mismatch, wrong answer, or unexpected
error triggers three additional paired repetitions. Raw initial runs remain
in the data. Rechecked cases are summarized by the median of the three new
runtimes per mode, not by a best-of run. Mixed statuses are reported as
unstable failures. This is not a statistical significance test.

The original suites used four shards pinned to physical cores 16, 20, 24,
28. The controlled suite used cores 17, 21, 25, 29, with a 16 GiB per-process
virtual-memory safety limit applied to both modes. Up to eight independent
instance pairs ran concurrently; each pair itself was sequential on one core.
`PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` were set throughout. These are shared-server measurements,
not measurements on an otherwise dedicated host.

## Files and reproduction

- `paper/shard-*/`: original-suite raw runs, metadata, completion markers.
- `controlled/shard-*/`: controlled-suite raw runs and metadata.
- `analysis/all-runs.csv`: all initial and repeated trials, unfiltered.
- `analysis/paired.csv`: per-case results after the declared aggregation.
- `analysis/summary.json`: coverage, runtimes, PAR-2, common-solved speedups,
  rechecks, and detected regressions.
- `historical/shard-*/` and `analysis/historical-*.{csv,json}`: the separate
  three-pair historical rechecks and their frozen baseline.
- `provenance/bag_ablation-paper.py`: exact driver used for the original
  suite, before the optional memory-limit CLI argument was added for the
  controlled runs. The reconstructed full source hash was verified against
  its metadata: `05c936cef57a953308e82f51a6c231cf557053b3b17c8e2f4d2dbf2003d1c41f`.
- `provenance/bag_ablation-controlled.py`: exact controlled-suite driver;
  its full source hash was also verified against metadata:
  `7b2a3e259c1db90eb412d575f052ff59652f29799e4c48654787b5719862611e`.
- `provenance/experiment-source.tar.gz`: source, runners, generators, manifests,
  problem files, relevant tests, and dependency lockfile. Restore the matching
  archived driver to reproduce an original full-source hash. For the paper
  run, omit the later `bag_lifting_cases.py` and `analyze_bag_ablation.py` from
  the hash input; for the controlled run, omit only `analyze_bag_ablation.py`.
  The current driver additionally supports the separate `--repeat-only`
  historical audit.

The paper's Section 10.2.4, Figure 10, Table 11, and Appendix E were updated.
The PDF compiles successfully; the updated experiment/appendix pages were
rendered and visually inspected. No new overfull boxes, unresolved references,
or multiply defined labels were introduced. The ten pre-existing overfull
boxes elsewhere in the manuscript remain outside this task's scope.

Commands to regenerate the suite, run all four shards, and analyze/plot the
results are in `scripts/benchmarks/README.md` under "Paired bag-lifting
ablation". The paper keeps the earlier five-backend comparison separate;
this study reruns the two WFOMC configurations, not the four unchanged
competing backends. The new off/on table must not be silently substituted
into the older comparison plots.
