# Development Guide

This guide covers the commands and test layout used by Cofola contributors.

## Commands

```bash
uv sync
uv run pytest
COFOLA_ALL_TESTS=1 uv run pytest tests/benchmarks/test_all_problems.py
uv run pyright
```

`uv run pyright` requires the `pyright` executable to be available in the
environment. The current `dev` dependency group contains `pytest`; install or
add Pyright separately when type checking locally.

## Test Layout

Tests are grouped by the layer under test (see `tests/README.md` for the full
map):

- `tests/frontend/` — parser diagnostics (`test_parser_errors.py`),
  type-checking failures (`test_type_check.py`), and public builder-API
  examples (`test_problem_builder_usage.py`).
- `tests/planning/` — pass infrastructure (`test_pass_infrastructure.py`) plus
  focused modules for reference utilities, analysis inference, optimization
  passes, lowering passes, analysis boundaries, and transform invariants.
- `tests/backends/` — `wfomc/` (profile, collection semantics, sequence
  patterns, choice/membership, encoding boundaries) and `test_backend_coso.py`.
- `tests/benchmarks/` — benchmark case discovery and runner behavior, plus
  `test_all_problems.py`, which runs representative `.cfl` examples from
  `problems/real/corpus.json` (examples tagged `benchmark` by default; set
  `COFOLA_ALL_TESTS=1` for the larger dataset).

Shared `Problem`-navigation helpers live in `tests/helpers.py`.

When adding a feature, prefer one small parser/type-check test and one semantic
solver test. Add planning or backend tests only when the behavior depends on a
specific internal transformation or encoding invariant.

## Architecture

```text
parser      .cfl text -> frontend Problem
frontend    immutable problem model, refs, constraints, validation
planing     analyses, simplification, lowering, decomposition, solve schedule
backend     WFOMC encoding, solving, decoding
```

The frontend `Problem` is the shared model. The planning layer transforms it but
does not introduce a separate IR type hierarchy.

## Backend Notes

The WFOMC backend expects lowered, component-sized problems. Important encoding
rules:

- bag union `+` is max multiplicity per entity
- additive bag union `++` is sum multiplicity per entity
- bag count atoms should resolve through the shared bag multiplicity helper
- tuple index constraints should never reach the backend
- sequence relation count predicates should be generated through the shared
  relation-count helper

Backend regressions should include a small source program and expected integer
answer whenever possible.
