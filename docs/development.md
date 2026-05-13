# Development Guide

This guide covers the commands and test layout used by Cofola contributors.

## Commands

```bash
uv sync
uv run pytest
COFOLA_ALL_TESTS=1 uv run pytest tests/test_all_problems.py
uv run pyright
```

`uv run pyright` requires the `pyright` executable to be available in the
environment. The current `dev` dependency group contains `pytest`; install or
add Pyright separately when type checking locally.

## Test Layout

- `tests/test_all_problems.py` runs representative `.cfl` examples from
  `problems/all.json`. By default it runs examples tagged `benchmark`; set
  `COFOLA_ALL_TESTS=1` for the larger dataset.
- `tests/test_type_check.py` covers invalid user programs that should fail
  validation before solving.
- `tests/test_parser_errors.py` covers parser and transformer diagnostics.
- `tests/test_problem_builder_usage.py` contains executable examples for the
  public Python builder API.
- `tests/test_pass_infrastructure.py` covers `AnalysisManager`,
  `RefAllocator`, and fixed-point pass behavior.
- `tests/test_planning_utilities.py` covers planning analyses and lowering
  policies.
- `tests/test_backend_wfomc.py` covers WFOMC backend boundary behavior and
  semantic regressions.

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
