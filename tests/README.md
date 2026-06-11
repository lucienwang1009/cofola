# Test Layout

The suite is organized by the layer under test. Use this map when adding a
regression so the intent stays visible.

## Frontend And Validation

- `test_parser_errors.py`: parser diagnostics and grammar edge cases.
- `test_type_check.py`: frontend type-checking failures.
- `test_problem_builder_usage.py`: public builder API and hand-built problem
  validation.

## Planning

- `test_pass_infrastructure.py`: analysis manager, fixed-point passes, and pass
  invalidation behavior.
- `test_planning_utilities.py`: planner helper utilities, analysis inference,
  optimization passes, lowering passes, and cross-pass invariants.

## Backends

- `test_backend_wfomc.py`: WFOMC backend profile, object semantics, sequence
  pattern semantics, encoding invariants, and backend error boundaries.
- `test_backend_coso.py`: CoSo backend encoding, routing, supported cases, and
  fallback behavior.

## Benchmarks And Corpus

- `test_benchmark_cases.py`: generated benchmark case discovery and manifest
  round trips.
- `test_benchmark_run.py`: benchmark runner behavior and result classification.
- `test_all_problems.py`: corpus-level parse/type/solve coverage for encodable
  `.cfl` problems.

## Useful Commands

```bash
uv run pytest tests/test_backend_wfomc.py tests/test_planning_utilities.py -q
uv run pytest tests/test_parser_errors.py tests/test_type_check.py -q
uv run pytest tests/test_benchmark_cases.py tests/test_all_problems.py -q
uv run pytest
```
