# Size Inequality Negation Implementation Plan

**Goal:** Make negating a cardinality inequality preserve its logical meaning.

**Architecture:** Extend the existing comparison-complement map in the planning
pipeline. Keep equality negation as two disjoint integer ranges, with no changes
to parsing, encoding, or decoding.

**Tech Stack:** Python, pytest, SymPy, WFOMC.

## Design

Before this fix, `_negate_constraint` treated `!=` like `==`. Consequently,
choosing a subset `A` of a three-element set under `not (|A| != 1)` returned 5
instead of 3. The same problem affected false atoms in Shannon branches of
disjunctions.

Add `"!=": "=="` to the existing complement map. The right-hand side and all
linear terms must remain unchanged. Replacing all comparison handling with a
generic negation representation would require broader pipeline changes and is
unnecessary for this fix. Negating partwise constraints is outside this change.

## Implementation and verification

1. Extend `tests/test_planning_utilities.py` with comparison-complement tests,
   including preservation of coefficients and the existing equality split.
2. Extend `tests/test_backend_wfomc.py` with exact-count regressions for negated
   size comparisons, boundary sizes, Boolean combinations, and linear terms.
3. Run `uv run --locked pytest tests/test_planning_utilities.py
   tests/test_backend_wfomc.py -k 'size_constraint_negation or negated_size or
   size_inequality_boolean'`. Confirm that the new inequality cases fail.
4. In `src/cofola/planing/pipeline.py`, use the complement map
   `{"<": ">=", "<=": ">", ">": "<=", ">=": "<", "!=": "=="}`.
   Include `!=` in the documented comparators of `SizeConstraint`.
5. Rerun the focused tests, then `uv run --locked pytest` and
   `uv run --locked pyright`. Compare any typing failures with the base branch.
6. Commit and open one pull request against `clean-main`. Leave `experiments`
   and the paper unchanged.

## Verification results

- New tests before the fix had 8 failures and 12 passes. After the fix, all 20
  passed.
- The complete default test suite passed all 368 tests. The opt-in larger
  problem corpus was not run.
- Pyright reported 0 errors and 116 warnings. Its normalized diagnostics were
  identical to the `clean-main` baseline at `00aa36c`.
