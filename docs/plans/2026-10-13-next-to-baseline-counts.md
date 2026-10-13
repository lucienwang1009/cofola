# Next-to Baseline Counts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Count each matching adjacent position pair once in the ASP and Essence backends, including overlapping pattern arguments.

**Architecture:** Leave directional matching for existence and occurrence coverage unchanged. Normalize ASP count matches to increasing position pairs, and combine the two Essence endpoint tests with Boolean disjunction before converting a match to an integer.

**Tech Stack:** Python, pytest, Clingo, Essence/Conjure/Savile Row/Minion.

---

### Task 1: Synchronize the base

1. Fetch origin, fast-forward local `clean-main`, and preserve a backup of `experiments`.
2. Rebase `experiments` onto `clean-main`, migrating PR #30 regressions into the layered test layout.
3. Verify the migrated tests and update remote `experiments` using an explicit force-with-lease.

### Task 2: Reproduce the counting error

**Files:** Create `tests/backends/test_next_to_counts.py`.

1. Add shared end-to-end regressions for both direct backends. Use small explicit or brute-force expected counts for identical and overlapping groups, repeated entities, reversed matches, empty/one-position sequences, and multiple counted patterns.
2. Protect predecessor, existence, negation, and both coverage anchors, including simultaneous count and coverage constraints.
3. Add an always-on Essence encoding check that requires one Boolean disjunction per adjacent pair, independent of external tool availability.
4. Run the new tests before implementation and confirm the overlap regressions fail.

### Task 3: Fix only the count encodings

**Files:** Modify `src/cofola/backend/asp/encoder.py` and `src/cofola/backend/essence/encoder.py`.

1. For ASP next-to counts, derive one auxiliary atom with increasing endpoints from either directional match and aggregate those atoms.
2. For Essence next-to counts, generate `toInt(forward or reverse)` per successor pair, not two separate integer contributions.
3. Do not modify the match relations used by existence and coverage, or predecessor's directional semantics.

### Task 4: Verify and publish

1. Run `pytest tests/backends/test_next_to_counts.py` and both existing direct-backend suites.
2. Exercise the Essence regressions with real Conjure where installed, not just a mocked solver.
3. Run the full suite and type checking, comparing unrelated failures with the rebased base.
4. Commit the focused patch, push the fix branch, and open a PR with base `experiments`, recording validation results and any environment limitations.

## Validation Results

- Rebased base: `clean-main` at `6251ace`, `experiments` at `2a8d2cb`. The 20 migrated PR #30 test cases pass.
- Before the fix, the new local regressions produced 24 failures, 12 passes, and 34 skips because Conjure is not installed locally.
- The rebased base full suite passed with 590 passes and one optional Conjure skip.
- The fixed full local suite passed with 626 passes and 35 optional Conjure skips.
- On `school_server`, the new regressions and existing ASP/Essence suites passed all 100 tests, with no skips, using real Conjure 2.6, Savile Row, and Minion in an isolated temporary checkout. Existing server code and environments were left unchanged.
- Pyright reported zero errors and 110 warnings, identical to the rebased base with no added diagnostics.
- No benchmark measurements or paper sources were changed. The real-problem benchmark corpus was not rerun in full.
