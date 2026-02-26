# Cofola Package Refactor Design

**Date:** 2026-02-26
**Goal:** Comprehensive refactor — file size compliance (200–400 lines), code quality, and architectural improvements (pipeline design).

---

## Motivation

| Issue | Location | Detail |
|---|---|---|
| Oversized file | `problem.py` | 898 lines — CofolaProblem class + 10+ standalone functions |
| Oversized file | `parser/parser.py` | 703 lines — transformer class + parse() + __main__ test |
| Oversized file | `objects/bag.py` | 534 lines |
| Oversized file | `objects/sequence.py` | 447 lines |
| Misplaced function | `solver.py` | `decompose_problem` belongs with the problem model |
| Dead code | `problem.py` | `decompose()` stub (body is just `pass`) |
| Dead code | `problem.py` | Unreachable code block in `transform_tuples` after `return True` (lines 531–577) |
| Dead code | `parser/parser.py` | 6 commented-out methods (lines 617–650) + `__main__` block |
| Dead code | `parser/bak/` | 6 old backup files never used |
| Global mutable state | `objects/utils.py` | `AUX_COUNTER` module-level int |
| Global mutable state | `utils.py` | `AUX_PRED_CNT` module-level int |
| No pipeline clarity | `solver.py` | `solve_single_problem` is an imperative list of steps with no structure |
| Inconsistent pass API | various | `optimize()` returns None, `simplify()` returns new problem, `transform()` returns same problem |
| Unused import | `problem.py` | `from os import replace` |
| Typo | `problem.py`, `encoder.py` | `propogate` → `propagate` |

---

## Target File Structure

```
src/cofola/
├── solver.py                    # solve(), main() — ~130 lines (was 245)
├── pipeline.py                  # NEW: SolvePipeline, solve_single_problem — ~120 lines
├── problem.py                   # CofolaProblem class only — ~290 lines (was 898)
├── passes.py                    # NEW: sanity_check, fold_constants, optimize, simplify,
│                                #      workaround, infer_max_size, decompose_problem — ~380 lines
├── transforms.py                # NEW: transform_*, transform_once, transform — ~320 lines
├── encoder.py                   # unchanged
├── decoder.py                   # unchanged
├── wfomc_solver.py              # unchanged
├── context.py                   # unchanged
├── utils.py                     # AUX_PRED_CNT → _AuxPredCounter class
└── objects/
│   ├── base.py                  # unchanged
│   ├── bag.py                   # BagInit, BagChoose, SizeConstraint, BagMultiplicity — ~200 lines (was 534)
│   ├── bag_ops.py               # NEW: BagUnion/Intersection/Difference/AdditiveUnion/Support + bag constraints — ~250 lines
│   ├── set.py                   # unchanged
│   ├── sequence.py              # SequenceImpl, SequenceConstraint, SequencePattern base — ~200 lines (was 447)
│   ├── sequence_patterns.py     # NEW: TogetherPattern, LessThanPattern, NextToPattern, etc. — ~250 lines
│   ├── function.py              # unchanged
│   ├── tuple.py                 # unchanged
│   ├── partition.py             # unchanged
│   └── utils.py                 # AUX_COUNTER → _AuxCounter class
└── parser/
    ├── grammar.py               # unchanged
    ├── common.py                # unchanged
    ├── transformer.py           # NEW: CofolaTransformer class (core + dispatch) — ~220 lines
    ├── transformer_objects.py   # NEW: object construction methods — ~280 lines
    ├── transformer_constraints.py  # NEW: constraint construction methods — ~200 lines
    └── parser.py                # slimmed: constants + exceptions + parse() — ~50 lines (was 703)
```

---

## SolvePipeline Class

New `pipeline.py` replaces the imperative `solve_single_problem` function with a class that makes the transformation pipeline explicit.

```python
class SolvePipeline:
    def __init__(self, wfomc_algo: Algo,
                 use_partition_constraint: bool,
                 lifted: bool) -> None: ...

    def run(self, problem: CofolaProblem) -> int:
        """Top-level entry: build, simplify, decompose, then solve each sub-problem."""
        problem.build()
        problem = simplify(problem)
        final = 1
        for sub in decompose_problem(problem):
            result = self._solve_sub(sub)
            if result == 0:
                return 0
            final *= result
        return final

    def _solve_sub(self, problem: CofolaProblem) -> int:
        """Apply the full pass sequence to a single decomposed sub-problem."""
        problem.build()
        if problem.is_unsat():
            return 0
        problem = simplify(problem)
        optimize(problem)
        infer_max_size(problem)
        sanity_check(problem)
        problem = transform(problem)
        optimize(problem)
        workaround(problem)
        problem = simplify(problem)
        sanity_check(problem)
        wfomc_problem, decoder = encode(problem, self.lifted)
        if wfomc_problem.contain_linear_order_axiom() and \
                self.wfomc_algo not in (Algo.INCREMENTAL, Algo.RECURSIVE):
            self.wfomc_algo = Algo.INCREMENTAL
            self.use_partition_constraint = True
        result = solve_wfomc(wfomc_problem, self.wfomc_algo,
                             self.use_partition_constraint)
        return decoder.decode_result(result) or 0
```

**Pass return-type consistency:** All in-place passes (`optimize`, `infer_max_size`, `workaround`) are updated to return `CofolaProblem` so the pipeline API is uniform.

---

## Global Counter Encapsulation

Replace module-level mutable integers with class-based counters. Public API stays the same.

```python
# objects/utils.py
class _AuxCounter:
    _count: int = 0

    @classmethod
    def next_name(cls) -> str:
        name = f"AUX_{cls._count}"
        cls._count += 1
        return name

    @classmethod
    def reset(cls, start_from: int = 0) -> None:
        cls._count = start_from

def aux_obj_name() -> str:
    return _AuxCounter.next_name()

def reset_aux_obj_counter(start_from: int = 0) -> None:
    _AuxCounter.reset(start_from)
```

Same pattern for `_AuxPredCounter` in `utils.py`.

---

## Parser Split: Mixin Pattern

`CofolaTransformer` (703 lines) is split via Python mixins:

```python
# parser/transformer.py
from .transformer_objects import ObjectTransformerMixin
from .transformer_constraints import ConstraintTransformerMixin

class CofolaTransformer(CommonTransformer, ObjectTransformerMixin, ConstraintTransformerMixin):
    def __init__(self): ...   # core state
    def cofola(self): ...     # top-level dispatch
    def identity(self): ...   # name resolution
    def operations(self): ...
    # ... core methods only
```

`ObjectTransformerMixin` (in `transformer_objects.py`): `base_object_init`, `entities_body`, `func_init`, `common_operations`, `binary_operations`, `indexing`, `image`, `inverse_image`, `count`, etc.

`ConstraintTransformerMixin` (in `transformer_constraints.py`): `size_constraint`, `membership_constraint`, `subset_constraint`, `disjoint_constraint`, `equivalence_constraint`, `seq_constraint`, `seq_pattern`, `together`, `less_than`, `next_to`, `predecessor`, `negation_constraint`, `binary_constraint`, etc.

---

## Code Quality Changes

| Change | Detail |
|---|---|
| Remove `parser/bak/` | Delete 6 legacy backup files |
| Remove `decompose()` stub | Dead function at bottom of `problem.py` |
| Remove unreachable code | `transform_tuples` lines 531–577 (after `return True`) |
| Remove commented-out code | Parser methods lines 617–650, `__main__` block lines 658–702 |
| Remove unused import | `from os import replace` in `problem.py` |
| Fix typo | `propogate` → `propagate` throughout |

---

## Testing Strategy

Use `scripts/solve.py` as regression harness before and after each major step:

```bash
# baseline (run before starting)
uv run python scripts/solve.py -i problems/all.json

# after each step, re-run to verify no regressions
uv run python scripts/solve.py -i problems/all.json
```

---

## Implementation Order

1. Run baseline regression test
2. Split `problem.py` → `problem.py` + `passes.py` + `transforms.py`
3. Move `decompose_problem` from `solver.py` into `passes.py`
4. Create `pipeline.py` with `SolvePipeline`, slim `solver.py`
5. Split `objects/bag.py` → `bag.py` + `bag_ops.py`
6. Split `objects/sequence.py` → `sequence.py` + `sequence_patterns.py`
7. Split `parser/parser.py` → `transformer.py` + `transformer_objects.py` + `transformer_constraints.py` + slimmed `parser.py`
8. Encapsulate global counters in `objects/utils.py` and `utils.py`
9. Code quality cleanup (dead code, imports, typo)
10. Final regression test
