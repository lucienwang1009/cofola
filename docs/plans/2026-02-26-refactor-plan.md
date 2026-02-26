# Cofola Package Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the cofola package to comply with the 200–400 line file limit, introduce a `SolvePipeline` class, encapsulate global mutable state, and remove all dead code.

**Architecture:** Split `problem.py` (898 lines) into `problem.py` + `passes.py` + `transforms.py`; split oversized objects and parser files via the same pattern; introduce `pipeline.py` to make the solve stages explicit; use Python classmethods to replace module-level mutable counters. No behavior changes — `scripts/solve.py` is the regression harness.

**Tech Stack:** Python 3.11+, uv, pytest, logzero, wfomc (git dep), lark (parser), sympy, scipy, numpy

**Design doc:** `docs/plans/2026-02-26-refactor-design.md`

---

## Pre-flight: Read Before Touching Any Code

- Design doc: `docs/plans/2026-02-26-refactor-design.md`
- CLAUDE.md project instructions (especially solve pipeline, object hierarchy)
- Verify current state: `uv run pyright` and `uv run python scripts/solve.py -i problems/all.json` must both pass before you start

---

## Task 1: Baseline Regression

**Files:** No changes.

**Step 1: Run the regression suite**

```bash
cd /Users/lucien/Repos/cofola
uv run python scripts/solve.py -i problems/all.json
```

Expected: Runs all problems without errors, CSV written to `scripts/results.csv`.
Record the number of passing problems — all must still pass after every subsequent task.

**Step 2: Run type checker**

```bash
uv run pyright
```

Record any pre-existing errors so you don't introduce new ones.

---

## Task 2: Split `problem.py` — Extract `transforms.py`

Extract all `transform_*` functions from `problem.py` into a new `transforms.py`.

**Files:**
- Create: `src/cofola/transforms.py`
- Modify: `src/cofola/problem.py` (remove extracted functions)
- Modify: `src/cofola/solver.py` (update imports)

**Step 1: Create `src/cofola/transforms.py`**

Move these functions verbatim from `problem.py` into the new file (in this order):
`transform_tuples`, `transform_sequences`, `transform_functions`, `transform_size_constraint`, `transform_once`, `transform`

Remove the dead unreachable block in `transform_tuples` (the code after `return True` on the `BagChoose` branch — lines 531–577 in the original). This code was already bypassed by the `return True` two lines above it.

The file header and imports:

```python
from __future__ import annotations

from logzero import logger

from cofola.objects.bag import BagChoose, BagInit, BagMultiplicity, BagSupport, SizeConstraint
from cofola.objects.base import Bag, CombinatoricsObject, Entity, Sequence, Set, Tuple
from cofola.objects.function import FuncImage, FuncInit, FuncInverseImage, FuncPairConstraint
from cofola.objects.set import (
    DisjointConstraint, SetChoose, SetEqConstraint, SetInit, SetIntersection
)
from cofola.objects.tuple import (
    TupleCount, TupleImpl, TupleIndex,
    TupleIndexEqConstraint, TupleMembershipConstraint
)
from cofola.objects.utils import IDX_PREFIX
from cofola.problem import CofolaProblem
```

**Step 2: Remove extracted functions from `problem.py`**

Delete from `problem.py`: `transform_tuples`, `transform_sequences`, `transform_functions`, `transform_size_constraint`, `transform_once`, `transform`.

Also delete the `decompose` stub at the bottom (body is `pass`).

Also remove `from os import replace` (unused import, line 3).

**Step 3: Update `solver.py` imports**

In `solver.py`, change:
```python
from cofola.problem import CofolaProblem, infer_max_size, optimize, \
    sanity_check, simplify, transform, workaround
```
to:
```python
from cofola.problem import CofolaProblem, infer_max_size, optimize, \
    sanity_check, simplify, workaround
from cofola.transforms import transform
```

**Step 4: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

Expected: Same number of passing problems as baseline.

**Step 5: Commit**

```bash
git add src/cofola/transforms.py src/cofola/problem.py src/cofola/solver.py
git commit -m "refactor(transforms): extract transform passes from problem.py"
```

---

## Task 3: Split `problem.py` — Extract `passes.py`

Extract the remaining standalone functions from `problem.py` into `passes.py`.

**Files:**
- Create: `src/cofola/passes.py`
- Modify: `src/cofola/problem.py` (remove extracted functions)
- Modify: `src/cofola/solver.py` (update imports)

**Step 1: Create `src/cofola/passes.py`**

Move these functions verbatim from `problem.py` (in this order):
`sanity_check`, `fold_constants`, `add_disjoint_constraints`, `infer_max_size`, `workaround`, `optimize`, `simplify`

Also move `decompose_problem` from `solver.py` here (see Step 2b).

File header and imports:

```python
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from logzero import logger

from cofola.objects.bag import (
    BagAdditiveUnion, BagChoose, BagDifference, BagInit,
    BagIntersection, BagMultiplicity, BagSupport, SizeConstraint
)
from cofola.objects.base import (
    Bag, CombinatoricsBase, CombinatoricsConstraint, CombinatoricsObject,
    Entity, Sequence, Set, SetChooseReplace, Tuple
)
from cofola.objects.function import FuncImage
from cofola.objects.partition import BagPartition
from cofola.objects.sequence import SequenceConstraint
from cofola.objects.set import (
    DisjointConstraint, SetChooseReplace, SetDifference,
    SetInit, SetIntersection, SetUnion
)
from cofola.objects.tuple import TupleCount
from cofola.problem import CofolaProblem
```

Note: `SetChooseReplace` is in `cofola.objects.set`, not `cofola.objects.base` — fix the import in `simplify` accordingly.

**Update pass return types:** Change `optimize`, `infer_max_size`, and `workaround` to return `CofolaProblem`:

```python
def optimize(problem: CofolaProblem) -> CofolaProblem:
    """..."""
    optimizing = True
    while optimizing:
        optimizing = fold_constants(problem)
    return problem


def infer_max_size(problem: CofolaProblem) -> CofolaProblem:
    """..."""
    # ... existing body unchanged ...
    return problem


def workaround(problem: CofolaProblem) -> CofolaProblem:
    """..."""
    # ... existing body unchanged ...
    return problem
```

**Step 2a: Move `decompose_problem` from `solver.py` to `passes.py`**

Cut the `decompose_problem` function (and its `visit` helper) from `solver.py` and append it to `passes.py`.

It needs this import (already in passes.py header above):
```python
from cofola.objects.base import CombinatoricsBase, CombinatoricsConstraint, CombinatoricsObject
```

**Step 2b: Remove extracted functions from `problem.py`**

Delete from `problem.py`: `sanity_check`, `fold_constants`, `add_disjoint_constraints`, `infer_max_size`, `workaround`, `optimize`, `simplify`.

`problem.py` should now contain only `CofolaProblem` (~290 lines).

**Step 3: Update imports in `solver.py`**

Replace:
```python
from cofola.problem import CofolaProblem, infer_max_size, optimize, \
    sanity_check, simplify, workaround
from cofola.transforms import transform
```
with:
```python
from cofola.problem import CofolaProblem
from cofola.passes import (
    decompose_problem, infer_max_size, optimize,
    sanity_check, simplify, workaround
)
from cofola.transforms import transform
```

Remove the now-dead `from collections import defaultdict` import from `solver.py` if it's no longer used.

**Step 4: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

Expected: Same result as baseline.

**Step 5: Commit**

```bash
git add src/cofola/passes.py src/cofola/problem.py src/cofola/solver.py
git commit -m "refactor(passes): extract analysis passes from problem.py, move decompose_problem"
```

---

## Task 4: Create `pipeline.py` with `SolvePipeline`

Introduce the `SolvePipeline` class in a new `pipeline.py` and refactor `solver.py` to use it.

**Files:**
- Create: `src/cofola/pipeline.py`
- Modify: `src/cofola/solver.py`

**Step 1: Create `src/cofola/pipeline.py`**

```python
from __future__ import annotations

from logzero import logger
from wfomc import Algo

from cofola.encoder import encode
from cofola.passes import (
    decompose_problem, infer_max_size, optimize,
    sanity_check, simplify, workaround
)
from cofola.problem import CofolaProblem
from cofola.transforms import transform
from cofola.wfomc_solver import solve as solve_wfomc


class SolvePipeline(object):
    """
    Encapsulates the sequence of transformation passes for solving a single CofolaProblem.

    Usage:
        pipeline = SolvePipeline(Algo.FASTv2, use_partition_constraint=True, lifted=False)
        answer = pipeline.run(problem)
    """

    def __init__(
        self,
        wfomc_algo: Algo = Algo.FASTv2,
        use_partition_constraint: bool = True,
        lifted: bool = False,
    ) -> None:
        super().__init__()
        self.wfomc_algo = wfomc_algo
        self.use_partition_constraint = use_partition_constraint
        self.lifted = lifted

    def run(self, problem: CofolaProblem) -> int:
        """
        Solve a single problem: build, simplify, decompose into independent
        sub-problems, then multiply their counts.

        :param problem: the combinatorics problem (may contain compound constraints)
        :return: the count
        """
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
        """
        Apply the full pass sequence to a single decomposed sub-problem.

        :param problem: an independent sub-problem
        :return: the count for this sub-problem (0 if unsatisfiable)
        """
        problem.build()
        if problem.is_unsat():
            logger.info("The sub-problem is unsatisfiable")
            return 0
        logger.info("Simplifying...")
        problem = simplify(problem)
        logger.info(problem)
        logger.info("Optimizing...")
        optimize(problem)
        logger.info("Inferring max sizes...")
        infer_max_size(problem)
        logger.info("Transforming...")
        sanity_check(problem)
        problem = transform(problem)
        logger.info(problem)
        logger.info("Optimizing after transform...")
        optimize(problem)
        workaround(problem)
        logger.info("Simplifying after transform...")
        problem = simplify(problem)
        logger.info(problem)
        sanity_check(problem)
        logger.info(f"Problem for encoding:\n{problem}")
        wfomc_problem, decoder = encode(problem, self.lifted)
        logger.info(f"Encoded WFOMC problem:\n{wfomc_problem}")
        logger.info(f"Result decoder:\n{decoder}")
        if wfomc_problem.contain_linear_order_axiom() and \
                self.wfomc_algo not in (Algo.INCREMENTAL, Algo.RECURSIVE):
            logger.warning(
                "Linear order axiom detected; switching to INCREMENTAL algorithm."
            )
            self.wfomc_algo = Algo.INCREMENTAL
            self.use_partition_constraint = True
        result = solve_wfomc(wfomc_problem, self.wfomc_algo, self.use_partition_constraint)
        logger.debug(f"WFOMC result: {result}")
        decoded = decoder.decode_result(result)
        if decoded is None:
            logger.info("Sub-problem is unsatisfiable (decoder returned None)")
            return 0
        logger.info(f"Sub-problem answer: {decoded}")
        return decoded
```

**Step 2: Refactor `solver.py` to use `SolvePipeline`**

Replace `solve_single_problem` with a thin wrapper that constructs and runs a `SolvePipeline`:

```python
def solve_single_problem(
    problem: CofolaProblem,
    wfomc_algo: Algo,
    use_partition_constraint: bool = True,
    lifted: bool = True,
) -> int:
    """Solve a single problem with no compound constraints."""
    return SolvePipeline(wfomc_algo, use_partition_constraint, lifted).run(problem)
```

Update `solve()` in `solver.py` to import and use `SolvePipeline` for the sub-problems too (the deepcopy branch).

Update imports at the top of `solver.py`:

```python
from cofola.pipeline import SolvePipeline
```

**Step 3: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 4: Commit**

```bash
git add src/cofola/pipeline.py src/cofola/solver.py
git commit -m "refactor(pipeline): introduce SolvePipeline class, slim solver.py"
```

---

## Task 5: Split `objects/bag.py` → `bag.py` + `bag_ops.py`

**Files:**
- Create: `src/cofola/objects/bag_ops.py`
- Modify: `src/cofola/objects/bag.py`
- Modify: any files importing from `cofola.objects.bag` that use the moved classes

**Step 1: Identify what moves**

Move to `bag_ops.py`:
- `BagUnion` (if present, check bag.py)
- `BagAdditiveUnion`
- `BagIntersection`
- `BagDifference`
- `BagSupport`
- `BagSubsetConstraint`
- `BagEqConstraint`

Keep in `bag.py`:
- `BagInit`
- `BagChoose`
- `BagMultiplicity`
- `SizeConstraint`

**Step 2: Create `src/cofola/objects/bag_ops.py`**

```python
from __future__ import annotations

from wfomc import fol_parse as parse, Const, Expr
from typing import TYPE_CHECKING

from cofola.objects.base import AtomicConstraint, Bag, Entity, Set, SizedObject
from cofola.objects.utils import invert_comparator, parse_comparator

if TYPE_CHECKING:
    from cofola.context import Context
```

Then add the moved classes.

**Step 3: Update `bag.py`** to remove moved classes. Add at bottom:

```python
# Re-export for backwards compatibility
from cofola.objects.bag_ops import (
    BagAdditiveUnion, BagDifference, BagIntersection,
    BagSupport, BagSubsetConstraint, BagEqConstraint
)
```

This keeps all existing `from cofola.objects.bag import ...` statements working without any changes to the rest of the codebase.

**Step 4: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 5: Commit**

```bash
git add src/cofola/objects/bag.py src/cofola/objects/bag_ops.py
git commit -m "refactor(objects): split bag.py into bag.py + bag_ops.py"
```

---

## Task 6: Split `objects/sequence.py` → `sequence.py` + `sequence_patterns.py`

**Files:**
- Create: `src/cofola/objects/sequence_patterns.py`
- Modify: `src/cofola/objects/sequence.py`

**Step 1: Identify what moves to `sequence_patterns.py`**

Move:
- `SequencePattern` (base class for patterns)
- `SequenceSizedPattern`
- `TogetherPattern`
- `LessThanPattern`
- `NextToPattern`
- `PredecessorPattern`
- `SequencePatternCount`

Keep in `sequence.py`:
- `SequenceImpl`
- `SequenceConstraint`

**Step 2: Create `src/cofola/objects/sequence_patterns.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from cofola.objects.base import Entity, Set, SizedObject

if TYPE_CHECKING:
    from cofola.context import Context
```

Then add the moved pattern classes.

**Step 3: Update `sequence.py`**

Remove moved classes. Add re-exports at bottom:

```python
# Re-export for backwards compatibility
from cofola.objects.sequence_patterns import (
    SequencePattern, SequenceSizedPattern,
    TogetherPattern, LessThanPattern, NextToPattern,
    PredecessorPattern, SequencePatternCount
)
```

**Step 4: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 5: Commit**

```bash
git add src/cofola/objects/sequence.py src/cofola/objects/sequence_patterns.py
git commit -m "refactor(objects): split sequence.py into sequence.py + sequence_patterns.py"
```

---

## Task 7: Split `parser/parser.py` — Extract Mixin Classes

Split the 703-line `CofolaTransformer` into three files using Python mixins.

**Files:**
- Create: `src/cofola/parser/transformer.py`
- Create: `src/cofola/parser/transformer_objects.py`
- Create: `src/cofola/parser/transformer_constraints.py`
- Modify: `src/cofola/parser/parser.py` (slim to ~50 lines)

**Step 1: Create `src/cofola/parser/transformer_objects.py`**

This is a mixin — it uses `self.problem`, `self.id2obj`, `self._check_obj_type`, etc. from the main class.

```python
from __future__ import annotations

from collections import defaultdict

from cofola.objects.bag import BagChoose, BagInit
from cofola.objects.base import CombinatoricsObject, Entity, Partition, Sequence, SizedObject, Tuple
from cofola.objects.function import FuncImage, FuncInverseImage, FuncInit, Function
from cofola.objects.partition import BagPartition, SetPartition
from cofola.objects.sequence import SequenceImpl
from cofola.objects.set import Set, SetChoose, SetChooseReplace
from cofola.objects.tuple import TupleImpl, TupleIndex


class ObjectTransformerMixin:
    """Mixin providing object-construction methods for CofolaTransformer."""
```

Move these methods into the mixin:
`base_object_init`, `entities_body`, `slicing_entities`, `duplicate_entities`, `entity`, `func_init`, `common_operations`, `binary_operations`, `indexing`, `inverse_object`, `image`, `inverse_image`, `count`

**Step 2: Create `src/cofola/parser/transformer_constraints.py`**

```python
from __future__ import annotations

from cofola.objects.bag import BagAdditiveUnion, BagDifference, BagIntersection, BagMultiplicity, BagSubsetConstraint, BagEqConstraint, SizeConstraint
from cofola.objects.base import CombinatoricsConstraint, Entity, Sequence, SizedObject, Tuple
from cofola.objects.function import Function
from cofola.objects.sequence import SequenceConstraint, SequenceImpl
from cofola.objects.sequence_patterns import (
    LessThanPattern, NextToPattern, PredecessorPattern,
    SequenceSizedPattern, TogetherPattern
)
from cofola.objects.set import (
    BagEqConstraint, DisjointConstraint, MembershipConstraint,
    SetEqConstraint, Set, SubsetConstraint
)
from cofola.objects.tuple import (
    TupleCount, TupleIndex, TupleIndexEqConstraint, TupleMembershipConstraint
)


class ConstraintTransformerMixin:
    """Mixin providing constraint-construction methods for CofolaTransformer."""
```

Move these methods into the mixin:
`size_constraint`, `size_atom`, `size_atomic_expr`, `size_add`, `size_sub`, `count`, `in_or_not`, `membership_constraint`, `subset_constraint`, `disjoint_constraint`, `equivalence_constraint`, `count_parameter`, `seq_constraint`, `seq_pattern`, `together`, `less_than`, `next_to`, `predecessor`, `negation_constraint`, `binary_constraint`, `part_constraint`

**Step 3: Create `src/cofola/parser/transformer.py`**

This contains the `CofolaTransformer` class with only its core methods:

```python
from __future__ import annotations

from logzero import logger
from lark import Lark

from cofola.objects.base import CombinatoricsConstraint, CombinatoricsObject, Entity, Partition
from cofola.parser.common import CommonTransformer
from cofola.parser.transformer_objects import ObjectTransformerMixin
from cofola.parser.transformer_constraints import ConstraintTransformerMixin
from cofola.problem import CofolaProblem


RESERVED_KEYWORDS = [
    "set", "bag", "choose", "choose_replace", "count", "in", "subset",
    "disjoint", "supp", "compose", "partition", "tuple", "choose_tuple",
    "choose_replace_tuple", "sequence", "choose_sequence",
    "choose_replace_sequence", "together", "not", "and", "or",
]

RESERVED_PREFIXES = ["AUX_", "IDX_"]


class CofolaParsingError(Exception):
    pass


class CofolaTypeMismatchError(CofolaParsingError):
    def __init__(self, expected_types, actual):
        if isinstance(expected_types, tuple):
            expected_types = [t.__name__ for t in expected_types]
            expected_type = " or ".join(expected_types)
        else:
            expected_type = expected_types.__name__
        super().__init__(
            f"Expect a {expected_type} object, but got {actual} of type {type(actual)}."
        )


class CofolaTransformer(CommonTransformer, ObjectTransformerMixin, ConstraintTransformerMixin):
    def __init__(self, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)
        self.problem: CofolaProblem = CofolaProblem()
        self.id2obj: dict[str, CombinatoricsObject] = dict()
        self._processing_partition: Partition = None
        self._processing_part_name: str = None

    def left_identity(self, args): ...
    def object_declaration(self, args): ...
    def operations(self, args): ...
    def identity(self, args): ...
    def cofola(self, args): ...
    def _op_or_constraint_on_list(self, op_or_constraint, *args): ...
    def _get_obj_by_id(self, obj_id: str): ...
    def _attach_obj(self, name: str, obj: CombinatoricsObject): ...
    def _check_id(self, id: str): ...
    def _check_obj_type(self, obj: object, *expected_types: type): ...
    def _transform_tree(self, tree): ...
```

(Fill in the actual method bodies from the original file.)

**Step 4: Slim down `parser/parser.py`**

Replace the entire content of `parser/parser.py` with:

```python
from __future__ import annotations

from lark import Lark

from cofola.parser.grammar import grammar
from cofola.parser.transformer import CofolaTransformer
from cofola.problem import CofolaProblem

# Re-export for backwards compatibility
from cofola.parser.transformer import (
    CofolaParsingError, CofolaTypeMismatchError, RESERVED_KEYWORDS, RESERVED_PREFIXES
)


def parse(text: str) -> CofolaProblem:
    """Parse a .cfl program text into a CofolaProblem."""
    parser = Lark(grammar, start="cofola")
    tree = parser.parse(text)
    return CofolaTransformer().transform(tree)
```

**Step 5: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 6: Commit**

```bash
git add src/cofola/parser/
git commit -m "refactor(parser): split CofolaTransformer into mixin files"
```

---

## Task 8: Encapsulate Global Counters

Replace module-level mutable integers with class-based counters. Public API stays identical.

**Files:**
- Modify: `src/cofola/objects/utils.py`
- Modify: `src/cofola/utils.py`

**Step 1: Update `src/cofola/objects/utils.py`**

Replace:
```python
AUX_COUNTER = 0

def reset_aux_obj_counter(start_from: int = 0):
    global AUX_COUNTER
    AUX_COUNTER = start_from

def aux_obj_name() -> str:
    global AUX_COUNTER
    name = 'AUX_' + str(AUX_COUNTER)
    AUX_COUNTER += 1
    return name
```

With:
```python
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

**Step 2: Update `src/cofola/utils.py`**

Replace:
```python
AUX_PRED_PREFIX = '$cofola_aux_'
AUX_PRED_CNT = 0

def reset_aux_pred_cnt(start_from: int = 0):
    global AUX_PRED_CNT
    AUX_PRED_CNT = start_from

def create_aux_pred(arity: int, aux_pred_prefix: str = AUX_PRED_PREFIX) -> Pred:
    global AUX_PRED_CNT
    cnt = AUX_PRED_CNT
    AUX_PRED_CNT += 1
    return create_cofola_pred(f"{aux_pred_prefix}_" + str(cnt), arity)
```

With:
```python
_AUX_PRED_PREFIX = '$cofola_aux_'


class _AuxPredCounter:
    _count: int = 0

    @classmethod
    def next(cls) -> int:
        cnt = cls._count
        cls._count += 1
        return cnt

    @classmethod
    def reset(cls, start_from: int = 0) -> None:
        cls._count = start_from


def reset_aux_pred_cnt(start_from: int = 0) -> None:
    _AuxPredCounter.reset(start_from)


def create_aux_pred(arity: int, aux_pred_prefix: str = _AUX_PRED_PREFIX) -> Pred:
    cnt = _AuxPredCounter.next()
    return create_cofola_pred(f"{aux_pred_prefix}_{cnt}", arity)
```

**Step 3: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 4: Commit**

```bash
git add src/cofola/objects/utils.py src/cofola/utils.py
git commit -m "refactor(utils): encapsulate global mutable counters in class-based counters"
```

---

## Task 9: Fix Typo `propogate` → `propagate`

**Files:**
- Modify: `src/cofola/problem.py`
- Modify: `src/cofola/passes.py`
- Modify: `src/cofola/encoder.py`
- (any other callers found by grep)

**Step 1: Find all occurrences**

```bash
grep -rn "propogate" src/cofola/
```

**Step 2: Rename method and update all call sites**

In `problem.py`: rename `def propogate(self)` → `def propagate(self)`.

Update every call to `problem.propogate()` or `p.propogate()` throughout passes.py, transforms.py, encoder.py, problem.py itself.

**Step 3: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 4: Commit**

```bash
git add -p
git commit -m "fix: rename propogate → propagate throughout"
```

---

## Task 10: Remove Dead Code

**Files:**
- Delete: `src/cofola/parser/bak/` directory

**Step 1: Delete backup directory**

```bash
git rm -r src/cofola/parser/bak/
```

**Step 2: Verify nothing imports from `bak/`**

```bash
grep -rn "from cofola.parser.bak" src/ || echo "No imports found"
```

Expected: `No imports found`

**Step 3: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 4: Commit**

```bash
git commit -m "chore: remove parser/bak legacy backup files"
```

---

## Task 11: Final Verification

**Step 1: Run type checker**

```bash
uv run pyright
```

Expected: No new errors compared to baseline.

**Step 2: Run full regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

Expected: Same number of passing problems as baseline (Task 1).

**Step 3: Verify line counts comply with the 200–400 line guideline**

```bash
wc -l src/cofola/*.py src/cofola/objects/*.py src/cofola/parser/*.py
```

Expected: No file exceeds 400 lines (except `objects/base.py` at ~395 which is acceptable).

**Step 4: Confirm `problem.py` is clean**

```bash
python -c "import cofola.problem; print('OK')"
grep -n "propogate\|from os import" src/cofola/problem.py || echo "Clean"
```

**Step 5: Commit if any final tweaks were made**

```bash
git add -p
git commit -m "chore: final cleanup after refactor"
```

---

---

## Task 12: Redesign `CombinatoricsBase` — Eliminate `_assign_args`

Replace the `_assign_args` boilerplate (~35 implementations) with a `_fields` class variable. Make `args` a computed property. Make `combinatorially_eq` abstract on `CombinatoricsObject`.

**Files:**
- Modify: `src/cofola/objects/base.py`
- Modify: every concrete subclass in `objects/` (bag.py, set.py, function.py, tuple.py, sequence.py, partition.py, sequence_patterns.py, bag_ops.py)

**Background — the current pattern and why it's ugly:**

Every subclass has this identical boilerplate:
```python
class BagChoose(Bag):
    def __init__(self, obj_from: Bag, size: int = None) -> None:
        super().__init__(obj_from, size)   # stores in self.args

    def _assign_args(self) -> None:
        self.obj_from, self.size = self.args  # unpacks back
```

This exists because `subs_args` mutates `self.args` and calls `_assign_args` to re-sync named attrs. The fix: declare `_fields`, set attrs directly in `__init__`, make `args` a property.

**Step 1: Rewrite `CombinatoricsBase` in `base.py`**

```python
class CombinatoricsBase(object):
    _fields: tuple[str, ...] = ()

    def __init__(self) -> None:
        # Subclasses must set all _fields attributes BEFORE calling super().__init__()
        super().__init__()
        self.dependences: set[CombinatoricsObject] = set()
        self.descendants: set[CombinatoricsBase] = set()
        self._build_dependences()

    @property
    def args(self) -> tuple:
        """Computed from named attributes declared in _fields."""
        return tuple(getattr(self, f) for f in self._fields)

    def _build_dependences(self) -> None:
        self.dependences = set(
            getattr(self, f) for f in self._fields
            if isinstance(getattr(self, f, None), CombinatoricsObject)
        )
        for dep in self.dependences:
            dep.descendants.add(self)

    def combinatorially_eq(self, o: CombinatoricsBase) -> bool:
        return False  # safe default for constraints

    def inherit(self) -> None:
        """Template hook: inherit properties from dependences during propagation."""
        pass

    def subs_args(self, *new_vals) -> None:
        for name, val in zip(self._fields, new_vals):
            setattr(self, name, val)
        self._build_dependences()

    def subs_obj(self, old_obj: CombinatoricsObject,
                 new_obj: CombinatoricsObject) -> None:
        for name in self._fields:
            if getattr(self, name, None) is old_obj:
                setattr(self, name, new_obj)
        self._build_dependences()

    def encode(self, context: "Context") -> "Context":
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)
```

**Step 2: Update `CombinatoricsObject`**

```python
from abc import abstractmethod

class CombinatoricsObject(CombinatoricsBase):
    def __init__(self) -> None:
        self.name = aux_obj_name()
        super().__init__()

    @abstractmethod
    def combinatorially_eq(self, o: CombinatoricsBase) -> bool:
        """Whether this object is combinatorially identical to o."""
        ...

    def body_str(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.name} = {self.body_str()}"

    def is_uncertain(self) -> bool:
        return False
```

Note: Make `CombinatoricsBase` inherit from `ABC` or just use `@abstractmethod` without ABC (Python allows this with a metaclass trick, but simpler to just `from abc import ABC, abstractmethod` and add `ABC` to `CombinatoricsBase`'s bases).

**Step 3: Update `SizedObject` — use class-level defaults**

```python
class SizedObject(CombinatoricsObject):
    # Class-level defaults; instances override via self.size = x
    size: int = None
    max_size: int = float("inf")

    def __init__(self) -> None:
        super().__init__()

    def encode_size_var(self, context: "Context") -> tuple["Context", Expr]:
        raise NotImplementedError
```

This avoids `SizedObject.__init__` overwriting `size` that a subclass already set.

**Step 4: Update intermediate base classes (`Set`, `Bag`, `Function`, `Partition`, etc.)**

Change each to `def __init__(self) -> None` with no `*args`, just:

```python
class Bag(SizedObject):
    def __init__(self) -> None:
        self.p_entities_multiplicity: dict[Entity, int] = None
        self.dis_entities: set[Entity] = None
        self.indis_entities: dict[int, set[Entity]] = None
        super().__init__()

class Set(SizedObject):
    def __init__(self) -> None:
        self.p_entities: set[Entity] = None
        super().__init__()
```

**Step 5: Update every concrete subclass**

For each class with `_assign_args`, apply this pattern:

Before:
```python
class BagChoose(Bag):
    def __init__(self, obj_from: Bag, size: int = None) -> None:
        super().__init__(obj_from, size)
    def _assign_args(self) -> None:
        self.obj_from, self.size = self.args
```

After:
```python
class BagChoose(Bag):
    _fields = ('obj_from', 'size')
    def __init__(self, obj_from: Bag, size: int = None) -> None:
        self.obj_from = obj_from
        self.size = size
        super().__init__()
    def combinatorially_eq(self, o: CombinatoricsBase) -> bool:
        return False  # or real logic if applicable
```

Apply to all ~35 classes. Classes that had meaningful `combinatorially_eq` logic keep it; others return `False` explicitly.

**Key subclasses that need attention:**

- `BagInit`: sets `self.size = sum(self.p_entities_multiplicity.values())` — do this in `__init__` before `super()`
- `SizeConstraint`: has complex `_assign_args` — carefully migrate
- `SequenceImpl`: has 7 fields (`obj_from, choose, replace, size, circular, reflection, flatten_obj`) — `_fields = ('obj_from', 'choose', 'replace', 'size', 'circular', 'reflection', 'flatten_obj')`
- `Partition` in `base.py`: also overrides `subs_obj` to handle `partitioned_objs` — keep that override
- `BinaryConstraint` in `base.py`: `_fields = ('first_constraint', 'second_constraint')`, op_name is not in fields (not a CombinatoricsBase)

**Step 6: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 7: Run type checker**

```bash
uv run pyright
```

**Step 8: Commit**

```bash
git add src/cofola/objects/
git commit -m "refactor(objects): replace _assign_args pattern with _fields + make combinatorially_eq abstract"
```

---

## Task 13: Clean Up Passes — Dispatch Tables + Transform Helpers

**Files:**
- Modify: `src/cofola/passes.py` (fold_constants → dispatch table)
- Modify: `src/cofola/transforms.py` (transform_tuples → helpers)

**Step 1: Refactor `fold_constants` with a dispatch table**

Replace the chain of `isinstance` checks with a registry of fold functions:

```python
from typing import Callable

# Each folder: (obj, problem) -> bool (True if folded)
_CONSTANT_FOLDERS: dict[tuple[type, type | None], Callable] = {}


def _folder(obj_type: type, from_type: type | None = None):
    """Register a constant-folding function for a specific type pair."""
    def decorator(fn: Callable) -> Callable:
        _CONSTANT_FOLDERS[(obj_type, from_type)] = fn
        return fn
    return decorator


def _lookup_folder(obj) -> Callable | None:
    key = (type(obj), type(getattr(obj, 'obj_from', None)))
    if key in _CONSTANT_FOLDERS:
        return _CONSTANT_FOLDERS[key]
    return _CONSTANT_FOLDERS.get((type(obj), None))


@_folder(BagSupport, BagInit)
def _fold_bag_support(obj: BagSupport, problem: CofolaProblem) -> bool:
    new_obj = SetInit(obj.obj_from.keys())
    problem.replace(obj, new_obj)
    logger.info(f"Folded {obj} to {new_obj}")
    return True


@_folder(SetUnion)
def _fold_set_union(obj: SetUnion, problem: CofolaProblem) -> bool:
    if not all(isinstance(o, SetInit) for o in [obj.first, obj.second]):
        return False
    new_obj = SetInit(set.union(obj.first.p_entities, obj.second.p_entities))
    problem.replace(obj, new_obj)
    logger.info(f"Folded {obj} to {new_obj}")
    return True


@_folder(SetIntersection)
def _fold_set_intersection(obj, problem): ...

@_folder(SetDifference)
def _fold_set_difference(obj, problem): ...

@_folder(BagAdditiveUnion)
def _fold_bag_additive_union(obj, problem): ...

@_folder(BagIntersection)
def _fold_bag_intersection(obj, problem): ...

@_folder(BagDifference)
def _fold_bag_difference(obj, problem): ...


def fold_constants(problem: CofolaProblem) -> bool:
    ret = False
    for obj in list(problem.objects):   # copy — list may change during folding
        folder = _lookup_folder(obj)
        if folder and folder(obj, problem):
            ret = True
    # fold size constraints (unchanged)
    for constraint in list(problem.constraints):
        if isinstance(constraint, SizeConstraint):
            ...
    problem.build()
    return ret
```

**Step 2: Break `transform_tuples` into helper functions**

The 200-line function has three distinct branches. Extract each:

```python
def _transform_tuple_choose_from_set(
    obj: Tuple, problem: CofolaProblem
) -> bool:
    """Handle: choose_tuple(Set, size)"""
    obj_from = obj.obj_from
    indices = problem.add_object(
        SetInit(Entity(f"{IDX_PREFIX}{i}") for i in range(obj.size))
    )
    mapping = problem.add_object(
        FuncInit(indices, obj_from, injective=(not obj.replace))
    )
    logger.info(f"Transformed {obj} to {mapping}")
    problem.replace(obj, TupleImpl(
        obj_from, obj.choose, obj.replace, obj.size, indices, mapping
    ))
    return True


def _transform_tuple_choose_from_bag(
    obj: Tuple, problem: CofolaProblem
) -> bool:
    """Handle: choose_tuple(Bag, size) — decomposes to BagChoose first."""
    choosing_obj = problem.add_object(BagChoose(obj.obj_from, obj.size))
    obj.subs_args(choosing_obj, False, False, None)
    return True


def _transform_tuple_permute(
    obj: Tuple, problem: CofolaProblem
) -> bool:
    """Handle: tuple(Set) or tuple(Bag) — permutation of all elements."""
    obj_from = obj.obj_from
    indices = problem.add_object(
        SetInit(Entity(f"{IDX_PREFIX}{i}") for i in range(obj_from.size))
    )
    if isinstance(obj_from, Set):
        mapping = problem.add_object(
            FuncInit(indices, obj_from, surjective=True)
        )
    else:
        mapping = _build_bag_permute_mapping(obj, obj_from, indices, problem)
    logger.info(f"Transformed {obj} to {mapping}")
    problem.replace(obj, TupleImpl(
        obj_from, obj.choose, obj.replace, obj.size, indices, mapping
    ))
    return True


def _build_bag_permute_mapping(obj, obj_from, indices, problem):
    """Build the mapping + injectiveness constraints for a bag permutation."""
    support = problem.add_object(BagSupport(obj_from))
    mapping = problem.add_object(FuncInit(indices, support))
    singletons = set()
    reverse_images = []
    for entity in obj_from.p_entities_multiplicity:
        if entity in problem.singletons:
            singletons.add(entity)
        else:
            reverse_image = problem.add_object(FuncInverseImage(mapping, entity))
            reverse_images.append(reverse_image)
            if isinstance(obj_from, BagInit):
                problem.add_constraint(SizeConstraint(
                    [(reverse_image, 1)], "==",
                    obj_from.p_entities_multiplicity[entity]
                ))
            else:
                entity_mul = problem.add_object(BagMultiplicity(obj_from, entity))
                problem.add_constraint(SizeConstraint(
                    [(reverse_image, 1), (entity_mul, -1)], "==", 0
                ))
    for i, img1 in enumerate(reverse_images):
        for j, img2 in enumerate(reverse_images):
            if i < j:
                problem.add_constraint(DisjointConstraint(img1, img2))
    if singletons:
        image = problem.add_object(FuncImage(mapping, indices))
        problem.add_constraint(SetEqConstraint(image, support))
    return mapping


def transform_tuples(problem: CofolaProblem) -> bool:
    for obj in problem.objects:
        if not isinstance(obj, Tuple) or obj.mapping is not None:
            continue
        if obj.choose:
            if isinstance(obj.obj_from, Set):
                return _transform_tuple_choose_from_set(obj, problem)
            else:
                return _transform_tuple_choose_from_bag(obj, problem)
        else:
            return _transform_tuple_permute(obj, problem)
    # handle TupleIndexEqConstraint and TupleMembershipConstraint (unchanged)
    ...
    return False
```

**Step 3: Run regression test**

```bash
uv run python scripts/solve.py -i problems/all.json
```

**Step 4: Commit**

```bash
git add src/cofola/passes.py src/cofola/transforms.py
git commit -m "refactor(passes): dispatch table for fold_constants, split transform_tuples into helpers"
```

---

## Checklist Before Calling Done

- [ ] All problems in `problems/all.json` pass `scripts/solve.py`
- [ ] `uv run pyright` has no new errors
- [ ] No file in `src/cofola/` exceeds 400 lines
- [ ] `parser/bak/` directory deleted
- [ ] `decompose()` stub removed from `problem.py`
- [ ] `from os import replace` removed from `problem.py`
- [ ] `propogate` renamed to `propagate` everywhere
- [ ] `AUX_COUNTER` and `AUX_PRED_CNT` replaced with class-based counters
- [ ] `SolvePipeline` class exists in `pipeline.py`
- [ ] `decompose_problem` lives in `passes.py` (not `solver.py`)
- [ ] `transforms.py` exists with all `transform_*` functions
- [ ] `bag_ops.py` and `sequence_patterns.py` exist
- [ ] Parser transformer split into mixin files
- [ ] `_assign_args` eliminated from all ~35 subclasses (replaced by `_fields`)
- [ ] `combinatorially_eq` is `@abstractmethod` on `CombinatoricsObject`
- [ ] `fold_constants` uses dispatch table (no long isinstance chain)
- [ ] `transform_tuples` split into `_transform_tuple_choose_from_set`, `_transform_tuple_choose_from_bag`, `_transform_tuple_permute`, `_build_bag_permute_mapping`
