# Design: Frontend / IR Architectural Separation

## Goal

Split the current monolithic `ir/` module into two distinct layers:

- **`frontend/`** — pure problem *definition* (types, objects, constraints, problem, rewriter)
- **`ir/`** — pure problem *analysis and processing* (passes, pipeline, analysis)

After this change, `solve()` accepts a `frontend.Problem` directly instead of raw text.

---

## Motivation

The current `ir/` module conflates two concerns:

1. **Representation**: What is a combinatorial problem? (`types.py`, `objects.py`, `constraints.py`, `problem.py`, `rewriter.py`)
2. **Processing**: How do we analyse and transform it? (`passes/`, `analysis/`, `pipeline.py`)

Separating these allows:
- Constructing `Problem` directly in Python without parsing (useful for embedding cofola as a library)
- Cleaner dependency graph: frontend has no dep on ir; ir depends on frontend; backend depends on both
- Explicit public API: users import from `cofola.frontend`; internals live in `cofola.ir`

---

## Current State

```
src/cofola/
├── ir/
│   ├── types.py          # ObjRef, Entity
│   ├── objects.py        # SetInit, BagInit, TupleDef, FuncDef, PartitionDef, ...
│   ├── constraints.py    # SizeConstraint, MembershipConstraint, ...
│   ├── problem.py        # Problem, ProblemBuilder
│   ├── rewriter.py       # Rewriter, RewriterWithSubstitution
│   ├── pipeline.py       # IRPipeline.solve(text: str) → int
│   ├── analysis/
│   │   ├── entities.py   # EntityAnalysis, AnalysisResult, SetInfo, BagInfo
│   │   ├── max_size.py   # MaxSizeInference
│   │   └── bag_classify.py  # BagClassification
│   └── passes/
│       ├── optimize.py   # ConstantFolder
│       ├── simplify.py   # SimplifyPass
│       └── lowering.py   # LoweringPass
├── solver.py             # solve(text: str) → int
└── parser/parser.py      # parse(text) → ir.Problem
```

---

## Target Structure

```
src/cofola/
├── frontend/                     # NEW — problem definition types
│   ├── __init__.py               # exports all public types
│   ├── types.py                  # ObjRef, Entity (moved from ir/types.py)
│   ├── objects.py                # SetInit, BagInit, TupleDef, etc. (moved from ir/objects.py)
│   ├── constraints.py            # SizeConstraint, MembershipConstraint, etc.
│   ├── problem.py                # Problem, ProblemBuilder (moved from ir/problem.py)
│   └── rewriter.py               # Rewriter, RewriterWithSubstitution
├── ir/                           # analysis + processing only
│   ├── __init__.py               # re-exports from frontend + analysis + passes + pipeline
│   ├── pipeline.py               # IRPipeline.solve(problem: Problem) → int
│   ├── analysis/                 # unchanged
│   └── passes/                   # unchanged (imports updated)
├── solver.py                     # solve(problem: Problem) → int
│                                 # parse_and_solve(text: str) → int
└── parser/parser.py              # parse(text) → frontend.Problem
```

---

## API Changes

### `solver.py`

```python
# Before
def solve(text: str) -> int:
    return IRPipeline().solve(text)

# After
from cofola.frontend import Problem

def solve(problem: Problem) -> int:
    """Solve a combinatorics problem from a Problem IR node."""
    return IRPipeline().solve(problem)

def parse_and_solve(text: str) -> int:
    """Parse .cfl text and solve. Convenience wrapper."""
    from cofola.parser.parser import parse
    return solve(parse(text))
```

`main()` calls `parse_and_solve(text)` instead of `solve(text)`.

### `ir/pipeline.py`

```python
# Before
def solve(self, text: str) -> int:
    from cofola.parser.parser import parse
    problem = parse(text)
    ...

# After
def solve(self, problem: Problem) -> int:
    # No parse stage — caller provides Problem
    ...
```

### `parser/parser.py`

```python
# Before
from cofola.ir.problem import ...   (implicitly)
def parse(text: str) -> ir.Problem

# After
from cofola.frontend.problem import Problem
def parse(text: str) -> Problem
```

---

## File Move Plan

| Source (ir/) | Destination (frontend/) | Notes |
|---|---|---|
| `ir/types.py` | `frontend/types.py` | No changes to content |
| `ir/objects.py` | `frontend/objects.py` | No changes to content |
| `ir/constraints.py` | `frontend/constraints.py` | No changes to content |
| `ir/problem.py` | `frontend/problem.py` | `from .types import ObjRef` stays relative |
| `ir/rewriter.py` | `frontend/rewriter.py` | Update relative imports |

After moving, create stub `frontend/__init__.py` with full `__all__`.

---

## Import Chain Updates

### `ir/passes/*.py` and `ir/analysis/*.py`

All `from ..types`, `from ..objects`, `from ..problem`, `from ..constraints` relative imports
must become absolute `cofola.frontend.*` imports.

**Pattern**: Replace `from \.\.(types|objects|problem|constraints)` → `from cofola.frontend.\1`

Example in `ir/passes/lowering.py`:
```python
# Before
from ..types import Entity, ObjRef
from ..objects import TupleDef, SetInit, FuncDef
from ..problem import Problem

# After
from cofola.frontend.types import Entity, ObjRef
from cofola.frontend.objects import TupleDef, SetInit, FuncDef
from cofola.frontend.problem import Problem
```

Note: `from ..analysis.entities import AnalysisResult` stays as-is (analysis remains in ir/).

### `ir/pipeline.py`

```python
# Before
from cofola.ir.constraints import OrConstraint

# After
from cofola.frontend.constraints import OrConstraint
```

### `parser/transformer*.py`

```python
# Before
from cofola.ir.problem import ProblemBuilder
from cofola.ir.types import Entity, ObjRef
from cofola.ir.objects import ...
from cofola.ir.constraints import ...

# After
from cofola.frontend.problem import ProblemBuilder
from cofola.frontend.types import Entity, ObjRef
from cofola.frontend.objects import ...
from cofola.frontend.constraints import ...
```

### `backend/wfomc/encoder_ir.py`

```python
# Before
import cofola.ir.objects as ir_obj
import cofola.ir.constraints as ir_cst
from cofola.ir.types import ObjRef, Entity as IREntity
from cofola.ir.problem import Problem

# After
import cofola.frontend.objects as ir_obj
import cofola.frontend.constraints as ir_cst
from cofola.frontend.types import ObjRef, Entity as IREntity
from cofola.frontend.problem import Problem
```

### `backend/wfomc/context_ir.py`

```python
# Before
from cofola.ir.problem import Problem
from cofola.ir.types import ObjRef, Entity as IREntity
from cofola.ir.objects import SequenceDef, PartRef

# After
from cofola.frontend.problem import Problem
from cofola.frontend.types import ObjRef, Entity as IREntity
from cofola.frontend.objects import SequenceDef, PartRef
```

### `backend/base.py`

```python
# Before
from cofola.ir.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult

# After
from cofola.frontend.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult
```

### `ir/__init__.py`

Update to re-export from `cofola.frontend` for backward compatibility:
```python
# Re-export definition types from frontend
from cofola.frontend import (
    Entity, ObjRef, RefOrEntity,
    Problem, ProblemBuilder,
    SetInit, BagInit, ...  # all ObjDef types
    SizeConstraint, ...    # all Constraint types
    Rewriter, RewriterWithSubstitution,
)
# Analysis + passes (local)
from .analysis import EntityAnalysis, ...
from .passes import ConstantFolder, ...
from .pipeline import IRPipeline
```

---

## Dependency Graph (After)

```
cofola.frontend          (no deps on ir)
       ↓
cofola.ir.analysis       (depends on frontend)
cofola.ir.passes         (depends on frontend + ir.analysis)
cofola.ir.pipeline       (depends on frontend + ir.analysis + ir.passes + backend)
       ↓
cofola.backend.wfomc     (depends on frontend + ir.analysis)
       ↓
cofola.parser            (depends on frontend only)
       ↓
cofola.solver            (depends on frontend + ir.pipeline + parser)
```

---

## Implementation Steps

1. Create `src/cofola/frontend/` directory
2. Copy `ir/types.py` → `frontend/types.py` (no content changes)
3. Copy `ir/objects.py` → `frontend/objects.py` (no content changes)
4. Copy `ir/constraints.py` → `frontend/constraints.py` (no content changes)
5. Copy `ir/problem.py` → `frontend/problem.py` (update `from .types` relative import)
6. Copy `ir/rewriter.py` → `frontend/rewriter.py` (update relative imports to `cofola.frontend.*`)
7. Create `frontend/__init__.py` with full re-exports
8. Update `ir/passes/*.py`: replace `from ..(types|objects|problem|constraints)` with `cofola.frontend.*`
9. Update `ir/analysis/*.py`: same pattern
10. Update `ir/pipeline.py`: remove parse stage, accept `Problem`; update imports
11. Update `ir/__init__.py`: re-export definition types from `cofola.frontend`
12. Update `parser/transformer*.py` and `parser/parser.py`
13. Update `backend/wfomc/encoder_ir.py` and `context_ir.py`
14. Update `backend/base.py`
15. Update `solver.py`: split into `solve(problem)` + `parse_and_solve(text)`
16. Delete `ir/types.py`, `ir/objects.py`, `ir/constraints.py`, `ir/problem.py`, `ir/rewriter.py`
17. Run `uv run pytest` — all 22 tests must pass

---

## Notes

- Keep `ir/__init__.py` re-exporting from `frontend` for backward compatibility.
  External code doing `from cofola.ir import Problem` should still work.
- The `rewriter.py` has no IR-specific deps (it's a pure graph rewriter on `Problem` structure).
  Moving it to `frontend` is clean.
- `ir/analysis/max_size.py` only imports `from ..problem import Problem` and
  `from ..constraints import SizeConstraint` — these become `cofola.frontend.*`.
- `ir/passes/optimize.py` imports `from ..analysis.entities import EntityAnalysis`
  — this stays as `from cofola.ir.analysis.entities import EntityAnalysis`.
