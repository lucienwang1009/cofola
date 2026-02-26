# Cofola Refactoring Session Summary

## Current Status (2026-03-05)

**Branch**: `refactor/architecture`
**All 22 tests passing**: `uv run pytest tests/test_all_problems.py`

Both refactoring plans are **COMPLETE**:
- `plan/backend-ir-native.md` — All 5 phases ✅
- `plan/frontend-ir-separation.md` — Complete ✅

---

## Session Accomplishments (2026-03-05)

### 1. Problem Class String Representation
Added `__str__` and `__repr__` to `frontend/problem.py:Problem`:
- `__repr__`: Unambiguous representation with all fields
- `__str__`: Readable multi-line format with named objects

### 2. Public API Update
Added `parse` to `cofola.__init__.py` exports:
- `solve(problem: Problem) → int`
- `parse_and_solve(text: str) → int`
- `parse(text: str) → Problem`

### 3. Documentation Updates
- Fixed MEMORY.md deleted files list (frontend/ moved, not deleted)
- Updated plan/backend-ir-native.md with correct file status

---

## Final Architecture

```
frontend/           # Problem definition (stable API)
├── types.py        # ObjRef, Entity
├── objects.py      # SetInit, BagInit, TupleDef, etc.
├── constraints.py  # SizeConstraint, etc.
├── problem.py      # Problem, ProblemBuilder
└── rewriter.py     # Rewriter

ir/                 # Analysis + processing only
├── pipeline.py     # IRPipeline.solve(problem) → int
├── analysis/       # EntityAnalysis, BagClassification, MaxSizeInference
└── passes/         # LoweringPass, SimplifyPass, ConstantFolder

backend/wfomc/      # IR-native WFOMC encoder
├── encoder_ir.py   # Match-case dispatch on IR dataclasses
├── context_ir.py   # ObjRef-based Context
└── backend.py      # WFOMCBackend

parser/             # IR-native transformer
├── transformer.py  # CofolaTransformer (ProblemBuilder + id2ref)
├── transformer_objects.py
└── transformer_constraints.py
```

---

## Data Flow

```
text → parse() [parser/parser.py → CofolaTransfomer → frontend.Problem]
     → EntityAnalysis → AnalysisResult
     → ConstantFolder (optimize)
     → MaxSizeInference
     → LoweringPass (TupleDef → FuncDef+SetInit, etc.)
     → SimplifyPass
     → BagClassification
     → WFOMCBackend.solve(problem, analysis) → int
```

---

## Deleted Legacy Code

- `src/cofola/objects/` — deleted
- `src/cofola/backend/wfomc/encoder.py` — deleted (replaced by encoder_ir.py)
- `src/cofola/backend/wfomc/context.py` — deleted (replaced by context_ir.py)
- `src/cofola/pipeline.py` — deleted
- `src/cofola/ir/parser_adapter.py` — deleted
- `src/cofola/ir/types.py` — moved to frontend/
- `src/cofola/ir/objects.py` — moved to frontend/
- `src/cofola/ir/constraints.py` — moved to frontend/
- `src/cofola/ir/problem.py` — moved to frontend/
- `src/cofola/ir/rewriter.py` — moved to frontend/

---

## Key Bug Fixes (Historical)

1. **TupleMembershipConstraint**: Two distinct meanings handled correctly
2. **LoweringPass._lower_tuple_constraints**: Update constraints after TupleDef→FuncDef
3. **BagClassification**: PartitionDef non-liftable; SetChooseReplace all dis_entities
4. **EntityAnalysis._compute_singletons**: PartitionDef → no singletons; mult>1 removal
5. **SetChooseReplace encoding**: Single shared variable for all entity weights
6. **OrConstraint**: Inclusion-exclusion expansion in `_solve_ir`
7. **count() on PartRef**: Look up partition source type