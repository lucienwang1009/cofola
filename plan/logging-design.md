# Logging Design — Cofola Pipeline

## Overview

This document specifies where and what to log across the entire Cofola solve pipeline using `logzero`. The goal is to make every pipeline stage observable during debug (`-d`) sessions without polluting normal output.

### Log Level Convention

| Level | Usage |
|-------|-------|
| `DEBUG` | Per-object / per-constraint detail; verbose intermediate state |
| `INFO` | Stage entry/exit; high-level counts; key decisions |
| `WARNING` | Fallback behavior; unhandled cases (already present) |

All new logs use `from logzero import logger` at module level (not inline). Where inline imports currently exist in `encoder_ir.py`, they should be moved to module level.

---

## 1. `src/cofola/solver.py`

### Current state
- Logs input file name and final answer at INFO.
- Sets log level based on `-d` flag.

### Missing
- Problem name (if available) at solve start.
- Duration / timing summary (optional, stretch goal — not in this plan).

### Additions

```python
# In solve()
logger.info(f"Solving problem with {len(problem.defs)} objects, "
            f"{len(problem.constraints)} constraints")

# In parse_and_solve()
logger.debug("Parsing input text (%d chars)", len(text))
```

**Location**: `solve()` (line 11), `parse_and_solve()`.

---

## 2. `src/cofola/parser/parser.py`

### Current state
- Uses `logging.getLogger(__name__)` — inconsistent with rest of codebase (should use `logzero`).
- No actual log calls.

### Additions

```python
# Replace at top of file:
from logzero import logger

# In parse():
logger.debug("Grammar parsed successfully; tree has %d tokens", ...)
logger.info("Parsed problem: %d objects, %d constraints",
            len(problem.defs), len(problem.constraints))
```

**Fix**: Replace `logging.getLogger(__name__)` with `from logzero import logger`.

---

## 3. `src/cofola/ir/pipeline.py`

### Current state
- `logger` is imported but **never used** in any method.
- Pipeline stages execute silently.

### Missing (critical)
Every stage transition should be logged so a developer can trace which pass transformed the problem and what the state looked like before/after.

### Additions — `IRPipeline.solve()`

Add at each stage boundary:

```python
# Stage 1
logger.info("[Stage 1] EntityAnalysis — %d objects", len(problem.defs))
analysis = EntityAnalysis().run(problem)
logger.debug("[Stage 1] entities=%s, singletons=%s",
             {e.name for e in analysis.all_entities},
             {e.name for e in analysis.singletons})

# Stage 2
logger.info("[Stage 2] ConstantFolder")
problem = ConstantFolder().rewrite(problem)
logger.debug("[Stage 2] after fold: %d objects", len(problem.defs))

# Re-analyze
logger.debug("[Stage 2b] Re-running EntityAnalysis after fold")
analysis = EntityAnalysis().run(problem)

# Stage 3
logger.info("[Stage 3] MaxSizeInference")
max_sizes = MaxSizeInference().run(problem, analysis)
logger.debug("[Stage 3] inferred max_sizes=%s",
             {ref.id: sz for ref, sz in max_sizes.items()})
self._merge_max_sizes(analysis, max_sizes)
logger.debug("[Stage 3] merged max_sizes into analysis")

# Stage 4
logger.info("[Stage 4] LoweringPass — %d objects before", len(problem.defs))
problem = LoweringPass().run(problem, analysis)
logger.info("[Stage 4] LoweringPass done — %d objects after", len(problem.defs))

# Re-analyze
logger.debug("[Stage 4b] Re-running EntityAnalysis after lowering")
analysis = EntityAnalysis().run(problem)

# Stage 5
logger.info("[Stage 5] SimplifyPass — %d objects before", len(problem.defs))
problem = SimplifyPass().run(problem)
logger.info("[Stage 5] SimplifyPass done — %d objects after", len(problem.defs))

# Stage 6
logger.info("[Stage 6] BagClassification")
analysis = BagClassification().run(problem, analysis)
logger.debug("[Stage 6] dis_entities=%s",
             {ref.id: {e.name for e in info.dis_entities}
              for ref, info in analysis.bag_info.items()})

# Stage 7
logger.info("[Stage 7] Solving via WFOMCBackend")
result = self._solve_ir(problem, analysis)
logger.info("[Stage 7] result=%d", result)
return result
```

### Additions — `IRPipeline._solve_ir()`

```python
# At entry to _solve_ir:
logger.debug("_solve_ir: %d constraints", len(problem.constraints))

# When OrConstraint is expanded:
logger.info("Expanding OrConstraint[%d] via inclusion-exclusion", i)
logger.debug("  left=%s  right=%s", c.left, c.right)

# After recursion:
logger.debug("  |A|=%d  |B|=%d  |A∩B|=%d  result=%d",
             count_a, count_b, count_ab, count_a + count_b - count_ab)
```

### Additions — `IRPipeline._merge_max_sizes()`

```python
logger.debug("_merge_max_sizes: updating %d refs", len(max_sizes))
for ref, size in max_sizes.items():
    # existing logic
    logger.debug("  ref=%s: old=%s -> new=%s", ref.id, current, ...)
```

---

## 4. `src/cofola/ir/analysis/entities.py`

### Current state
- No logging at all.

### Missing
- Which objects are processed and what type they resolve to.
- Final entity set and singleton set.

### Additions — `EntityAnalysis.run()`

```python
logger.debug("EntityAnalysis.run: processing %d objects in topological order",
             len(list(problem.topological_order())))

# In the dispatch loop, after each handler:
logger.debug("  analyzed %s ref=%s", type(defn).__name__, ref.id)

# After computing singletons:
logger.debug("EntityAnalysis result: all_entities=%s, singletons=%s",
             {e.name for e in all_entities},
             {e.name for e in singletons})

# Per set_info/bag_info entry (very verbose, DEBUG only):
for ref, info in set_info.items():
    logger.debug("  SetInfo ref=%s: p_entities=%s, max_size=%s",
                 ref.id, {e.name for e in info.p_entities}, info.max_size)
for ref, info in bag_info.items():
    logger.debug("  BagInfo ref=%s: p_entities=%s, max_size=%s",
                 ref.id,
                 {e.name: m for e, m in info.p_entities_multiplicity.items()},
                 info.max_size)
```

**File location**: `src/cofola/ir/analysis/entities.py`
**Import to add**: `from logzero import logger`

---

## 5. `src/cofola/ir/analysis/max_size.py`

### Current state
- No logging at all.

### Missing
- Whether any size constraints were found.
- LP result per ref (success/failure, inferred size).

### Additions — `MaxSizeInference.run()`

```python
logger.debug("MaxSizeInference: %d size_constraints, %d constrained_refs",
             len(size_constraints), len(constrained_refs))

# Early exit:
if not size_constraints:
    logger.debug("MaxSizeInference: no size constraints, skipping LP")
    return {}

# Per LP solve:
for ref in constrained_refs:
    # existing LP solve
    if ret.success:
        logger.debug("  LP ref=%s: max_size=%d (initial_max=%s)",
                     ref.id, size, initial_max)
    else:
        logger.debug("  LP ref=%s: infeasible or unbounded", ref.id)

logger.info("MaxSizeInference: inferred sizes for %d refs", len(inferred_sizes))
```

**File location**: `src/cofola/ir/analysis/max_size.py`
**Import to add**: `from logzero import logger`

---

## 6. `src/cofola/ir/analysis/bag_classify.py`

### Current state
- No logging at all.

### Missing
- Which refs are marked non-liftable and why.
- Which entities are marked distinguishable at each step.
- Final dis/indis classification per bag.

### Additions — `BagClassification.run()`

```python
logger.info("BagClassification.run: %d refs", len(list(problem.refs())))

# Step 1:
logger.debug("[Step 1] identifying non-liftable refs")
# After building non_lifted_refs:
logger.debug("  non_lifted_refs=%s", {r.id for r in non_lifted_refs})

# Step 2:
logger.debug("[Step 2] marking entities in non-liftable bags as dis")
for ref in non_lifted_refs:
    if ref in analysis.bag_info:
        logger.debug("  bag ref=%s -> all entities marked dis", ref.id)

# Step 3:
logger.debug("[Step 3] marking BagCountAtom entities as dis")
for constraint in problem.constraints:
    if isinstance(constraint, SizeConstraint):
        for term, _ in constraint.terms:
            if isinstance(term, BagCountAtom):
                logger.debug("  BagCountAtom bag=%s entity=%s",
                             term.bag.id, term.entity.name)

# Step 4:
logger.debug("[Step 4] classifying BagInit/SetChooseReplace entities by multiplicity")

# Step 5:
logger.debug("[Step 5] propagating dis_entities in topological order")

# Final summary:
for ref, info in analysis.bag_info.items():
    logger.debug("  BagInfo ref=%s: dis=%s, indis=%s",
                 ref.id,
                 {e.name for e in info.dis_entities},
                 {e.name: m for e, m in info.indis_entities.items()})
```

**File location**: `src/cofola/ir/analysis/bag_classify.py`
**Import to add**: `from logzero import logger`

---

## 7. `src/cofola/ir/passes/optimize.py`

### Current state
- Already logs each fold at INFO: `logger.info(f"Folded {defn} to {folded}")`.

### Missing
- How many fold iterations ran to reach fixed point.
- Total objects folded count.

### Additions — `ConstantFolder.rewrite()`

```python
iteration = 0
while changed:
    iteration += 1
    current, changed = self._fold_once(current)
    logger.debug("ConstantFolder iteration %d: changed=%s", iteration, changed)

logger.info("ConstantFolder: converged after %d iterations", iteration)
```

---

## 8. `src/cofola/ir/passes/simplify.py`

### Current state
- No logging at all.

### Missing
- Which objects were removed (key diagnostic for debugging encoding errors).

### Additions — `SimplifyPass.run()`

```python
logger.info("SimplifyPass: %d objects before", len(list(problem.iter_objects())))

# After computing used_refs:
logger.debug("SimplifyPass: %d used refs out of %d total",
             len(used_refs), len(list(problem.iter_objects())))

# Log removed refs:
all_refs = {ref for ref, _ in problem.iter_objects()}
removed = all_refs - used_refs
if removed:
    logger.info("SimplifyPass: removed %d unused objects: %s",
                len(removed), {r.id for r in removed})
else:
    logger.debug("SimplifyPass: no objects removed")

logger.info("SimplifyPass: %d objects after", len(new_defs))
```

**File location**: `src/cofola/ir/passes/simplify.py`
**Import to add**: `from logzero import logger`

---

## 9. `src/cofola/ir/passes/lowering.py`

### Current state
- Has `logger.info()` for TupleDef, SequenceDef, FuncDef lowering events (lines 265, 388, 457).

### Missing
- Entry/exit of each lowering iteration.
- Which object type triggered which lowering rule.

### Additions — `LoweringPass.run()`

```python
logger.info("LoweringPass.run: %d objects before", len(problem.defs))
iteration = 0
while True:
    iteration += 1
    logger.debug("LoweringPass iteration %d", iteration)
    result = self._lower_once(problem, analysis)
    if result is None:
        break
    problem, analysis = result
logger.info("LoweringPass done: %d iterations, %d objects after",
            iteration, len(problem.defs))
```

**Note**: `LoweringPass.run()` body must be inspected to match actual loop structure — the above is the design intent.

---

## 10. `src/cofola/backend/wfomc/backend.py`

### Current state
- Warns on auto-switch to INCREMENTAL algorithm.

### Missing
- Log the WFOMC problem structure (predicate count, formula complexity).
- Log raw WFOMC result before decoding.
- Log decoded final result.

### Additions — `WFOMCBackend.solve()`

```python
logger.info("WFOMCBackend.solve: encoding IR problem (%d objects, %d constraints)",
            len(list(problem.iter_objects())), len(problem.constraints))

wfomc_problem, decoder = encode_ir(problem, analysis, self.lifted)

logger.debug("WFOMCBackend: wfomc_problem has %d predicates, algo=%s",
             len(wfomc_problem.predicates), algo)

# After solve_wfomc:
logger.debug("WFOMCBackend: raw wfomc result = %s", raw)

# After decode:
logger.debug("WFOMCBackend: decoded result = %s", result)

if result is None:
    logger.info("WFOMCBackend: result is None (unsatisfiable) -> 0")
else:
    logger.info("WFOMCBackend: final result = %d", result)
```

---

## 11. `src/cofola/backend/wfomc/encoder_ir.py`

### Current state
- `logger` imported inline (inside function body) at lines 118, 296, 1083, 1289, 1308.
- Logs singletons and each object type at encoding time (INFO).
- Warns on unhandled node/constraint types.

### Changes required
1. **Move all inline `from logzero import logger` to module-level** (top of file, once).
2. Change encoding progress logs from INFO to DEBUG (there can be many objects).

### Additions — `encode_ir()`

```python
# At module level (replace all inline imports):
from logzero import logger

# In encode_ir():
logger.debug("encode_ir: %d objects to encode", len(list(problem.iter_objects())))

# Existing per-object log (downgrade to DEBUG):
# BEFORE: logger.info(f"Encoding {type(defn).__name__} ref={ref.id}")
# AFTER:  logger.debug(f"Encoding {type(defn).__name__} ref={ref.id}")

# After encoding all objects:
logger.debug("encode_ir: encoding %d constraints", len(problem.constraints))

# After full encoding:
logger.info("encode_ir complete: %d clauses generated",
            len(wfomc_problem.clauses) if hasattr(wfomc_problem, 'clauses') else -1)
```

---

## Summary Table

| File | Import needed | New log calls | Severity | Priority |
|------|--------------|---------------|----------|----------|
| `solver.py` | — (exists) | 2 | DEBUG/INFO | Low |
| `parser/parser.py` | Replace stdlib with logzero | 2 | DEBUG/INFO | Low |
| `ir/pipeline.py` | — (exists, unused) | ~12 | DEBUG/INFO | **High** |
| `ir/analysis/entities.py` | Add logzero | ~6 | DEBUG | **High** |
| `ir/analysis/max_size.py` | Add logzero | 4 | DEBUG/INFO | **High** |
| `ir/analysis/bag_classify.py` | Add logzero | ~8 | DEBUG/INFO | **High** |
| `ir/passes/optimize.py` | — (exists) | 2 | DEBUG/INFO | Medium |
| `ir/passes/simplify.py` | Add logzero | 4 | INFO/DEBUG | **High** |
| `ir/passes/lowering.py` | — (exists) | 3 | DEBUG/INFO | Medium |
| `backend/wfomc/backend.py` | — (exists) | 5 | DEBUG/INFO | **High** |
| `backend/wfomc/encoder_ir.py` | Move to module level | 3 + downgrade 1 | DEBUG | Medium |

---

## Implementation Order

Implement in this order for maximum immediate debug utility:

1. `ir/pipeline.py` — stage-level observability (most impactful)
2. `ir/analysis/bag_classify.py` — classification errors are most common bug source
3. `ir/analysis/entities.py` — entity analysis is the foundation of all passes
4. `ir/passes/simplify.py` — removed objects are a frequent source of confusion
5. `ir/analysis/max_size.py` — LP failures are opaque without logs
6. `backend/wfomc/backend.py` — see raw WFOMC results
7. `backend/wfomc/encoder_ir.py` — cleanup inline imports + downgrade INFO→DEBUG
8. `ir/passes/optimize.py` — fold iteration count
9. `ir/passes/lowering.py` — entry/exit logs
10. `solver.py` + `parser/parser.py` — cosmetic improvements

---

## Testing the Logs

After implementation, verify with:

```bash
# Should show all DEBUG logs across all stages
uv run cofola -i problems/others/bag.cfl -d 2>&1 | head -100

# Should show only INFO and above
uv run cofola -i problems/others/gifts.cfl 2>&1

# Check a complex problem with OrConstraint / Partition
uv run cofola -i problems/others/cards.cfl -d 2>&1 | grep -E "\[Stage|Expand|BagClass"
```
