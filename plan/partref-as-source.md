# PartRef as Source — Design Plan

## Problem Statement

`PartRef` (a part of a `PartitionDef`/composition) cannot currently be used as the source
for other objects (e.g., `TupleDef`, `SetChoose`, `BagChoose`).

### Root Cause

Three gaps, all in analysis passes:

| Layer | Gap | Consequence |
|---|---|---|
| `EntityAnalysis` | `PartRef` not dispatched → no `SetInfo`/`BagInfo` produced | `LoweringPass._try_lower_tuples` hits `else: continue`, silently skips `TupleDef` |
| `BagClassification` | `PartRef` not marked non-liftable, not propagated | Parts incorrectly treated as liftable |
| `encoder_ir.py` | `case ir_obj.PartRef(): pass` | Silent no-op, opaque |

### Confirmed Bug (error.cfl)

```
aux_1 = set(e1, e2, e3, e4)
aux_2 = compose(aux_1, 2)
t = tuple(aux_2[0])
t[0] == e1
```

**Bug chain:**
1. `EntityAnalysis` produces no `SetInfo` for `PartRef` (`aux_2[0]`)
2. `LoweringPass._try_lower_tuples`: `source not in set_info and source not in bag_info` → `continue`
3. `TupleDef` never lowered → encoder emits "unhandled IR node type" warning
4. `TupleIndexEq` constraint silently ignored
5. Answer is **16** (all compositions, no filtering) instead of the correct value

The encoder already handles the `PartRef` predicate correctly — `_encode_partition` creates
predicates and entity vars for all parts before their `PartRef` nodes are processed in
topological order. The problem is purely in the analysis passes.

---

## Pipeline Context

```
Stage 1: EntityAnalysis              ← gap here (no SetInfo/BagInfo for PartRef)
Stage 2: ConstantFolder
Stage 2b: EntityAnalysis (re-run)    ← gap here (same)
Stage 3: MaxSizeInference            ← would work once SetInfo exists
Stage 4: LoweringPass                ← FAILS silently without SetInfo
Stage 4b: EntityAnalysis (re-run)
Stage 5: SimplifyPass
Stage 6: BagClassification           ← gap here (PartRef not non-liftable)
Stage 7: WFOMCBackend
```

---

## Design

### Change 1 — `src/cofola/ir/analysis/entities.py`

Add dispatch in `EntityAnalysis.run` (after the `BagSupport` branch):

```python
elif isinstance(defn, PartRef):
    self._analyze_part_ref(ref, defn, set_info, bag_info, problem)
```

Implement the method:

```python
def _analyze_part_ref(
    self,
    ref: ObjRef,
    defn: PartRef,
    set_info: dict[ObjRef, SetInfo],
    bag_info: dict[ObjRef, BagInfo],
    problem: Problem,
) -> None:
    """PartRef: inherit entity information from the partition's source.

    Each part has the same *potential* entities as the source (any entity
    could end up in any part). max_size is conservatively the source's
    max_size — MaxSizeInference can tighten this later via SizeConstraints.

    Set partition  → SetInfo
    Bag partition  → BagInfo
    """
    partition_defn = problem.get_object(defn.partition)
    source = partition_defn.source

    if source in set_info:
        src = set_info[source]
        set_info[ref] = SetInfo(
            p_entities=set(src.p_entities),
            max_size=src.max_size,
        )
    elif source in bag_info:
        src = bag_info[source]
        bag_info[ref] = BagInfo(
            p_entities_multiplicity=dict(src.p_entities_multiplicity),
            max_size=src.max_size,
        )
```

**Why topological order is safe**: `PartRef.partition` is an `ObjRef` field, so
`Problem.get_refs(PartRef)` returns `[partition_ref]`. Kahn's algorithm places
`PartitionDef` before `PartRef`, and `PartitionDef.source` before `PartitionDef`.
By the time `_analyze_part_ref` runs, the source's `SetInfo`/`BagInfo` is already populated.

---

### Change 2 — `src/cofola/ir/analysis/bag_classify.py`

**2a — Step 1: mark PartRef as non-liftable**

```python
# In BagClassification.run, Step 1 loop — add after the PartitionDef branch:
elif isinstance(defn, PartRef):
    non_lifted_refs.add(ref)
    self._mark_dependencies_non_lifted(ref, problem, non_lifted_refs)
```

**2b — Step 5: propagate dis_entities into PartRef**

```python
# In BagClassification._propagate_from_sources — add a new elif:
elif isinstance(defn, PartRef):
    partition_defn = problem.get_object(defn.partition)
    src_info = analysis.bag_info.get(partition_defn.source)
    if src_info is not None:
        info.dis_entities = src_info.dis_entities.copy()
        info.indis_entities = {k: v.copy() for k, v in src_info.indis_entities.items()}
```

---

### Change 3 — `src/cofola/backend/wfomc/encoder_ir.py`

**3a — Replace `pass` with explicit no-op function**

```python
def _encode_part_ref(
    ref: ObjRef,
    defn: ir_obj.PartRef,
    context: ContextIR,
) -> None:
    """PartRef: predicate and entity vars already registered by _encode_partition.

    _encode_partition runs for defn.partition before this node is reached
    (PartitionDef precedes PartRef in topological order) and has already called:
      context.get_pred(ref, create=True)       # main predicate
      context.get_entity_var(ref, entity)      # per-entity vars (bag partitions)
    Nothing additional is needed here.
    """
    context.get_pred(ref, use=False)  # defensive: verify predicate exists
```

**3b — Update `_encode_object` dispatch**

```python
case ir_obj.PartRef():
    _encode_part_ref(ref, defn, context)
```

---

### Change 4 — LoweringPass: warn on variable-size PartRef tuple sources

`_encode_bag_choose` already handles `PartRef` as source correctly via its `else` branch
(`context.get_entity_var(defn.source, entity)` finds the entity var created in `_encode_partition`).
No change needed there.

However, `tuple(PartRef)` with no explicit size has a semantic ambiguity: the part size is
variable (0..N), but `_try_lower_tuples` uses `max_size` as the fixed tuple size, which creates
a surjective bijection requiring the part to always have exactly `max_size` elements.

Add a warning (not a hard error) to surface this:

```python
# In _try_lower_tuples, after size is determined from set_info/bag_info:
source_defn_check = problem.get_object(source)
if isinstance(source_defn_check, PartRef) and defn.size is None:
    logger.warning(
        "TupleDef {}: source is PartRef with potentially variable size. "
        "Results may be incorrect unless the part size is constrained. "
        "Use 'choose k tuple from ...' with explicit k for reliable results.",
        ref.id,
    )
```

---

## Compatibility Matrix

| Pass | Current | After change |
|---|---|---|
| `EntityAnalysis` | PartRef silently skipped | ✓ `_analyze_part_ref` produces SetInfo/BagInfo |
| `BagClassification` | PartRef not non-liftable, not propagated | ✓ Step 1 + `_propagate_from_sources` |
| `MaxSizeInference` | LP operates on SizeConstraint terms directly | ✓ No change needed |
| `_compute_singletons` | Returns `∅` if any `PartitionDef` exists | ✓ No change needed |
| `LoweringPass` | Skips TupleDef silently | ✓ Can now find size from SetInfo |
| `_encode_partition` | Already creates preds + entity vars for all parts | ✓ No change needed |
| `_encode_set_choose` | Only needs `get_pred(source)` | ✓ No change needed |
| `_encode_bag_choose` | else branch handles PartRef entity vars | ✓ No change needed |
| Topological order | PartRef depends on PartitionDef via `partition` field | ✓ Already correct |

---

## Semantic Limitation: `tuple(PartRef)` with variable-size part

After this fix, `tuple(PartRef)` with no explicit size uses `max_size` from `SetInfo`
as the tuple size. This is **only correct** when the part has a fixed (constrained) size.

| Source of tuple | Behavior |
|---|---|
| `SetInit` | Always fixed size — correct |
| `SetChoose(size=k)` | Fixed size k — correct |
| `PartRef` + `SizeConstraint(\|part\| == k)` | MaxSizeInference tightens max_size to k — correct |
| `PartRef` with no size constraint | max_size = source.max_size (conservative upper bound) — **may be wrong** |

For variable-size parts, users should write:

```
# Correct: specify size explicitly
t = choose 2 tuple from aux_2[0]

# Or: constrain part size first
|aux_2[0]| == 2
t = tuple(aux_2[0])
```

---

## Implementation Order

1. Change 1 (`EntityAnalysis._analyze_part_ref`) — unblocks all downstream passes
2. Change 2a (BagClassification step 1) — marks non-liftability
3. Change 2b (`_propagate_from_sources`) — propagates dis_entities
4. Change 3 (`_encode_part_ref`) — replaces opaque `pass`
5. Change 4 (LoweringPass warning) — surfaces variable-size ambiguity

Estimated scope: ~50 lines added, 2 lines changed. Existing 241 passing tests should be
unaffected — the new analysis path only activates when `PartRef` objects exist and have
downstream dependents.
