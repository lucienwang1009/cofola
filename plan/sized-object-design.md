# Design Plan: Introducing `exact_size` (analogous to `SizedObject`)

## Problem Statement

The legacy `SizedObject` distinguished between:
- `size: int | None` — exact known size
- `max_size: int` — conservative upper bound

The current IR analysis only has `max_size` in `SetInfo`/`BagInfo`. When the size of an
object is definitively known at analysis time (e.g., `SetInit`, `BagInit`,
`SetChoose(size=k)`), the encoder still creates a WFOMC polynomial variable (`get_obj_var`)
where an integer constant would suffice — or worse, the `max_size` is imprecise (e.g.,
`SetDifference.max_size` inherits from `left.max_size` without subtracting `right`'s
contribution, which causes wrong TupleDef domain sizes in lowering).

Additionally, `LoweringPass` currently requires tuple sources to have a statically-known
size. The `exact_size` field unifies how this is communicated: whether the size comes from
the object definition itself (EntityAnalysis) or is derived from a constraint (`|S| == k`)
via MaxSizeInference.

---

## Change 1 — Add `exact_size` to `SetInfo` and `BagInfo`

**File**: `src/cofola/ir/analysis/entities.py`

```python
@dataclass
class SetInfo:
    p_entities: set[Entity]
    max_size: int
    exact_size: int | None = None   # NEW: known if size is always fixed

@dataclass
class BagInfo:
    p_entities_multiplicity: dict[Entity, int]
    max_size: int
    dis_entities: set[Entity] = field(default_factory=set)
    indis_entities: dict[int, set[Entity]] = field(default_factory=dict)
    exact_size: int | None = None   # NEW
```

`exact_size = None` means the size is variable (not statically known).
`exact_size = k` means every instance of this object always has exactly `k` elements.

---

## Change 2 — Propagate `exact_size` in `EntityAnalysis`

**File**: `src/cofola/ir/analysis/entities.py`

| Object type | `exact_size` | Notes |
|---|---|---|
| `SetInit(entities)` | `len(entities)` | Always fixed |
| `BagInit(entity_multiplicity)` | `sum(mults)` | Always fixed |
| `SetChoose(source, size=k)` | `k` if given, else `None` | `size=None` → variable |
| `BagChoose(source, size=k)` | `k` if given, else `None` | Same |
| `SetChooseReplace(source, size=k)` | `k` if given, else `None` | Same |
| `SetUnion(A, B)` | `None` | Overlap unknown |
| `SetIntersection(A, B)` | `None` | Overlap unknown |
| `SetDifference(A, B)` | `None`* | See note below |
| `BagUnion/Intersection/Difference` | `None` | Conservative |
| `BagAdditiveUnion(A, B)` | `A.exact_size + B.exact_size` if both known **and entity sets are disjoint** | Disjointness required: if A and B share an entity, the combined multiplicity may not equal A.exact + B.exact |
| `BagSupport(source)` | `None` | Depends on multiplicities |
| `PartRef` | `None` | Variable by construction |

\* **`SetDifference` `max_size` improvement** (bonus, fixes Problem 254):
If the right operand has a known `exact_size`, tighten the bound:
`max_size = max(0, left.max_size - right.exact_size)`.
Currently `_analyze_set_difference` copies `left.max_size` verbatim — too loose.
Note: `exact_size` for `SetDifference` itself remains `None` unless the LP separately
pins it to an exact value (overlap between left and right is unknown at analysis time).

---

## Change 3 — Extend `MaxSizeInference` to infer `exact_size`

**File**: `src/cofola/ir/analysis/max_size.py`

### 3a — New return type

Replace `dict[ObjRef, int]` with a result dataclass:

```python
@dataclass
class SizeInferenceResult:
    max_sizes: dict[ObjRef, int]       # tighter upper bounds (existing)
    exact_sizes: dict[ObjRef, int]     # NEW: LP-proven exact sizes
    unsatisfiable: bool = False        # NEW: True if constraints are contradictory
```

### 3b — Algorithm extension

After solving LP maximization (existing), also solve LP **minimization** for each ref:

```python
# Minimize: find smallest possible size for this variable
c_min = np.zeros(n_vars)
c_min[index] = 1   # minimize x[index]
ret_min = linprog(c_min, A_ub=A_u, b_ub=b_u, A_eq=A_e, b_eq=b_e)
```

Then:
- If `max_result == min_result` (both LP solves succeed and give the same integer value)
  → `exact_sizes[ref] = max_result`
- If the LP is infeasible (`not ret.success` due to infeasibility, not just unboundedness)
  → `result.unsatisfiable = True`

### 3c — Conflict detection with EntityAnalysis

After computing LP-based `exact_sizes`, check against existing `exact_size` from EntityAnalysis:

```python
for ref, lp_exact in exact_sizes.items():
    info = analysis.set_info.get(ref) or analysis.bag_info.get(ref)
    if info is not None and info.exact_size is not None:
        if info.exact_size != lp_exact:
            logger.info(
                "MaxSizeInference: conflict on ref={}: "
                "EntityAnalysis.exact_size={} vs LP exact_size={}",
                ref.id, info.exact_size, lp_exact,
            )
            result.unsatisfiable = True
```

**Example**: `S = choose(people, 3)` → `EntityAnalysis.exact_size = 3`.
If the problem also has `|S| == 4` → LP infers `exact_size = 4`.
Conflict detected → `unsatisfiable = True` → pipeline returns 0.

---

## Change 4 — Update pipeline to use `SizeInferenceResult`

**File**: `src/cofola/ir/pipeline.py`

### 4a — Rename `_merge_max_sizes` → `_merge_size_inference`

```python
def _merge_size_inference(
    self, analysis: AnalysisResult, result: SizeInferenceResult
) -> bool:
    """Merge inferred sizes into analysis. Returns True if unsatisfiable."""
    if result.unsatisfiable:
        return True

    # Merge max_sizes (existing logic)
    for ref, size in result.max_sizes.items():
        if ref in analysis.set_info:
            analysis.set_info[ref].max_size = min(analysis.set_info[ref].max_size, size)
        elif ref in analysis.bag_info:
            analysis.bag_info[ref].max_size = min(analysis.bag_info[ref].max_size, size)

    # NEW: Merge exact_sizes
    for ref, exact in result.exact_sizes.items():
        if ref in analysis.set_info:
            info = analysis.set_info[ref]
            if info.exact_size is not None and info.exact_size != exact:
                return True  # conflict
            info.exact_size = exact
            info.max_size = min(info.max_size, exact)  # tighten upper bound too
        elif ref in analysis.bag_info:
            info = analysis.bag_info[ref]
            if info.exact_size is not None and info.exact_size != exact:
                return True  # conflict
            info.exact_size = exact
            info.max_size = min(info.max_size, exact)

    return False
```

### 4b — Early return in `solve` when unsatisfiable

```python
# Stage 3: Infer max sizes (LP)
result = MaxSizeInference().run(problem, analysis)
if self._merge_size_inference(analysis, result):
    logger.info("MaxSizeInference: problem is unsatisfiable (size conflict) → 0")
    return 0
```

---

## Change 5 — Add `get_size_expr` to `ContextIR`

**File**: `src/cofola/backend/wfomc/context_ir.py`

Add a new method alongside `get_obj_var`:

```python
def get_size_expr(self, ref: ObjRef) -> Expr | int:
    """Return the size expression for an object.

    If the object has a known exact_size in the analysis, returns the integer
    directly (no WFOMC variable created).  Otherwise falls back to get_obj_var()
    to create/reuse a polynomial variable.
    """
    info = self.analysis.set_info.get(ref) or self.analysis.bag_info.get(ref)
    if info is not None and info.exact_size is not None:
        return info.exact_size   # plain Python int — no symbolic variable needed
    return self.get_obj_var(ref)
```

This is the direct IR equivalent of `SizedObject.encode_size_var()` from the legacy system,
which also returned a constant when the size was known.

---

## Change 6 — Update encoder to use `get_size_expr`

**File**: `src/cofola/backend/wfomc/encoder_ir.py`

Replace `get_obj_var` with `get_size_expr` at size-related call sites:

| Line | Current | Change to |
|---|---|---|
| 772 | `domain_size = context.get_obj_var(defn.domain)` | `domain_size = context.get_size_expr(defn.domain)` |
| 383 | `var = context.get_obj_var(ref)` (SetChoose) | `var = context.get_size_expr(ref)` |
| 1074 | `var = context.get_obj_var(part)` (PartitionDef parts) | `var = context.get_size_expr(part)` |
| 1207 | `var = context.get_obj_var(term)` (SizeConstraint term) | `var = context.get_size_expr(term)` |

**Do NOT change** the `get_obj_var` call at the `SetChooseReplace` encoder (~line 420):
```python
shared_var = context.get_obj_var(ref, set_weight=False)
```
This variable is used as a **symbolic weight in the WFOMC polynomial**
(`1 + x + x² + ...`) and must remain symbolic regardless of `exact_size`.
If an exact size is known, the validator's `Eq(shared_var, defn.size)` will constrain it.

When `get_size_expr` returns an `int`, the WFOMC validator receives `Eq(obj_var, k)` with a
literal — valid, and the Decoder coefficient extraction becomes trivial for that term.

---

## Change 7 — Update `LoweringPass` to use `exact_size`

**File**: `src/cofola/ir/passes/lowering.py`

Both `_try_lower_tuples` and `_try_lower_sequences` need the same fix.

**Current `_try_lower_tuples`** falls back to `max_size` when `defn.size is None`, then
(after `partref-as-source.md`) would silently use an imprecise upper bound.

**Current `_try_lower_sequences`** (line ~457–462) does the same:
```python
if size is None:
    bag_info = analysis.bag_info.get(defn.source)
    if bag_info is None:
        continue
    size = bag_info.max_size   # ← imprecise; should use exact_size
```

**Fix for both**: require `exact_size` to be known; raise otherwise.

```python
# _try_lower_tuples — replace the size=None block:
size = defn.size
if size is None:
    info = analysis.set_info.get(source) or analysis.bag_info.get(source)
    if info is not None and info.exact_size is not None:
        size = info.exact_size
    else:
        raise ValueError(
            f"TupleDef {ref.id}: tuple size must be specified explicitly. "
            "Use 'choose k tuple from <source>' with an explicit k, "
            "or add a size constraint '|<source>| == k'."
        )

# _try_lower_sequences — replace the size=None block:
size = defn.size
if size is None:
    info = analysis.bag_info.get(defn.source) or analysis.set_info.get(defn.source)
    if info is not None and info.exact_size is not None:
        size = info.exact_size
    else:
        raise ValueError(
            f"SequenceDef {ref.id}: sequence size must be specified explicitly. "
            "Use 'choose k sequence from <source>' with an explicit k, "
            "or add a size constraint '|<source>| == k'."
        )
```

**Outcome table** (applies to both TupleDef and SequenceDef):

| Source type | `exact_size` source | Result |
|---|---|---|
| `SetInit{a,b,c}` | EntityAnalysis → `3` | ✓ size = 3 |
| `SetChoose(S, size=k)` | EntityAnalysis → `k` | ✓ size = k |
| `BagInit{...}` | EntityAnalysis → `sum(mults)` | ✓ size = sum |
| `SetChoose(S)` (no size) | `None` | ✗ raises ValueError |
| `PartRef` with `\|part\| == k` | MaxSizeInference → `k` | ✓ size = k |
| `PartRef` without constraint | `None` | ✗ raises ValueError |

---

## Change 8 — Update `all.json` problem programs

**File**: `problems/all.json`

36 existing problem programs use bare `tuple(source)` where the source is a `SetInit` or
`SetChoose(size=k)`. Since `exact_size` will now be propagated for these cases, no change
to the CFL programs is needed — `LoweringPass` will find `exact_size` from EntityAnalysis
and proceed correctly.

No changes to `all.json` are required.

---

## Summary of Changes

```
entities.py          — SetInfo + BagInfo gain `exact_size: int | None` field
                       EntityAnalysis propagates exact_size during bottom-up pass
                       SetDifference.max_size tightened: max(0, left.max_size - right.exact_size)
                       BagAdditiveUnion.exact_size = A + B only when entity sets are disjoint

max_size.py          — New SizeInferenceResult dataclass (max_sizes, exact_sizes, unsatisfiable)
                       Also solves minimization LP: if min==max → exact_size
                       Conflict detection: LP exact_size vs EntityAnalysis exact_size → unsat
                       unsatisfiable flag when LP is infeasible or constraints contradict

pipeline.py          — _merge_max_sizes → _merge_size_inference (returns bool)
                       Early return 0 if unsatisfiable (before lowering / Shannon)

context_ir.py        — new get_size_expr(ref) → int (exact) or Expr (symbolic)

encoder_ir.py        — 4 call sites changed: get_obj_var → get_size_expr
                       (SetChooseReplace ~line 420 intentionally excluded — needs symbolic var)

lowering.py          — _try_lower_tuples: use exact_size; raise ValueError if None
                       _try_lower_sequences: same fix (was silently using max_size)
```

## What Does NOT Change

- `frontend/objects.py` — object definitions remain immutable, no size fields added there
- `ir/passes/optimize.py`, `simplify.py` — no changes needed
- `AnalysisResult` dataclass fields — `set_info`/`bag_info` dicts remain the same structure,
  just the values gain `exact_size`
- `problems/all.json` — existing programs are unaffected (exact_size inferred automatically)
- Public API (`solve`, `parse_and_solve`) — no interface changes

## Interaction with PartRef

After implementing the `PartRef` analysis fix (`partref-as-source.md`), `PartRef` objects
get `SetInfo`/`BagInfo` from EntityAnalysis with `exact_size = None` (since part sizes are
variable). MaxSizeInference will set `exact_size = k` for a PartRef if the problem contains
`|part| == k`. LoweringPass then picks up that exact_size, making `tuple(part)` valid
when the part size is constrained.

```
# Valid after this design (size determined via MaxSizeInference):
partition P = partition S into 3
|P[0]| == 2
t = tuple(P[0])     # exact_size=2 inferred → size=2 ✓

# Still raises ValueError (no constraint, exact_size=None):
t = tuple(P[0])
```
