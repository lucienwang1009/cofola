# Cofola IR Refactor: Immutable Dataclass IR + Analysis Passes

## 1. Requirements Restatement

### Pain Points (from user)
1. **`CofolaProblem` is not a dataclass** — frontend passes mutate it in-place, causing hard-to-trace side effects
2. **`_fields` / `dependences` / `subs_args` / `subs_obj` too ugly** — hand-rolled dataclass + dependency graph + visitor pattern
3. **`inherit()` called everywhere** — derived properties (p_entities, max_size, dis_entities) mixed into IR structure

### Constraints (from user)
- **Backend untouched**: the WFOMC encoder (`backend/wfomc/encoder.py`, `context.py`) must NOT be modified. The interface it sees must be identical.
- **Tuple→Function lowering migrated**: `LoweringPass` moves to the new IR
- **Full syntax compatibility**: parser output changes internally, but `.cofola` syntax is identical

### Core Design Principle
> **Separate structure (what the IR IS) from analysis (what we can COMPUTE about it).**

---

## 2. Architecture Overview

```
                    ┌──────────────┐
   .cofola text ──▶ │   Parser     │──▶ ProblemBuilder (mutable)
                    └──────────────┘          │
                                              ▼ .build()
                                     ┌─────────────────┐
                                     │ Problem (frozen) │
                                     └────────┬────────┘
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
                    ▼                         ▼                          ▼
             ConstantFolder           EntityAnalysis              MaxSizeAnalysis
             (Rewriter)               (Analysis Pass)            (Analysis Pass)
                    │                         │                          │
                    ▼                         ▼                          ▼
             Problem' (new frozen)    dict[ObjRef, EntityInfo]   dict[ObjRef, int]
                    │                         │                          │
                    └─────────────────────────┼──────────────────────────┘
                                              ▼
                                     ┌─────────────────┐
                                     │   LoweringPass   │
                                     └────────┬────────┘
                                              ▼
                                     Problem'' (lowered)
                                              │
                                     ┌────────┴────────┐
                                     │  Adapter Layer   │ ← converts to legacy objects
                                     └────────┬────────┘
                                              ▼
                                     CofolaProblem (legacy)
                                              │
                                     ┌────────┴────────┐
                                     │  WFOMC Encoder   │ ← UNTOUCHED
                                     └─────────────────┘
```

The key insight: an **Adapter Layer** converts the new frozen IR into the legacy mutable objects that the backend expects. This allows a clean separation without touching the encoder.

---

## 3. New IR Design

### 3.1 Core Types (`cofola/ir/types.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Union

@dataclass(frozen=True)
class ObjRef:
    """Unique reference to an object in the IR graph."""
    id: int

    def __hash__(self): return self.id
    def __eq__(self, o): return isinstance(o, ObjRef) and self.id == o.id

@dataclass(frozen=True)
class Entity:
    """An atomic entity (e.g., 'A', 'math1', 'key3')."""
    name: str
```

### 3.2 Object Definitions (`cofola/ir/objects.py`)

All object definitions are frozen dataclasses. They store **only structural information** — no derived properties like `p_entities`, `max_size`, `dis_entities`.

```python
# --- Base Sets ---
@dataclass(frozen=True)
class SetInit:
    entities: frozenset[Entity]

@dataclass(frozen=True)
class SetChoose:
    source: ObjRef
    size: int | None = None

@dataclass(frozen=True)
class SetChooseReplace:
    source: ObjRef
    size: int | None = None

@dataclass(frozen=True)
class SetUnion:
    left: ObjRef
    right: ObjRef

@dataclass(frozen=True)
class SetIntersection:
    left: ObjRef
    right: ObjRef

@dataclass(frozen=True)
class SetDifference:
    left: ObjRef
    right: ObjRef

# --- Bags ---
@dataclass(frozen=True)
class BagInit:
    entity_multiplicity: tuple[tuple[Entity, int], ...]
    # Use tuple of pairs for hashability; ordered by entity name

@dataclass(frozen=True)
class BagChoose:
    source: ObjRef
    size: int | None = None

@dataclass(frozen=True)
class BagAdditiveUnion:
    left: ObjRef
    right: ObjRef

@dataclass(frozen=True)
class BagUnion:
    left: ObjRef
    right: ObjRef

@dataclass(frozen=True)
class BagIntersection:
    left: ObjRef
    right: ObjRef

@dataclass(frozen=True)
class BagDifference:
    left: ObjRef
    right: ObjRef

@dataclass(frozen=True)
class BagSupport:
    source: ObjRef

# --- Functions ---
@dataclass(frozen=True)
class FuncDef:
    domain: ObjRef
    codomain: ObjRef
    injective: bool = False
    surjective: bool = False

@dataclass(frozen=True)
class FuncImage:
    func: ObjRef
    argument: ObjRef  # Set or Entity ref

@dataclass(frozen=True)
class FuncInverseImage:
    func: ObjRef
    argument: ObjRef

# --- Tuples ---
@dataclass(frozen=True)
class TupleDef:
    source: ObjRef
    choose: bool = False
    replace: bool = False
    size: int | None = None

# --- Sequences ---
@dataclass(frozen=True)
class SequenceDef:
    source: ObjRef
    choose: bool = False
    replace: bool = False
    size: int | None = None
    circular: bool = False
    reflection: bool = False

# --- Partitions ---
@dataclass(frozen=True)
class PartitionDef:
    source: ObjRef
    num_parts: int
    ordered: bool  # True = composition

@dataclass(frozen=True)
class PartRef:
    """Reference to the i-th part of a partition/composition."""
    partition: ObjRef
    index: int

# Union of all object definition types
ObjDef = Union[
    SetInit, SetChoose, SetChooseReplace,
    SetUnion, SetIntersection, SetDifference,
    BagInit, BagChoose, BagAdditiveUnion,
    BagUnion, BagIntersection, BagDifference, BagSupport,
    FuncDef, FuncImage, FuncInverseImage,
    TupleDef, SequenceDef,
    PartitionDef, PartRef,
]
```

### 3.3 Constraint Definitions (`cofola/ir/constraints.py`)

```python
@dataclass(frozen=True)
class SizeConstraint:
    terms: tuple[tuple[ObjRef, int], ...]  # (obj_ref, coefficient)
    comparator: str  # "==", "<", "<=", ">", ">="
    rhs: int

@dataclass(frozen=True)
class MembershipConstraint:
    entity: Entity
    container: ObjRef
    positive: bool = True

@dataclass(frozen=True)
class SubsetConstraint:
    sub: ObjRef
    sup: ObjRef
    positive: bool = True

@dataclass(frozen=True)
class DisjointConstraint:
    left: ObjRef
    right: ObjRef
    positive: bool = True

@dataclass(frozen=True)
class EqualityConstraint:
    left: ObjRef
    right: ObjRef
    positive: bool = True

@dataclass(frozen=True)
class TupleIndexEq:
    tuple_ref: ObjRef
    index: int
    entity: Entity
    positive: bool = True

@dataclass(frozen=True)
class TupleIndexMembership:
    tuple_ref: ObjRef
    index: int
    container: ObjRef
    positive: bool = True

@dataclass(frozen=True)
class SequencePatternConstraint:
    seq: ObjRef
    pattern: SeqPattern
    positive: bool = True

@dataclass(frozen=True)
class FuncPairConstraint:
    func: ObjRef
    arg_entity: Entity
    result: ObjRef | Entity
    positive: bool = True

# --- Sequence Patterns ---
@dataclass(frozen=True)
class TogetherPattern:
    group: ObjRef

@dataclass(frozen=True)
class LessThanPattern:
    left: ObjRef | Entity
    right: ObjRef | Entity

@dataclass(frozen=True)
class PredecessorPattern:
    first: ObjRef | Entity
    second: ObjRef | Entity

@dataclass(frozen=True)
class NextToPattern:
    first: ObjRef | Entity
    second: ObjRef | Entity

SeqPattern = Union[TogetherPattern, LessThanPattern, PredecessorPattern, NextToPattern]

# --- Size atoms for ordered objects ---
@dataclass(frozen=True)
class TupleCountAtom:
    """Size atom: T.count(S) — positions in T occupied by elements of S."""
    tuple_ref: ObjRef
    count_obj: ObjRef
    deduplicate: bool = False

@dataclass(frozen=True)
class SeqPatternCountAtom:
    """Size atom: seq.count(pattern) — occurrences of pattern in seq."""
    seq: ObjRef
    pattern: SeqPattern

@dataclass(frozen=True)
class BagCountAtom:
    """Size atom: B.count(e) — multiplicity of entity e in bag B."""
    bag: ObjRef
    entity: Entity

# Compound constraints
@dataclass(frozen=True)
class NotConstraint:
    sub: Constraint

@dataclass(frozen=True)
class AndConstraint:
    left: Constraint
    right: Constraint

@dataclass(frozen=True)
class OrConstraint:
    left: Constraint
    right: Constraint

# Part-level constraints
@dataclass(frozen=True)
class ForAllParts:
    """Constraint applied to every part: `cst for part in P`."""
    partition: ObjRef
    constraint_template: Constraint  # Contains a sentinel PartRef

Constraint = Union[
    SizeConstraint, MembershipConstraint, SubsetConstraint,
    DisjointConstraint, EqualityConstraint,
    TupleIndexEq, TupleIndexMembership,
    SequencePatternConstraint, FuncPairConstraint,
    NotConstraint, AndConstraint, OrConstraint,
    ForAllParts,
]
```

### 3.4 Problem & Builder (`cofola/ir/problem.py`)

```python
from dataclasses import dataclass, fields
from types import MappingProxyType

@dataclass(frozen=True)
class Problem:
    """Immutable IR for a combinatorics counting problem."""
    defs: dict[ObjRef, ObjDef]
    constraints: tuple[Constraint, ...]
    names: dict[ObjRef, str]  # user-given identifiers (for debugging/encoding)

    # --- Query helpers (all pure, no mutation) ---

    def get_refs(self, defn: ObjDef) -> list[ObjRef]:
        """Extract all ObjRef fields from an IR node."""
        result = []
        for f in fields(defn):
            val = getattr(defn, f.name)
            if isinstance(val, ObjRef):
                result.append(val)
        return result

    def dep_graph(self) -> dict[ObjRef, list[ObjRef]]:
        """Build adjacency list: ref → list of refs it depends on."""
        return {ref: self.get_refs(defn) for ref, defn in self.defs.items()}

    def topological_order(self) -> list[ObjRef]:
        """Kahn's algorithm over dependency graph."""
        graph = self.dep_graph()
        in_degree = {ref: 0 for ref in self.defs}
        for ref, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[ref] += 1  # 这里要注意：dep → ref 的边
        # ... standard Kahn's ...

    def substitute(self, old_ref: ObjRef, new_ref: ObjRef) -> 'Problem':
        """Replace all uses of old_ref with new_ref. Returns new Problem."""
        def sub_field(val):
            if isinstance(val, ObjRef) and val == old_ref:
                return new_ref
            return val

        new_defs = {}
        for ref, defn in self.defs.items():
            if ref == old_ref:
                continue
            new_fields = {
                f.name: sub_field(getattr(defn, f.name))
                for f in fields(defn)
            }
            new_defs[ref] = type(defn)(**new_fields)
        if new_ref not in new_defs and new_ref in self.defs:
            new_defs[new_ref] = self.defs[new_ref]
        # Also substitute in constraints...
        new_constraints = tuple(
            self._sub_constraint(c, old_ref, new_ref) for c in self.constraints
        )
        return Problem(defs=new_defs, constraints=new_constraints, names=self.names)


class ProblemBuilder:
    """Mutable builder used by the parser to construct a Problem."""

    def __init__(self):
        self._next_id: int = 0
        self._defs: dict[ObjRef, ObjDef] = {}
        self._constraints: list[Constraint] = []
        self._names: dict[ObjRef, str] = {}

    def add(self, defn: ObjDef, name: str | None = None) -> ObjRef:
        ref = ObjRef(self._next_id)
        self._next_id += 1
        self._defs[ref] = defn
        if name:
            self._names[ref] = name
        return ref

    def add_constraint(self, c: Constraint) -> None:
        self._constraints.append(c)

    def find_equivalent(self, defn: ObjDef) -> ObjRef | None:
        """Find existing ref with an equivalent definition (deduplication)."""
        for ref, existing in self._defs.items():
            if existing == defn:  # frozen dataclass __eq__ does structural comparison
                return ref
        return None

    def build(self) -> Problem:
        return Problem(
            defs=dict(self._defs),
            constraints=tuple(self._constraints),
            names=dict(self._names),
        )
```

---

## 4. Analysis Passes (replace `inherit()`)

### 4.1 Entity Analysis (`cofola/ir/analysis/entities.py`)

Replaces: `Set.p_entities`, `Bag.p_entities_multiplicity`, `problem.entities`, `problem.singletons`

```python
@dataclass
class SetInfo:
    p_entities: set[Entity]
    max_size: int

@dataclass
class BagInfo:
    p_entities_multiplicity: dict[Entity, int]
    max_size: int
    dis_entities: set[Entity] = field(default_factory=set)
    indis_entities: dict[int, set[Entity]] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    set_info: dict[ObjRef, SetInfo]
    bag_info: dict[ObjRef, BagInfo]
    all_entities: set[Entity]
    singletons: set[Entity]

class EntityAnalysis:
    def run(self, problem: Problem) -> AnalysisResult:
        """Single-pass bottom-up analysis over topological order."""
        set_info, bag_info = {}, {}
        for ref in problem.topological_order():
            defn = problem.defs[ref]
            match defn:
                case SetInit(entities=e):
                    set_info[ref] = SetInfo(p_entities=set(e), max_size=len(e))
                case SetChoose(source=src, size=k):
                    parent = set_info[src]
                    max_s = min(k, parent.max_size) if k else parent.max_size
                    set_info[ref] = SetInfo(p_entities=parent.p_entities.copy(), max_size=max_s)
                case SetUnion(left=l, right=r):
                    set_info[ref] = SetInfo(
                        p_entities=set_info[l].p_entities | set_info[r].p_entities,
                        max_size=set_info[l].max_size + set_info[r].max_size,
                    )
                case BagInit(entity_multiplicity=em):
                    em_dict = dict(em)
                    bag_info[ref] = BagInfo(
                        p_entities_multiplicity=em_dict,
                        max_size=sum(em_dict.values()),
                    )
                # ... all other cases ...
        all_entities = set()
        for info in set_info.values():
            all_entities |= info.p_entities
        for info in bag_info.values():
            all_entities |= set(info.p_entities_multiplicity.keys())
        singletons = self._compute_singletons(set_info, bag_info)
        return AnalysisResult(set_info, bag_info, all_entities, singletons)
```

### 4.2 Max Size Inference (`cofola/ir/analysis/max_size.py`)

Replaces: `InferMaxSizePass` — uses LP to tighten `max_size` bounds from size constraints.

```python
class MaxSizeInference:
    def run(self, problem: Problem, entity_result: AnalysisResult) -> dict[ObjRef, int]:
        """LP-based max_size inference. Returns refined max_size per ref."""
        # Build LP from SizeConstraints in problem.constraints
        # Uses entity_result for initial bounds
        ...
```

### 4.3 Bag Preprocessing (`cofola/ir/analysis/bag_classify.py`)

Replaces: `preprocess_bags()` in encoder — classifies entities into `dis_entities` / `indis_entities`.

```python
class BagClassification:
    def run(self, problem: Problem, entity_result: AnalysisResult) -> AnalysisResult:
        """Classify bag entities as distinguishable/indistinguishable.
        Mutates entity_result.bag_info in-place and returns it."""
        ...
```

---

## 5. Rewriter Framework (replace `subs_obj` / `problem.replace`)

### 5.1 Base Rewriter (`cofola/ir/rewriter.py`)

```python
class Rewriter:
    """Base class for IR-to-IR transformation passes."""

    def rewrite(self, problem: Problem) -> Problem:
        new_defs = dict(problem.defs)
        removals = set()
        additions = {}  # new ObjRef → ObjDef
        new_constraints = list(problem.constraints)

        for ref in problem.topological_order():
            defn = problem.defs[ref]
            result = self.visit(ref, defn, problem)
            if result is not None:
                new_defs[ref] = result

        # Apply removals and additions
        for ref in removals:
            del new_defs[ref]
        new_defs.update(additions)

        return Problem(
            defs=new_defs,
            constraints=tuple(new_constraints),
            names=problem.names,
        )

    def visit(self, ref: ObjRef, defn: ObjDef, problem: Problem) -> ObjDef | None:
        """Override in subclass. Return new ObjDef to replace, or None to keep."""
        return None
```

### 5.2 Constant Folder (`cofola/ir/passes/optimize.py`)

```python
class ConstantFolder(Rewriter):
    def visit(self, ref, defn, problem):
        match defn:
            case SetUnion(left=l, right=r) if (
                isinstance(problem.defs[l], SetInit) and
                isinstance(problem.defs[r], SetInit)
            ):
                return SetInit(
                    entities=problem.defs[l].entities | problem.defs[r].entities
                )
            case BagAdditiveUnion(left=l, right=r) if (
                isinstance(problem.defs[l], BagInit) and
                isinstance(problem.defs[r], BagInit)
            ):
                # Merge multiplicities
                ...
            case BagSupport(source=src) if isinstance(problem.defs[src], BagInit):
                bag = problem.defs[src]
                return SetInit(entities=frozenset(e for e, _ in bag.entity_multiplicity))
        return None
```

### 5.3 Lowering Pass (`cofola/ir/passes/lowering.py`)

Migrates `frontend/passes/transform.py` to new IR. This is the most complex pass.

```python
class LoweringPass:
    """Lowers TupleDef → FuncDef + SetInit(indices), etc."""

    def __init__(self):
        self._next_id = 10000  # Avoid collision with builder IDs

    def run(self, problem: Problem, analysis: AnalysisResult) -> Problem:
        new_defs = dict(problem.defs)
        new_constraints = list(problem.constraints)
        new_names = dict(problem.names)

        for ref in problem.topological_order():
            defn = problem.defs[ref]
            match defn:
                case TupleDef(source=src, choose=choose, replace=replace, size=size):
                    self._lower_tuple(ref, defn, new_defs, new_constraints,
                                      new_names, analysis)
                case SequenceDef():
                    self._lower_sequence(ref, defn, new_defs, new_constraints,
                                         new_names, analysis)
                case FuncDef(injective=True):
                    self._lower_injective_func(ref, defn, new_defs,
                                                new_constraints, analysis)

        return Problem(
            defs=new_defs,
            constraints=tuple(new_constraints),
            names=new_names,
        )

    def _lower_tuple(self, ref, defn, defs, constraints, names, analysis):
        """TupleDef → indices SetInit + FuncDef(indices → source)."""
        size = defn.size or analysis.get_size(defn.source)
        # Create index entities
        idx_entities = frozenset(Entity(f"idx_{i}") for i in range(size))
        idx_ref = self._new_ref()
        defs[idx_ref] = SetInit(entities=idx_entities)
        names[idx_ref] = f"{names.get(ref, 'T')}_indices"

        # Create mapping function
        func_ref = self._new_ref()
        defs[func_ref] = FuncDef(
            domain=idx_ref,
            codomain=defn.source,
            injective=not defn.replace,
        )
        names[func_ref] = f"{names.get(ref, 'T')}_mapping"

        # Replace TupleDef with a marker that references indices + mapping
        # The backend will see FuncDef and SetInit, not TupleDef
        del defs[ref]
        # Store mapping ref so constraints can reference it
        self._tuple_to_func[ref] = func_ref
        self._tuple_to_indices[ref] = idx_ref

        # Transform TupleIndexEq constraints → FuncPairConstraint
        ...
```

---

## 6. Adapter Layer (new IR → legacy objects for backend)

This is the **key bridge** that allows the backend to remain untouched.

### 6.1 Adapter (`cofola/ir/adapter.py`)

```python
from cofola.frontend.ir import CofolaProblem
from cofola.objects.base import Entity as LegacyEntity
from cofola.objects.set import SetInit as LegacySetInit, SetChoose as LegacySetChoose, ...
from cofola.objects.bag import BagInit as LegacyBagInit, ...
# ... all legacy object imports

class LegacyAdapter:
    """Converts new IR Problem + AnalysisResult → legacy CofolaProblem."""

    def convert(self, problem: Problem, analysis: AnalysisResult) -> CofolaProblem:
        self._ref_to_legacy: dict[ObjRef, CombinatoricsObject] = {}
        self._entity_map: dict[Entity, LegacyEntity] = {}

        legacy_problem = CofolaProblem()

        # Convert entities
        for e in analysis.all_entities:
            self._entity_map[e] = LegacyEntity(e.name)

        # Convert objects in topological order
        for ref in problem.topological_order():
            defn = problem.defs[ref]
            legacy_obj = self._convert_obj(ref, defn, problem, analysis)
            legacy_obj.name = problem.names.get(ref, legacy_obj.name)
            self._ref_to_legacy[ref] = legacy_obj
            legacy_problem.objects.append(legacy_obj)

        # Inject analysis results into legacy objects
        for ref, info in analysis.set_info.items():
            obj = self._ref_to_legacy[ref]
            if isinstance(obj, (LegacySet,)):
                obj.p_entities = {self._entity_map[e] for e in info.p_entities}
                obj.size = info.max_size if info.max_size == len(info.p_entities) else obj.size
                obj.max_size = info.max_size

        for ref, info in analysis.bag_info.items():
            obj = self._ref_to_legacy[ref]
            if isinstance(obj, (LegacyBag,)):
                obj.p_entities_multiplicity = {
                    self._entity_map[e]: m for e, m in info.p_entities_multiplicity.items()
                }
                obj.max_size = info.max_size
                obj.dis_entities = {self._entity_map[e] for e in info.dis_entities}
                obj.indis_entities = {
                    k: {self._entity_map[e] for e in v}
                    for k, v in info.indis_entities.items()
                }

        # Convert constraints
        for c in problem.constraints:
            legacy_c = self._convert_constraint(c, problem)
            legacy_problem.constraints.append(legacy_c)

        legacy_problem.entities = {self._entity_map[e] for e in analysis.all_entities}
        legacy_problem.singletons = {self._entity_map[e] for e in analysis.singletons}

        return legacy_problem

    def _convert_obj(self, ref, defn, problem, analysis):
        match defn:
            case SetInit(entities=e):
                return LegacySetInit(*[self._entity_map[ent] for ent in e])
            case SetChoose(source=src, size=k):
                return LegacySetChoose(self._ref_to_legacy[src], k)
            case SetUnion(left=l, right=r):
                return LegacySetUnion(self._ref_to_legacy[l], self._ref_to_legacy[r])
            case BagInit(entity_multiplicity=em):
                return LegacyBagInit(
                    *[(self._entity_map[e], m) for e, m in em]
                )
            case FuncDef(domain=d, codomain=c, injective=inj, surjective=sur):
                return LegacyFuncInit(
                    self._ref_to_legacy[d], self._ref_to_legacy[c], inj, sur
                )
            # ... all object types
```

---

## 7. New Pipeline (`cofola/ir/pipeline.py`)

```python
class IRPipeline:
    """New pipeline using immutable IR."""

    def solve(self, text: str) -> int:
        # Stage 1: Parse → ProblemBuilder → Problem
        builder = parse_cofola(text)  # returns ProblemBuilder
        problem = builder.build()

        # Stage 2: Analysis
        entity_analysis = EntityAnalysis().run(problem)

        # Stage 3: Optimize (constant folding)
        problem = ConstantFolder().rewrite(problem)
        entity_analysis = EntityAnalysis().run(problem)  # Re-analyze

        # Stage 4: Infer max sizes (LP)
        max_sizes = MaxSizeInference().run(problem, entity_analysis)
        # Merge into entity_analysis
        for ref, ms in max_sizes.items():
            if ref in entity_analysis.set_info:
                entity_analysis.set_info[ref].max_size = min(
                    entity_analysis.set_info[ref].max_size, ms
                )

        # Stage 5: Lowering (tuple→func, injective→constraint, etc.)
        problem = LoweringPass().run(problem, entity_analysis)
        entity_analysis = EntityAnalysis().run(problem)  # Re-analyze lowered

        # Stage 6: Simplify (remove unused objects)
        problem = SimplifyPass().run(problem)

        # Stage 7: Bag classification
        entity_analysis = BagClassification().run(problem, entity_analysis)

        # Stage 8: Convert to legacy and solve
        legacy_problem = LegacyAdapter().convert(problem, entity_analysis)
        return self._solve_legacy(legacy_problem)

    def _solve_legacy(self, problem: CofolaProblem) -> int:
        """Delegate to existing WFOMC backend (untouched)."""
        from cofola.backend.wfomc.encoder import WFOMCEncoder
        ...
```

---

## 8. Parser Migration (`cofola/parser/transformer.py`)

The parser's grammar file (`grammar.py`) stays **completely unchanged** — same `.cofola` syntax.

Only the transformer changes: instead of creating legacy objects, it creates `ObjDef` instances and adds them to a `ProblemBuilder`.

```python
class NewCofolaTransformer(Transformer):
    def __init__(self):
        self.builder = ProblemBuilder()
        self.id2ref: dict[str, ObjRef] = {}

    def object_declaration(self, args):
        obj_id, defn = args
        ref = self.builder.add(defn, name=str(obj_id))
        self.id2ref[str(obj_id)] = ref
        return ref

    def base_object_init(self, args):
        obj_type, entities = args[0], args[1:]
        if obj_type == "set":
            return SetInit(entities=frozenset(entities))
        elif obj_type == "bag":
            return BagInit(entity_multiplicity=tuple(entities))

    def common_operations(self, args):
        op_type, ref, size, op_arg = ...
        match op_type:
            case "choose":
                # Determine if source is set or bag from builder
                return SetChoose(source=ref, size=size)
            case "tuple":
                return TupleDef(source=ref)
            case "choose_tuple":
                return TupleDef(source=ref, choose=True, size=size)
            case "sequence":
                return SequenceDef(source=ref)
            case "circle":
                return SequenceDef(source=ref, circular=True, reflection=op_arg)
            case "partition":
                return PartitionDef(source=ref, num_parts=size, ordered=False)
            case "compose":
                return PartitionDef(source=ref, num_parts=size, ordered=True)
            # ...

    def size_constraint(self, args):
        expr, comp, param = args
        terms = tuple((ref, coef) for ref, coef in expr)
        self.builder.add_constraint(SizeConstraint(terms=terms, comparator=comp, rhs=param))

    def cofola(self, args):
        return self.builder
```

---

## 9. Implementation Phases

### Phase 1: New IR Core (no behavior change)
**Files**: `cofola/ir/types.py`, `cofola/ir/objects.py`, `cofola/ir/constraints.py`, `cofola/ir/problem.py`

- Define all frozen dataclasses
- Implement `Problem` with `topological_order()`, `substitute()`, `dep_graph()`
- Implement `ProblemBuilder`
- **Test**: Unit tests for Problem construction, topological order, substitute

**Estimated effort**: ~400 lines, 1-2 days

### Phase 2: Analysis Passes
**Files**: `cofola/ir/analysis/entities.py`, `cofola/ir/analysis/max_size.py`, `cofola/ir/analysis/bag_classify.py`

- Implement `EntityAnalysis` (replace `inherit()` + `propagate()`)
- Implement `MaxSizeInference` (port from `InferMaxSizePass`)
- Implement `BagClassification` (port from `preprocess_bags`)
- **Test**: Given a hand-built `Problem`, verify analysis results match legacy

**Estimated effort**: ~500 lines, 2-3 days

### Phase 3: Rewriter & Passes
**Files**: `cofola/ir/rewriter.py`, `cofola/ir/passes/optimize.py`, `cofola/ir/passes/simplify.py`, `cofola/ir/passes/lowering.py`

- Implement base `Rewriter`
- Port `ConstantFolder` (from `optimization.py`)
- Port `SimplifyPass` (from `simplification.py`)
- Port `LoweringPass` (from `transform.py`) — most complex, ~300 lines
- **Test**: Verify pass outputs match legacy pipeline

**Estimated effort**: ~600 lines, 3-4 days

### Phase 4: Adapter Layer
**Files**: `cofola/ir/adapter.py`

- Implement `LegacyAdapter.convert()` for all object/constraint types
- Inject analysis results into legacy objects
- **Test**: `new_pipeline(text)` produces same `CofolaProblem` as `old_pipeline(text)` (structural equivalence)

**Estimated effort**: ~400 lines, 2 days

### Phase 5: Parser Migration
**Files**: `cofola/parser/transformer_new.py` (new file, keep old as fallback)

- Rewrite transformer to produce `ProblemBuilder` output
- Grammar file **unchanged**
- **Test**: Parse all 293 problems from `all.json`, verify identical results

**Estimated effort**: ~350 lines, 2 days

### Phase 6: New Pipeline Integration
**Files**: `cofola/ir/pipeline.py`, `cofola/solver.py` (add flag)

- Wire everything together in `IRPipeline`
- Add `use_new_ir=True` flag to `Solver` for gradual rollout
- Run full test suite against both pipelines
- **Test**: All 293 problems produce identical counts

**Estimated effort**: ~150 lines, 1 day

### Phase 7: Cleanup
- Remove legacy `objects/` (except what backend imports)
- Remove legacy `frontend/ir.py`, `frontend/passes/`
- Keep legacy object *classes* (needed by backend adapter), but remove their `inherit()`, `subs_obj()` etc.
- Update imports

**Estimated effort**: 1-2 days

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Adapter layer introduces subtle bugs | HIGH | Run all 293 problems as regression test |
| Lowering pass is complex to port | MEDIUM | Port function-by-function, test each |
| New parser transformer misses edge cases | MEDIUM | Compare AST output for all 293 problems |
| Performance regression from creating new Problem objects | LOW | Problems are small (~10-50 objects); dict copy is O(n) |
| Backend relies on undocumented object attributes | MEDIUM | Thorough attribute audit already completed above |

---

## 11. File Structure

```
cofola/
├── ir/                          # NEW: Immutable IR
│   ├── __init__.py
│   ├── types.py                 # ObjRef, Entity
│   ├── objects.py               # All ObjDef frozen dataclasses
│   ├── constraints.py           # All Constraint frozen dataclasses
│   ├── problem.py               # Problem, ProblemBuilder
│   ├── rewriter.py              # Base Rewriter class
│   ├── adapter.py               # LegacyAdapter (new IR → legacy objects)
│   ├── pipeline.py              # IRPipeline (new entry point)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── entities.py          # EntityAnalysis (replaces inherit)
│   │   ├── max_size.py          # MaxSizeInference (replaces InferMaxSizePass)
│   │   └── bag_classify.py      # BagClassification (replaces preprocess_bags)
│   └── passes/
│       ├── __init__.py
│       ├── optimize.py          # ConstantFolder
│       ├── simplify.py          # SimplifyPass
│       └── lowering.py          # LoweringPass (tuple→func, etc.)
├── objects/                     # KEPT: legacy object classes (for backend)
│   └── ...                      # Stripped of inherit/subs_obj (adapter sets attrs)
├── frontend/                    # KEPT temporarily: legacy pipeline
│   └── ir.py                    # CofolaProblem (adapter target)
├── backend/                     # UNTOUCHED
│   └── wfomc/
│       ├── encoder.py
│       └── context.py
├── parser/
│   ├── grammar.py               # UNCHANGED
│   ├── transformer.py           # Legacy (kept as fallback)
│   └── transformer_new.py       # NEW: produces ProblemBuilder
└── solver.py                    # Updated to support both pipelines
```

---

## 12. Test

- Write tests for each new component (Problem, ProblemBuilder, EntityAnalysis, Rewriters, Adapter)
- Run full regression test on all 293 problems after Phase 4 (Adapter) to verify structural equivalence of `CofolaProblem` (`uv run python scripts/solve.py -i problem/all.json`)
- Run full regression test on all 293 problems after Phase 6 (new pipeline) to verify identical counts
- Add new tests for any edge cases discovered during migration

## 13. Summary: How Each Pain Point Is Resolved

| Pain Point | Current | After Refactor |
|-----------|---------|---------------|
| Problem not dataclass, side effects | `CofolaProblem` is mutable class with `add_object()`, `replace()` | `Problem` is `@dataclass(frozen=True)`, `ProblemBuilder` for construction |
| `_fields` / `dependences` / `subs_obj` ugly | Hand-rolled per-class metadata, manual dep tracking | `@dataclass(frozen=True)` gives `fields()` for free; `dep_graph()` computed from `ObjRef` fields; `substitute()` is generic |
| `inherit()` called everywhere | Analysis mixed into objects, must call after any mutation | `EntityAnalysis` is a separate pass, run once after each pipeline stage, no object mutation needed |
