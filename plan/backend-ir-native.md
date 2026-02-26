# Cofola Backend IR-Native Redesign

---

## 0. 当前进度

### 🎉 全部完成（Phases 1–5）

**测试状态**：22 个 encodable 测试全部通过（`uv run pytest tests/test_all_problems.py`）
**最后提交**：`a314606` — refactor(arch): complete backend IR-native redesign (Phases 1-5)

---

#### Phase 2 ✅ — `backend/wfomc/context_ir.py`
ObjRef-based Context（替代 legacy Context(CofolaProblem)）。

#### Phase 3 ✅ — `backend/wfomc/encoder_ir.py`
IR-native encoder，match-case 分派，直接消费 `ir.Problem + AnalysisResult`。

#### Phase 4 ✅ — 接入 + IRPipeline 全面通过
- `backend/base.py`：`solve(problem: Problem, analysis: AnalysisResult) -> int`
- `WFOMCBackend.solve()`：纯 IR-native 单路径
- 主要 Bug 修复：TupleMembershipConstraint、LoweringPass tuple 约束同步、
  BagClassification（PartitionDef / SetChooseReplace / BagCountAtom）、
  EntityAnalysis singletons、SetChooseReplace 编码、OrConstraint inclusion-exclusion

#### Phase 5 ✅ — Parser 直接产出 IR（消除双程转换）
- `parser/transformer.py`：重写为 IR-native，使用 `ProblemBuilder` + `id2ref`；
  引入 `TupleIndexSentinel` 处理 `T[i]` 表达式
- `parser/transformer_objects.py`：所有方法直接返回 `ObjRef`（不再产出 legacy 对象）
- `parser/transformer_constraints.py`：所有方法直接返回 IR constraint dataclasses
- `parser/parser.py`：`parse(text)` 直接返回 `ir.Problem`
- `ir/pipeline.py`：Stage 1 直接调用 `parse()`
- `ir/problem.py`：新增 `ProblemBuilder.set_name()`
- `utils.py`：移除 legacy 对象类型辅助函数

#### 遗留代码 ✅ — 全部删除
| 已删除 | 说明 |
|---|---|
| `src/cofola/objects/` | 整目录 |
| `src/cofola/backend/wfomc/encoder.py` | 旧 encoder |
| `src/cofola/backend/wfomc/context.py` | 旧 context |
| `src/cofola/pipeline.py` | 顶层旧 pipeline |
| `src/cofola/ir/parser_adapter.py` | legacy→IR 桥接 |
| `src/cofola/ir/types.py` | 移至 frontend/ |
| `src/cofola/ir/objects.py` | 移至 frontend/ |
| `src/cofola/ir/constraints.py` | 移至 frontend/ |
| `src/cofola/ir/problem.py` | 移至 frontend/ |
| `src/cofola/ir/rewriter.py` | 移至 frontend/ |

---

## 1. 目标与约束

### 目标
- **删除 `objects/` 和 `frontend/`**：后端直接消费 `ir.Problem + ir.AnalysisResult`
- **删除 `LegacyAdapter`**：消除 IR → legacy → encoder 的双程转换
- **新 WFOMC encoder**：基于 IR dataclass 的 match-case 分派，读取 `AnalysisResult` 中的分析数据
- **新 Context**：用 `ObjRef` 作字典 key，替代 legacy 对象实例

### 约束
- `wfomc` 库（外部依赖）完全不动
- `backend/wfomc/solver.py` 和 `backend/wfomc/decoder.py` 不动
- 277 个测试在整个过程中始终保持通过
- 每个阶段可独立运行测试验证

---

## 2. 实现后的数据流（已达成）

```
.cofola text
  → parser/parser.py: parse()
      CofolaTransfomer (ProblemBuilder-based, IR-native)
  → ir.Problem (frozen)
  → EntityAnalysis → AnalysisResult
  → ConstantFolder (optimize)
  → MaxSizeInference
  → LoweringPass  (TupleDef → FuncDef+SetInit, etc.)
  → SimplifyPass
  → BagClassification
  → WFOMCBackend.solve(problem, analysis)
      → encoder_ir.py: encode_ir()
      → ContextIR
      → WFOMCProblem + Decoder
  → int
```

所有旧的中间步骤（CofolaProblem、parser_adapter、LegacyAdapter、旧 encoder）均已删除。

---

## 3. 实现后目录结构（已达成）

```
cofola/
├── parser/
│   ├── grammar.py                 # 不变
│   ├── transformer.py             # ✅ IR-native，ProblemBuilder + id2ref
│   ├── transformer_objects.py     # ✅ 产出 ObjRef
│   ├── transformer_constraints.py # ✅ 产出 IR constraint dataclasses
│   ├── common.py                  # 不变
│   └── parser.py                  # ✅ parse() → ir.Problem
│
├── ir/
│   ├── types.py                   # ObjRef, Entity
│   ├── objects.py                 # IR object dataclasses
│   ├── constraints.py             # IR constraint dataclasses（含 BagSubset/BagEq）
│   ├── problem.py                 # Problem + ProblemBuilder（含 set_name()）
│   ├── rewriter.py
│   ├── pipeline.py                # ✅ IRPipeline，直接调用 parse()
│   ├── analysis/
│   │   ├── entities.py
│   │   ├── max_size.py
│   │   └── bag_classify.py
│   └── passes/
│       ├── optimize.py
│       ├── simplify.py
│       └── lowering.py
│
├── backend/
│   ├── base.py                    # solve(Problem, AnalysisResult) → int
│   └── wfomc/
│       ├── context_ir.py          # ObjRef-based Context
│       ├── encoder_ir.py          # IR-native encoder（match-case 分派）
│       ├── backend.py             # WFOMCBackend，纯 IR-native
│       ├── solver.py              # 不变
│       └── decoder.py             # 不变
│
├── solver.py                      # ✅ solve() → IRPipeline().solve()
└── utils.py                       # ✅ 移除 legacy 对象辅助函数
```

---

## 4. Phase 1：补充 IR 节点 + Lowering

### 4.1 补充 `ir/objects.py`

添加以下节点：

```python
@dataclass(frozen=True)
class FuncInverse:
    """f⁻¹ — 函数的逆映射。domain/codomain 对调。"""
    func: ObjRef

# union 类型中添加 FuncInverse
ObjDef = Union[
    ...,
    FuncDef, FuncInverse, FuncImage, FuncInverseImage,
    ...
]
```

**注意**：`FuncSubsetConstrainedImage`、`FuncDisjointConstrainedImage`、
`FuncEqConstrainedImage`、`FuncMembershipConstraintedImage` 在新设计中不进入 IR，
由新 parser 直接产出 `FuncInverseImage + SubsetConstraint` 等组合（lowering 处理遗留情况）。

### 4.2 补充 `ir/constraints.py`

```python
@dataclass(frozen=True)
class BagSubsetConstraint:
    """B₁ ⊆ B₂（按 multiplicity）"""
    sub: ObjRef
    sup: ObjRef
    positive: bool = True

@dataclass(frozen=True)
class BagEqConstraint:
    """B₁ == B₂（按 multiplicity）"""
    left: ObjRef
    right: ObjRef
    positive: bool = True

# 更新 Constraint union
Constraint = Union[
    ...,
    BagSubsetConstraint, BagEqConstraint,
    ...
]
```

### 4.3 更新 `ir/passes/lowering.py`：处理 constrained image types

`parser_adapter.py` 当前对 `FuncSubsetConstrainedImage` 等类型**未处理**（静默跳过）。
在 lowering pass 中添加一个扫描：若问题来自 legacy parser，这些类型应在 parser_adapter 阶段
就转换为 `FuncInverseImage + SubsetConstraint` 组合。

更新 `ir/parser_adapter.py` 中的 `_convert_object`：

```python
elif isinstance(obj, LegacyFuncSubsetConstrainedImage):
    func = legacy_to_ref.get(obj.func)
    subset = legacy_to_ref.get(obj.subset)
    if func and subset:
        inv_img = FuncInverseImage(func=func, argument=subset)
        inv_ref = builder.add(inv_img)
        # obj.inverse indicates direction
        builder.add_constraint(SubsetConstraint(sub=inv_ref, sup=subset, positive=obj.inverse))
        return inv_ref

elif isinstance(obj, LegacyFuncDisjointConstrainedImage):
    func = legacy_to_ref.get(obj.func)
    disj = legacy_to_ref.get(obj.disjoint_set)
    if func and disj:
        inv_img = FuncInverseImage(func=func, argument=...)
        # ...
```

### 4.4 更新 `ir/parser_adapter.py`：FuncInverse

```python
elif isinstance(obj, LegacyFuncInverse):
    func = legacy_to_ref.get(obj.func)
    if func:
        defn = FuncInverse(func=func)
```

**测试 Phase 1**：`uv run pytest` — 277 passed。

---

## 5. Phase 2：新 Context（`backend/wfomc/context_ir.py`）

新 Context 以 `ObjRef` 作为字典 key，接受 `ir.Problem + ir.AnalysisResult`。

### 5.1 接口设计

```python
# backend/wfomc/context_ir.py
from cofola.ir.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult
from cofola.ir.types import ObjRef, Entity as IREntity

class ContextIR:
    def __init__(self, problem: Problem, analysis: AnalysisResult) -> None:
        self.problem: Problem = problem
        self.analysis: AnalysisResult = analysis

        # WFOMC domain：所有实体 → Const
        self.singletons: frozenset[IREntity] = analysis.singletons
        self.domain: set[Const] = {Const(e.name) for e in analysis.all_entities}

        # 核心映射（ObjRef 作 key，替换 legacy 对象实例）
        self.ref2pred: dict[ObjRef, Pred] = {}
        self.ref_entity2pred: dict[tuple[ObjRef, IREntity], Pred] = {}
        self.ref2var: dict[ObjRef, Expr] = {}
        self.ref_entity2var: dict[tuple[ObjRef, IREntity], Expr] = {}
        self.ref_mul2var: dict[tuple[ObjRef, int], Expr] = {}
        self.used_refs: set[ObjRef] = set()

        # WFOMC 句子
        self.sentence: Formula = top
        self.weighting: dict[Pred, tuple] = {}
        self.unary_evidence: set[AtomicFormula] = set()
        self.overcount: Rational = Rational(1, 1)
        self.validator: list = []
        self.indis_vars: list = []
        self.gen_vars: list = []

        # 序列专用
        self.sequence_ref: ObjRef | None = self._find_sequence_ref()
        self.leq_pred = Pred('LEQ', 2)
        self.pred_pred = Pred('PRED', 2)
        self.circular_pred = Pred('CIRCULAR_PRED', 2)
        self.circle_len: int = len(self.domain)

    def _find_sequence_ref(self) -> ObjRef | None:
        from cofola.ir.objects import SequenceDef
        for ref in self.problem.refs():
            if isinstance(self.problem.get_object(ref), SequenceDef):
                return ref
        return None

    # --- Pred 管理 ---
    def get_pred(self, ref: ObjRef, *, create: bool = False, use: bool = True) -> Pred:
        if ref not in self.ref2pred:
            if not create:
                raise ValueError(f"ObjRef {ref} not found in ref2pred")
            name = self._get_name(ref)
            self.ref2pred[ref] = create_cofola_pred(name, 1)
        if use:
            self.used_refs.add(ref)
        return self.ref2pred[ref]

    def get_entity_pred(self, ref: ObjRef, entity: IREntity) -> Pred:
        key = (ref, entity)
        if key not in self.ref_entity2pred:
            name = f"{self._get_name(ref)}_{entity.name}"
            self.ref_entity2pred[key] = create_cofola_pred(name, 1)
        return self.ref_entity2pred[key]

    # --- Var 管理 ---
    def get_obj_var(self, ref: ObjRef, *, set_weight: bool = True) -> Expr:
        if ref not in self.ref2var:
            self.ref2var[ref] = self.create_var(self._get_name(ref))
        if set_weight:
            pred = self.get_pred(ref)
            self.weighting[pred] = (self.ref2var[ref], 1)
        return self.ref2var[ref]

    def get_entity_var(self, ref: ObjRef, entity: IREntity | None = None) -> Expr | dict:
        if entity is None:
            return {e: v for (r, e), v in self.ref_entity2var.items() if r == ref}
        key = (ref, entity)
        if key not in self.ref_entity2var:
            self.ref_entity2var[key] = self.create_var(f"{self._get_name(ref)}_{entity.name}")
        return self.ref_entity2var[key]

    def get_indis_entity_var(self, ref: ObjRef, multiplicity: int | None = None) -> Expr | dict:
        if multiplicity is None:
            return {m: v for (r, m), v in self.ref_mul2var.items() if r == ref}
        key = (ref, multiplicity)
        if key not in self.ref_mul2var:
            self.ref_mul2var[key] = self.create_var(f"{self._get_name(ref)}#{multiplicity}")
        return self.ref_mul2var[key]

    # --- 序列专用谓词 ---
    def get_leq_pred(self, seq_ref: ObjRef) -> Pred:
        defn = self.problem.get_object(seq_ref)
        source_pred = self.get_pred(defn.source)
        leq_pred = self.create_pred(f"{self._get_name(seq_ref)}_LEQ", 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({leq_pred}(X,Y) <-> "
            f"({source_pred}(X) & {source_pred}(Y) & {self.leq_pred}(X,Y))))"
        )
        return leq_pred

    def get_predecessor_pred(self, seq_ref: ObjRef) -> Pred:
        defn = self.problem.get_object(seq_ref)
        source_pred = self.get_pred(defn.source)
        pred_pred = self.circular_pred if defn.circular else self.pred_pred
        seq_pred = self.create_pred(f"{self._get_name(seq_ref)}_PRED", 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({seq_pred}(X,Y) <-> "
            f"({source_pred}(X) & {source_pred}(Y) & {pred_pred}(X,Y))))"
        )
        return seq_pred

    def get_next_to_pred(self, seq_ref: ObjRef) -> Pred:
        pred_pred = self.get_predecessor_pred(seq_ref)
        next_to = self.create_pred(f"{self._get_name(seq_ref)}_NEXT_TO", 2)
        self.sentence = self.sentence & parse(
            f"\\forall X: (\\forall Y: ({next_to}(X,Y) <-> "
            f"({pred_pred}(X,Y) | {pred_pred}(Y,X))))"
        )
        return next_to

    # --- 工具 ---
    def _get_name(self, ref: ObjRef) -> str:
        return self.problem.get_name(ref) or f"obj_{ref.id}"

    def create_pred(self, name: str, arity: int) -> Pred:
        return create_cofola_pred(name, arity)

    def create_var(self, name: str, *, use_gen: bool = True) -> Expr:
        var = create_cofola_var(name)
        if use_gen:
            self.gen_vars.append(var)
        return var

    def prune_evidence(self) -> None:
        used_preds = self.sentence.preds()
        self.unary_evidence = {e for e in self.unary_evidence if e.pred in used_preds}

    def build(self) -> tuple[WFOMCProblem, Decoder]:
        self.sentence = to_sc2(self.sentence)
        self.prune_evidence()
        new_domain = {Const(f"c_{c.name}") for c in self.domain}
        new_evidence = {
            AtomicFormula(a.pred, (Const(f"c_{a.args[0].name}"),), a.positive)
            for a in self.unary_evidence
        }
        wfomc_problem = WFOMCProblem(
            self.sentence, new_domain, self.weighting,
            unary_evidence=new_evidence, circle_len=self.circle_len
        )
        decoder = Decoder(self.overcount, self.gen_vars, self.validator, self.indis_vars)
        return wfomc_problem, decoder

    def get_parts_of(self, partition_ref: ObjRef) -> list[ObjRef]:
        """获取 PartitionDef 的所有 PartRef（按 index 排序）"""
        from cofola.ir.objects import PartRef
        parts = [
            (r, defn.index)
            for r in self.problem.refs()
            if isinstance((defn := self.problem.get_object(r)), PartRef)
            and defn.partition == partition_ref
        ]
        return [r for r, _ in sorted(parts, key=lambda x: x[1])]
```

**测试 Phase 2**：只写 context_ir.py，尚不接入 encoder，277 passed（context_ir 尚未使用）。

---

## 6. Phase 3：新 Encoder（`backend/wfomc/encoder_ir.py`）

这是工作量最大的阶段（~1000 行）。核心是把 `encoder.py` 的每个 `isinstance` 分支
改写为 `match-case` on IR dataclasses，分析数据从 `AnalysisResult` 读取。

### 6.1 入口函数

```python
# backend/wfomc/encoder_ir.py
def encode_ir(
    problem: ir.Problem,
    analysis: ir.AnalysisResult,
    lifted: bool = False,
) -> tuple[WFOMCProblem, Decoder]:
    context = ContextIR(problem, analysis)
    _encode_entities(analysis, context)

    for ref in problem.topological_order():
        defn = problem.get_object(ref)
        if defn is None:
            continue
        _encode_object(ref, defn, problem, analysis, context)

    for c in problem.constraints:
        _encode_constraint(c, problem, analysis, context)

    return context.build()
```

### 6.2 对象编码分派

```python
def _encode_object(ref, defn, problem, analysis, context):
    match defn:
        # ── Sets ──────────────────────────────────────────
        case ir.SetInit(entities=entities):
            _encode_set_init(ref, entities, context)

        case ir.SetChoose(source=src, size=size):
            _encode_set_choose(ref, src, size, context)

        case ir.SetChooseReplace(source=src, size=size):
            bag_info = analysis.bag_info.get(ref)  # SetChooseReplace 视作 bag
            _encode_set_choose_replace(ref, src, size, bag_info, context)

        case ir.SetUnion(left=l, right=r):
            _encode_set_union(ref, l, r, context)

        case ir.SetIntersection(left=l, right=r):
            _encode_set_intersection(ref, l, r, context)

        case ir.SetDifference(left=l, right=r):
            _encode_set_difference(ref, l, r, context)

        # ── Bags ──────────────────────────────────────────
        case ir.BagInit():
            bag_info = analysis.bag_info[ref]
            _encode_bag_init(ref, defn, bag_info, context)

        case ir.BagChoose(source=src, size=size):
            bag_info = analysis.bag_info[ref]
            _encode_bag_choose(ref, src, size, bag_info, context)

        case ir.BagUnion(left=l, right=r):
            bag_info = analysis.bag_info[ref]
            _encode_bag_union(ref, l, r, bag_info, context)

        case ir.BagAdditiveUnion(left=l, right=r):
            bag_info = analysis.bag_info[ref]
            _encode_bag_additive_union(ref, l, r, bag_info, context)

        case ir.BagIntersection(left=l, right=r):
            bag_info = analysis.bag_info[ref]
            _encode_bag_intersection(ref, l, r, bag_info, context)

        case ir.BagDifference(left=l, right=r):
            bag_info = analysis.bag_info[ref]
            _encode_bag_difference(ref, l, r, bag_info, context)

        case ir.BagSupport(source=src):
            _encode_bag_support(ref, src, context)

        # ── Functions ─────────────────────────────────────
        case ir.FuncDef(domain=d, codomain=c, surjective=sur):
            _encode_func_def(ref, d, c, sur, context)

        case ir.FuncInverse(func=func):
            _encode_func_inverse(ref, func, context)

        case ir.FuncImage(func=func, argument=arg):
            _encode_func_image(ref, func, arg, problem, analysis, context)

        case ir.FuncInverseImage(func=func, argument=arg):
            _encode_func_inverse_image(ref, func, arg, context)

        # ── Sequences ─────────────────────────────────────
        case ir.SequenceDef():
            set_info = analysis.set_info.get(ref)
            bag_info = analysis.bag_info.get(ref)
            _encode_sequence(ref, defn, problem, analysis, context)

        # ── Partitions ────────────────────────────────────
        case ir.PartitionDef():
            _encode_partition(ref, defn, problem, analysis, context)

        case ir.PartRef():
            pass  # 由 PartitionDef 编码时统一处理
```

### 6.3 分析数据读取对照表

新 encoder 从 `AnalysisResult` 读取以前注入到对象的属性：

| Legacy 对象属性 | 新 IR 来源 |
|---|---|
| `bag_obj.dis_entities` | `analysis.bag_info[ref].dis_entities` |
| `bag_obj.p_entities_multiplicity` | `analysis.bag_info[ref].p_entities_multiplicity` |
| `bag_obj.indis_entities` | `analysis.bag_info[ref].indis_entities` |
| `bag_obj.max_size` | `analysis.bag_info[ref].max_size` |
| `set_obj.p_entities` | `analysis.set_info[ref].p_entities` |
| `set_obj.max_size` | `analysis.set_info[ref].max_size` |
| `obj.name` | `problem.get_name(ref)` |
| `bag_obj.multiplicity(e)` | `analysis.bag_info[ref].p_entities_multiplicity[e]` |
| `problem.entities` | `analysis.all_entities` |
| `problem.singletons` | `analysis.singletons` |
| `partition.partitioned_objs` | `context.get_parts_of(partition_ref)` |
| `seq.flatten_obj` | `defn.source`（LoweringPass 已处理） |
| `seq.obj_from` | `defn.source` |

### 6.4 约束编码分派

```python
def _encode_constraint(c, problem, analysis, context):
    match c:
        case ir.SizeConstraint(terms=terms, comparator=comp, rhs=rhs):
            _encode_size_constraint(terms, comp, rhs, context)

        case ir.MembershipConstraint(entity=e, container=cont, positive=pos):
            _encode_membership(e, cont, pos, context)

        case ir.SubsetConstraint(sub=sub, sup=sup, positive=pos):
            _encode_subset(sub, sup, pos, context)

        case ir.DisjointConstraint(left=l, right=r, positive=pos):
            _encode_disjoint(l, r, pos, context)

        case ir.EqualityConstraint(left=l, right=r, positive=pos):
            _encode_eq(l, r, pos, context)

        case ir.FuncPairConstraint(func=func, arg_entity=e, result=res, positive=pos):
            _encode_func_pair(func, e, res, pos, problem, context)

        case ir.SequencePatternConstraint(seq=seq, pattern=pat, positive=pos):
            _encode_seq_pattern(seq, pat, pos, context)

        case ir.BagSubsetConstraint(sub=sub, sup=sup, positive=pos):
            _encode_bag_subset(sub, sup, pos, context)

        case ir.BagEqConstraint(left=l, right=r, positive=pos):
            _encode_bag_eq(l, r, pos, context)
```

### 6.5 Size constraint 的 terms 处理

在 IR 中，`SizeConstraint.terms` 的每个元素是 `(ObjRef, int)` 。
编码时要区分 size atom 类型：

```python
def _encode_size_constraint(terms, comp, rhs, context):
    # terms: tuple[tuple[ObjRef | BagCountAtom | SeqPatternCountAtom, int], ...]
    # 直接用 ObjRef 查 context.get_obj_var(ref)
    expr = []
    for term_ref, coef in terms:
        if isinstance(term_ref, ir.BagCountAtom):
            var = _get_bag_count_var(term_ref, context)
        elif isinstance(term_ref, ir.SeqPatternCountAtom):
            var = _get_seq_pattern_count_var(term_ref, context)
        else:  # ObjRef
            var = context.get_obj_var(term_ref)
        expr.append((var, coef))
    # 构造约束...
```

**测试 Phase 3**（关键）：
```bash
# 双 encoder 并行验证
uv run python -c "
from cofola.ir.pipeline import IRPipeline
from cofola.backend.wfomc.encoder_ir import encode_ir

# 对每个 problem 文件运行两个 encoder，比对结果
"
uv run pytest  # 全量 277 passed
```

---

## 7. Phase 4：接入 + 删除遗留代码

### 7.1 更新 `backend/base.py`

```python
# backend/base.py
from cofola.ir.problem import Problem
from cofola.ir.analysis.entities import AnalysisResult

class Backend(ABC):
    @abstractmethod
    def solve(self, problem: Problem, analysis: AnalysisResult) -> int: ...
```

### 7.2 更新 `backend/wfomc/backend.py`

```python
class WFOMCBackend(Backend):
    def solve(self, problem: Problem, analysis: AnalysisResult) -> int:
        from cofola.backend.wfomc.encoder_ir import encode_ir
        wfomc_problem, decoder = encode_ir(problem, analysis, self.lifted)
        # ... algo 选择逻辑不变 ...
        raw = solve_wfomc(wfomc_problem, algo, use_partition_constraint)
        result = decoder.decode_result(raw)
        return result if result is not None else 0
```

### 7.3 更新 `ir/pipeline.py`

```python
# IRPipeline._solve_legacy → _solve_ir
def _solve_ir(self, problem: Problem, analysis: AnalysisResult) -> int:
    backend = WFOMCBackend(algo=self.algo, lifted=self.lifted)
    return backend.solve(problem, analysis)
```

删除 `ir/adapter.py` 的调用，删除 `LegacyAdapter` 导入。

### 7.4 删除遗留代码（有顺序）

```bash
# 1. 删除 frontend/
rm -rf src/cofola/frontend/

# 2. 删除 objects/
rm -rf src/cofola/objects/

# 3. 删除旧 encoder 和 context
rm src/cofola/backend/wfomc/encoder.py
rm src/cofola/backend/wfomc/context.py

# 4. 删除 adapter
rm src/cofola/ir/adapter.py
```

**测试 Phase 4**：`uv run pytest` — 277 passed（全量验证）。

---

## 8. Phase 5：Parser 直接产出 IR（消除双程转换）

### 8.1 目标

当前 `parser_adapter.py` 把 legacy parser 输出转换为 IR，这是临时桥接。
Phase 5 直接让 parser transformer 产出 `ProblemBuilder`。

### 8.2 新 Transformer 设计

```python
# parser/transformer.py（更新版）
from lark import Transformer
from cofola.ir.problem import ProblemBuilder
from cofola.ir.objects import SetInit, SetChoose, BagInit, ...
from cofola.ir.constraints import SizeConstraint, ...

class CofolaTransformer(Transformer):
    def __init__(self):
        self.builder = ProblemBuilder()
        self.id2ref: dict[str, ObjRef] = {}

    # 每个 rule 直接 return ObjDef / Constraint，不再 return legacy 对象
    def set_init(self, args):
        entities = frozenset(Entity(e.name) for e in args)
        defn = SetInit(entities=entities)
        return defn

    def object_declaration(self, args):
        name, defn = args
        ref = self.builder.add(defn, name=str(name))
        self.id2ref[str(name)] = ref
        return ref

    def size_constraint(self, args):
        ...
        self.builder.add_constraint(SizeConstraint(...))

    def cofola(self, args):
        return self.builder.build()  # 直接返回 ir.Problem
```

### 8.3 删除 parser_adapter.py

```bash
rm src/cofola/ir/parser_adapter.py
```

更新 `ir/pipeline.py` 的 Stage 1：
```python
# 旧
from .parser_adapter import parse_cofola
problem = parse_cofola(text)

# 新
from cofola.parser.parser import parse
problem = parse(text)  # 直接返回 ir.Problem
```

**测试 Phase 5**：`uv run pytest` — 277 passed。

---

## 9. 测试策略

### 每个 Phase 的验收标准

| Phase | 验收条件 |
|---|---|
| 1 | `uv run pytest` 277 passed；新 IR 节点有单元测试 |
| 2 | `uv run pytest` 277 passed；context_ir.py 有独立单元测试 |
| 3 | `uv run pytest` 277 passed；新旧 encoder 对比脚本通过 |
| 4 | `uv run pytest` 277 passed；所有遗留代码已删除，无 import 引用 |
| 5 | `uv run pytest` 277 passed；`parser_adapter.py` 不存在 |

### Phase 3 对比验证脚本

```python
# scripts/compare_encoders.py
import json
from pathlib import Path
from cofola.solver import solve as solve_legacy
from cofola.ir.pipeline import IRPipeline

problems = json.loads(Path("problems/all.json").read_text())
pipeline = IRPipeline()
errors = []
for p in problems:
    try:
        expected = p["answer"]
        got = pipeline.solve(p["problem"])
        if got != expected:
            errors.append(f"MISMATCH: {p['name']}: expected {expected}, got {got}")
    except Exception as e:
        errors.append(f"ERROR: {p['name']}: {e}")

if errors:
    for e in errors: print(e)
else:
    print(f"All {len(problems)} problems match!")
```

---

## 10. 风险与缓解

| 风险 | 严重度 | 缓解 |
|---|---|---|
| encoder_ir.py 某个对象类型行为与旧版不同 | HIGH | 对比脚本逐题验证，新旧 encoder 可并行运行 |
| `SequenceDef.flatten_obj` 处理不同 | MEDIUM | 充分测试序列问题；LoweringPass 完善 choose/replace 路径 |
| `context.py` 中 `used_objs` pruning 逻辑与 `context_ir.py` 不同 | MEDIUM | 对比生成的 WFOMCProblem.sentence |
| FuncInverse 等节点 parser_adapter 未覆盖 | LOW | Phase 1 验证时检查 `objects/function.py` 中有哪些类型需处理 |
| partition.partitioned_objs 顺序不同 | LOW | `get_parts_of` 按 index 排序保证一致 |

---

## 11. 实施优先级总结

```
Phase 1（2-3天）：补充 IR 节点 + parser_adapter 完整性修复
Phase 2（1-2天）：context_ir.py
Phase 3（5-7天）：encoder_ir.py（核心工作）
Phase 4（1天）  ：接入 + 删除遗留
Phase 5（2-3天）：新 transformer，消除双程转换
```

总计约 **2-3 周**工作量（主要在 Phase 3 encoder 重写）。

---

## 12. 与 plan/ir-refactor.md 的关系

| ir-refactor Phase | 状态 | 本计划 |
|---|---|---|
| Phase 1-4（IR core + passes） | ✅ 已完成 | 基础 |
| Phase 5（parser 迁移） | ⏳ 待做 | → 本计划 Phase 5 |
| Phase 6（pipeline 集成） | ✅ 已完成 | 基础 |
| Phase 7（Cleanup） | ⏳ 待做 | → 本计划 Phase 4 |
| **Phase 8（Backend IR-native）** | **新增** | → 本计划 Phase 2-4 |
