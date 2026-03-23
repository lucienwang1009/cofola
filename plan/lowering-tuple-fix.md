# Lowering Pass — Tuple 修复与优化计划

## 背景

`_try_lower_tuples` 对照 `cofola_main/src/cofola/problem.py:transform_tuples` 发现三个 Bug 及若干可简化点。

---

## Bug 1 — choose=True + Bag source：应分解为两步

**问题**：当前一步处理，用 `<=` 约束，缺少 surjectivity 强制，entity 可能被选少。

**修复**：仿照 reference，直接分解为 `BagChoose + TupleDef(choose=False)`，下一轮再处理：

```python
if defn.choose and isinstance(source_defn, (BagInit, BagChoose)):
    chosen_ref = self._new_ref()
    new_defs.append((chosen_ref, BagChoose(source=source, size=size)))
    new_tuple_defn = TupleDef(source=chosen_ref, choose=False, replace=False, size=size)
    new_defs = [(r, d) if r != ref else (ref, new_tuple_defn) for r, d in new_defs]
    return Problem(...), True  # 下一轮再处理 choose=False
```

---

## Bug 2 + 简化 — choose=False + Bag source：用 surjective 替代 singleton 特殊处理

**问题**：当前 singleton 收集后 `pass`，`SetEqConstraint` 从未生成，singleton 不被强制覆盖。

**正确性分析**：
- `mapping: indices → support, surjective=True` 保证每个 singleton ≥ 1 个 index 指向
- `|indices| = Σ mult_e`（by construction），non-singleton inv_img 约束消耗完对应 index 后
- 剩余 index 数 = singleton 数量 → 每个 singleton 恰好被指向一次
- 因此无需任何 singleton 特殊处理

**修复**：`mapping` 标记 `surjective=True`，所有 entity 统一处理：

```python
mapping_defn = FuncDef(domain=indices_ref, codomain=support_ref,
                       injective=False, surjective=True)

inv_img_refs = []
for entity, mult in bag_info.p_entities_multiplicity.items():
    # 全部 entity 统一处理，不区分 singleton/non-singleton
    inv_img_ref = self._new_ref()
    new_defs.append((inv_img_ref, FuncInverseImage(func=mapping_ref, argument=entity)))
    inv_img_refs.append(inv_img_ref)
    new_constraints.append(SizeConstraint(
        terms=((inv_img_ref, 1),),
        comparator="==",
        rhs=mult,  # 见 Bug 3 对 BagChoose source 的处理
    ))

for i, ri in enumerate(inv_img_refs):
    for j, rj in enumerate(inv_img_refs):
        if i < j:
            new_constraints.append(DisjointConstraint(left=ri, right=rj, positive=True))
# 不再需要 singleton 分支和 SetEqConstraint
```

---

## Bug 3 — BagChoose source 的动态 multiplicity

**问题**：当前用 `analysis.bag_info[source].p_entities_multiplicity[entity]`（静态上界），对 BagChoose source 不准确（实际 multiplicity 是运行时值）。

**修复**：按 source 类型分支：

```python
if isinstance(source_defn, BagInit):
    rhs = bag_info.p_entities_multiplicity[entity]
    new_constraints.append(SizeConstraint(
        terms=((inv_img_ref, 1),), comparator="==", rhs=rhs
    ))
else:  # BagChoose source
    bag_mult_ref = self._new_ref()
    new_defs.append((bag_mult_ref, BagMultiplicity(bag=source, entity=entity)))
    new_constraints.append(SizeConstraint(
        terms=((inv_img_ref, 1), (bag_mult_ref, -1)), comparator="==", rhs=0
    ))
```

注意：需确认 `BagMultiplicity` 是否已在 frontend IR 中定义，若无需补充。

---

## 优化 — 状态合并为 dataclass

**现状**：三个并行 dict 散落在 `LoweringPass` 中，`_try_lower_size_constraints` 处查询繁琐。

```python
# 现在
self._tuple_to_mapping: dict[ObjRef, ObjRef]
self._tuple_to_indices:  dict[ObjRef, ObjRef]
self._tuple_choose:      dict[ObjRef, bool]

# 改为
@dataclass
class _TupleInfo:
    mapping_ref: ObjRef
    indices_ref: ObjRef
    choose: bool

self._tuple_info: dict[ObjRef, _TupleInfo]
```

---

## 整体流程变化

```
choose=True  + Set  → FuncDef(injective / non-injective)        [不变]
choose=True  + Bag  → BagChoose + TupleDef(choose=False)        [修复 Bug 1]
choose=False + Set  → FuncDef(surjective=True)                  [不变]
choose=False + Bag  → FuncDef(surjective=True)                  [修复 Bug 2]
                      + |inv_img| == mult（全部 entity，无分支） [修复 Bug 2]
                      + 动态 BagMultiplicity（BagChoose source） [修复 Bug 3]
```

---

## 实施顺序

1. 确认 `BagMultiplicity` 是否存在于 frontend IR
2. 修复 Bug 1（choose+Bag 分解）
3. 修复 Bug 2（surjective + 统一 entity 处理）
4. 修复 Bug 3（BagChoose source 动态 multiplicity）
5. 优化 `_TupleInfo` dataclass
6. 运行 `uv run pytest tests/` 验证全部通过
