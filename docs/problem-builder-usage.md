# Building Problems With `ProblemBuilder`

This guide shows how to build Cofola problems directly from Python instead of
writing `.cfl` source text.

Use this API when you want to generate problems programmatically, write focused
tests, or integrate Cofola into another Python workflow.

## Core Idea

`ProblemBuilder` is a mutable builder for the frontend problem model. Each call
to `builder.add(...)` stores an object definition and returns an `ObjRef`.
Later objects and constraints refer to those `ObjRef`s.

```python
from cofola.frontend import Entity, ProblemBuilder, SetInit

builder = ProblemBuilder()

a = Entity("a")
b = Entity("b")

S = builder.add(
    SetInit(entities=frozenset({a, b})),
    name="S",
)

problem = builder.build()
```

Important rule: keep and reuse the returned refs (`S` above). Do not pass the
object definition itself where Cofola expects an object reference.

## Set Example

This Python model is equivalent to:

```cfl
S = set(a, b, c)
T = choose(S, 2)
|T| == 2
a in T
```

```python
from cofola.frontend import (
    Entity,
    MembershipConstraint,
    ProblemBuilder,
    SetChoose,
    SetInit,
    SizeConstraint,
    validate_problem,
)
from cofola.solver import solve

builder = ProblemBuilder()

a = Entity("a")
b = Entity("b")
c = Entity("c")

S = builder.add(
    SetInit(entities=frozenset({a, b, c})),
    name="S",
)

T = builder.add(
    SetChoose(source=S, size=2),
    name="T",
)

builder.add_constraint(
    SizeConstraint(
        terms=((T, 1),),
        comparator="==",
        rhs=2,
    )
)

builder.add_constraint(
    MembershipConstraint(
        entity=a,
        container=T,
        positive=True,
    )
)

problem = builder.build()
validate_problem(problem)

answer = solve(problem, validate=False)
```

## Bag Count Example

Use `BagCountAtom` inside a `SizeConstraint` to constrain a multiplicity.

This Python model is equivalent to:

```cfl
B = bag(a: 2, b)
C = choose(B)
C.count(a) == 1
```

```python
from cofola.frontend import (
    BagChoose,
    BagCountAtom,
    BagInit,
    Entity,
    ProblemBuilder,
    SizeConstraint,
    validate_problem,
)
from cofola.solver import solve

builder = ProblemBuilder()

a = Entity("a")
b = Entity("b")

B = builder.add(
    BagInit(entity_multiplicity=((a, 2), (b, 1))),
    name="B",
)

C = builder.add(
    BagChoose(source=B),
    name="C",
)

builder.add_constraint(
    SizeConstraint(
        terms=((BagCountAtom(bag=C, entity=a), 1),),
        comparator="==",
        rhs=1,
    )
)

problem = builder.build()
validate_problem(problem)

answer = solve(problem, validate=False)
```

## Tuple Index Example

Tuple indexing constraints use dedicated constraint nodes.

```python
from cofola.frontend import (
    Entity,
    ProblemBuilder,
    SetInit,
    TupleDef,
    TupleIndexEq,
    validate_problem,
)

builder = ProblemBuilder()

a = Entity("a")
b = Entity("b")

S = builder.add(SetInit(entities=frozenset({a, b})), name="S")
T = builder.add(TupleDef(source=S, choose=True, size=2), name="T")

builder.add_constraint(
    TupleIndexEq(
        tuple_ref=T,
        index=0,
        entity=a,
        positive=True,
    )
)

problem = builder.build()
validate_problem(problem)
```

## Validation

Parsed `.cfl` programs are type-checked automatically by `parse()`.
`solve(problem)` also validates hand-built problems by default. Call
`validate_problem(problem)` explicitly when you want an early validation step or
want to validate once before a later `solve(problem, validate=False)` call:

```python
from cofola.frontend import validate_problem

validate_problem(problem)
```

This catches type errors such as using a tuple where a set or bag is expected.

## Common Patterns

- `builder.add(defn, name="X")` returns an `ObjRef`.
- Object definitions such as `SetChoose`, `BagChoose`, and `TupleDef` refer to
  sources by `ObjRef`.
- Constraints refer to objects by `ObjRef`.
- `Entity("a")` represents an atomic element, not an object definition.
- Use `positive=False` for negated relation constraints such as `a not in S`
  or `T[0] != a`.

## Current Limitations

- The builder is intentionally low-level. It mirrors Cofola's frontend model
  rather than providing a fluent modeling DSL.
