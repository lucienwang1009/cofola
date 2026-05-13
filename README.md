# Cofola

Cofola is a Python DSL and solver for combinatorial counting problems. It lets
you model sets, bags, tuples, sequences, partitions, functions, and constraints,
then counts the number of satisfying configurations through a Weighted
First-Order Model Counting (WFOMC) backend.

## Install

Cofola uses Python 3.11+ and `uv`.

```bash
uv sync
```

## Solve A Problem

Write a `.cfl` file:

```cfl
items = bag(quarter: 3, nickel: 2, penny: 3)
payment = choose(items)
|payment| > 0
payment.count(nickel) == 1
```

Run it:

```bash
uv run cofola -i problems/others/bag.cfl
uv run cofola -i problems/others/bag.cfl -d
```

You can also solve source text or hand-built problems from Python:

```python
from cofola.solver import parse_and_solve

answer = parse_and_solve("""
S = set(a, b, c)
T = choose(S, 2)
a in T
""")
```

## Language At A Glance

```cfl
S = set(a, b, c)
B = bag(a: 2, b: 1)

T = choose(S, 2)          # choose without replacement
M = choose_replace(S, 3)  # bag chosen with replacement from a set

U = S + set(d)            # set union
D = B + bag(a: 1)         # bag union: max multiplicity per entity
A = B ++ bag(a: 1)        # additive bag union: sum multiplicities
I = S & set(a)
R = S - set(c)

P = partition(S, 2)       # unordered parts
C = compose(S, 2)         # ordered parts
Tup = choose_tuple(S, 2)
Seq = sequence(S)
Ring = circle(S, reflection=True)

|T| == 2
a in T
T subset S
Tup[0] == a
a < b in Seq
Seq.count(next_to(a, b)) == 1
```

For the full grammar and semantics, see
[Language Reference](docs/language-reference.md).

## Python API

For programmatic construction, use `ProblemBuilder`:

```python
from cofola.frontend import Entity, ProblemBuilder, SetChoose, SetInit
from cofola.solver import solve

builder = ProblemBuilder()
a = Entity("a")
b = Entity("b")

S = builder.add(SetInit(entities=frozenset({a, b})), name="S")
T = builder.add(SetChoose(source=S, size=1), name="T")

answer = solve(builder.build())
```

See [Building Problems With `ProblemBuilder`](docs/problem-builder-usage.md).

## Architecture

The solve path is:

```text
.cfl source
  -> parser
  -> frontend Problem + type checking
  -> planing pipeline: analyses, simplification, lowering, decomposition
  -> WFOMC backend encoding
  -> decoded integer count
```

The middle layer is named `cofola.planing` intentionally. It plans the frontend
problem into backend-ready components. See
[Cofola Planning Layer](docs/planning-layer.md).

## Development

Useful commands:

```bash
uv run pytest
COFOLA_ALL_TESTS=1 uv run pytest tests/test_all_problems.py
uv run pyright
```

See [Development Guide](docs/development.md) for test organization and backend
notes.

## WFOMC Backend

Cofola currently uses the `for_cofola` branch of
[yuanhong-wang/WFOMC](https://github.com/yuanhong-wang/WFOMC).
