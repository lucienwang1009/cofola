# Cofola Language Reference

This document describes the user-facing `.cfl` language. Parsed programs are
validated before solving, so type errors are reported before planning and
backend encoding.

## Comments

```cfl
# full-line comment
S = set(a, b)  # inline comment
```

## Objects

### Sets

Sets contain distinct entities.

```cfl
S = set(a, b, c)
Range = set(item1...5)  # item1, item2, item3, item4
```

Duplicate set entries are rejected.

### Bags

Bags are multisets. Omitted multiplicity means `1`.

```cfl
B = bag(a: 2, b, c: 3)
```

### Choose

```cfl
T = choose(S)       # any subset of set S
T = choose(S, 2)    # subset of size 2
C = choose(B)       # sub-bag of bag B
C = choose(B, 3)    # sub-bag with total multiplicity 3
M = choose_replace(S, 3)
```

`choose_replace` chooses with replacement from a set and produces a bag.
Choosing with replacement from a bag is rejected.

### Binary Operations

For sets:

```cfl
U = A + B   # union
I = A & B   # intersection
D = A - B   # difference
```

For bags:

```cfl
U = A + B    # bag union: max multiplicity per entity
S = A ++ B   # additive union: sum multiplicities
I = A & B    # min multiplicity per entity
D = A - B    # max(left - right, 0) per entity
```

### Support

```cfl
S = supp(B)
```

The support of a bag is the set of entities whose multiplicity is positive.

### Tuples, Sequences, And Circles

```cfl
T = tuple(S)
T = choose_tuple(S, 2)
T = choose_replace_tuple(S, 3)

Seq = sequence(S)
Seq = choose_sequence(S, 2)
Seq = choose_replace_sequence(S, 3)

Ring = circle(S)
Bracelet = circle(S, reflection=True)
```

Tuples support indexed constraints such as `T[0] == a`. Sequences and circles
support relative positional constraints such as `a < b in Seq`.

### Partitions And Compositions

```cfl
P = partition(S, 2)  # unordered parts
C = compose(S, 2)    # ordered parts
```

Apply one constraint to every part of a named partition or composition:

```cfl
(|part| > 0) for part in P
```

Indexing unordered partitions is rejected. Use `compose` when part order matters.

## Constraints

### Size Constraints

```cfl
|S| == 3
|T| >= 1
2 |A| + |B| == 5
```

Comparators: `==`, `!=`, `<`, `<=`, `>`, `>=`.

### Count Atoms

```cfl
B.count(a) == 2
T.count(a) > 0
T.dedup_count(A) == 1
Seq.count(a < b) == 1
```

`dedup_count` is for tuples. `seq.count(together(...))` is rejected because
`together` has no count variant.

### Relations

```cfl
a in S
a not in S
A subset B
A disjoint B
A == B
A != B
```

Set equality/subset/disjoint constraints are set-like. Bag subset/equality use
multiplicity semantics.

### Tuple Indexing

```cfl
T[0] == a
T[1] != b
T[0] in A
```

Tuple index constraints are lowered before backend encoding.

### Sequence Patterns

```cfl
a < b in Seq
next_to(a, b) in Seq
(a, b) in Seq          # predecessor: a immediately precedes b
together(A) in Seq

Seq.count(a < b) == 1
Seq.count(next_to(a, b)) >= 1
Seq.count((a, b)) == 0
```

Circles reject strict ordering patterns like `a < b`.

### Logic

```cfl
not (a in S)
(a in S) and (b in S)
(a in S) or (b in S)
```

The planner expands supported logical structure before backend solving.

## Validation Notes

- Object names cannot use reserved keywords such as `set`, `bag`, `choose`,
  `tuple`, `sequence`, `circle`, `partition`, or `compose`.
- Undefined object names are rejected during parsing.
- Type mismatches are rejected before solving.
- Hand-built Python `Problem` objects should be checked with
  `cofola.validate_problem(problem)` or solved with the default
  `solve(problem)` validation enabled.
