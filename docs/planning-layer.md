# Cofola Planning Layer

The current `cofola.planing` package is the planning layer. It operates over the
frontend `Problem` model; it does not define a separate IR data model.

## Layer Boundaries

```text
parser     .cfl source -> frontend Problem
frontend   core problem model and validation
planning   analysis, normalization, lowering, decomposition, solve scheduling
backend    WFOMC encoding and execution
```

The package is named `cofola.planing` to match its architectural role.

## Contract

Input:

- a validated `cofola.frontend.Problem`

Output:

- a `SolveSchedule` containing backend-ready `(Problem, BagClassification)`
  components

The planning layer may transform the problem, split it into Shannon branches,
decompose independent components, and compute analyses required by the backend.

## Transform-Pass Expectations

Transform passes should:

- return either the original `Problem` or a new `Problem`
- not mutate the input `Problem`
- preserve semantic metadata such as `names` when rebuilding a problem
- allocate generated refs through a shared allocator
- avoid relying on stale analyses after changing the problem

Source locations are frontend diagnostic metadata. Planning passes may preserve
`locs` when convenient, but `locs` are not part of the planning correctness
contract.

## Size Atom Policy

`SizeConstraint.terms` can contain raw object refs or derived size atoms.
`MaxSizeInference` only reasons about raw `ObjRef` cardinality terms. Constraints
containing size atoms such as `BagCountAtom`, `TupleCountAtom`, or
`SeqPatternCountAtom` are skipped by LP inference until another pass lowers or
handles them.
