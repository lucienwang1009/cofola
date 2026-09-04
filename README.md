# Cofola

Cofola (COmbinatorial counting with First-Order logic LAnguage) is a declarative language and solver for modeling and solving combinatorial counting problems using Weighted First-Order Model Counting (WFOMC). It allows you to define problems involving sets, multisets (bags), functions, and constraints in a natural way, and then automatically computes the number of solutions.

## Features

- **Declarative Modeling**: Define your problem using high-level concepts like sets, bags, and mappings.
- **First-Order Logic Backend**: Leverages the power of Weighted First-Order Model Counting (WFOMC) to solve complex counting problems efficiently.
- **Unary-Function Backend**: An experimental UnaryFOMC adapter solves the shared monadic set fragment and exposes its generated logical sentence.
- **Constraint Support**: Easily specify cardinality constraints, functional dependencies, and more.

## Installation

This project uses `uv` for dependency management.

1.  Ensure you have Python 3.11 or higher installed.
2.  Install `uv` if you haven't already.
3.  Sync dependencies:

```bash
uv sync
```

To enable the experimental UnaryFOMC backend as well:

```bash
uv sync --extra unaryfomc
```

## Usage

To solve a problem defined in a `.cfl` file, use the `cofola` command via `uv run`:

```bash
uv run cofola -i <path_to_problem_file>
```

The same program can be sent to UnaryFOMC when it belongs to the supported
set fragment:

```bash
uv run cofola -i <path_to_problem_file> --backend unaryfomc
```

### Options

- `-i`, `--input_file`: Path to the input `.cfl` file (required).
- `-d`, `--debug`: Enable debug logging.
- `--backend`: Select `wfomc` (the default) or `unaryfomc`.

The backend is also selectable through the Python API:

```python
from cofola import parse_and_solve

program = """
people = set(person0...8)
committee = choose(people, 3)
"""

answer = parse_and_solve(program, backend="unaryfomc")
assert answer == 56
```

### UnaryFOMC Backend

The first UnaryFOMC integration deliberately targets the normalized,
unweighted monadic set fragment. It currently supports:

- explicit sets and subset choice, with or without a fixed size;
- union, intersection, and difference;
- linear constraints over set cardinalities;
- membership, subset, disjointness, and set equality constraints; and
- Boolean combinations of supported constraints, through Cofola's existing
  IR decomposition.

Bags, functions, tuples, sequences, circles, and partitions still use the
default WFOMC backend. Asking UnaryFOMC to solve one of those constructs raises
an error naming the first unsupported IR construct rather than silently
changing its meaning.

This basic adapter uses Cofola's finite list of explicit entities as the model
domain. Symbolic domain-size parameters and direct asymptotic analysis of
families of Cofola programs are planned extensions; they are not inferred by
the current API.

## Language Examples

### Example 1: Coin Selection

**Problem:**
You have a bag of items containing quarters, nickels, pennies, and other items. You want to choose a non-empty subset of these items such that you have exactly one nickel.

**Cofola Code:**

```plaintext
items = bag(quarter: 3, nickel: 2, penny: 3, A: 3, B: 2, D: 3, E: 3, F: 3, G: 3)
payment = choose(items)
|payment| > 0
payment.count(nickel) == 1
```

### Example 2: Worker Assignment

**Problem:**
How many distinct three-letter sequences with at least one $T$ can be formed by using three of the six letters of $TARGET$? One such sequence is $TRT$.

**Cofola Code:**

```plaintext
letters = bag(T, A, R, G, E, T)
S = choose_tuple(letters, 3)
S.count(T) > 0
```

## Syntax Guide

Cofola uses a declarative syntax to define objects and constraints.

### Comments

Use `#` for comments.

```plaintext
# This is a comment
s = set(a, b) # Inline comment
```

### Object Declaration

-   **Sets**:
    ```plaintext
    s = set(a, b, c)
    s2 = set(item1...5)  # Creates item1, item2, ..., item4
    ```

-   **Bags (Multisets)**:
    ```plaintext
    b = bag(a: 2, b: 3)  # 2 'a's and 3 'b's
    ```

### Operations

-   **Selection**:
    ```plaintext
    sub = choose(s)      # Any subset of s
    sub = choose(s, k)   # Subset of size k
    sub = choose_replace(s, k) # Selection with replacement
    ```

-   **Partitioning**:
    ```plaintext
    p = partition(s, k)  # Partition s into k parts
    c = compose(s, k)    # Ordered partition (composition) of s into k parts
    ```

-   **Permutations and Sequences**:
    ```plaintext
    t = tuple(s)         # Tuple (ordered, support indexes)
    seq = sequence(s)    # Sequence (ordered, support relative positional constraints)
    c = circle(s)        # Circular arrangement
    c_ref = circle(s, reflection=True) # Reflexive circular arrangement
    
    # Variants with selection
    t = choose_tuple(s, k)
    t = choose_replace_tuple(s, k)
    seq = choose_sequence(s, k)
    c = choose_circle(s, k)
    ```

-   **Support**:
    ```plaintext
    s = supp(b) # Support set of bag b
    ```

-   **Binary Operations**:
    ```plaintext
    union = A + B       # Set union or Bag additive union
    inter = A & B       # Intersection
    diff = A - B        # Difference
    ```

### Constraints

-   **Cardinality**:
    ```plaintext
    |sub| == 5
    |sub| >= 1
    ```

-   **Counting**:
    ```plaintext
    obj.count(x) == 2        # Count occurrences of x in obj
    obj.dedup_count(x) == 1  # Count unique occurrences
    ```

-   **Set Relations**:
    ```plaintext
    x in sub
    x not in sub
    sub1 subset sub2
    sub1 disjoint sub2
    A == B
    A != B
    ```
-  **Indexing**:
    ```plaintext
    t[0] == a            # First element of tuple t is a
    ```

-   **Relative Positional Constraints**:
    ```plaintext
    pattern in seq  # seq matches the given pattern
    seq.count(pattern) == k  # pattern occurs k times in seq
    ```
    where `pattern` can be specified using:
    ```plaintext
    together(x)          # Elements in x are adjacent
    next_to(a, b)        # a is next to b
    predecessor(a, b)    # a immediately precedes b
    a < b                # a appears before b
    ```

-   **Logical Connectives**:
    ```plaintext
    not (constraint)
    constraint1 and constraint2
    constraint1 or constraint2
    ```

-   **Quantifiers (Part Constraints)**:
    Apply a constraint to every part of a partition or composition.
    ```plaintext
    (|part| > 0) for part in p
    ```

## References

The default backend uses the WFOMC implementation available
[here](https://github.com/yuanhong-wang/WFOMC). The experimental monadic set
backend uses [UnaryFOMC](https://github.com/supertweety/UnaryFOMC), implementing
the unary-function counting methods developed in
[Unary Functions and Unlabeled Counting](https://arxiv.org/abs/2608.30580).
