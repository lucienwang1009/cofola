# Cofola

Cofola is a declarative language and solver for modeling and solving combinatorial counting problems using Weighted First-Order Model Counting (WFOMC). It allows you to define problems involving sets, multisets (bags), functions, and constraints in a natural way, and then automatically computes the number of solutions.

## Features

- **Declarative Modeling**: Define your problem using high-level concepts like sets, bags, and mappings.
- **First-Order Logic Backend**: Leverages the power of Weighted First-Order Model Counting (WFOMC) to solve complex counting problems efficiently.
- **Constraint Support**: Easily specify cardinality constraints, functional dependencies, and more.

## Installation

This project uses `uv` for dependency management.

1.  Ensure you have Python 3.11 or higher installed.
2.  Install `uv` if you haven't already.
3.  Sync dependencies:

```bash
uv sync
```

## Usage

To solve a problem defined in a `.cfl` file, use the `cofola` command via `uv run`:

```bash
uv run cofola -i <path_to_problem_file>
```

### Options

- `-i`, `--input_file`: Path to the input `.cfl` file (required).
- `-d`, `--debug`: Enable debug logging.
- `-p`, `--use_partition_constraint`: Use partition constraints to potentially speed up the solver.

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
Assign 15 workers to 4 jobs such that Job 1 gets exactly 7 workers and Job 2 gets exactly 5 workers.

**Cofola Code:**

```plaintext
workers = set(worker1...15)
jobs = set(job1...4)

assign = workers -> jobs
|assign-1(job1)| = 7
|assign-1(job2)| = 5
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
    s2 = set(item1...5)  # Creates item1, item2, ..., item5
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
    t = tuple(s)         # Tuple (ordered, with replacement if from set?) - check implementation
    seq = sequence(s)    # Sequence (ordered, no replacement if from set)
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

-   **Sequence Patterns**:
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

## Project Structure

- `src/cofola`: Source code for the parser, solver, and encoder.
- `problems/`: Example problem definitions in `.cfl` format.
- `scripts/`: Utility scripts for batch processing and plotting.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
