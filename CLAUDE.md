# Cofola — Claude Code Instructions

## Project Overview

**Cofola** (COmbinatorial counting with First-Order logic LAnguage) is a Python DSL and solver for combinatorial counting problems via Weighted First-Order Model Counting (WFOMC).

- **Language**: Python 3.11+
- **Package manager**: `uv`
- **Entry point**: `cofola.solver:main`
- **WFOMC backend**: [yuanhong-wang/WFOMC](https://github.com/yuanhong-wang/WFOMC) (branch `for_cofola`)

---

## Essential Commands

```bash
# Install / sync dependencies
uv sync

# Run solver on a .cfl problem file
uv run cofola -i problems/others/bag.cfl
uv run cofola -i problems/others/bag.cfl -d   # debug logging

# Type checking
uv run pyright

# Tests
uv run pytest
uv run pytest tests/test_specific.py::test_name -v
```

---

## Repository Layout

```
cofola/
├── src/cofola/
│   ├── solver.py          # CLI entry point + solve() / solve_single_problem()
│   ├── problem.py         # CofolaProblem: build, simplify, optimize, transform
│   ├── encoder.py         # Encode CofolaProblem → WFOMC problem
│   ├── decoder.py         # Decode WFOMC result → integer count
│   ├── wfomc_solver.py    # Thin wrapper around wfomc.solve()
│   ├── context.py         # Global context / entity registry
│   ├── utils.py           # Shared utilities
│   ├── objects/           # Combinatorial object hierarchy
│   │   ├── base.py        # CombinatoricsBase, CombinatoricsObject, constraints
│   │   ├── bag.py         # BagInit, SizeConstraint
│   │   ├── set.py
│   │   ├── tuple.py
│   │   ├── sequence.py
│   │   ├── partition.py
│   │   └── function.py
│   └── parser/
│       ├── grammar.py     # Lark grammar definition
│       └── parser.py      # parse() → CofolaProblem
├── problems/              # .cfl problem files for testing / benchmarking
│   └── others/            # Miscellaneous examples
├── check-points/          # Experiment logs and result CSVs
├── scripts/               # Utility / analysis scripts
├── AGENTS.md              # Agent-specific dev guide (more detailed)
└── pyproject.toml
```

---

## Solve Pipeline

```
.cfl text
  → parse()          [parser/parser.py]
  → CofolaProblem
  → solve()          [solver.py]
      → satisfiable() on compound constraints (sympy)
      → solve_single_problem() per truth-assignment
          → decompose_problem()   # connected components
          → simplify / optimize / infer_max_size / transform
          → encode()             [encoder.py]  → WFOMCProblem + Decoder
          → solve_wfomc()        [wfomc_solver.py]
          → decoder.decode_result()
  → integer answer
```

---

## Code Style (project-specific)

- Inherit from `object` explicitly: `class Foo(object):`
- Use `from __future__ import annotations` at the top of files
- Type hints: modern syntax (`list[X]`, `tuple[A, B]`) — Python 3.11+
- Logging: `from logzero import logger` — use `logger.info/debug/warning`
- Never use `print()` for diagnostic output
- Custom exceptions inherit from `Exception`
- Return early for unsatisfiable / degenerate cases (avoid deep nesting)

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `wfomc` | WFOMC solver (git dependency) |
| `sympy` | Symbolic constraint solving (`satisfiable`) |
| `scipy` | Linear programming |
| `numpy` | Numerical operations |
| `logzero` | Structured logging |

---

## Testing Problem Files

Problem files use the `.cfl` extension. Examples live in `problems/others/`.
Run any example directly:

```bash
uv run cofola -i problems/others/gifts.cfl
uv run cofola -i problems/others/cards.cfl -d
```

---

## Important Notes

- The solver uses `lifted=False` by default in `main()` (see `solver.py:244`).
- If a problem contains a linear order axiom (LEQ predicate), the algorithm auto-switches to `INCREMENTAL`.
- `decompose_problem()` finds connected components in the "problem graph" to split independent sub-problems and multiply their counts.
- Deep copying (`deepcopy`) is used when iterating over `satisfiable()` models to avoid side effects on the original problem.
