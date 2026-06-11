# Test Layout

Tests are organized into subpackages by the layer under test. Use this map when
adding a regression so the intent stays visible. Each module is kept focused on
a single concern.

```
tests/
├── helpers.py                     # shared Problem-navigation helpers
├── frontend/
│   ├── test_parser_errors.py      # parser/transformer diagnostics, grammar edges
│   ├── test_type_check.py         # frontend type-checking failures
│   └── test_problem_builder_usage.py  # public builder API + hand-built problems
├── planning/
│   ├── test_pass_infrastructure.py    # AnalysisManager, RefAllocator, fixed-point passes
│   ├── test_reference_utilities.py    # object_refs / constraint_refs / RefAllocator
│   ├── test_analysis_inference.py     # entity / max-size / merged analyses
│   ├── test_optimization_passes.py    # constant folding, full-choice, size folding, merge
│   ├── test_lowering_passes.py        # tuple/partition/function/linear-def lowering
│   ├── test_analysis_boundaries.py    # analysis boundary cases and policy edges
│   └── test_transform_invariants.py   # cross-pass invariants on the pipeline
├── backends/
│   ├── wfomc/
│   │   ├── test_profile.py             # backend planning-profile declaration
│   │   ├── test_collection_semantics.py  # set / bag / choice semantics
│   │   ├── test_sequence_patterns.py   # sequence/circle pattern syntax + semantics
│   │   ├── test_choice_membership.py   # choice objects, bag negation, tuple membership
│   │   └── test_encoding_boundaries.py # encoding invariants + error boundaries
│   └── test_backend_coso.py            # CoSo encoding, routing, fallback (needs `coso` extra)
└── benchmarks/
    ├── test_benchmark_cases.py    # generated benchmark discovery + manifest round trips
    ├── test_benchmark_run.py      # benchmark runner behavior and result classification
    └── test_all_problems.py       # corpus-level parse/type/solve coverage
```

## Useful Commands

```bash
uv run pytest                                   # everything
uv run pytest tests/frontend tests/planning -q  # frontend + planning
uv run pytest tests/backends/wfomc -q           # WFOMC backend
COFOLA_ALL_TESTS=1 uv run pytest tests/benchmarks/test_all_problems.py
```

## Conventions

- Group related test methods into a `Test*` class per concern; keep each module
  within ~400 lines and split by sub-concern when it grows past that.
- Shared `Problem`-navigation helpers (`_ref_named`, `_first_def_ref`, …) live in
  `tests/helpers.py`; import them rather than redefining per module.
- Tests import only from `cofola` (and `tests.helpers`); they do not import from
  one another.
