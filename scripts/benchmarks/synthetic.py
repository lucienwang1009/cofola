"""Deterministic stratified synthetic benchmark generation.

The suite is a robustness benchmark, not a scaling benchmark.  It balances
object types and dependency shapes explicitly, while retaining controlled
random variation inside each structural stratum.  Expected answers are
computed from combinatorial formulae in this module rather than by invoking a
Cofola backend.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from scripts.benchmarks.cases import BenchmarkCase, save_cases


logger = logging.getLogger(__name__)


DEFAULT_SYNTHETIC_SEED = 20260811
SYNTHETIC_DOMAINS = (8, 12, 16)
SYNTHETIC_DIFFICULTIES = ("small", "medium", "large")
SYNTHETIC_VARIANTS_PER_CELL = 8
SYNTHETIC_STRATA = (
    "set_choice_intersection",
    "bag_choice_multiplicity",
    "set_tuple_position_count",
    "bag_tuple_multiplicity",
    "set_sequence_patterns",
    "bag_sequence_adjacency",
    "set_partition_min_size",
    "set_composition_sizes",
    "dependent_choice_disjoint",
    "dependent_tuple_position_count",
    "dependent_sequence_patterns",
    "dependent_partition_min_size",
    "dependent_composition_sizes",
    "dependent_bag_tuple",
    "nested_choice_chain",
    "dependent_selected_subsets",
)
DEPENDENT_SYNTHETIC_STRATA = frozenset(SYNTHETIC_STRATA[8:])


@dataclass(frozen=True)
class _GeneratedCase(object):
    program: str
    expected: int
    features: tuple[str, ...]
    operator_count: int
    constraint_count: int
    depth: int


_Generator = Callable[[random.Random, int], _GeneratedCase]


def stratified_synthetic_cases(
    *,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    variants_per_cell: int = SYNTHETIC_VARIANTS_PER_CELL,
) -> list[BenchmarkCase]:
    """Generate the balanced synthetic suite for a fixed seed.

    Every stratum has ``variants_per_cell`` cases at each of the three fixed
    domains.  Duplicate programs and trivial expected answers are rejected
    during generation so a manifest cannot silently accumulate low-value
    cases.
    """

    if variants_per_cell <= 0:
        raise ValueError("variants_per_cell must be positive")

    generators = _stratum_generators()
    if tuple(generators) != SYNTHETIC_STRATA:
        raise AssertionError("Synthetic stratum registry is out of order")

    cases: list[BenchmarkCase] = []
    seen_programs: set[str] = set()
    for difficulty, domain in zip(
        SYNTHETIC_DIFFICULTIES,
        SYNTHETIC_DOMAINS,
        strict=True,
    ):
        for stratum, generator in generators.items():
            for variant in range(variants_per_cell):
                generated = _generate_unique_case(
                    generator,
                    seed=seed,
                    stratum=stratum,
                    domain=domain,
                    variant=variant,
                    seen_programs=seen_programs,
                )
                tags = (
                    "synthetic",
                    "stratified",
                    f"stratum={stratum}",
                    f"family={stratum}",
                    f"difficulty={difficulty}",
                    f"domain={domain}",
                    f"variant={variant}",
                    f"seed={seed}",
                    f"operatorcount={generated.operator_count}",
                    f"constraintcount={generated.constraint_count}",
                    f"depth={generated.depth}",
                    *generated.features,
                )
                if stratum in DEPENDENT_SYNTHETIC_STRATA:
                    tags = (*tags, "dependent_configuration", "outside_cola_fragment")
                cases.append(
                    BenchmarkCase(
                        suite="synthetic",
                        case_id=f"syn-{stratum}-{difficulty}-{variant:02d}",
                        program=generated.program,
                        expected=generated.expected,
                        tags=tags,
                        source=f"stratified synthetic generator seed={seed}",
                    )
                )

    _validate_suite(cases, variants_per_cell=variants_per_cell)
    return cases


def materialize_synthetic_suite(
    cases: Sequence[BenchmarkCase],
    directory: Path,
) -> None:
    """Replace a materialized synthetic suite without deleting unrelated files."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        records = data.get("cases", []) if isinstance(data, dict) else data
        for record in records:
            program_path = record.get("program_path")
            if not program_path:
                continue
            stale_path = _safe_manifest_program_path(output_dir, str(program_path))
            if stale_path.suffix != ".cfl":
                raise ValueError(f"Refusing to remove non-CFL manifest path: {stale_path}")
            stale_path.unlink(missing_ok=True)

    save_cases(cases, output_dir, suite_subdirectories=False)


def _safe_manifest_program_path(directory: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Manifest program path must be relative: {relative_path}")
    resolved_directory = directory.resolve()
    resolved_path = (directory / path).resolve()
    if resolved_path.parent != resolved_directory:
        raise ValueError(f"Manifest program path escapes suite directory: {relative_path}")
    return resolved_path


def _generate_unique_case(
    generator: _Generator,
    *,
    seed: int,
    stratum: str,
    domain: int,
    variant: int,
    seen_programs: set[str],
) -> _GeneratedCase:
    for attempt in range(1000):
        rng = random.Random(f"{seed}:{stratum}:{domain}:{variant}:{attempt}")
        generated = generator(rng, domain)
        program = generated.program.strip()
        if generated.expected <= 1 or generated.constraint_count <= 0:
            continue
        if program in seen_programs:
            continue
        seen_programs.add(program)
        return _GeneratedCase(
            program=program,
            expected=generated.expected,
            features=generated.features,
            operator_count=generated.operator_count,
            constraint_count=generated.constraint_count,
            depth=generated.depth,
        )
    raise RuntimeError(
        f"Could not generate a unique non-trivial case for {stratum}, "
        f"domain={domain}, variant={variant}"
    )


def _validate_suite(cases: Sequence[BenchmarkCase], *, variants_per_cell: int) -> None:
    expected_count = len(SYNTHETIC_STRATA) * len(SYNTHETIC_DOMAINS) * variants_per_cell
    if len(cases) != expected_count:
        raise AssertionError(f"Expected {expected_count} synthetic cases, got {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise AssertionError("Synthetic case ids must be unique")
    if len({case.program for case in cases}) != len(cases):
        raise AssertionError("Synthetic programs must be unique")
    if any(case.expected is None or case.expected <= 1 for case in cases):
        raise AssertionError("Synthetic cases must have non-trivial independent oracles")


def _stratum_generators() -> dict[str, _Generator]:
    return {
        "set_choice_intersection": _set_choice_intersection,
        "bag_choice_multiplicity": _bag_choice_multiplicity,
        "set_tuple_position_count": _set_tuple_position_count,
        "bag_tuple_multiplicity": _bag_tuple_multiplicity,
        "set_sequence_patterns": _set_sequence_patterns,
        "bag_sequence_adjacency": _bag_sequence_adjacency,
        "set_partition_min_size": _set_partition_min_size,
        "set_composition_sizes": _set_composition_sizes,
        "dependent_choice_disjoint": _dependent_choice_disjoint,
        "dependent_tuple_position_count": _dependent_tuple_position_count,
        "dependent_sequence_patterns": _dependent_sequence_patterns,
        "dependent_partition_min_size": _dependent_partition_min_size,
        "dependent_composition_sizes": _dependent_composition_sizes,
        "dependent_bag_tuple": _dependent_bag_tuple,
        "nested_choice_chain": _nested_choice_chain,
        "dependent_selected_subsets": _dependent_selected_subsets,
    }


def _set_choice_intersection(rng: random.Random, domain: int) -> _GeneratedCase:
    left_size = rng.randint(3, domain - 3)
    right_size = domain - left_size
    choose_size = rng.randint(3, min(domain - 1, 8))
    lower = max(1, choose_size - right_size)
    upper = min(left_size, choose_size - 1)
    if lower > upper:
        return _set_choice_intersection(rng, domain)
    left_chosen = rng.randint(lower, upper)
    expected = math.comb(left_size, left_chosen) * math.comb(
        right_size,
        choose_size - left_chosen,
    )
    program = "\n".join(
        (
            f"left = set(a0...{left_size})",
            f"right = set(b0...{right_size})",
            f"picked = choose(left + right, {choose_size})",
            f"|picked & left| == {left_chosen}",
        )
    )
    return _GeneratedCase(program, expected, ("set", "choose", "intersection", "size"), 1, 1, 1)


def _bag_choice_multiplicity(rng: random.Random, domain: int) -> _GeneratedCase:
    distinguished_cap = rng.randint(2, min(5, domain - 3))
    singletons = domain - distinguished_cap
    choose_size = rng.randint(3, min(domain - 1, 8))
    lower = max(1, choose_size - singletons)
    upper = min(distinguished_cap, choose_size - 1)
    if lower > upper:
        return _bag_choice_multiplicity(rng, domain)
    distinguished_count = rng.randint(lower, upper)
    expected = math.comb(singletons, choose_size - distinguished_count)
    items = [f"e0: {distinguished_cap}", *(f"e{i}" for i in range(1, singletons + 1))]
    program = "\n".join(
        (
            f"stock = bag({', '.join(items)})",
            f"picked = choose(stock, {choose_size})",
            f"picked.count(e0) == {distinguished_count}",
        )
    )
    return _GeneratedCase(program, expected, ("bag", "choose", "count"), 1, 1, 1)


def _set_tuple_parameters(
    rng: random.Random,
    domain: int,
) -> tuple[int, int, int, int]:
    tuple_size = rng.randint(4, min(domain - 1, 8))
    marked_size = rng.randint(2, domain - 3)
    unmarked_size = domain - 1 - marked_size
    lower = max(1, tuple_size - 1 - unmarked_size)
    upper = min(marked_size, tuple_size - 2)
    if lower > upper:
        return _set_tuple_parameters(rng, domain)
    marked_count = rng.randint(lower, upper)
    expected = (
        math.comb(marked_size, marked_count)
        * math.comb(unmarked_size, tuple_size - 1 - marked_count)
        * math.factorial(tuple_size - 1)
    )
    return tuple_size, marked_size, marked_count, expected


def _set_tuple_position_count(rng: random.Random, domain: int) -> _GeneratedCase:
    tuple_size, marked_size, marked_count, expected = _set_tuple_parameters(rng, domain)
    marked = ", ".join(f"e{i}" for i in range(1, marked_size + 1))
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            f"marked = set({marked})",
            f"row = choose_tuple(items, {tuple_size})",
            "row[0] == e0",
            f"row.count(marked) == {marked_count}",
        )
    )
    return _GeneratedCase(program, expected, ("set", "tuple", "position", "count"), 1, 2, 1)


def _bag_tuple_parameters(
    rng: random.Random,
    domain: int,
) -> tuple[list[int], int, int, int]:
    # Tuple lowering is exponential in the number of distinct bag entities,
    # not merely in total multiplicity.  Three types still exercise genuine
    # multiset permutations while keeping all three difficulty tiers usable.
    capacities = _random_positive_composition(rng, domain, 3)
    capacities.sort(reverse=True)
    marker_count = rng.randint(4, min(8, domain))
    witness_size = rng.randint(2, marker_count - 1)
    permutations = math.factorial(domain) // math.prod(
        math.factorial(capacity) for capacity in capacities
    )
    witnesses = math.comb(marker_count - 1, witness_size - 1)
    return capacities, marker_count, witness_size, permutations * witnesses


def _bag_tuple_multiplicity(rng: random.Random, domain: int) -> _GeneratedCase:
    capacities, marker_count, witness_size, expected = _bag_tuple_parameters(rng, domain)
    program = "\n".join(
        (
            f"stock = bag({_bag_items(capacities)})",
            "row = tuple(stock)",
            f"markers = set(m0...{marker_count})",
            f"witness = choose(markers, {witness_size})",
            "m0 in witness",
        )
    )
    return _GeneratedCase(
        program,
        expected,
        ("bag", "tuple", "set", "choose", "membership", "decomposition"),
        2,
        1,
        1,
    )


def _sequence_parameters(
    rng: random.Random,
    domain: int,
    *,
    require_proper_subset: bool,
) -> tuple[int, int, int]:
    max_size = min(domain - int(require_proper_subset), 10)
    sequence_size = rng.randint(5, max_size)
    max_pairs = min(3, (sequence_size - 2) // 2, (domain - 2) // 2)
    before_pairs = rng.randint(1, max_pairs)
    required = 2 + 2 * before_pairs
    expected = (
        math.comb(domain - required, sequence_size - required)
        * 2
        * math.factorial(sequence_size - 1)
        // (2**before_pairs)
    )
    return sequence_size, before_pairs, expected


def _sequence_constraints(
    rng: random.Random,
    domain: int,
    before_pairs: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    ordered_entities = rng.sample(range(2, domain), 2 * before_pairs)
    precedence: list[str] = []
    for pair_index in range(before_pairs):
        left, right = ordered_entities[2 * pair_index : 2 * pair_index + 2]
        if rng.random() < 0.5:
            left, right = right, left
        precedence.append(f"row.before(e{left}, e{right})")
    return (
        ("next_to(e0, e1) in row", *precedence),
        (0, 1, *ordered_entities),
    )


def _set_sequence_patterns(rng: random.Random, domain: int) -> _GeneratedCase:
    before_pairs = rng.randint(1, min(3, (domain - 2) // 2))
    expected = 2 * math.factorial(domain - 1) // (2**before_pairs)
    constraints, _ = _sequence_constraints(rng, domain, before_pairs)
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            "row = sequence(items)",
            *constraints,
        )
    )
    return _GeneratedCase(
        program,
        expected,
        ("set", "sequence", "adjacency", "order"),
        1,
        len(constraints),
        1,
    )


def _bag_sequence_adjacency(rng: random.Random, domain: int) -> _GeneratedCase:
    # Keep the number of distinct types fixed so the three tiers scale by
    # multiplicity rather than receiving accidental, non-monotone hardness.
    other_caps = _random_positive_composition(rng, domain - 2, 3)
    capacities = [1, 1, *other_caps]
    expected = 2 * math.factorial(domain - 1) // math.prod(
        math.factorial(capacity) for capacity in other_caps
    )
    program = "\n".join(
        (
            f"stock = bag({_bag_items(capacities)})",
            "row = sequence(stock)",
            "next_to(e0, e1) in row",
        )
    )
    return _GeneratedCase(program, expected, ("bag", "sequence", "adjacency"), 1, 1, 1)


def _partition_parameters(
    rng: random.Random,
    total: int,
) -> tuple[int, int, int]:
    parts = rng.randint(2, min(5, total // 2))
    min_size = rng.randint(1, total // parts)
    expected = _restricted_set_partitions(total, parts, min_size)
    return parts, min_size, expected


def _set_partition_min_size(rng: random.Random, domain: int) -> _GeneratedCase:
    parts, min_size, expected = _partition_parameters(rng, domain)
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            f"groups = partition(items, {parts})",
            f"|part| >= {min_size} for part in groups",
        )
    )
    return _GeneratedCase(program, expected, ("set", "partition", "size"), 1, 1, 1)


def _composition_parameters(
    rng: random.Random,
    total: int,
) -> tuple[list[int], int]:
    parts = rng.randint(2, min(5, total - 1))
    sizes = _random_positive_composition(rng, total, parts)
    expected = math.factorial(total) // math.prod(math.factorial(size) for size in sizes)
    return sizes, expected


def _composition_constraints(name: str, sizes: Sequence[int]) -> tuple[str, ...]:
    return tuple(f"|{name}[{index}]| == {size}" for index, size in enumerate(sizes))


def _set_composition_sizes(rng: random.Random, domain: int) -> _GeneratedCase:
    sizes, expected = _composition_parameters(rng, domain)
    constraints = _composition_constraints("groups", sizes)
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            f"groups = compose(items, {len(sizes)})",
            *constraints,
        )
    )
    return _GeneratedCase(
        program,
        expected,
        ("set", "composition", "size"),
        1,
        len(constraints),
        1,
    )


def _dependent_choice_disjoint(rng: random.Random, domain: int) -> _GeneratedCase:
    team_size = rng.randint(4, min(domain - 1, 9))
    committee_size = rng.randint(1, team_size - 2)
    expected = (
        math.comb(domain, team_size)
        * team_size
        * math.comb(team_size - 1, committee_size)
    )
    program = "\n".join(
        (
            f"people = set(e0...{domain})",
            f"team = choose(people, {team_size})",
            "captain = choose(team, 1)",
            f"committee = choose(team, {committee_size})",
            "captain disjoint committee",
        )
    )
    return _GeneratedCase(program, expected, ("set", "choose", "disjoint"), 3, 1, 2)


def _dependent_tuple_position_count(rng: random.Random, domain: int) -> _GeneratedCase:
    tuple_size, marked_size, marked_count, expected = _set_tuple_parameters(rng, domain)
    marked = ", ".join(f"e{i}" for i in range(1, marked_size + 1))
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            f"marked = set({marked})",
            f"selected = choose(items, {tuple_size})",
            f"row = choose_tuple(selected, {tuple_size})",
            "row[0] == e0",
            f"row.count(marked) == {marked_count}",
        )
    )
    return _GeneratedCase(program, expected, ("set", "choose", "tuple", "position", "count"), 2, 2, 2)


def _dependent_sequence_patterns(rng: random.Random, domain: int) -> _GeneratedCase:
    sequence_size, before_pairs, expected = _sequence_parameters(
        rng,
        domain,
        require_proper_subset=True,
    )
    constraints, required_entities = _sequence_constraints(rng, domain, before_pairs)
    membership_constraints = tuple(
        f"e{entity} in selected" for entity in required_entities
    )
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            f"selected = choose(items, {sequence_size})",
            "row = sequence(selected)",
            *membership_constraints,
            *constraints,
        )
    )
    return _GeneratedCase(
        program,
        expected,
        ("set", "choose", "sequence", "adjacency", "order"),
        2,
        len(membership_constraints) + len(constraints),
        2,
    )


def _dependent_partition_min_size(rng: random.Random, domain: int) -> _GeneratedCase:
    selected_size = rng.randint(5, min(domain - 1, 11))
    parts, min_size, partition_count = _partition_parameters(rng, selected_size)
    expected = math.comb(domain, selected_size) * partition_count
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            f"selected = choose(items, {selected_size})",
            f"groups = partition(selected, {parts})",
            f"|part| >= {min_size} for part in groups",
        )
    )
    return _GeneratedCase(program, expected, ("set", "choose", "partition", "size"), 2, 1, 2)


def _dependent_composition_sizes(rng: random.Random, domain: int) -> _GeneratedCase:
    selected_size = rng.randint(5, min(domain - 1, 11))
    sizes, composition_count = _composition_parameters(rng, selected_size)
    constraints = _composition_constraints("groups", sizes)
    expected = math.comb(domain, selected_size) * composition_count
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            f"selected = choose(items, {selected_size})",
            f"groups = compose(selected, {len(sizes)})",
            *constraints,
        )
    )
    return _GeneratedCase(
        program,
        expected,
        ("set", "choose", "composition", "size"),
        2,
        len(constraints),
        2,
    )


def _dependent_bag_tuple(rng: random.Random, domain: int) -> _GeneratedCase:
    capacities, marker_count, witness_size, expected = _bag_tuple_parameters(rng, domain)
    program = "\n".join(
        (
            f"stock = bag({_bag_items(capacities)})",
            f"picked = choose(stock, {domain})",
            "row = tuple(picked)",
            f"markers = set(m0...{marker_count})",
            f"witness = choose(markers, {witness_size})",
            "m0 in witness",
        )
    )
    return _GeneratedCase(
        program,
        expected,
        ("bag", "choose", "tuple", "set", "membership", "decomposition"),
        3,
        1,
        2,
    )


def _nested_choice_chain(rng: random.Random, domain: int) -> _GeneratedCase:
    team_size = rng.randint(4, min(domain - 1, 9))
    shortlist_size = rng.randint(2, team_size - 1)
    expected = (
        math.comb(domain - 1, team_size - 1)
        * math.comb(team_size, shortlist_size)
        * shortlist_size
    )
    program = "\n".join(
        (
            f"people = set(e0...{domain})",
            f"team = choose(people, {team_size})",
            "e0 in team",
            f"shortlist = choose(team, {shortlist_size})",
            "captain = choose(shortlist, 1)",
        )
    )
    return _GeneratedCase(program, expected, ("set", "choose", "membership"), 3, 1, 3)


def _dependent_selected_subsets(rng: random.Random, domain: int) -> _GeneratedCase:
    selected_size = rng.randint(5, min(domain - 1, 10))
    left_size = rng.randint(1, selected_size - 2)
    right_size = rng.randint(1, selected_size - left_size - 1)
    expected = (
        math.comb(domain, selected_size)
        * math.comb(selected_size, left_size)
        * math.comb(selected_size - left_size, right_size)
    )
    program = "\n".join(
        (
            f"items = set(e0...{domain})",
            f"selected = choose(items, {selected_size})",
            f"left = choose(selected, {left_size})",
            f"right = choose(selected, {right_size})",
            "left disjoint right",
        )
    )
    return _GeneratedCase(program, expected, ("set", "choose", "disjoint"), 3, 1, 2)


def _bag_items(capacities: Sequence[int]) -> str:
    return ", ".join(
        f"e{index}: {capacity}" if capacity > 1 else f"e{index}"
        for index, capacity in enumerate(capacities)
    )


def _random_positive_composition(
    rng: random.Random,
    total: int,
    parts: int,
) -> list[int]:
    if parts <= 0 or total < parts:
        raise ValueError("Positive composition requires total >= parts > 0")
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    bounds = [0, *cuts, total]
    return [bounds[index + 1] - bounds[index] for index in range(parts)]


def _restricted_set_partitions(total: int, parts: int, min_size: int) -> int:
    cache: dict[tuple[int, int], int] = {}

    def rec(remaining: int, remaining_parts: int) -> int:
        key = (remaining, remaining_parts)
        if key in cache:
            return cache[key]
        if remaining_parts == 0:
            return int(remaining == 0)
        if remaining < remaining_parts * min_size:
            return 0
        max_first_size = remaining - (remaining_parts - 1) * min_size
        value = sum(
            math.comb(remaining - 1, first_size - 1)
            * rec(remaining - first_size, remaining_parts - 1)
            for first_size in range(min_size, max_first_size + 1)
        )
        cache[key] = value
        return value

    return rec(total, parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("problems/benchmarks/synthetic"),
        help="Directory receiving flat .cfl files and JSON/CSV manifests.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument(
        "--variants-per-cell",
        type=int,
        default=SYNTHETIC_VARIANTS_PER_CELL,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = stratified_synthetic_cases(
        seed=args.seed,
        variants_per_cell=args.variants_per_cell,
    )
    materialize_synthetic_suite(cases, args.output_dir)
    logger.info("Materialized %d synthetic cases in %s", len(cases), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
