"""Benchmark case generation for Cofola experiments.

Three experiment families:

- real-world combinatorics problems,
- growing-domain variants (parameterised scaling families), and
- random synthetic CoLa-style benchmarks.

This module keeps those case definitions deterministic and independent from
the runner so experiments can be reproduced with the same input JSON and seed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkCase(object):
    """One benchmark instance."""

    suite: str
    case_id: str
    program: str
    expected: int | None = None
    tags: tuple[str, ...] = ()
    source: str = ""


def load_saved_cases(path: Path) -> list[BenchmarkCase]:
    """Load benchmark cases from a saved manifest file or directory."""

    manifest_path = path / "manifest.json" if path.is_dir() else path
    data = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent
    records = data["cases"] if isinstance(data, dict) and "cases" in data else data
    cases: list[BenchmarkCase] = []
    for record in records:
        program = str(record.get("program", ""))
        program_path = record.get("program_path")
        if program_path:
            program = (base_dir / str(program_path)).read_text().strip()
        tags = record.get("tags", ())
        expected = record.get("expected")
        cases.append(
            BenchmarkCase(
                suite=str(record["suite"]),
                case_id=str(record["case_id"]),
                program=program.strip(),
                expected=None if expected in (None, "") else int(expected),
                tags=tuple(str(tag) for tag in tags),
                source=str(record.get("source", "")),
            )
        )
    return cases


def save_cases(cases: Iterable[BenchmarkCase], directory: Path) -> None:
    """Persist benchmarks as reusable .cfl files plus JSON/CSV manifests."""

    case_list = list(cases)
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for case in case_list:
        suite_dir = directory / _safe_path_part(case.suite)
        suite_dir.mkdir(parents=True, exist_ok=True)
        program_rel = Path(_safe_path_part(case.suite)) / f"{_safe_path_part(case.case_id)}.cfl"
        program_text = case.program.strip() + "\n"
        (directory / program_rel).write_text(program_text)
        records.append(
            {
                **asdict(case),
                "program": program_text.rstrip("\n"),
                "program_path": str(program_rel),
                "program_sha256": hashlib.sha256(program_text.encode()).hexdigest(),
            }
        )

    manifest = {
        "format": "cofola-benchmark-manifest-v1",
        "num_cases": len(records),
        "cases": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_manifest_csv(directory / "manifest.csv", records)


def _write_manifest_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fields = (
        "suite",
        "case_id",
        "expected",
        "tags",
        "source",
        "program_path",
        "program_sha256",
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fields}
            row["expected"] = "" if row["expected"] is None else row["expected"]
            row["tags"] = ";".join(record.get("tags", ()))
            writer.writerow(row)


def _safe_path_part(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def load_real_world_cases(path: Path = Path("problems/real/corpus.json")) -> list[BenchmarkCase]:
    """Load real-world benchmark cases from problems/real/corpus.json."""

    data = json.loads(path.read_text())
    cases: list[BenchmarkCase] = []
    for problem_id, item in data.items():
        program = str(item.get("program", "")).strip()
        if not program:
            continue
        answer = item.get("answer")
        if answer in (None, ""):
            continue
        expected = int(answer)
        tags = tuple(str(tag) for tag in item.get("tags", ()))
        cases.append(
            BenchmarkCase(
                suite="real",
                case_id=str(problem_id),
                program=program,
                expected=expected,
                tags=tags,
                source=str(path),
            )
        )
    return cases


def growing_domain_cases() -> list[BenchmarkCase]:
    """Generate deterministic growing-domain benchmark families."""

    cases: list[BenchmarkCase] = []
    for i in range(11):
        vowels = 3 + i
        other_consonants = 5 + i
        choose_size = 4
        cases.append(
            BenchmarkCase(
                suite="growing",
                case_id=f"mathcounts_{i:02d}",
                program="\n".join(
                    (
                        f"vowels = bag(v0...{vowels})",
                        f"consonants = bag(t: 2, c0...{other_consonants})",
                        "magnets = vowels ++ consonants",
                        f"chosen = choose(magnets, {choose_size})",
                        f"|chosen & vowels| <= {(choose_size - 1) // 2}",
                    )
                ),
                expected=_mathcounts_expected(
                    vowels=vowels,
                    other_consonants=other_consonants,
                    choose_size=choose_size,
                ),
                tags=("mathcounts", "bag", "choose", "count", "growing"),
                source="MATHCOUNTS bag choose",
            )
        )

    for i in range(11):
        n_letters = 6 + i
        cases.append(
            BenchmarkCase(
                suite="growing",
                case_id=f"fourletter_{i:02d}",
                program="\n".join(
                    (
                        f"letters = set(a0...{n_letters})",
                        "arr = choose_tuple(letters, 4)",
                        "arr[0] == a0",
                        "arr.count(a1) > 0",
                    )
                ),
                expected=3 * (n_letters - 2) * (n_letters - 3),
                tags=("fourletter", "set", "tuple", "position", "count", "growing"),
                source="four-letter arrangements",
            )
        )

    for i in range(11):
        defective = 3 + i
        working = 9 + i
        cases.append(
            BenchmarkCase(
                suite="growing",
                case_id=f"tvs_{i:02d}",
                program="\n".join(
                    (
                        f"defective = set(d0...{defective})",
                        f"working = set(w0...{working})",
                        "purchase = choose(defective + working, 5)",
                        "|purchase & defective| >= 2",
                    )
                ),
                expected=_p3_expected(defective, working),
                tags=("tvs", "set", "choose", "count", "growing"),
                source="defective TVs purchase",
            )
        )

    sizes = [7, 5, 2]
    for i in range(11):
        if i > 0:
            sizes[(i - 1) % 3] += 1
        n_workers = sum(sizes)
        cases.append(
            BenchmarkCase(
                suite="growing",
                case_id=f"workers_{i:02d}",
                program="\n".join(
                    (
                        f"workers = set(w0...{n_workers})",
                        "groups = compose(workers, 3)",
                        f"|groups[0]| == {sizes[0]}",
                        f"|groups[1]| == {sizes[1]}",
                        f"|groups[2]| == {sizes[2]}",
                    )
                ),
                expected=math.factorial(n_workers)
                // math.prod(math.factorial(size) for size in sizes),
                tags=("workers", "set", "composition", "index", "size", "growing"),
                source="workers composition",
            )
        )

    capacities = {"B": 2, "A": 6, "N": 4}
    for k in range(2, 13):
        cases.append(
            BenchmarkCase(
                suite="growing",
                case_id=f"banana_{k:02d}",
                program="\n".join(
                    (
                        "letters = bag(B: 2, A: 6, N: 4)",
                        f"word = choose_tuple(letters, {k})",
                    )
                ),
                expected=_bounded_multiset_permutations(capacities.values(), k),
                tags=("banana", "bag", "tuple", "growing"),
                source="BANANA bag tuple",
            )
        )

    return cases


def synthetic_cases(seed: int = 0) -> list[BenchmarkCase]:
    """Generate deterministic random synthetic benchmarks in the Sec. 6.2.2 style."""

    rng = random.Random(seed)
    cases: list[BenchmarkCase] = []
    config_types = ("ss", "sr", "ms", "pm", "bm", "sq", "pt", "cp")
    universe_sizes = (10, 15, 20)
    reference_sizes = (5, 10, 15)
    for config_type in config_types:
        for universe_size in universe_sizes:
            for reference_size in reference_sizes:
                case = _synthetic_case(
                    rng,
                    config_type=config_type,
                    universe_size=universe_size,
                    reference_size=reference_size,
                    seed=seed,
                )
                cases.append(case)
    return cases


def select_cases(
    *,
    suites: Iterable[str],
    real_path: Path = Path("problems/real/corpus.json"),
    ids: set[str] | None = None,
    synthetic_seed: int = 0,
) -> list[BenchmarkCase]:
    """Build the requested benchmark case list."""

    selected: list[BenchmarkCase] = []
    suite_set = set(suites)
    if "all" in suite_set or "real" in suite_set:
        selected.extend(load_real_world_cases(Path(real_path)))
    if "all" in suite_set or "growing" in suite_set:
        selected.extend(growing_domain_cases())
    if "all" in suite_set or "synthetic" in suite_set:
        selected.extend(synthetic_cases(seed=synthetic_seed))

    if ids is not None:
        selected = [case for case in selected if case.case_id in ids]
    return selected


def _p3_expected(defective: int, working: int) -> int:
    return sum(
        math.comb(defective, d) * math.comb(working, 5 - d)
        for d in range(2, min(defective, 5) + 1)
        if 0 <= 5 - d <= working
    )


def _bounded_multiset_permutations(capacities: Iterable[int], size: int) -> int:
    caps = tuple(capacities)

    def rec(index: int, remaining: int, chosen: list[int]) -> int:
        if index == len(caps):
            if remaining != 0:
                return 0
            denom = math.prod(math.factorial(count) for count in chosen)
            return math.factorial(size) // denom
        total = 0
        for count in range(min(caps[index], remaining) + 1):
            chosen.append(count)
            total += rec(index + 1, remaining - count, chosen)
            chosen.pop()
        return total

    return rec(0, size, [])


def _mathcounts_expected(
    *,
    vowels: int,
    other_consonants: int,
    choose_size: int,
) -> int:
    max_vowels = (choose_size - 1) // 2
    return sum(
        math.comb(vowels, num_vowels)
        * _one_doubleton_bag_subsets(other_consonants, choose_size - num_vowels)
        for num_vowels in range(max_vowels + 1)
    )


def _one_doubleton_bag_subsets(singletons: int, size: int) -> int:
    if size < 0:
        return 0
    return sum(
        math.comb(singletons, size - doubleton_count)
        for doubleton_count in range(3)
        if 0 <= size - doubleton_count <= singletons
    )


def _synthetic_case(
    rng: random.Random,
    *,
    config_type: str,
    universe_size: int,
    reference_size: int,
    seed: int,
) -> BenchmarkCase:
    multiplicities = _random_multiplicities(rng, universe_size)
    if config_type == "bm":
        multiplicities = [2, 2, 1, 1]
    bag_items = ", ".join(
        f"e{i}: {multiplicity}" if multiplicity > 1 else f"e{i}"
        for i, multiplicity in enumerate(multiplicities)
    )
    set_based = {"ss", "sr", "pm", "sq", "pt", "cp"}
    domain_count = universe_size if config_type in set_based else len(multiplicities)
    properties = [
        sorted(rng.sample(range(domain_count), k=rng.randint(1, domain_count)))
        for _ in range(3)
    ]
    comparators = ("==", ">=", "<=")
    size_cmp = "=="
    count_cmp = rng.choice(comparators)

    if config_type in set_based:
        set_items = ", ".join(f"e{i}" for i in range(domain_count))
        lines = [f"U = set({set_items})"]
    else:
        lines = [f"U = bag({bag_items})"]
    for idx, prop in enumerate(properties):
        labels = ", ".join(f"e{i}" for i in prop)
        lines.append(f"P{idx} = set({labels})")

    if config_type == "ss":
        size = min(reference_size, domain_count)
        lower = rng.randint(0, min(size, len(properties[0])))
        upper = rng.randint(0, min(size, len(properties[2])))
        lines.append(f"cfg = choose(U, {size})")
        lines.append(f"|cfg & P0| >= {lower}")
        lines.append(f"|cfg & P2| <= {upper}")
        if rng.random() < 0.5:
            lines.append(f"e{properties[1][0]} in cfg")
        else:
            lines.append("cfg disjoint P1")
    elif config_type == "sr":
        lines.append(f"cfg = choose_replace(U, {reference_size})")
        lines.append(f"cfg.count(e0) {count_cmp} {rng.randint(0, reference_size)}")
        lines.append(f"cfg.count(e{properties[0][0]}) <= {reference_size}")
    elif config_type == "ms":
        lines.append("cfg = choose(U)")
        lines.append(f"|cfg| {size_cmp} {reference_size}")
        lines.append(f"cfg.count(e0) {count_cmp} {rng.randint(0, min(multiplicities[0], reference_size))}")
    elif config_type == "pm":
        lines.append(f"cfg = choose_tuple(U, {min(reference_size, universe_size)})")
        max_count = min(reference_size, len(properties[0]))
        lines.append(f"cfg.count(P0) {count_cmp} {rng.randint(0, max_count)}")
        lines.append(f"cfg[{rng.randrange(min(reference_size, universe_size))}] in P1")
    elif config_type == "bm":
        size = min(reference_size, 4)
        lines.append(f"cfg = choose_tuple(U, {size})")
        max_count = min(size, sum(1 for idx in properties[0] if idx < len(multiplicities)))
        lines.append(f"cfg.count(P0) {count_cmp} {rng.randint(0, max_count)}")
        lines.append(f"cfg[{rng.randrange(size)}] in P1")
    elif config_type == "sq":
        lines.append(f"cfg = choose_replace_tuple(U, {reference_size})")
        max_count = reference_size if properties[0] else 0
        lines.append(f"cfg.count(P0) {count_cmp} {rng.randint(0, max_count)}")
        lines.append(f"cfg[{rng.randrange(reference_size)}] in P1")
    elif config_type == "pt":
        parts = max(2, min(reference_size, 5))
        lines.append(f"cfg = partition(U, {parts})")
    elif config_type == "cp":
        parts = max(2, min(reference_size, 5))
        part_sizes = _random_positive_partition(rng, domain_count, parts)
        lines.append(f"cfg = compose(U, {parts})")
        for idx, part_size in enumerate(part_sizes):
            lines.append(f"|cfg[{idx}]| == {part_size}")
    else:
        raise ValueError(f"Unknown synthetic config type: {config_type}")

    return BenchmarkCase(
        suite="synthetic",
        case_id=f"{config_type}_{universe_size}_{reference_size}_seed{seed}",
        program="\n".join(lines),
        expected=None,
        tags=(config_type, f"u={universe_size}", f"s={reference_size}", "synthetic"),
        source="synthetic generator",
    )


def _random_multiplicities(rng: random.Random, target_size: int) -> list[int]:
    multiplicities: list[int] = []
    total = 0
    while total < target_size:
        remaining = target_size - total
        multiplicity = rng.randint(1, min(4, remaining))
        multiplicities.append(multiplicity)
        total += multiplicity
    return multiplicities


def _random_positive_partition(rng: random.Random, total: int, parts: int) -> list[int]:
    if parts <= 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    bounds = [0, *cuts, total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]
