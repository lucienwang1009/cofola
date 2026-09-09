"""Controlled bag-lifting workloads with independent integer-DP answers.

Entity sweep: n = 2, 4, 8, 16, 32 at m = 2.
Multiplicity sweep: m = 2, 4, 8, 16 at n = 4 (the common point is shared).
Four dependency shapes plus named-entity and unsupported-operation controls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from loguru import logger

from scripts.benchmarks.cases import BenchmarkCase, save_cases


SHAPES = ("choose", "chain", "hidden", "intersection")


def coefficient(states: Counter, count: int, target: tuple[int, ...]) -> int:
    """Count vectors of entity multiplicities, without invoking Cofola/WFOMC."""
    zero = (0,) * len(target)
    table = {zero: 1}
    for _ in range(count):
        updated: Counter = Counter()
        for totals, ways in table.items():
            for state, multiplicity in states.items():
                combined = tuple(a + b for a, b in zip(totals, state))
                if all(a <= b for a, b in zip(combined, target)):
                    updated[combined] += ways * multiplicity
        table = updated
    return table.get(target, 0)


def make_case(shape: str, n: int, m: int) -> BenchmarkCase:
    size = n * m // 2
    half = size // 2
    root = "B = bag(" + ", ".join(f"e{i}: {m}" for i in range(n)) + ")"
    if shape == "choose":
        lines = [f"X = choose(B, {size})"]
        states = Counter({(q,): 1 for q in range(m + 1)})
        target = (size,)
    elif shape == "chain":
        lines = [f"X = choose(B, {size})", f"Y = choose(X, {half})"]
        states = Counter({(x, y): 1 for x in range(m + 1) for y in range(x + 1)})
        target = (size, half)
    elif shape == "hidden":
        lines = ["X = choose(B)", f"Y = choose(X, {size})"]
        states = Counter({(y,): m - y + 1 for y in range(m + 1)})
        target = (size,)
    elif shape == "intersection":
        lines = [f"X = choose(B, {size})", f"Y = choose(B, {size})", "Z = X & Y", f"|Z| == {half}"]
        states = Counter({(x, y, min(x, y)): 1 for x in range(m + 1) for y in range(m + 1)})
        target = (size, size, half)
    elif shape == "named":
        lines = [f"X = choose(B, {size})", "X.count(e0) == 1"]
        states = Counter({(q,): 1 for q in range(m + 1)})
        target = (size - 1,)
    elif shape == "fallback":
        lines = ["X = choose(B)", "Y = choose(B)", "Z = X + Y", f"|Z| == {size}"]
        # Cofola's multiset union takes max, not the sum of multiplicities.
        states = Counter((max(x, y),) for x in range(m + 1) for y in range(m + 1))
        target = (size,)
    else:
        raise ValueError(shape)
    expected = coefficient(states, n - 1 if shape == "named" else n, target)
    return BenchmarkCase(
        suite="bag_lifting", case_id=f"{shape}_n{n:02d}_m{m:02d}",
        program="\n".join([root, *lines]), expected=expected,
        tags=(f"shape={shape}", f"entities={n}", f"multiplicity={m}", f"size={size}"),
        source="Deterministic integer DP over entity multiplicity vectors; independent of solver encodings",
    )


def generate_cases() -> list[BenchmarkCase]:
    cases = []
    points = [(n, 2) for n in (2, 4, 8, 16, 32)]
    points += [(4, m) for m in (4, 8, 16)]
    for shape in SHAPES:
        for n, m in points:
            logger.info("Generating {} n={} m={}", shape, n, m)
            cases.append(make_case(shape, n, m))
    for shape in ("named", "fallback"):
        for n in (2, 4, 8, 16):
            cases.append(make_case(shape, n, 2))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("problems/benchmarks/bag-lifting"))
    args = parser.parse_args()
    cases = generate_cases()
    save_cases(cases, args.output_dir, suite_subdirectories=False)
    logger.info("Saved {} controlled cases to {}", len(cases), args.output_dir)


if __name__ == "__main__":
    main()
