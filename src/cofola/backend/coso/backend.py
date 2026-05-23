"""CoSo backend - implements Backend ABC by emitting CoLa."""
from __future__ import annotations

import itertools
import math

from loguru import logger

from cofola.backend.base import Backend
from cofola.backend.coso.encoder import encode
from cofola.backend.coso.solver import run_coso_program
from cofola.frontend.constraints import SizeConstraint
from cofola.frontend.objects import (
    BagIntersection,
    BagPartDef,
    CompositionDef,
    Entity,
    ObjRef,
    PartitionDef,
    SetIntersection,
    SetPartDef,
)
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.passes.lowering import ForAllPartsExpansionStep, LoweringPass
from cofola.planing.passes.merge_identical import MergeIdenticalObjects
from cofola.planing.passes.optimize import ConstantFolder, SizeConstraintFolder
from cofola.planing.passes.simplify import SimplifyPass
from cofola.planing.pipeline import PlanningProfile

__all__ = ["COSO_GLOBAL_PASSES", "COSO_LOCAL_PASSES", "CoSoBackend"]


COSO_GLOBAL_PASSES = (
    FixedPointPass(ConstantFolder),
    MergeIdenticalObjects,
)


class _CoSoForAllPartsLoweringPass(LoweringPass):
    """Expand ``for part in ...`` without lowering CoSo-native tuple objects."""

    STEP_CLASSES = (ForAllPartsExpansionStep,)


# CoSo can directly represent tuple/permutation configurations with absolute
# positional and counting constraints. Preserve frontend TupleDef nodes instead
# of lowering them to FuncDef, and let the encoder reject sequence/circle
# constructs with relative positional constraints.
COSO_LOCAL_PASSES = (
    FixedPointPass(_CoSoForAllPartsLoweringPass),
    SizeConstraintFolder,
    MergeIdenticalObjects,
    SimplifyPass,
)


class CoSoBackend(Backend):
    """Solves a single-configuration problem by translating it to CoLa."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @property
    def name(self) -> str:
        """Human-readable backend identifier."""
        return "coso"

    def planning_profile(self) -> PlanningProfile:
        """Return the CoSo-compatible planning profile."""

        return PlanningProfile(
            global_passes=COSO_GLOBAL_PASSES,
            local_passes=COSO_LOCAL_PASSES,
        )

    def solve(
        self,
        problem: Problem,
        analysis: AnalysisResult,
    ) -> int:
        """Encode and solve a single atomic planning problem via CoSo."""

        logger.info(
            "CoSoBackend.solve: encoding planning problem ({} objects, {} constraints)",
            len(list(problem.iter_objects())),
            len(problem.constraints),
        )
        direct = _direct_level2_count(problem, analysis)
        if direct is not None:
            return direct

        program = encode(problem, analysis)
        if program.is_trivial:
            logger.debug("CoSoBackend: trivial component -> {}", program.trivial_count)
            return program.trivial_count

        logger.debug("CoSoBackend: generated CoLa program:\n{}", program.cola)
        result = run_coso_program(program.cola, debug=self.debug)
        if program.count_divisor != 1:
            if result % program.count_divisor != 0:
                raise ValueError(
                    "CoSo result is not divisible by indexed composition "
                    f"normalization factor {program.count_divisor}: {result}"
                )
            result //= program.count_divisor
        logger.info("CoSoBackend: final result = {}", result)
        return result


def _direct_level2_count(
    problem: Problem,
    analysis: AnalysisResult,
) -> int | None:
    config_items = [
        (ref, defn)
        for ref, defn in problem.defs
        if isinstance(defn, (CompositionDef, PartitionDef))
    ]
    if len(config_items) != 1:
        return None

    config_ref, config = config_items[0]
    info = analysis.set_info.get(config.source) or analysis.bag_info.get(config.source)
    if info is None:
        return None

    multiplicities = getattr(info, "p_entities_multiplicity", None)
    if multiplicities is None:
        multiplicities = {entity: 1 for entity in info.p_entities}

    parts = config.num_parts
    constraints = list(problem.constraints)
    evaluators = [
        _level2_constraint_evaluator(c, config_ref, problem, analysis)
        for c in constraints
    ]
    if any(evaluator is None for evaluator in evaluators):
        return None

    if not constraints and isinstance(config, CompositionDef):
        return math.prod(
            math.comb(multiplicity + parts - 1, parts - 1)
            for multiplicity in multiplicities.values()
        )

    distribution_options = [
        _weak_compositions(multiplicity, parts)
        for _, multiplicity in sorted(multiplicities.items(), key=lambda item: item[0].name)
    ]
    search_space = math.prod(len(options) for options in distribution_options)
    if search_space > 2_000_000:
        return None

    entities = [
        entity
        for entity, _ in sorted(multiplicities.items(), key=lambda item: item[0].name)
    ]
    seen_partitions: set[tuple[tuple[tuple[str, int], ...], ...]] = set()
    count = 0
    for distributions in itertools.product(*distribution_options):
        boxes = [
            {
                entity: distribution[part_index]
                for entity, distribution in zip(entities, distributions, strict=True)
                if distribution[part_index] > 0
            }
            for part_index in range(parts)
        ]
        if not all(evaluator(boxes) for evaluator in evaluators if evaluator is not None):
            continue
        if isinstance(config, PartitionDef):
            key = _partition_key(boxes)
            if key in seen_partitions:
                continue
            seen_partitions.add(key)
        count += 1
    return count


def _level2_constraint_evaluator(
    constraint: object,
    config_ref: ObjRef,
    problem: Problem,
    analysis: AnalysisResult,
):
    if not isinstance(constraint, SizeConstraint):
        return None
    atoms = [
        (_level2_size_atom_evaluator(atom, config_ref, problem, analysis), coeff)
        for atom, coeff in constraint.terms
    ]
    if any(atom is None for atom, _ in atoms):
        return None

    def evaluate(boxes: list[dict[Entity, int]]) -> bool:
        value = sum(coeff * atom(boxes) for atom, coeff in atoms if atom is not None)
        return _compare(value, constraint.comparator, constraint.rhs)

    return evaluate


def _level2_size_atom_evaluator(
    atom: object,
    config_ref: ObjRef,
    problem: Problem,
    analysis: AnalysisResult,
):
    if isinstance(atom, ObjRef):
        part_index = _part_index(atom, config_ref, problem)
        if part_index is not None:
            return lambda boxes: sum(boxes[part_index].values())

        defn = problem.get_object(atom)
        if isinstance(defn, (SetIntersection, BagIntersection)):
            left_index = _part_index(defn.left, config_ref, problem)
            if left_index is not None:
                entities = _positive_entities(defn.right, analysis)
                return lambda boxes: sum(
                    count for entity, count in boxes[left_index].items() if entity in entities
                )
            right_index = _part_index(defn.right, config_ref, problem)
            if right_index is not None:
                entities = _positive_entities(defn.left, analysis)
                return lambda boxes: sum(
                    count for entity, count in boxes[right_index].items() if entity in entities
                )
    return None


def _part_index(ref: ObjRef, config_ref: ObjRef, problem: Problem) -> int | None:
    defn = problem.get_object(ref)
    if isinstance(defn, (SetPartDef, BagPartDef)) and defn.partition == config_ref:
        return defn.index
    return None


def _positive_entities(ref: ObjRef, analysis: AnalysisResult) -> set[Entity]:
    set_info = analysis.set_info.get(ref)
    if set_info is not None:
        return set(set_info.p_entities)
    bag_info = analysis.bag_info.get(ref)
    if bag_info is not None:
        return {
            entity
            for entity, multiplicity in bag_info.p_entities_multiplicity.items()
            if multiplicity > 0
        }
    return set()


def _compare(lhs: int, comparator: str, rhs: int) -> bool:
    return {
        "==": lhs == rhs,
        "!=": lhs != rhs,
        "<": lhs < rhs,
        "<=": lhs <= rhs,
        ">": lhs > rhs,
        ">=": lhs >= rhs,
    }[comparator]


def _partition_key(
    boxes: list[dict[Entity, int]],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    return tuple(
        sorted(
            tuple(sorted((entity.name, count) for entity, count in box.items()))
            for box in boxes
        )
    )


def _weak_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        return ((total,),)
    result: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _weak_compositions(total - first, parts - 1):
            result.append((first,) + rest)
    return tuple(result)
