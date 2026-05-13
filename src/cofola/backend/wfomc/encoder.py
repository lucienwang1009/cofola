"""WFOMC encoder public entry point."""
from __future__ import annotations

from copy import deepcopy

from wfomc import MultinomialCoefficients, WFOMCProblem

from cofola.backend.wfomc.constraint_encoders import _encode_constraint
from cofola.backend.wfomc.context import Context
from cofola.backend.wfomc.decoder import Decoder
from cofola.backend.wfomc.encoding_helpers import _encode_entities, _encode_singleton
from cofola.backend.wfomc.object_encoders import _encode_object
from cofola.frontend.problem import Problem
from cofola.planing.analysis.entities import AnalysisResult
from loguru import logger


# =============================================================================
# Public entry point
# =============================================================================


def encode(
    problem: Problem,
    analysis: AnalysisResult,
    lifted: bool = False,
) -> tuple[WFOMCProblem, Decoder]:
    """Encode a planning Problem + AnalysisResult to a WFOMC problem and decoder."""
    logger.debug("encode: {} objects to encode", len(list(problem.iter_objects())))

    # When not using lifted inference, all indistinguishable entities must be
    # treated as distinguishable. Work on a copy so backend encoding does not
    # mutate cached planning analyses.
    encoding_analysis = deepcopy(analysis)
    if not lifted:
        for info in encoding_analysis.bag_info.values():
            for entities in info.indis_entities.values():
                info.dis_entities.update(entities)
            info.indis_entities = {}

    context = Context(problem, encoding_analysis)
    MultinomialCoefficients.setup(len(context.domain))
    _encode_entities(encoding_analysis, context)

    logger.debug("WFOMC encode: singletons={}", context.singletons)

    if context.singletons:
        _encode_singleton(context)

    for ref in problem.topological_order():
        defn = problem.get_object(ref)
        if defn is None:
            continue
        logger.debug("Encoding {} ref={}", type(defn).__name__, ref.id)
        _encode_object(ref, defn, problem, encoding_analysis, context)

    logger.debug(
        "encode: objects encoded — predicates={}, entity_preds={}, "
        "evidence={}, weighting={}, sentence_len={}",
        len(context.ref2pred),
        len(context.ref_entity2pred),
        len(context.unary_evidence),
        len(context.weighting),
        len(str(context.sentence)),
    )

    logger.debug("encode: encoding {} constraints", len(problem.constraints))
    for c in problem.constraints:
        _encode_constraint(c, problem, encoding_analysis, context)

    logger.debug(
        "encode: constraints encoded — validator={}, gen_vars={}, overcount={}",
        len(context.validator),
        len(context.gen_vars),
        context.overcount,
    )

    result = context.build()
    logger.info("encode complete: encoding done")
    return result
