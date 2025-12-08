from __future__ import annotations
from functools import reduce
from itertools import product
from logzero import logger
from wfomc import WFOMCProblem, X, Const, QuantifiedFormula, \
    Universal, top, MultinomialCoefficients

from cofola.context import Context
from cofola.decoder import Decoder
from cofola.objects.bag import BagAdditiveUnion, BagChoose, BagDifference, BagInit, BagMultiplicity, SizeConstraint
from cofola.objects.base import Bag, Entity, Set, Tuple
from cofola.objects.partition import BagPartition
from cofola.objects.set import SetChooseReplace

from cofola.problem import CofolaProblem
from cofola.utils import create_aux_pred


def preprocess_bags(problem: CofolaProblem,
                    lifted: bool = True):
    # if not lifted, we view all entities in bags as distinguishable
    if not lifted:
        for obj in problem.objects:
            if isinstance(obj, Bag):
                obj.dis_entities = set(obj.p_entities_multiplicity.keys())
        return
    # 1. the objects derived from BagAdditiveUnion, BagDifference, BagPartition and Tuple are non-liftable
    non_lifted_bags = set()
    for obj in problem.objects:
        if isinstance(obj, BagAdditiveUnion) or \
                isinstance(obj, BagDifference) or \
                isinstance(obj, BagPartition) or \
                isinstance(obj, Tuple):
            non_lifted_bags.add(obj)
    # the dependences of non-liftable objects are also non-liftable
    # TODO: a possible optimization: the entities with maximum multiplicity 1 can be lifted
    non_lifted_bags.update(
        problem.get_all_dependences(*non_lifted_bags)
    )
    for obj in problem.objects:
        if obj in non_lifted_bags and isinstance(obj, Bag):
            obj.dis_entities = set(obj.p_entities_multiplicity.keys())
    # 2. the entities with multiplicity constraints cannot be lifted
    for constraint in problem.constraints:
        if isinstance(constraint, SizeConstraint):
            for bag_mul, _ in constraint.expr:
                if isinstance(bag_mul, BagMultiplicity):
                    non_lifted_entity = bag_mul.entity
                    for obj in problem.get_all_dependences(bag_mul):
                        if isinstance(obj, Bag):
                            obj.dis_entities.add(non_lifted_entity)
    # 3. the remaining entities in BagInit and SetChooseReplace with the same multiplicity, i.e., indistinguishable entities, can be lifted
    for obj in problem.objects:
        if isinstance(obj, (BagInit, SetChooseReplace)):
            multis = set(obj.p_entities_multiplicity.values())
            for multiplicity in multis:
                entities = set(
                    e for e, m in obj.p_entities_multiplicity.items()
                    if m == multiplicity and e not in obj.dis_entities
                )
                # if there are more than one entities with the same multiplicity, they are indistinguishable
                # for entities with multiplicity 1, they are handled by singletons
                if len(entities) > 1 and multiplicity > 1:
                    obj.indis_entities[multiplicity] = entities
                else:
                    obj.dis_entities.update(entities)
    # 4. propogate the non-liftable entities
    # TODO: Here, for simplicity, we discriminate distinguishable entities and indistinguishable entities in a bag
    # Actually, an entity can be distinguishable and indistinguishable in the same bag for different branchs, e.g.,
    #               Bag(a:2, b:2)
    #               /           \
    #        BagChoose1     BagChoose2
    #            |                \
    # MultiplicityConstraint(a=1)  BagChoose3
    # when we encode the left branch, the entity a is distinguishable, while in the right branch, a and b are indistinguishable
    problem.propogate()
    # 5. now the bag objects forms several trees connected by `choose` operations
    # we need to find the root of each tree for lifting
    root_bags = list()
    for obj in problem.objects:
        # a bag that is not formed by `choose` operations is a root
        if isinstance(obj, (BagChoose, SetChooseReplace)) and \
                not isinstance(obj.obj_from, BagChoose) and \
                len(obj.indis_entities) > 0:
            root_bags.append(obj)
    return root_bags


def encode_set_entities(entities: set[Entity], context: Context) -> Context:
    """
    Encode a set of entities

    :param entities: the set of entities
    :return: the encoding
    """
    pred = context.get_pred(entities, create=True)
    for e in context.problem.entities:
        if e in entities:
            context.unary_evidence.add(pred(Const(e.name)))
        else:
            context.unary_evidence.add(~pred(Const(e.name)))
    return pred, context


def lifted_encode_bags(
    context: Context,
    root_bags: list[Bag],
) -> Context:
    def all_configs(bag: Bag, root: bool = False):
        if len(bag.descendants) == 0 or \
                all(not isinstance(obj, BagChoose) for obj in bag.descendants):
            # an entity can in a bag or not
            yield tuple(), (bag, )
            if not root:
                yield (bag, ), tuple()
            return
        descent_configs = [
            all_configs(obj) for obj in bag.descendants if isinstance(obj, BagChoose)
        ]
        for descent_config in product(*descent_configs):
            neg = sum((neg for neg, _ in descent_config), ())
            pos = sum((pos for _, pos in descent_config), ())
            # if the entity is in some descendants, the entity must be in the bag
            # if the bag is a root, the entity must be in the bag
            if len(pos) > 0 or root:
                yield neg, pos + (bag, )
            else:
                yield neg + (bag, ), pos
                yield neg, pos + (bag, )

    def get_symbolic_weight(bag: Bag, context: Context, pos_bags: set[Bag], multi: int, max_multi: int):
        var = context.get_indis_entity_var(bag, multi)
        if len(bag.descendants) == 0 or \
                all(descent not in pos_bags for descent in bag.descendants):
            return sum(var ** i for i in range(1, max_multi + 1))
        ret = 0
        for i in range(1, max_multi + 1):
            descendant_weight = 0
            for descendant in bag.descendants:
                if descendant in pos_bags:
                    descendant_weight += get_symbolic_weight(descendant, context, pos_bags, multi, i)
            ret += var ** i * descendant_weight
        return ret

    for root_bag in root_bags:
        for multi, entities in root_bag.indis_entities.items():
            indis_pred, context = encode_set_entities(entities, context)
            for neg_bags, pos_bags in all_configs(root_bag, True):
                pred = create_aux_pred(1, f'{root_bag.name}_indis_{multi}')
                pos_formula = reduce(
                    lambda x, y: x & y,
                    [context.get_pred(bag)(X) for bag in pos_bags],
                    top
                )
                neg_formula = reduce(
                    lambda x, y: x & y,
                    [~context.get_pred(bag)(X) for bag in neg_bags],
                    top
                )
                context.sentence = context.sentence & QuantifiedFormula(
                    Universal(X),
                    pred(X).equivalent(pos_formula & neg_formula & indis_pred(X))
                )
                context.weighting[pred] = (get_symbolic_weight(root_bag, context, pos_bags, multi, multi), 1)
    return context


def encode_singleton(context: Context) -> Context:
    if len(context.singletons) == 0:
        return context
    _, context = encode_set_entities(context.singletons, context)
    return context


def encode(problem: CofolaProblem,
           lifted: bool = True,
           old_pred_encoding: bool = False) -> tuple[WFOMCProblem, Decoder]:
    """
    Encode the combinatorics problem to the WFOMC problem

    :param problem: the combinatorics problem
    :param lifted: whether to lift the entities in bags
    :return: the WFOMC problem and the decoder
    """
    objs = problem.objects
    constraints = problem.constraints

    # setup context
    context = Context(problem, old_pred_encoding=old_pred_encoding)
    # initialize the cache for multinomial coefficients
    MultinomialCoefficients.setup(len(context.domain))

    logger.info(f'The singletons: {context.singletons}')
    # preprocess
    logger.info('Preprocessing bags...')
    root_bags = preprocess_bags(problem, lifted)

    for obj in objs:
        if isinstance(obj, Set):
            logger.info(f"potential entities of {obj.name}: {obj.p_entities}")
        elif isinstance(obj, Bag):
            logger.info(f"potential entities of {obj.name}: {obj.p_entities_multiplicity}")
            logger.info(f"distinguishable entities of {obj.name}: {obj.dis_entities}")
            logger.info(f"indistinguishable entities of {obj.name}: {obj.indis_entities}")

    # encode singletons
    logger.info('Encoding singletons...')
    context = encode_singleton(context)
    # encode objects
    for obj in objs:
        logger.info(f'Encoding object {obj.name}[{type(obj).__name__}]...')
        # NOTE: for bags, here only encode their distinguishable entities
        context = obj.encode(context)

    # if lifted, encode bags
    if lifted:
        logger.info('Lifting entities in bags...')
        context = lifted_encode_bags(context, root_bags)

    # encode constraints
    for constraint in constraints:
        logger.info(
            f'Encoding constraint {constraint}:{type(constraint).__name__}...'
        )
        context = constraint.encode(context)

    return context.build()
