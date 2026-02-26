"""Object transformer mixin for Cofola parser."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, List

from cofola.objects.bag import Bag, BagChoose, BagInit, BagSupport, BagMultiplicity
from cofola.objects.base import CombinatoricsObject, Entity, Partition, Sequence, Set, Tuple
from cofola.objects.function import FuncImage, FuncInit, FuncInverseImage, Function
from cofola.objects.partition import BagPartition, SetPartition
from cofola.objects.sequence import SequenceImpl, SequenceSizedPattern, SequencePatternCount
from cofola.objects.set import SetChoose, SetChooseReplace, SetInit
from cofola.objects.tuple import TupleImpl, TupleIndex, TupleCount

if TYPE_CHECKING:
    from cofola.parser.transformer import CofolaTransfomer


class ObjectTransformerMixin:
    """Mixin providing object transformation methods for CofolaTransformer."""

    def base_object_init(self: "CofolaTransfomer", args):
        obj_type, _, obj_init, _ = args
        if obj_type == "set":
            entities = []
            for atom in obj_init:
                self._check_obj_type(atom, Entity)
                entities.append(atom)
            # check if entities are duplicated
            if len(entities) != len(set(entities)):
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError(f"Duplicate entities are not allowed in a set object.")
            obj = SetInit(entities)
        if obj_type == "bag":
            entity_multiplicity = defaultdict(lambda: 0)
            for atom in obj_init:
                if isinstance(atom, tuple):
                    entity, multiplicity = atom
                    entity_multiplicity[entity] = multiplicity
                else:
                    entity_multiplicity[atom] += 1
            obj = BagInit(entity_multiplicity)
        obj = self.problem.add_object(obj)
        return obj

    def entities_body(self: "CofolaTransfomer", args):
        entities = []
        for atom in args:
            if isinstance(atom, Entity):
                entities.append(atom)
            elif isinstance(atom, list):
                entities.extend(atom)
            else:
                entities.append(atom)
        return entities

    def slicing_entities(self: "CofolaTransfomer", args):
        name = ""
        start = 0
        end = 0
        if len(args) == 2:
            start = args[0]
            end = args[1]
        elif len(args) == 3:
            name = args[0].value
            start = args[1]
            end = args[2]
        entities = list(
            Entity(f'{name}{i}') for i in range(start, end)
        )
        return entities

    def duplicate_entities(self: "CofolaTransfomer", args):
        entity, count = args
        return entity, int(count)

    def entity(self: "CofolaTransfomer", args):
        from cofola.parser.parser import CofolaParsingError

        item = args[0]
        entity_name = str(item)
        if entity_name in self.id2obj:
            raise CofolaParsingError(f"Entity {entity_name} has already used as an object name.")
        ret = Entity(entity_name)
        return ret

    def func_init(self: "CofolaTransfomer", args):
        domain, func_type, codomain = args
        injection = ((func_type == "|->") or (func_type == "|->|"))
        surjection = ((func_type == "->|") or (func_type == "|->|"))
        obj = FuncInit(domain, codomain, injection, surjection)
        return self.problem.add_object(obj)

    def identity(self: "CofolaTransfomer", args):
        obj_id = str(args[0].value)
        # if the object is a part of a partition,
        # return the partitioned objects
        if self._processing_part_name is not None and \
                obj_id == self._processing_part_name:
            ret_objs = []
            for part in self._processing_partition.partitioned_objs:
                ret_objs.append(self.problem.add_object(part))
            return ret_objs
        if self.problem.contains_entity(Entity(obj_id)):
            return Entity(obj_id)
        return self._get_obj_by_id(obj_id)

    def operations(self: "CofolaTransfomer", args):
        objs = args[-1]
        ret_objs = []
        if isinstance(objs, list):
            for obj in objs:
                ret_objs.append(self.problem.add_object(obj))
            return ret_objs
        return self.problem.add_object(objs)

    def common_operations(self: "CofolaTransfomer", args):
        from cofola.parser.parser import CofolaParsingError

        op_type = args[0]
        obj = args[2]
        self._check_obj_type(obj, Set, Bag)
        if len(args) == 4:
            size = None
            op_arg = False
        elif len(args) == 5:
            if isinstance(args[3], bool):
                size = None
                op_arg = args[3]
            else:
                size = args[3]
                op_arg = False
        else:
            size = args[3]
            op_arg = args[4]

        if op_type == "choose":
            if isinstance(obj, Set):
                return SetChoose(obj, size)
            else:
                return BagChoose(obj, size)
        if op_type == 'choose_replace':
            # operation `choose_replace` only supports Set object
            self._check_obj_type(obj, Set)
            return SetChooseReplace(obj, size)
        if op_type == 'supp':
            return BagSupport(obj)
        if op_type == 'compose':
            if size is None:
                raise CofolaParsingError(f"The size of a composition must be specified.")
            if isinstance(obj, Set):
                return SetPartition(obj, size, True)
            else:
                return BagPartition(obj, size, True)
        if op_type == 'partition':
            if size is None:
                raise CofolaParsingError(f"The size of a partition must be specified.")
            if isinstance(obj, Set):
                return SetPartition(obj, size, False)
            else:
                return BagPartition(obj, size, False)
        if op_type == 'tuple':
            return TupleImpl(
                obj, size=size, choose=False, replace=False
            )
        if op_type == 'choose_tuple':
            return TupleImpl(
                obj, size=size, choose=True, replace=False
            )
        if op_type == 'choose_replace_tuple':
            self._check_obj_type(obj, Set)
            return TupleImpl(
                obj, size=size, choose=True, replace=True
            )
        if op_type == 'sequence':
            self._check_obj_type(obj, Set, Bag)
            return SequenceImpl(
                obj, size=size, choose=False, replace=False
            )
        if op_type == 'choose_sequence':
            self._check_obj_type(obj, Set, Bag)
            return SequenceImpl(
                obj, size=size, choose=True, replace=False
            )
        if op_type == 'choose_replace_sequence':
            self._check_obj_type(obj, Set)
            return SequenceImpl(
                obj, size=size, choose=True, replace=True
            )
        if op_type == 'circle':
            self._check_obj_type(obj, Set, Bag)
            return SequenceImpl(
                obj, size=size, choose=False, replace=False,
                circular=True, reflection=op_arg
            )
        if op_type == 'choose_circle':
            self._check_obj_type(obj, Set, Bag)
            return SequenceImpl(
                obj, size=size, choose=True, replace=False,
                circular=True, reflection=op_arg
            )
        if op_type == 'choose_replace_circle':
            self._check_obj_type(obj, Set)
            return SequenceImpl(
                obj, size=size, choose=True, replace=True,
                circular=True, reflection=op_arg
            )

    def binary_operations(self: "CofolaTransfomer", args):
        from cofola.parser.parser import CofolaParsingError
        from cofola.objects.bag import BagAdditiveUnion, BagIntersection, BagDifference
        from cofola.objects.set import SetUnion, SetIntersection, SetDifference

        objs1, op, objs2 = args
        def single_operation(obj1, obj2):
            self._check_obj_type(obj1, Set, Bag)
            self._check_obj_type(obj2, Set, Bag)
            if (isinstance(obj1, Set) and isinstance(obj2, Bag)) or \
                    (isinstance(obj1, Bag) and isinstance(obj2, Set)):
                raise CofolaParsingError(f"Set and Bag objects cannot be operated together")
            if isinstance(obj1, Set):
                if op == "+":
                    return SetUnion(obj1, obj2)
                elif op == "++":
                    raise CofolaParsingError(f"Additive union operation is not supported for set objects.")
                elif op == "&":
                    return SetIntersection(obj1, obj2)
                elif op == "-":
                    return SetDifference(obj1, obj2)
                else:
                    raise CofolaParsingError(f"Unknown set operation {op}.")
            else:
                if op == "+":
                    return BagAdditiveUnion(obj1, obj2)
                # elif op == "++":
                #     return BagAdditiveUnion(obj1, obj2)
                elif op == "&":
                    return BagIntersection(obj1, obj2)
                elif op == "-":
                    return BagDifference(obj1, obj2)
                else:
                    raise CofolaParsingError(f"Unknown bag operation {op}.")
        return self._op_or_constraint_on_list(
            single_operation, objs1, objs2)

    def indexing(self: "CofolaTransfomer", args):
        obj, index = args[0], args[2]
        self._check_obj_type(obj, Partition, Tuple)
        if isinstance(obj, Partition):
            if index == '*':
                return obj.partitioned_objs
            # if not obj.ordered:
            #     raise CofolaParsingError("Indexing operation is only supported for ordered partitions.")
            index_obj = obj.partitioned_objs[int(index)]
        else:
            index_obj = TupleIndex(obj, int(index))
        return self.problem.add_object(index_obj)

    def inverse_object(self: "CofolaTransfomer", args):
        obj = args[0]
        return obj

    def image(self: "CofolaTransfomer", args):
        obj, set_or_entity = args[0], args[2]
        self._check_obj_type(obj, Function)
        return FuncImage(obj, set_or_entity)

    def inverse_image(self: "CofolaTransfomer", args):
        obj, set_or_entity = args[0], args[2]
        self._check_obj_type(obj, Function)
        return FuncInverseImage(obj, set_or_entity)

    def count(self: "CofolaTransfomer", args):
        from cofola.parser.parser import CofolaParsingError

        objs, count_name, _, entity_or_obj, _ = args
        deduplicate = count_name == "dedup_count"
        def single_operation(obj):
            if deduplicate:
                self._check_obj_type(obj, Tuple)
            if isinstance(obj, Bag) and isinstance(entity_or_obj, Entity):
                count_obj = BagMultiplicity(obj, entity_or_obj)
            elif isinstance(obj, Tuple) and (
                isinstance(entity_or_obj, Entity) or isinstance(entity_or_obj, Set)
            ):
                count_obj = TupleCount(obj, entity_or_obj, deduplicate)
            elif isinstance(obj, Sequence) and isinstance(entity_or_obj, SequenceSizedPattern):
                if not entity_or_obj._has_size(obj):
                    raise CofolaParsingError(
                        f"Count constraint must be applied to a sequence pattern with a size."
                    )
                count_obj = SequencePatternCount(obj, entity_or_obj)
            else:
                raise CofolaParsingError(
                    f"Count constraint is not supported for the given objects: {obj}, {entity_or_obj}"
                )
            return self.problem.add_object(count_obj)
        return self._op_or_constraint_on_list(
            single_operation, objs)