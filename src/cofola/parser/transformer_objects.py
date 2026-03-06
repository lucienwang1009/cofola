"""Object transformer mixin for Cofola parser - IR-native version."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from cofola.frontend.types import Entity, ObjRef
from cofola.frontend.objects import (
    SetInit, SetChoose, SetChooseReplace,
    SetUnion, SetIntersection, SetDifference,
    BagInit, BagChoose, BagUnion, BagAdditiveUnion,
    BagIntersection, BagDifference, BagSupport,
    FuncImage,
    TupleDef, SequenceDef, PartitionDef, PartRef,
)

if TYPE_CHECKING:
    from cofola.parser.transformer import CofolaTransfomer


class ObjectTransformerMixin:
    """Mixin providing object transformation methods for CofolaTransformer (IR-native)."""

    def base_object_init(self: "CofolaTransfomer", args):
        obj_type, _, obj_init, _ = args
        if str(obj_type) == "set":
            if isinstance(obj_init, ObjRef):
                # set(S) — initialize from existing set reference; treat as alias
                return obj_init
            entities = []
            for atom in obj_init:
                self._check_obj_type(atom, Entity)
                entities.append(atom)
            if len(entities) != len(set(entities)):
                from cofola.parser.parser import CofolaParsingError
                raise CofolaParsingError("Duplicate entities are not allowed in a set object.")
            defn = SetInit(entities=frozenset(entities))
        elif str(obj_type) == "bag":
            if isinstance(obj_init, ObjRef):
                return obj_init
            entity_multiplicity: dict[Entity, int] = defaultdict(lambda: 0)
            for atom in obj_init:
                if isinstance(atom, tuple):
                    entity, multiplicity = atom
                    entity_multiplicity[entity] = multiplicity
                else:
                    entity_multiplicity[atom] += 1
            em = tuple(sorted(entity_multiplicity.items(), key=lambda x: x[0].name))
            defn = BagInit(entity_multiplicity=em)
        else:
            from cofola.parser.parser import CofolaParsingError
            raise CofolaParsingError(f"Unknown object type: {obj_type}")
        return self.builder.add(defn)

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
        entities = [Entity(f'{name}{i}') for i in range(start, end)]
        for e in entities:
            self.entities.add(e.name)
        return entities

    def duplicate_entities(self: "CofolaTransfomer", args):
        entity, count = args
        return entity, int(count)

    def entity(self: "CofolaTransfomer", args):
        from cofola.parser.parser import CofolaParsingError
        item = args[0]
        entity_name = str(item)
        if entity_name in self.id2ref:
            raise CofolaParsingError(
                f"Entity {entity_name} has already been used as an object name."
            )
        e = Entity(entity_name)
        self.entities.add(entity_name)
        return e

    def identity(self: "CofolaTransfomer", args):
        obj_id = str(args[0].value)
        if self._processing_part_name is not None and obj_id == self._processing_part_name:
            part_refs = [
                r for r, defn in self.builder._defs
                if isinstance(defn, PartRef) and defn.partition == self._processing_partition_ref
            ]
            part_refs.sort(key=lambda r: self.builder.get_object(r).index)
            return part_refs
        if obj_id in self.entities:
            return Entity(obj_id)
        return self._get_ref_by_id(obj_id)

    def operations(self: "CofolaTransfomer", args):
        # Child already called builder.add() and returned an ObjRef (or list for partition expansion)
        return args[-1]

    def common_operations(self: "CofolaTransfomer", args):
        from cofola.parser.parser import CofolaParsingError

        op_type = str(args[0])
        obj = args[2]

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

        obj_cat = self._ref_category(obj) if isinstance(obj, ObjRef) else 'unknown'

        if op_type == "choose":
            if obj_cat == 'set':
                return self.builder.add(SetChoose(source=obj, size=size))
            elif obj_cat == 'bag':
                return self.builder.add(BagChoose(source=obj, size=size))
            else:
                raise CofolaParsingError(f"choose() requires Set or Bag, got {obj_cat}")

        elif op_type == 'choose_replace':
            if obj_cat != 'set':
                raise CofolaParsingError(f"choose_replace() requires Set, got {obj_cat}")
            return self.builder.add(SetChooseReplace(source=obj, size=size))

        elif op_type == 'supp':
            if obj_cat != 'bag':
                raise CofolaParsingError(f"supp() requires Bag, got {obj_cat}")
            return self.builder.add(BagSupport(source=obj))

        elif op_type in ('compose', 'partition'):
            if size is None:
                raise CofolaParsingError(f"The size of a {op_type} must be specified.")
            ordered = (op_type == 'compose')
            partition_ref = self.builder.add(
                PartitionDef(source=obj, num_parts=size, ordered=ordered)
            )
            for i in range(size):
                self.builder.add(PartRef(partition=partition_ref, index=i))
            return partition_ref

        elif op_type == 'tuple':
            return self.builder.add(TupleDef(source=obj, choose=False, replace=False, size=size))

        elif op_type == 'choose_tuple':
            return self.builder.add(TupleDef(source=obj, choose=True, replace=False, size=size))

        elif op_type == 'choose_replace_tuple':
            if obj_cat != 'set':
                raise CofolaParsingError(
                    f"choose_replace_tuple() requires Set, got {obj_cat}"
                )
            return self.builder.add(TupleDef(source=obj, choose=True, replace=True, size=size))

        elif op_type == 'sequence':
            return self.builder.add(
                SequenceDef(source=obj, choose=False, replace=False, size=size)
            )

        elif op_type == 'choose_sequence':
            return self.builder.add(
                SequenceDef(source=obj, choose=True, replace=False, size=size)
            )

        elif op_type == 'choose_replace_sequence':
            if obj_cat != 'set':
                raise CofolaParsingError(
                    f"choose_replace_sequence() requires Set, got {obj_cat}"
                )
            return self.builder.add(
                SequenceDef(source=obj, choose=True, replace=True, size=size)
            )

        elif op_type == 'circle':
            return self.builder.add(
                SequenceDef(
                    source=obj, choose=False, replace=False, size=size,
                    circular=True, reflection=op_arg
                )
            )

        elif op_type == 'choose_circle':
            return self.builder.add(
                SequenceDef(
                    source=obj, choose=True, replace=False, size=size,
                    circular=True, reflection=op_arg
                )
            )

        elif op_type == 'choose_replace_circle':
            if obj_cat != 'set':
                raise CofolaParsingError(
                    f"choose_replace_circle() requires Set, got {obj_cat}"
                )
            return self.builder.add(
                SequenceDef(
                    source=obj, choose=True, replace=True, size=size,
                    circular=True, reflection=op_arg
                )
            )

        else:
            raise CofolaParsingError(f"Unknown operation type: {op_type}")

    def binary_operations(self: "CofolaTransfomer", args):
        from cofola.parser.parser import CofolaParsingError

        objs1, op, objs2 = args
        op = str(op)

        def single_operation(obj1, obj2):
            cat1 = self._ref_category(obj1)
            cat2 = self._ref_category(obj2)
            if (cat1 == 'set' and cat2 == 'bag') or (cat1 == 'bag' and cat2 == 'set'):
                raise CofolaParsingError("Set and Bag objects cannot be operated together")
            if cat1 == 'set' or cat2 == 'set':
                if op == "+":
                    return self.builder.add(SetUnion(left=obj1, right=obj2))
                elif op == "++":
                    raise CofolaParsingError(
                        "Additive union is not supported for set objects."
                    )
                elif op == "&":
                    return self.builder.add(SetIntersection(left=obj1, right=obj2))
                elif op == "-":
                    return self.builder.add(SetDifference(left=obj1, right=obj2))
                else:
                    raise CofolaParsingError(f"Unknown set operation {op}.")
            else:  # bag
                if op == "+":
                    return self.builder.add(BagAdditiveUnion(left=obj1, right=obj2))
                elif op == "++":
                    return self.builder.add(BagUnion(left=obj1, right=obj2))
                elif op == "&":
                    return self.builder.add(BagIntersection(left=obj1, right=obj2))
                elif op == "-":
                    return self.builder.add(BagDifference(left=obj1, right=obj2))
                else:
                    raise CofolaParsingError(f"Unknown bag operation {op}.")

        return self._op_or_constraint_on_list(single_operation, objs1, objs2)

    def indexing(self: "CofolaTransfomer", args):
        from cofola.parser.transformer import TupleIndexSentinel
        from cofola.parser.parser import CofolaParsingError

        obj, index = args[0], args[2]
        cat = self._ref_category(obj) if isinstance(obj, ObjRef) else 'unknown'

        if cat == 'partition':
            idx = int(index)
            for r, defn in self.builder._defs:
                if isinstance(defn, PartRef) and defn.partition == obj and defn.index == idx:
                    return r
            raise CofolaParsingError(f"Partition part index {idx} not found.")

        elif cat == 'tuple':
            return TupleIndexSentinel(tuple_ref=obj, index=int(index))

        else:
            raise CofolaParsingError(f"Indexing not supported for {cat} objects.")

    def count(self: "CofolaTransfomer", args):
        from cofola.parser.parser import CofolaParsingError
        from cofola.frontend.constraints import (
            TupleCountAtom, BagCountAtom, SeqPatternCountAtom,
            TogetherPattern, LessThanPattern, PredecessorPattern, NextToPattern,
        )

        objs, count_name, _, entity_or_obj, _ = args
        deduplicate = str(count_name) == "dedup_count"
        _SEQ_PATTERNS = (TogetherPattern, LessThanPattern, PredecessorPattern, NextToPattern)

        def _effective_cat(obj: ObjRef) -> str:
            """Get effective category, resolving PartRef to its source type."""
            cat = self._ref_category(obj)
            if cat == 'part':
                # PartRef — look at the partition's source to determine bag vs set
                defn = self.builder.get_object(obj)  # PartRef
                partition_defn = self.builder.get_object(defn.partition)  # PartitionDef
                return self._ref_category(partition_defn.source)
            return cat

        def single_operation(obj):
            cat = _effective_cat(obj) if isinstance(obj, ObjRef) else 'unknown'
            if deduplicate and cat != 'tuple':
                raise CofolaParsingError(
                    f"dedup_count() requires a Tuple, got {cat}"
                )
            if cat == 'bag' and isinstance(entity_or_obj, Entity):
                return BagCountAtom(bag=obj, entity=entity_or_obj)
            elif cat == 'tuple' and isinstance(entity_or_obj, (Entity, ObjRef)):
                return TupleCountAtom(
                    tuple_ref=obj, count_obj=entity_or_obj, deduplicate=deduplicate
                )
            elif cat == 'sequence' and isinstance(entity_or_obj, _SEQ_PATTERNS):
                return SeqPatternCountAtom(seq=obj, pattern=entity_or_obj)
            else:
                raise CofolaParsingError(
                    f"count() not supported for {cat} with {type(entity_or_obj).__name__}"
                )

        return self._op_or_constraint_on_list(single_operation, objs)
