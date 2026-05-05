"""Object transformer mixin for Cofola parser - IR-native version."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from cofola.frontend.constraints import (
    BagCountAtom,
    LessThanPattern,
    NextToPattern,
    PredecessorPattern,
    SeqPatternCountAtom,
    TogetherPattern,
    TupleCountAtom,
)
from cofola.frontend.objects import (
    BagAdditiveUnion,
    BagChoose,
    BagDifference,
    BagInit,
    BagIntersection,
    BagObjDef,
    BagPartRef,
    BagSupport,
    BagUnion,
    PartitionDef,
    PartRef,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetDifference,
    SetInit,
    SetIntersection,
    SetObjDef,
    SetPartRef,
    SetUnion,
    TupleDef,
)
from cofola.frontend.types import Entity, ObjRef
from cofola.parser.constants import TupleIndexSentinel
from cofola.ir.analysis.type_check import CofolaTypeError, TypeCheckError
from cofola.parser.common import CofolaParsingError

if TYPE_CHECKING:
    from cofola.parser.transformer import CofolaTransfomer


class ObjectTransformerMixin:
    """Mixin providing object transformation methods for CofolaTransformer (IR-native)."""

    def base_object_init(self: "CofolaTransfomer", args):
        obj_type, _, obj_init, _ = args
        obj_type = str(obj_type)
        if obj_type == "set":
            if isinstance(obj_init, ObjRef):
                # set(S) — initialize from existing set reference; treat as alias
                return obj_init
            entities: list[Entity] = []
            for atom in obj_init:
                if not isinstance(atom, Entity):
                    raise CofolaTypeError([TypeCheckError(
                        loc=None,
                        message=(
                            f"set(...) entries must be entities, "
                            f"got {atom!r} of type {type(atom).__name__}"
                        ),
                        node=atom,
                    )])
                entities.append(atom)
            if len(entities) != len(set(entities)):
                raise CofolaParsingError(
                    "Duplicate entities are not allowed in a set object."
                )
            defn = SetInit(entities=frozenset(entities))
        elif obj_type == "bag":
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
            raise CofolaParsingError(f"Unknown object type: {obj_type}")
        return self.builder.add(defn)

    def entities_body(self: "CofolaTransfomer", args):
        entities = []
        for atom in args:
            if isinstance(atom, list):
                entities.extend(atom)
            else:
                entities.append(atom)
        return entities

    def slicing_entities(self: "CofolaTransfomer", args):
        name = ""
        if len(args) == 2:
            start, end = args
        elif len(args) == 3:
            name = args[0].value
            start, end = args[1], args[2]
        else:
            raise CofolaParsingError(
                f"Unexpected slicing_entities args: {args}"
            )
        entities = [Entity(f"{name}{i}") for i in range(start, end)]
        for e in entities:
            self.entities.add(e.name)
        return entities

    def duplicate_entities(self: "CofolaTransfomer", args):
        entity, count = args
        return entity, int(count)

    def entity(self: "CofolaTransfomer", args):
        item = args[0]
        entity_name = str(item)
        if entity_name in self.id2ref:
            raise CofolaParsingError(
                f"Entity {entity_name} has already been used as an object name."
            )
        e = Entity(entity_name)
        self.entities.add(entity_name)
        return e

    def operations(self: "CofolaTransfomer", args):
        # Child already called builder.add() and returned an ObjRef.
        return args[-1]

    def common_operations(self: "CofolaTransfomer", args):
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

        # Dispatch on source kind. For choose/compose/partition we still need
        # to pick the right IR class (Set* vs Bag*); when the source is the
        # wrong kind we default to the Set branch and let TypeCheckPass raise
        # the proper post-parse error.
        defn = self._defn_of(obj)
        is_set = isinstance(defn, SetObjDef)
        is_bag = isinstance(defn, BagObjDef)

        if op_type == "choose":
            # Default to SetChoose for non-bag sources; TypeCheckPass enforces
            # source kind.
            if is_bag:
                return self.builder.add(BagChoose(source=obj, size=size))
            return self.builder.add(SetChoose(source=obj, size=size))

        if op_type == "choose_replace":
            # Always create SetChooseReplace; TypeCheckPass validates source is SET.
            return self.builder.add(SetChooseReplace(source=obj, size=size))

        if op_type == "supp":
            # Always create BagSupport; TypeCheckPass validates source is BAG.
            return self.builder.add(BagSupport(source=obj))

        if op_type in ("compose", "partition"):
            if size is None:
                raise CofolaParsingError(f"The size of a {op_type} must be specified.")
            ordered = op_type == "compose"
            # Default to SetPartRef for non-bag sources; TypeCheckPass enforces
            # source kind via PartitionDef signature.
            part_cls: type = BagPartRef if is_bag else SetPartRef
            partition_ref = self.builder.add(
                PartitionDef(source=obj, num_parts=size, ordered=ordered)
            )
            for i in range(size):
                self.builder.add(part_cls(partition=partition_ref, index=i))
            return partition_ref

        if op_type == "tuple":
            return self.builder.add(
                TupleDef(source=obj, choose=False, replace=False, size=size)
            )

        if op_type == "choose_tuple":
            return self.builder.add(
                TupleDef(source=obj, choose=True, replace=False, size=size)
            )

        if op_type == "choose_replace_tuple":
            return self.builder.add(
                TupleDef(source=obj, choose=True, replace=True, size=size)
            )

        if op_type == "sequence":
            return self.builder.add(
                SequenceDef(source=obj, choose=False, replace=False, size=size)
            )

        if op_type == "choose_sequence":
            return self.builder.add(
                SequenceDef(source=obj, choose=True, replace=False, size=size)
            )

        if op_type == "choose_replace_sequence":
            return self.builder.add(
                SequenceDef(source=obj, choose=True, replace=True, size=size)
            )

        if op_type == "circle":
            return self.builder.add(
                SequenceDef(
                    source=obj, choose=False, replace=False, size=size,
                    circular=True, reflection=op_arg,
                )
            )

        if op_type == "choose_circle":
            return self.builder.add(
                SequenceDef(
                    source=obj, choose=True, replace=False, size=size,
                    circular=True, reflection=op_arg,
                )
            )

        if op_type == "choose_replace_circle":
            return self.builder.add(
                SequenceDef(
                    source=obj, choose=True, replace=True, size=size,
                    circular=True, reflection=op_arg,
                )
            )

        raise CofolaParsingError(f"Unknown operation type: {op_type}")

    def binary_operations(self: "CofolaTransfomer", args):
        obj1, op, obj2 = args
        op = str(op)
        # Dispatch on operand kinds to pick Set* vs Bag* IR class. Mixed-kind
        # operands and ill-typed `++`/`+` usage are rejected by TypeCheckPass.
        d1 = self._defn_of(obj1)
        d2 = self._defn_of(obj2)
        is_bag1, is_bag2 = isinstance(d1, BagObjDef), isinstance(d2, BagObjDef)
        # Use bag branch only when both operands are bags; otherwise default
        # to the set branch (mixed-kind cases will be flagged by TypeCheckPass
        # via the SetUnion/SetIntersection/SetDifference signatures).
        if is_bag1 and is_bag2:
            if op == "+":
                return self.builder.add(BagUnion(left=obj1, right=obj2))
            if op == "++":
                return self.builder.add(BagAdditiveUnion(left=obj1, right=obj2))
            if op == "&":
                return self.builder.add(BagIntersection(left=obj1, right=obj2))
            if op == "-":
                return self.builder.add(BagDifference(left=obj1, right=obj2))
            raise CofolaParsingError(f"Unknown bag operation {op}.")
        # set (or mixed — TypeCheckPass will reject mixed)
        if op == "+":
            return self.builder.add(SetUnion(left=obj1, right=obj2))
        if op == "++":
            # `++` on non-bag operands: still build a BagAdditiveUnion so that
            # TypeCheckPass can report the proper "requires Bag" error.
            return self.builder.add(BagAdditiveUnion(left=obj1, right=obj2))
        if op == "&":
            return self.builder.add(SetIntersection(left=obj1, right=obj2))
        if op == "-":
            return self.builder.add(SetDifference(left=obj1, right=obj2))
        raise CofolaParsingError(f"Unknown set operation {op}.")

    def indexing(self: "CofolaTransfomer", args):
        obj, index = args[0], args[2]
        defn = self._defn_of(obj)

        if isinstance(defn, PartitionDef):
            idx = int(index)
            for r, d in self.builder.iter_defs():
                if isinstance(d, PartRef) and d.partition == obj and d.index == idx:
                    return r
            raise CofolaParsingError(f"Partition part index {idx} not found.")

        if isinstance(defn, TupleDef):
            # If the tuple's source is a PartRef, return that PartRef directly.
            # This allows `t[k]` to be used as the source partition part in
            # constraints like `|t[0]| == 2` or `t[0] == e1`.
            src_defn = self.builder.get_object(defn.source)
            if isinstance(src_defn, PartRef):
                return defn.source
            return TupleIndexSentinel(tuple_ref=obj, index=int(index))

        kind = type(defn).__name__ if defn is not None else "unknown"
        raise CofolaParsingError(f"Indexing not supported for {kind} objects.")

    def count(self: "CofolaTransfomer", args):
        obj, count_name, _, entity_or_obj, _ = args
        deduplicate = str(count_name) == "dedup_count"
        _SEQ_PATTERNS = (TogetherPattern, LessThanPattern, PredecessorPattern, NextToPattern)

        defn = self._defn_of(obj)
        kind = type(defn).__name__ if defn is not None else "unknown"
        if deduplicate and not isinstance(defn, TupleDef):
            raise CofolaParsingError(f"dedup_count() requires a Tuple, got {kind}")
        if isinstance(defn, BagObjDef) and isinstance(entity_or_obj, Entity):
            return BagCountAtom(bag=obj, entity=entity_or_obj)
        if isinstance(defn, TupleDef) and isinstance(entity_or_obj, (Entity, ObjRef)):
            return TupleCountAtom(
                tuple_ref=obj, count_obj=entity_or_obj, deduplicate=deduplicate
            )
        if isinstance(defn, SequenceDef) and isinstance(entity_or_obj, _SEQ_PATTERNS):
            return SeqPatternCountAtom(seq=obj, pattern=entity_or_obj)
        raise CofolaParsingError(
            f"count() not supported for {kind} with {type(entity_or_obj).__name__}"
        )
