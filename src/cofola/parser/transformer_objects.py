"""Object transformer mixin for Cofola parser."""
from __future__ import annotations

from collections import defaultdict

from cofola.frontend.constraints import (
    BagCountAtom,
    LessThanPattern,
    NextToPattern,
    PredecessorPattern,
    SeqPatternCountAtom,
    SizeAtom,
    TogetherPattern,
    TupleCountAtom,
)
from cofola.frontend.type_check import CofolaTypeError, TypeCheckError
from cofola.frontend.objects import (
    BagAdditiveUnion,
    BagChoose,
    BagDifference,
    BagInit,
    BagIntersection,
    BagObjDef,
    BagPartDef,
    BagSupport,
    BagUnion,
    CircleDef,
    CompositionDef,
    Grouped,
    Linear,
    ObjDef,
    PartitionDef,
    PartDef,
    SequenceDef,
    SetChoose,
    SetChooseReplace,
    SetDifference,
    SetInit,
    SetIntersection,
    SetPartDef,
    SetUnion,
    TupleDef,
)
from cofola.frontend.objects import Entity, ObjRef
from cofola.parser.utils import CofolaParsingError, TupleIndexSentinel

_SEQ_PATTERNS = (
    TogetherPattern,
    LessThanPattern,
    PredecessorPattern,
    NextToPattern,
)

_ORDERED_COLLECTION_OPS = {
    "tuple": (TupleDef, False, False),
    "choose_tuple": (TupleDef, True, False),
    "choose_replace_tuple": (TupleDef, True, True),
    "sequence": (SequenceDef, False, False),
    "choose_sequence": (SequenceDef, True, False),
    "choose_replace_sequence": (SequenceDef, True, True),
    "circle": (CircleDef, False, False),
    "choose_circle": (CircleDef, True, False),
    "choose_replace_circle": (CircleDef, True, True),
}

_BAG_BINARY_OPS = {
    "+": BagUnion,
    "++": BagAdditiveUnion,
    "&": BagIntersection,
    "-": BagDifference,
}

_SET_BINARY_OPS = {
    "+": SetUnion,
    "++": BagAdditiveUnion,
    "&": SetIntersection,
    "-": SetDifference,
}


def _resolve_collection_operation(
    *,
    builder,
    op_type: str,
    source: ObjRef,
    source_defn: ObjDef | None,
    size: int | None,
    op_arg: bool,
) -> ObjRef:
    is_bag = isinstance(source_defn, BagObjDef)

    if op_type == "choose":
        if is_bag:
            return builder.add(BagChoose(source=source, size=size))
        return builder.add(SetChoose(source=source, size=size))
    if op_type == "choose_replace":
        return builder.add(SetChooseReplace(source=source, size=size))
    if op_type == "supp":
        return builder.add(BagSupport(source=source))

    if op_type in ("compose", "partition"):
        if size is None:
            raise CofolaParsingError(f"The size of a {op_type} must be specified.")
        partition_cls = CompositionDef if op_type == "compose" else PartitionDef
        part_cls = BagPartDef if is_bag else SetPartDef
        partition_ref = builder.add(partition_cls(source=source, num_parts=size))
        for i in range(size):
            builder.add(part_cls(partition=partition_ref, index=i))
        return partition_ref

    op_spec = _ORDERED_COLLECTION_OPS.get(op_type)
    if op_spec is not None:
        cls, choose, replace = op_spec
        if cls is CircleDef:
            return builder.add(cls(
                source=source,
                choose=choose,
                replace=replace,
                size=size,
                reflection=op_arg,
            ))
        return builder.add(cls(
            source=source,
            choose=choose,
            replace=replace,
            size=size,
        ))

    raise CofolaParsingError(f"Unknown operation type: {op_type}")


def _resolve_binary_operation(
    *,
    builder,
    left: ObjRef,
    op: str,
    right: ObjRef,
    left_defn: ObjDef | None,
    right_defn: ObjDef | None,
) -> ObjRef:
    op_cls = _BAG_BINARY_OPS.get(op) if (
        isinstance(left_defn, BagObjDef) and isinstance(right_defn, BagObjDef)
    ) else _SET_BINARY_OPS.get(op)
    if op_cls is not None:
        return builder.add(op_cls(left=left, right=right))

    if isinstance(left_defn, BagObjDef) and isinstance(right_defn, BagObjDef):
        raise CofolaParsingError(f"Unknown bag operation {op}.")
    raise CofolaParsingError(f"Unknown set operation {op}.")


def _resolve_count_atom(
    *,
    target: ObjRef,
    count_arg: object,
    target_defn: ObjDef | None,
    deduplicate: bool,
) -> SizeAtom:
    kind = type(target_defn).__name__ if target_defn is not None else "unknown"
    if deduplicate and not isinstance(target_defn, TupleDef):
        raise CofolaParsingError(f"dedup_count() requires a Tuple, got {kind}")
    if isinstance(target_defn, BagObjDef) and isinstance(count_arg, Entity):
        return BagCountAtom(bag=target, entity=count_arg)
    if isinstance(target_defn, TupleDef) and isinstance(count_arg, (Entity, ObjRef)):
        return TupleCountAtom(
            tuple_ref=target, count_obj=count_arg, deduplicate=deduplicate
        )
    if isinstance(target_defn, Linear) and isinstance(count_arg, _SEQ_PATTERNS):
        return SeqPatternCountAtom(seq=target, pattern=count_arg)
    raise CofolaParsingError(
        f"count() not supported for {kind} with {type(count_arg).__name__}"
    )


class ObjectTransformerMixin:
    """Mixin providing object transformation methods for CofolaTransformer."""

    def base_object_init(self, args):
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

    def entities_body(self, args):
        entities = []
        for atom in args:
            if isinstance(atom, list):
                entities.extend(atom)
            else:
                entities.append(atom)
        return entities

    def slicing_entities(self, args):
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

    def duplicate_entities(self, args):
        entity, count = args
        return entity, int(count)

    def entity(self, args):
        item = args[0]
        entity_name = str(item)
        if entity_name in self.id2ref:
            raise CofolaParsingError(
                f"Entity {entity_name} has already been used as an object name."
            )
        e = Entity(entity_name)
        self.entities.add(entity_name)
        return e

    def operations(self, args):
        # Child already called builder.add() and returned an ObjRef.
        return args[-1]

    def common_operations(self, args):
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

        return _resolve_collection_operation(
            builder=self.builder,
            op_type=op_type,
            source=obj,
            source_defn=self._defn_of(obj),
            size=size,
            op_arg=op_arg,
        )

    def binary_operations(self, args):
        obj1, op, obj2 = args
        op = str(op)
        return _resolve_binary_operation(
            builder=self.builder,
            left=obj1,
            op=op,
            right=obj2,
            left_defn=self._defn_of(obj1),
            right_defn=self._defn_of(obj2),
        )

    def indexing(self, args):
        obj, index = args[0], args[2]
        defn = self._defn_of(obj)

        if isinstance(defn, Grouped):
            idx = int(index)
            for r, d in self.builder.iter_defs():
                if isinstance(d, PartDef) and d.partition == obj and d.index == idx:
                    return r
            raise CofolaParsingError(f"Partition part index {idx} not found.")

        if isinstance(defn, TupleDef):
            # If the tuple's source is a PartDef, return that PartDef directly.
            # This allows `t[k]` to be used as the source partition part in
            # constraints like `|t[0]| == 2` or `t[0] == e1`.
            src_defn = self.builder.get_object(defn.source)
            if isinstance(src_defn, PartDef):
                return defn.source
            return TupleIndexSentinel(tuple_ref=obj, index=int(index))

        kind = type(defn).__name__ if defn is not None else "unknown"
        raise CofolaParsingError(f"Indexing not supported for {kind} objects.")

    def count(self, args):
        obj, count_name, _, entity_or_obj, _ = args
        deduplicate = str(count_name) == "dedup_count"
        return _resolve_count_atom(
            target=obj,
            count_arg=entity_or_obj,
            target_defn=self._defn_of(obj),
            deduplicate=deduplicate,
        )
