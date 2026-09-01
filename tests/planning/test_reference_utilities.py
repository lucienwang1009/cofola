"""Planning reference/utility helpers (object_refs, constraint_refs, RefAllocator)."""
from __future__ import annotations

from cofola.frontend import (
    BagCountAtom,
    Entity,
    ObjRef,
    Problem,
    SetInit,
    SizeConstraint,
)
from cofola.frontend.utils import constraint_refs
from cofola.planing.pass_manager import RefAllocator


class TestPlanningReferenceUtilities(object):
    """Reference allocation and reference walking helpers."""

    def test_ref_allocator_starts_after_existing_refs(self) -> None:
        """Generated refs should not depend on a magic-number id range."""
        problem = Problem(
            defs=((ObjRef(10000), SetInit(entities=frozenset({Entity("a")}))),),
            constraints=(),
            names=(),
        )

        allocator = RefAllocator(problem)

        assert allocator.new_ref() == ObjRef(10001)
        assert allocator.new_ref() == ObjRef(10002)


    def test_constraint_refs_recurses_into_size_atoms(self) -> None:
        """Shared ref walking should see refs nested inside size atoms."""
        bag = ObjRef(1)
        constraint = SizeConstraint(
            terms=((BagCountAtom(bag=bag, entity=Entity("a")), 1),),
            comparator="==",
            rhs=1,
        )

        assert constraint_refs(constraint) == [bag]
