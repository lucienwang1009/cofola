"""Tests for the Backend ABC."""
from __future__ import annotations

import pytest


def test_backend_abc_cannot_be_instantiated():
    from cofola.backend.base import Backend

    with pytest.raises(TypeError):
        Backend()  # type: ignore[abstract]


def test_concrete_backend_must_implement_name_and_solve():
    from cofola.backend.base import Backend

    class IncompleteBackend(Backend):
        pass  # missing name and solve

    with pytest.raises(TypeError):
        IncompleteBackend()  # type: ignore[abstract]


def test_concrete_backend_can_be_instantiated():
    from cofola.backend.base import Backend

    class ConcreteBackend(Backend):
        name = "test"

        def solve(self, problem: object) -> int:
            return 42

    backend = ConcreteBackend()
    assert backend.name == "test"
    assert backend.solve(None) == 42
