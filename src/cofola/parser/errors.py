"""Cofola parser exception types."""
from __future__ import annotations


class CofolaParsingError(Exception):
    """Base exception for Cofola parsing errors."""


class CofolaTypeMismatchError(CofolaParsingError):
    """Raised when a parsed object has the wrong type."""

    def __init__(self, expected_types, actual) -> None:
        if isinstance(expected_types, tuple):
            names = [
                t.__name__ if hasattr(t, "__name__") else str(t)
                for t in expected_types
            ]
            expected = " or ".join(names)
        else:
            expected = (
                expected_types.__name__
                if hasattr(expected_types, "__name__")
                else str(expected_types)
            )
        super().__init__(
            f"Expect a {expected} object, but got {actual} of type {type(actual)}."
        )
