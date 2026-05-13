"""Cofola parser package — public API."""
from __future__ import annotations

from cofola.parser.utils import CofolaParsingError
from cofola.parser.parser import parse

__all__ = [
    "CofolaParsingError",
    "parse",
]
