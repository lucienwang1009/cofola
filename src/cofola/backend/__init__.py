"""Backend package — WFOMC translation and solving."""

from cofola.backend.base import Backend
from cofola.backend.wfomc.backend import WFOMCBackend

__all__ = [
    "Backend",
    "WFOMCBackend",
]