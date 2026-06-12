"""Thin wrapper around the optional clingo dependency for the ASP backend."""
from __future__ import annotations

from importlib import import_module

__all__ = ["ASPSolverError", "run_clingo"]


class ASPSolverError(Exception):
    """Raised when the optional clingo solver cannot be invoked."""


def run_clingo(program: str) -> int:
    """Count the stable models of an ASP ``program`` with clingo.

    clingo is an optional dependency (it brings a solver stack), so it is
    imported only when the ASP backend is actually used.
    """
    try:
        clingo = import_module("clingo")
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ASPSolverError(
            "clingo is not installed. Install the optional backend with "
            "`uv sync --extra asp` before using `--backend asp`."
        ) from exc

    ctl = clingo.Control(["--warn=none", "--models=0"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as handle:
        return sum(1 for _ in handle)
