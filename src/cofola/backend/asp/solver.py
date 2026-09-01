"""Run an ASP program with clingo for the ASP backend."""
from __future__ import annotations

import clingo

__all__ = ["run_clingo"]


def run_clingo(program: str) -> int:
    """Count the stable models of an ASP ``program`` with clingo."""
    ctl = clingo.Control(["--warn=none", "--models=0"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as handle:
        return sum(1 for _ in handle)
