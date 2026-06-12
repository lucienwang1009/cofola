"""Thin wrapper around the external Conjure/Savile Row toolchain."""
from __future__ import annotations

import os
import json
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = ["EssenceSolverConfig", "EssenceSolverError", "run_conjure"]


class EssenceSolverError(Exception):
    """Raised when Conjure cannot be invoked or does not return solutions."""


@dataclass(frozen=True)
class EssenceSolverConfig(object):
    """External tool locations for the Essence backend."""

    conjure_dir: Path | None = None
    java_bin: Path | None = None
    timeout: float | None = None


def run_conjure(program: str, config: EssenceSolverConfig | None = None) -> int:
    """Count solutions of an Essence ``program`` with Conjure.

    Conjure/Savile Row/Minion are intentionally external tools. If
    ``config.conjure_dir`` is omitted, ``COFOLA_CONJURE_DIR`` and then
    ``PATH`` are searched. ``COFOLA_JAVA_BIN`` can similarly point at Java.
    """

    config = config or EssenceSolverConfig()
    env = os.environ.copy()
    conjure = _resolve_conjure(config, env)
    _prepare_java(config, env)

    with tempfile.TemporaryDirectory(prefix="cofola-essence-") as tmp:
        tmp_path = Path(tmp)
        model = tmp_path / "model.essence"
        out_dir = tmp_path / "conjure-output"
        model.write_text(program)
        proc = subprocess.Popen(
            [
                str(conjure),
                "solve",
                "-ac",
                "-o",
                str(out_dir),
                "--number-of-solutions=all",
                "--savilerow-options=-solutions-to-null",
                "--log-level",
                "lognone",
                str(model),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=env,
            cwd=tmp_path,
        )
        try:
            stdout, stderr = proc.communicate(timeout=config.timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            timeout = config.timeout if config.timeout is not None else 0.0
            raise TimeoutError(f"Essence timed out after {timeout:.3f}s") from exc

        if proc.returncode != 0:
            raise EssenceSolverError(_summarize_output(stderr, stdout))

        solution_count = _solver_solution_count(out_dir)
        if solution_count is not None:
            return solution_count

        solution_file = out_dir / "model000001.solutions"
        if not solution_file.exists():
            raise EssenceSolverError("Conjure did not produce model000001.solutions")
        solutions = solution_file.read_text()
        return solutions.count("Count:") or solutions.count("$ Solution:")


def _summarize_output(stderr: str, stdout: str, limit: int = 8) -> str:
    lines = [line.strip() for line in (stderr + "\n" + stdout).splitlines()]
    non_empty = [line for line in lines if line]
    if not non_empty:
        return "Conjure failed"
    return " | ".join(non_empty[:limit])


def _solver_solution_count(out_dir: Path) -> int | None:
    stats_file = out_dir / "model000001.stats.json"
    if not stats_file.exists():
        return None
    data = json.loads(stats_file.read_text())
    info = data.get("savilerowInfo")
    if not isinstance(info, dict):
        return None
    solutions = info.get("SolverSolutionsFound")
    if isinstance(solutions, str) and solutions.isdecimal():
        return int(solutions)
    satisfiable = info.get("SolverSatisfiable")
    if satisfiable == "0":
        return 0
    return None


def _resolve_conjure(config: EssenceSolverConfig, env: dict[str, str]) -> Path:
    configured_dir = config.conjure_dir
    if configured_dir is None:
        env_dir = env.get("COFOLA_CONJURE_DIR")
        configured_dir = Path(env_dir) if env_dir else None

    if configured_dir is not None:
        configured_dir = configured_dir.resolve()
        conjure = configured_dir / "conjure"
        if not conjure.exists():
            raise EssenceSolverError(f"Essence backend requires Conjure at {conjure}")
        env["PATH"] = str(configured_dir) + os.pathsep + env.get("PATH", "")
        return conjure

    found = shutil.which("conjure", path=env.get("PATH"))
    if found is None:
        raise EssenceSolverError(
            "Conjure is not on PATH. Install it with "
            "`uv run cofola-install-conjure` or pass `--conjure-dir`."
        )
    return Path(found)


def _prepare_java(config: EssenceSolverConfig, env: dict[str, str]) -> None:
    java_bin = config.java_bin
    if java_bin is None:
        env_java = env.get("COFOLA_JAVA_BIN")
        java_bin = Path(env_java) if env_java else None
    if java_bin is not None:
        env["PATH"] = str(java_bin.parent.resolve()) + os.pathsep + env.get("PATH", "")
    if shutil.which("java", path=env.get("PATH")) is None:
        raise EssenceSolverError("Essence backend requires java on PATH for Savile Row")
