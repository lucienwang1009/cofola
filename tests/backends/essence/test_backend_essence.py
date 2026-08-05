"""Direct Essence backend encoding and routing tests."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from cofola.backend.essence.backend import (
    ESSENCE_GLOBAL_PASSES,
    ESSENCE_LOCAL_PASSES,
    EssenceBackend,
)
from cofola.backend.essence.encoder import EssenceEncoder
from cofola.frontend.objects import ObjRef, SetChooseReplace, SetInit
from cofola.frontend.problem import Problem
from cofola.parser.parser import parse
from cofola.planing.analysis.entities import AnalysisResult, BagInfo, SetInfo
from cofola.planing.pipeline import PlaningPipeline
from cofola.solver import parse_and_solve


def _available_conjure_dir() -> Path | None:
    env_dir = os.environ.get("COFOLA_CONJURE_DIR")
    if env_dir is not None:
        configured = Path(env_dir)
        if (configured / "conjure").exists():
            return configured

    repo_root = Path(__file__).resolve().parents[3]
    local_conjures = (repo_root / ".tools" / "conjure").glob(
        "conjure-v*-*-combined/conjure"
    )
    for conjure in sorted(local_conjures):
        return conjure.parent

    found = shutil.which("conjure")
    return Path(found).parent if found is not None else None


def _encode_single_component(source: str) -> str:
    backend = EssenceBackend()
    schedule = PlaningPipeline(backend.planning_profile()).process(parse(source))
    assert len(schedule.branches) == 1
    assert len(schedule.branches[0].components) == 1
    problem, analysis = schedule.branches[0].components[0]
    return EssenceEncoder(problem, analysis).encode()


def test_essence_backend_declares_planning_profile() -> None:
    profile = EssenceBackend().planning_profile()

    assert profile.global_passes == ESSENCE_GLOBAL_PASSES
    assert profile.local_passes == ESSENCE_LOCAL_PASSES


def test_set_choice_uses_native_set_domain() -> None:
    model = _encode_single_component(
        """
S = set(a, b, c)
B = choose(S)
|B| == 2
"""
    )

    assert "find o_0 : set (size 3) of Entity" in model
    assert "find o_1 : set (size 2) of Entity" in model
    assert "o_1 subsetEq o_0" in model
    assert "mult(" not in model


def test_native_size_domain_is_not_duplicated_as_a_constraint() -> None:
    """Size lives in the domain (`set (size 2)`); the encoder must not also
    re-emit `|o| <= max` / `|o| = exact` bounds that duplicate it."""
    model = _encode_single_component(
        """
S = set(a, b, c)
T = set(b, c, d)
U = S & T
|U| == 2
"""
    )

    # The folded intersection keeps its native (size 2) domain plus the user's
    # own size constraint, but no encoder-injected redundant size bounds.
    assert "set (size 2) of Entity" in model
    assert "<=" not in model
    assert model.count("= 2") == 1  # only the user's |U| == 2


def test_membership_of_pruned_entity_folds_to_false() -> None:
    """A constraint may reference an entity that constant folding removed from
    the component (e.g. `a in S & T` once `S & T` folds to `{b, c}`). Such an
    entity cannot occur, so membership encodes as `false` instead of crashing."""
    model = _encode_single_component(
        """
S = set(a, b, c)
T = set(b, c, d)
U = S & T
a in U
"""
    )

    assert "$ entity" in model  # a is gone from the domain (only b, c remain)
    assert "= a" not in model and "in U" not in model
    assert "false" in model


def test_bag_choice_uses_native_mset_domain() -> None:
    model = _encode_single_component(
        """
B = bag(a: 1, b: 2)
C = choose(B)
C.count(b) == 1
"""
    )

    assert "find o_0 : mset (size 3, maxOccur 2) of Entity" in model
    assert "find o_1 : mset" in model
    assert "freq(o_1, e) <= freq(o_0, e)" in model


def test_sequence_uses_native_sequence_domain() -> None:
    model = _encode_single_component(
        """
S = set(a, b, c)
row = sequence(S)
row.before(a, b)
"""
    )

    assert "find o_1 : sequence (size 3) of Entity" in model
    assert "o_1(1) in o_0" in model
    assert "o_1(1) = " in model


def test_composition_uses_native_part_sets_with_coverage_constraint() -> None:
    model = _encode_single_component(
        """
S = set(a, b, c)
C = compose(S, 2)
"""
    )

    assert "find o_2 : set (maxSize 3) of Entity" in model
    assert "find o_3 : set (maxSize 3) of Entity" in model
    assert "sum([toInt(e in o_2), toInt(e in o_3)]) = toInt(e in o_0)" in model
    assert "partition from" not in model


def test_bag_partition_uses_native_part_msets_and_canonical_order() -> None:
    model = _encode_single_component(
        """
B = bag(a: 2, b: 1)
P = partition(B, 2)
"""
    )

    assert "find o_2 : mset (maxSize 3, maxOccur 2) of Entity" in model
    assert "find o_3 : mset (maxSize 3, maxOccur 2) of Entity" in model
    assert "sum([freq(o_2, e), freq(o_3, e)]) = freq(o_0, e)" in model
    assert "freq(o_3, 0) < freq(o_2, 0)" in model
    assert "freq(o_3, 0) = freq(o_2, 0)" in model


def test_empty_replace_choice_still_constrains_the_phantom_domain_value() -> None:
    source = ObjRef(0)
    chosen = ObjRef(1)
    problem = Problem(
        defs=(
            (source, SetInit(entities=frozenset())),
            (chosen, SetChooseReplace(source=source, size=1)),
        ),
        constraints=(),
        names=((source, "S"), (chosen, "B")),
    )
    analysis = AnalysisResult(
        set_info={source: SetInfo(p_entities=set(), max_size=0, exact_size=0)},
        bag_info={
            chosen: BagInfo(
                p_entities_multiplicity={},
                max_size=1,
                exact_size=1,
            )
        },
        all_entities=set(),
        singletons=set(),
    )
    model = EssenceEncoder(problem, analysis).encode()

    assert "letting Entity be domain int(0..0)" in model
    assert "forAll e : Entity . (freq(o_1, e) > 0 -> e in o_0)" in model


def test_parse_and_solve_routes_to_essence_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_run_conjure(program, config):
        seen.append(program)
        assert config.conjure_dir is None
        return 3

    import cofola.backend.essence.backend as backend_module

    monkeypatch.setattr(backend_module, "run_conjure", fake_run_conjure)

    result = parse_and_solve(
        """
A = set(a, b, c)
B = choose(A)
|B| == 2
""",
        backend="essence",
    )

    assert result == 3
    assert len(seen) == 1
    assert "find o_1 : set (size 2) of Entity" in seen[0]


def test_timeout_cleanup_escalates_to_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    import cofola.backend.essence.solver as solver_module

    monkeypatch.setattr(solver_module, "_TERMINATION_GRACE_SECONDS", 0.1)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(30)\n"
            ),
        ],
        start_new_session=True,
    )
    try:
        solver_module._terminate_process_group(proc)
        assert proc.wait(timeout=1) != 0
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=1)


def test_conjure_installer_copy_fallback_replaces_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cofola.backend.essence.install import _link_or_copy

    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new")
    target = tmp_path / "target"
    target.mkdir()
    (target / "old.txt").write_text("old")

    def fail_symlink(self, target, target_is_directory=False):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(Path, "symlink_to", fail_symlink)

    _link_or_copy(source, target)

    assert not (target / "old.txt").exists()
    assert (target / "new.txt").read_text() == "new"


def test_essence_backend_solves_tiny_model_with_real_conjure() -> None:
    conjure_dir = _available_conjure_dir()
    if conjure_dir is None:
        pytest.skip("Conjure is not installed")
    if shutil.which("java") is None and os.environ.get("COFOLA_JAVA_BIN") is None:
        pytest.skip("Java is not installed")

    result = parse_and_solve(
        """
A = set(a, b, c)
B = choose(A)
|B| == 2
""",
        backend=EssenceBackend(conjure_dir=conjure_dir, timeout=15.0),
    )

    assert result == 3
