"""WFOMC backend planning-profile declaration."""
from __future__ import annotations

from cofola.backend.wfomc.backend import (
    WFOMCBackend,
    WFOMC_GLOBAL_PASSES,
    WFOMC_LOCAL_PASSES,
)
from cofola.planing.pass_manager import FixedPointPass
from cofola.planing.passes.lowering import LoweringPass


class TestWFOMCBackendProfile(object):
    """Backend profile and planner integration."""

    def test_wfomc_backend_declares_default_planning_profile(self) -> None:
        profile = WFOMCBackend().planning_profile()

        assert profile.global_passes == WFOMC_GLOBAL_PASSES
        assert profile.local_passes == WFOMC_LOCAL_PASSES
        assert profile.local_passes is not None
        assert any(
            isinstance(pass_spec, FixedPointPass) and pass_spec.pass_cls is LoweringPass
            for pass_spec in profile.local_passes
        )
