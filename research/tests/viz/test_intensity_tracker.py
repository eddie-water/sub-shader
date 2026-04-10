"""Tests for IntensityTracker fixed-reference normalization.

IntensityTracker now holds a fixed reference value set at construction.
update() is a no-op — the value never changes during playback.
"""

import numpy as np
import pytest
from subshader.renderer.intensity import IntensityTracker


def test_fixed_max_is_set_at_construction():
    """IntensityTracker(fixed_max=42.0).global_max == 42.0 after construction."""
    tracker = IntensityTracker(fixed_max=42.0)
    assert tracker.global_max == 42.0


def test_update_with_loud_frame_returns_fixed_value():
    """update() with a loud frame does not change global_max."""
    tracker = IntensityTracker(fixed_max=42.0)
    loud_frame = np.ones((10, 10)) * 1000.0
    result = tracker.update(loud_frame)
    assert tracker.global_max == 42.0
    assert result == 42.0


def test_update_with_quiet_frame_returns_fixed_value():
    """update() with a quiet frame does not change global_max."""
    tracker = IntensityTracker(fixed_max=42.0)
    quiet_frame = np.zeros((10, 10))
    result = tracker.update(quiet_frame)
    assert tracker.global_max == 42.0
    assert result == 42.0


def test_zero_fixed_max_clamps_to_floor_value():
    """IntensityTracker(fixed_max=0.0) clamps to floor_value (1e-8)."""
    tracker = IntensityTracker(fixed_max=0.0)
    assert tracker.global_max == pytest.approx(1e-8)


def test_color_normalization_config_has_no_retention_rate():
    """ColorNormalizationConfig no longer has retention_rate field."""
    from subshader.config import ColorNormalizationConfig
    config = ColorNormalizationConfig()
    assert not hasattr(config, "retention_rate"), "retention_rate field still exists — should be removed"


def test_color_normalization_config_validate_passes():
    """ColorNormalizationConfig.validate() passes with valid params."""
    from subshader.config import ColorNormalizationConfig
    config = ColorNormalizationConfig()
    errors = config.validate()
    assert errors == [], f"Expected no errors, got: {errors}"
