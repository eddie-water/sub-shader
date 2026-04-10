"""Tests for IntensityTracker retention semantics.

These tests verify that retention_rate=0.95 means "retain 95% per frame",
not the old inverted decay_rate behavior that dropped to 5% per frame.
"""

import numpy as np
import pytest
from subshader.renderer.intensity import IntensityTracker


def test_retention_rate_single_frame():
    """retention_rate=0.95 on a zero-intensity frame retains 95% of global_max."""
    tracker = IntensityTracker(retention_rate=0.95)
    tracker.global_max = 100.0
    tracker.update(np.zeros((10, 10)))
    # The floor_value (1e-8) is below 95.0, so global_max should be ~95.0
    assert abs(tracker.global_max - 95.0) < 0.01, f"Expected ~95.0, got {tracker.global_max}"


def test_retention_rate_exponential_decay():
    """After 10 silent frames, global_max decays by 0.95^10 of initial."""
    tracker = IntensityTracker(retention_rate=0.95)
    tracker.global_max = 100.0
    for _ in range(10):
        tracker.update(np.zeros((10, 10)))
    expected = 100.0 * (0.95 ** 10)  # ~59.87
    assert abs(tracker.global_max - expected) < 0.01, f"Expected ~{expected:.2f}, got {tracker.global_max}"


def test_retention_rate_high_intensity_frames():
    """High-intensity frames drive global_max up; retention keeps it stable between frames."""
    tracker = IntensityTracker(retention_rate=0.95)
    high_frame = np.ones((10, 10)) * 50.0
    for _ in range(5):
        tracker.update(high_frame)
    # After several high frames, global_max should be near 50.0
    assert tracker.global_max >= 49.0, f"Expected >= 49.0, got {tracker.global_max}"


def test_config_field_is_retention_rate():
    """ColorNormalizationConfig exposes retention_rate, not decay_rate."""
    from subshader.config import ColorNormalizationConfig
    config = ColorNormalizationConfig()
    assert hasattr(config, "retention_rate"), "Missing retention_rate field"
    assert not hasattr(config, "decay_rate"), "Stale decay_rate field still exists"


def test_config_retention_rate_default_is_sensible():
    """Default retention_rate is 0.95 (retains 95% per frame)."""
    from subshader.config import ColorNormalizationConfig
    config = ColorNormalizationConfig()
    assert config.retention_rate == 0.95, f"Expected 0.95, got {config.retention_rate}"


def test_config_validate_accepts_retention_rate():
    """ColorNormalizationConfig.validate() passes with valid retention_rate."""
    from subshader.config import ColorNormalizationConfig
    config = ColorNormalizationConfig(retention_rate=0.9)
    errors = config.validate()
    assert errors == [], f"Expected no errors, got: {errors}"


def test_config_validate_rejects_invalid_retention_rate():
    """ColorNormalizationConfig.validate() rejects retention_rate outside (0, 1)."""
    from subshader.config import ColorNormalizationConfig
    config_zero = ColorNormalizationConfig(retention_rate=0.0)
    assert any("retention_rate" in e for e in config_zero.validate())
    config_one = ColorNormalizationConfig(retention_rate=1.0)
    assert any("retention_rate" in e for e in config_one.validate())
