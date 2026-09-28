"""
Tests for the CWT downsample stage: block max-pooling of the time axis.

These tests drive the fix for stride aliasing in the visualization path:
uniform index selection kept 1 of every ~128 columns, so a short transient
(a few ms of high-frequency CWT response) could fall entirely between kept
columns and vanish from the rendered frame. Block max-pooling guarantees
every column lands in exactly one block, so no event can disappear.
"""

import numpy as np
import pytest


class TestDownsampleShape:
    """Downsample must produce (freq_bins, target_width) for any input width."""

    def test_output_shape_divisible(self, cpu_cwt):
        coefs = np.random.rand(12, 8192)
        out = cpu_cwt.downsample(coefs, 64)
        assert out.shape == (12, 64)

    def test_output_shape_non_divisible(self, cpu_cwt):
        coefs = np.random.rand(12, 1000)
        out = cpu_cwt.downsample(coefs, 64)
        assert out.shape == (12, 64)

    def test_invalid_target_width_raises(self, cpu_cwt):
        coefs = np.random.rand(12, 100)
        with pytest.raises(ValueError):
            cpu_cwt.downsample(coefs, 0)
        with pytest.raises(ValueError):
            cpu_cwt.downsample(coefs, 101)


class TestDownsamplePreservesTransients:
    """No narrow event may vanish, regardless of where it lands in time."""

    def test_narrow_spike_survives_at_any_position(self, cpu_cwt):
        """A 5-column spike must appear in the output wherever it sits.

        Under index selection (stride 128), a spike placed between kept
        columns was silently dropped - this is the aliasing regression.
        """
        num_samples, target_width = 8192, 64
        stride = num_samples // target_width
        spike_positions = [
            stride // 2,            # centered between the first two kept columns
            3 * stride + 10,        # just after a kept column
            num_samples - stride,   # near the tail
        ]
        for pos in spike_positions:
            coefs = np.zeros((4, num_samples))
            coefs[2, pos : pos + 5] = 7.0
            out = cpu_cwt.downsample(coefs, target_width)
            assert out.max() == pytest.approx(7.0), (
                f"Spike at column {pos} vanished from downsampled output. "
                "Downsample must max-pool blocks, not select single columns."
            )

    def test_spike_lands_in_correct_block(self, cpu_cwt):
        num_samples, target_width = 8192, 64
        block = num_samples // target_width
        coefs = np.zeros((4, num_samples))
        coefs[1, 10 * block + 3] = 5.0
        out = cpu_cwt.downsample(coefs, target_width)
        assert out[1, 10] == pytest.approx(5.0)
        assert np.count_nonzero(out) == 1


class TestDownsampleSustainedContent:
    """Pooling must not distort locally-constant (sustained tone) magnitudes."""

    def test_constant_input_unchanged(self, cpu_cwt):
        coefs = np.full((8, 8192), 3.25)
        out = cpu_cwt.downsample(coefs, 64)
        assert np.allclose(out, 3.25)

    def test_slow_ramp_tracks_envelope(self, cpu_cwt):
        """Output of a slow ramp must stay within the input's range."""
        coefs = np.tile(np.linspace(0.0, 1.0, 8192), (3, 1))
        out = cpu_cwt.downsample(coefs, 64)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert np.all(np.diff(out, axis=1) > 0)
