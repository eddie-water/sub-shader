"""
Pytest tests for SubShader visualization components.

Covers test category 10:
  10. Buffer / visualization normalization
"""

import numpy as np

from subshader.config import ColorNormalizationConfig
from subshader.viz.plotter import CircularFrameBuffer


# =============================================================================
# TEST 10: BUFFER / VISUALIZATION NORMALIZATION
# =============================================================================

def test_buffer_normalization():
    """Verify CircularFrameBuffer preserves relative intensities."""
    n_freqs = 16
    n_time = 32
    n_frames = 10

    color_norm = ColorNormalizationConfig()
    buf = CircularFrameBuffer(
        frame_shape=(n_freqs, n_time),
        num_frames=n_frames,
        color_norm_config=color_norm,
    )

    for i in range(n_frames):
        frame = np.random.rand(n_freqs, n_time).astype(np.float32) * (i + 1)
        buf.push_frame(frame)

    imax = buf.get_intensity_max()
    print(f"  get_intensity_max() = {imax:.4f}")
    assert imax > 0, f"get_intensity_max() = {imax:.4f} (expect > 0)"

    flat = buf.get_flattened_buffer()
    print(f"  Flattened buffer shape: {flat.shape} (expect ({n_freqs}, {n_time * n_frames}))")
    assert flat.shape[0] == n_freqs, (
        f"Flattened buffer freq dimension: {flat.shape[0]} (expect {n_freqs})"
    )
    assert flat.shape[1] == n_time * n_frames, (
        f"Flattened buffer time dimension: {flat.shape[1]} (expect {n_time * n_frames})"
    )

    print(f"  Flattened buffer max = {flat.max():.2f}")
    assert flat.max() > 1.0, (
        f"Flattened buffer max = {flat.max():.2f} "
        f"(expect > 1.0, confirming no [0,1] normalization)"
    )
