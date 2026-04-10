"""
Pytest tests for SubShader visualization components.

Covers test category 10:
  10. Buffer / visualization normalization
"""

import numpy as np

from subshader.renderer.frame_buffer import CircularFrameBuffer


# =============================================================================
# TEST 10: BUFFER / VISUALIZATION NORMALIZATION
# =============================================================================

def test_buffer_preserves_frame_data():
    """Verify CircularFrameBuffer stores frames and produces correct flattened buffer."""
    n_freqs = 16
    n_time = 32
    n_frames = 10

    buf = CircularFrameBuffer(
        frame_shape=(n_freqs, n_time),
        num_frames=n_frames,
    )

    for i in range(n_frames):
        frame = np.random.rand(n_freqs, n_time).astype(np.float32) * (i + 1)
        buf.push_frame(frame)

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
