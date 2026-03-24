"""
Pytest tests for SubShader audio overlap handling.

Validates that hop-center extraction of CWT frames produces
a continuous spectrogram without redundant overlap.
"""

import numpy as np

from subshader.config import get_default_config
from subshader.dsp.wavelet import NumPyWavelet


def test_overlap_hop_center_extraction():
    """
    Given overlapping chunks from a synthetic chirp, hop-center extraction
    (taking only the trailing hop_size columns from each CWT frame's reliable
    region) should tile contiguously: n_chunks * hop_cols == expected_total_width.
    Also asserts no NaN values in the tiled output.
    """
    config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk_size = config.audio.chunk_size
    overlap_factor = config.audio.overlap_factor
    hop_size = int(chunk_size * (1.0 - overlap_factor))

    # Synthetic rising chirp: enough samples for several overlapping chunks
    n_chunks = 5
    total_samples = hop_size * n_chunks + chunk_size
    t = np.linspace(0, total_samples / sr, total_samples, endpoint=False)
    f0, f1 = 200.0, 4000.0
    from scipy.signal import chirp as scipy_chirp
    signal = scipy_chirp(t, f0=f0, f1=f1, t1=t[-1], method='linear').astype(np.float64)

    wavelet = NumPyWavelet(sample_rate=sr, input_n=chunk_size, config=config.wavelet)

    hop_center_frames = []
    for i in range(n_chunks):
        chunk_start = i * hop_size
        chunk = signal[chunk_start: chunk_start + chunk_size]

        # Run CWT through reliable-region trimming (skip final downsample)
        cwt_coefs = wavelet.class_specific_cwt(np.asarray(chunk, dtype=np.float64))
        cwt_coefs = wavelet.normalize_by_scale(cwt_coefs)
        mag_coefs = wavelet.compute_mag(cwt_coefs)
        reliable_coefs = wavelet.discard_unreliable_coefs(mag_coefs)

        # Extract trailing hop_size columns (non-overlapping new content)
        reliable_width = reliable_coefs.shape[1]
        hop_cols = min(hop_size, reliable_width)
        hop_center = reliable_coefs[:, -hop_cols:]
        hop_center_frames.append(hop_center)

    # All hop-center slices should have the same column count
    hop_cols_per_frame = hop_center_frames[0].shape[1]
    expected_total_cols = n_chunks * hop_cols_per_frame

    tiled = np.concatenate(hop_center_frames, axis=1)

    print(f"  hop_size={hop_size}, hop_cols_per_frame={hop_cols_per_frame}")
    print(f"  tiled shape: {tiled.shape}, expected cols: {expected_total_cols}")

    assert tiled.shape[1] == expected_total_cols, (
        f"Tiled width {tiled.shape[1]} != expected {expected_total_cols} "
        f"({n_chunks} chunks x {hop_cols_per_frame} cols)"
    )
    assert not np.any(np.isnan(tiled)), "NaN values found in tiled hop-center output"
