"""
Overlap Redundancy Diagnostic for SubShader CWT Pipeline.

Demonstrates the overlap redundancy bug and hop-center extraction fix.
For each overlap_factor in [0.0, 0.5, 0.75]:
  - Extracts consecutive overlapping chunks from a synthetic chirp signal
  - Computes CWT (NumPyWavelet) on each chunk
  - Shows side-by-side: full reliable region tiled vs hop-center tiled

Usage:
    cd /home/eddie-water/dev/python/sub-shader
    python research/overlap_diagnostic.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys

# Configure matplotlib BEFORE any pyplot imports (match benchmark.py pattern)
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

# Ensure the project root is on sys.path so subshader imports resolve when
# running as a plain script rather than via python -m.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from subshader.config import get_default_config
from subshader.dsp.wavelet import NumPyWavelet
from subshader.utils.logging import logger_init, get_logger

# =============================================================================
# LOGGING
# =============================================================================

logger_init(log_level="WARNING", console_output=True, file_output=False)
log = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 44100
CHUNK_SIZE = 16384
TARGET_WIDTH = 256
CHIRP_DURATION_S = 4.0      # long enough for multiple overlapping chunks
CHIRP_F0 = 440.0            # Hz
CHIRP_F1 = 880.0            # Hz
NUM_FRAMES = 5              # consecutive chunks to tile per cell
OVERLAP_FACTORS = [0.0, 0.5, 0.75]
OUTPUT_PATH = "assets/images/diagnostics/overlap_redundancy_diagnostic.png"

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def make_chirp(sample_rate: int, duration_s: float, f0: float, f1: float) -> np.ndarray:
    """Generate a linear chirp signal from f0 to f1 Hz."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Linear instantaneous frequency: f(t) = f0 + (f1-f0)*t/T
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t ** 2)
    return np.sin(phase).astype(np.float64)

# =============================================================================
# HOP-CENTER EXTRACTION
# =============================================================================

def extract_hop_center(reliable_coefs: np.ndarray, overlap_factor: float) -> np.ndarray:
    """
    Extract the hop-center portion of reliable CWT coefficients.

    The reliable region maps to the full input chunk. Only the center portion
    corresponding to the hop (non-overlapping new data) contains unique time
    content. Extract that center slice.

    Args:
        reliable_coefs: CWT coefficients after reliable-region trimming,
                        shape (num_freqs, reliable_width).
        overlap_factor: Fraction of consecutive chunks that overlap.

    Returns:
        Center slice of shape (num_freqs, center_width) where center_width
        proportional to (1 - overlap_factor) * reliable_width.
        Returns input unchanged if overlap_factor is 0.
    """
    if overlap_factor <= 0.0:
        return reliable_coefs

    reliable_width = reliable_coefs.shape[1]
    hop_fraction = 1.0 - overlap_factor
    center_width = max(1, int(reliable_width * hop_fraction))
    center_start = (reliable_width - center_width) // 2

    return reliable_coefs[:, center_start:center_start + center_width]

# =============================================================================
# PIPELINE HELPERS
# =============================================================================

def get_reliable_coefs(wavelet: NumPyWavelet, chunk: np.ndarray) -> np.ndarray:
    """
    Run the CWT pipeline up to (but not including) the downsample step.

    Returns the reliable-region coefficients before hop-center extraction,
    so the diagnostic can apply both extraction strategies for comparison.
    """
    cwt_coefs = wavelet.class_specific_cwt(np.asarray(chunk, dtype=np.float64))
    cwt_coefs = wavelet.normalize_by_scale(cwt_coefs)
    mag_coefs = wavelet.compute_mag(cwt_coefs)
    reliable_coefs = wavelet.discard_unreliable_coefs(mag_coefs)
    return reliable_coefs


def tile_frames(frames: list[np.ndarray], target_width: int) -> np.ndarray:
    """
    Downsample each frame to target_width columns, then concatenate horizontally.

    Args:
        frames: List of (num_freqs, width) arrays.
        target_width: Number of columns each frame is downsampled to.

    Returns:
        (num_freqs, target_width * len(frames)) tiled array.
    """
    downsampled = []
    for frame in frames:
        _, w = frame.shape
        if w < target_width:
            # Frame narrower than target — stretch via repeat (rare edge case)
            factor = max(1, target_width // w)
            frame = np.repeat(frame, factor, axis=1)[:, :target_width]
        hop = w / target_width
        indices = np.floor(np.arange(target_width) * hop).astype(int)
        indices = np.clip(indices, 0, w - 1)
        downsampled.append(frame[:, indices])
    return np.concatenate(downsampled, axis=1)

# =============================================================================
# MAIN DIAGNOSTIC
# =============================================================================

def run_diagnostic():
    config = get_default_config()

    # Override target_width for clarity
    config.wavelet.target_width = TARGET_WIDTH

    print("\nSubShader — Overlap Redundancy Diagnostic")
    print("=" * 60)

    # Generate synthetic chirp signal
    signal = make_chirp(SAMPLE_RATE, CHIRP_DURATION_S, CHIRP_F0, CHIRP_F1)

    # Build wavelet (shared across overlap factors — only chunk extraction changes)
    wavelet = NumPyWavelet(
        sample_rate=SAMPLE_RATE,
        input_n=CHUNK_SIZE,
        config=config.wavelet,
    )

    # Summary table header
    print(f"\n{'overlap_factor':>14} {'hop_size':>10} {'reliable_width':>15} {'center_width':>13} {'frame_count':>12}")
    print("-" * 66)

    results = {}

    for overlap_factor in OVERLAP_FACTORS:
        hop_size = int(CHUNK_SIZE * (1.0 - overlap_factor))

        # Extract NUM_FRAMES consecutive overlapping chunks
        full_frames = []
        hop_center_frames = []

        for i in range(NUM_FRAMES):
            chunk_start = i * hop_size
            chunk_end = chunk_start + CHUNK_SIZE
            if chunk_end > len(signal):
                break

            chunk = signal[chunk_start:chunk_end]
            reliable_coefs = get_reliable_coefs(wavelet, chunk)
            reliable_width = reliable_coefs.shape[1]

            hop_center = extract_hop_center(reliable_coefs, overlap_factor)
            center_width = hop_center.shape[1]

            full_frames.append(reliable_coefs)
            hop_center_frames.append(hop_center)

        frame_count = len(full_frames)
        reliable_width = full_frames[0].shape[1] if full_frames else 0
        center_width = hop_center_frames[0].shape[1] if hop_center_frames else 0

        print(f"{overlap_factor:>14.2f} {hop_size:>10} {reliable_width:>15} {center_width:>13} {frame_count:>12}")

        results[overlap_factor] = {
            "full_tiled": tile_frames(full_frames, TARGET_WIDTH),
            "hop_center_tiled": tile_frames(hop_center_frames, TARGET_WIDTH),
        }

    print()

    # ==========================================================================
    # FIGURE
    # ==========================================================================

    n_rows = len(OVERLAP_FACTORS)
    n_cols = 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(18, 4 * n_rows),
        constrained_layout=True,
    )

    col_titles = ["Full Reliable Region (current — redundant wings)", "Hop-Center Only (fix — unique content)"]

    for row_idx, overlap_factor in enumerate(OVERLAP_FACTORS):
        full_tiled = results[overlap_factor]["full_tiled"]
        hop_tiled = results[overlap_factor]["hop_center_tiled"]

        vmax = max(full_tiled.max(), hop_tiled.max())
        vmax = vmax if vmax > 0 else 1.0

        for col_idx, (data, title) in enumerate([(full_tiled, col_titles[0]), (hop_tiled, col_titles[1])]):
            ax = axes[row_idx][col_idx]
            ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(f"overlap={overlap_factor:.2f} — {title}", fontsize=9)
            ax.set_xlabel(f"Time columns ({data.shape[1]} total)")
            ax.set_ylabel("Frequency bin")

            # Draw frame boundaries as vertical lines
            frame_width = TARGET_WIDTH
            for f in range(1, NUM_FRAMES):
                ax.axvline(x=f * frame_width - 0.5, color="cyan", linewidth=0.6, alpha=0.6)

    fig.suptitle(
        "CWT Overlap Redundancy Diagnostic\n"
        "Cyan lines = frame boundaries. At overlap>0, full-region frames share content across boundaries.\n"
        "Hop-center frames tile cleanly.",
        fontsize=11,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120)
    plt.close(fig)

    print(f"Diagnostic figure saved to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    run_diagnostic()
