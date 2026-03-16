"""
Unit tests for SubShader components.

Currently includes:
  - NumPy vs CuPy CWT correctness verification
"""

import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from subshader.config import get_default_config
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import NumPyWavelet, CuPyWavelet

# =============================================================================
# CONSTANTS
# =============================================================================

BENCHMARKS_DIR   = "assets/images/benchmarks"
AUDIO_DEFAULT    = "assets/audio/daw/a2a3_a4_minor_scale.wav"

# =============================================================================
# GPU DETECTION
# =============================================================================

def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False

GPU_AVAILABLE = _gpu_available()

# =============================================================================
# NUMPY VS CUPY VERIFICATION
# =============================================================================

def verify_numpy_vs_cupy(save_figure: bool = True):
    """
    Verify that NumPyWavelet and CuPyWavelet produce numerically equivalent outputs.

    Checks:
      1. Global absolute / relative error statistics
      2. Per-frequency-bin Pearson correlation (all bins should be > 0.999)
      3. Optionally saves a 3-panel diff figure: NumPy | CuPy | abs(diff)

    Requires a GPU — exits early with a message if none is available.
    """
    if not GPU_AVAILABLE:
        print("[verify] Skipped — no GPU available. Reconnect GPU and re-run.")
        return

    config = get_default_config()
    config.audio.file_path = AUDIO_DEFAULT

    ai = AudioInput(
        path=config.audio.file_path,
        chunk_size=config.audio.chunk_size,
        overlap_factor=config.audio.overlap_factor,
    )
    sr   = ai.get_sample_rate()
    data = ai.get_chunk()

    npwt = NumPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)
    cpwt = CuPyWavelet(sample_rate=sr, input_n=config.audio.chunk_size, config=config.wavelet)

    print("\nVerifying NumPy vs CuPy CWT outputs …\n")

    np_coefs = npwt.cwt(data)
    cp_coefs = cpwt.cwt(data)

    diff      = np.abs(np_coefs - cp_coefs)
    max_val   = np_coefs.max() + 1e-12
    rel_error = diff.max() / max_val

    print(f"  shape          : {np_coefs.shape}")
    print(f"  max abs diff   : {diff.max():.2e}")
    print(f"  mean abs diff  : {diff.mean():.2e}")
    print(f"  rel error      : {rel_error:.2e}")

    # Per-frequency-bin Pearson correlation
    n_freq = np_coefs.shape[0]
    corrs  = np.array([
        np.corrcoef(np_coefs[i], cp_coefs[i])[0, 1]
        for i in range(n_freq)
    ])
    low_corr = np.where(corrs < 0.999)[0]
    print(f"  bins < 0.999 r : {len(low_corr)} / {n_freq}", end="")
    if len(low_corr):
        print(f"  ← bins: {low_corr[:10]}")
    else:
        print("  ✓")

    passed = rel_error < 1e-3 and len(low_corr) == 0
    print(f"\n  {'PASS ✓' if passed else 'FAIL ✗ — inspect diff figure'}")

    if save_figure:
        _save_diff_figure(np_coefs, cp_coefs, diff)


def _save_diff_figure(np_coefs, cp_coefs, diff):
    """Save 3-panel figure: NumPy | CuPy | abs(diff)."""
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("NumPy vs CuPy CWT — correctness check", fontsize=12)

    vmax = max(np_coefs.max(), cp_coefs.max())

    axes[0].imshow(np_coefs, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=vmax)
    axes[0].set_title("NumPyWavelet.cwt()")
    axes[0].set_ylabel("Frequency bin")
    axes[0].set_xlabel("Time bin")

    axes[1].imshow(cp_coefs, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=vmax)
    axes[1].set_title("CuPyWavelet.cwt()")
    axes[1].set_xlabel("Time bin")

    im = axes[2].imshow(diff, cmap="RdBu_r", aspect="auto", origin="lower")
    axes[2].set_title("abs(NumPy − CuPy)")
    axes[2].set_xlabel("Time bin")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="abs diff")

    fig.tight_layout()
    path = os.path.join(BENCHMARKS_DIR, "numpy_vs_cupy_diff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diff figure saved → {path}\n")


# =============================================================================
# RUN ALL
# =============================================================================

def run_all():
    """Run all unit tests."""
    print("\n=== Unit Tests ===\n")
    verify_numpy_vs_cupy()
    print("Unit tests complete.\n")


if __name__ == "__main__":
    run_all()
