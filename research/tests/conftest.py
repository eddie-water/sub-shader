"""Shared test helpers for SubShader pytest suite."""
import numpy as np
from subshader.config import CWTConfig


def generate_tone(freq_hz, sample_rate, num_samples):
    """Generate a pure sine tone at a given frequency."""
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)


def find_peak_bin(cwt_output, freqs):
    """Find the frequency bin with the highest mean energy.

    Handles both real-valued (post-processed) and complex (raw CWT) outputs by
    taking the absolute value before computing mean energy. This allows comparing
    PywaveletCWT (which returns raw complex coefficients from its stub post()) and
    CpuCWT (which returns real-valued magnitude output).
    """
    mean_energy = np.mean(np.abs(cwt_output), axis=1)
    peak_bin = np.argmax(mean_energy)
    return peak_bin, freqs[peak_bin]


def make_test_config():
    """Create a CWTConfig for tests without requiring a valid audio file.

    Tests that don't use audio file I/O should use this instead of
    get_default_config() to avoid file-existence validation errors.
    """
    # CWTConfig() uses default values; file_path not validated until validate() is called
    return CWTConfig()


def _make_wavelet(cls, config=None):
    """Construct a CWT instance with default config."""
    if config is None:
        config = make_test_config()
    # config is already a CWTConfig with sample_rate and chunk_size embedded
    return cls(config)
