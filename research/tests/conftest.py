"""Shared test helpers for SubShader pytest suite."""
import numpy as np
from subshader.config import get_default_config


def generate_tone(freq_hz, sample_rate, num_samples):
    """Generate a pure sine tone at a given frequency."""
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)


def find_peak_bin(cwt_output, freqs):
    """Find the frequency bin with the highest mean energy."""
    mean_energy = np.mean(cwt_output, axis=1)
    peak_bin = np.argmax(mean_energy)
    return peak_bin, freqs[peak_bin]


def _make_wavelet(cls, config=None):
    """Construct a wavelet instance with default config."""
    if config is None:
        config = get_default_config()
    sr = int(config.wavelet.typical_sampling_freq)
    chunk = config.audio.chunk_size
    return cls(sample_rate=sr, input_n=chunk, config=config.wavelet)
