"""Shared test fixtures for SubShader tests."""

import pytest
import os

from subshader.config import WaveletConfig
from subshader.dsp.wavelet import NpWavelet


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def valid_audio_path():
    """Return path to a valid test audio file."""
    return "assets/audio/daw/a2a3_a4_minor_scale.wav"


@pytest.fixture
def wavelet_config():
    """Return a WaveletConfig with default parameters."""
    return WaveletConfig()


@pytest.fixture
def numpy_wavelet(wavelet_config):
    """Return an NpWavelet with standard test parameters."""
    return NpWavelet(sample_rate=44100, input_n=16384, config=wavelet_config)
