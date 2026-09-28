"""Shared test fixtures for SubShader tests."""

import pytest
import os
from pathlib import Path

from subshader.config import CWTConfig
from subshader.dsp.cwt import CpuCWT


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return str(Path(__file__).resolve().parents[3])


@pytest.fixture
def valid_audio_path():
    """Return path to a valid test audio file."""
    return "assets/audio/daw/a2a3_a4_minor_scale.wav"


@pytest.fixture
def cwt_config():
    """Return a CWTConfig with default parameters."""
    return CWTConfig(chunk_size=16384, sample_rate=44100.0)


@pytest.fixture
def cpu_cwt(cwt_config):
    """Return a CpuCWT with standard test parameters."""
    return CpuCWT(cwt_config)
