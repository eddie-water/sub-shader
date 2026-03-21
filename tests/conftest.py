"""Shared test fixtures for SubShader tests."""

import pytest
import os


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def valid_audio_path():
    """Return path to a valid test audio file."""
    return "assets/audio/daw/a2a3_a4_minor_scale.wav"
