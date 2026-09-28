"""Tests for GPU detection and fallback behavior (PIPE-02, PIPE-03)."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestGpuAvailable:
    """Test gpu_available() utility function."""

    def test_returns_bool(self):
        from subshader.utils.gpu import gpu_available
        assert isinstance(gpu_available(), bool)

    def test_returns_false_when_cupy_unavailable(self):
        with patch.dict('sys.modules', {'cupy': None}):
            # Force reimport to pick up the mock
            import importlib
            from subshader.utils import gpu
            importlib.reload(gpu)
            assert gpu.gpu_available() is False
            # Restore
            importlib.reload(gpu)


def _make_mock_audio_stream(sample_rate: float = 44100.0) -> MagicMock:
    """Build a mock AudioStream whose constructor writes runtime config values."""
    mock_stream = MagicMock()

    def constructor(config):
        config.sample_rate = sample_rate
        config.total_samples = 16384 * 4
        return mock_stream

    return constructor, mock_stream


def _make_mock_dsp() -> MagicMock:
    """Build a mock CWT backend with a real output shape."""
    mock_dsp = MagicMock()
    mock_dsp.get_output_shape.return_value = (116, 64)
    return mock_dsp


def _build_pipeline(gpu: bool):
    """Construct SubShader with all heavyweight stages mocked except backend choice."""
    from subshader.config import CWTConfig
    from subshader.pipeline import SubShader

    constructor, mock_stream = _make_mock_audio_stream()
    config = CWTConfig()

    with patch('subshader.pipeline.gpu_available', return_value=gpu), \
         patch('subshader.pipeline.AudioStream', side_effect=constructor), \
         patch('subshader.pipeline.Renderer'), \
         patch('subshader.pipeline.GpuCWT', return_value=_make_mock_dsp()) as mock_gpu_cwt, \
         patch('subshader.pipeline.CpuCWT', return_value=_make_mock_dsp()) as mock_cpu_cwt, \
         patch.object(SubShader, '_prescan_intensity', return_value=1.0), \
         patch.object(SubShader, '_prime_pipeline'):
        shader = SubShader(config)
        return shader, mock_gpu_cwt, mock_cpu_cwt


class TestGpuFallback:
    """SubShader must select the CWT backend matching GPU availability."""

    def test_gpu_available_selects_gpu_cwt(self):
        """When GPU is available, GpuCWT should be constructed."""
        shader, mock_gpu_cwt, mock_cpu_cwt = _build_pipeline(gpu=True)
        mock_gpu_cwt.assert_called_once()
        mock_cpu_cwt.assert_not_called()

    def test_gpu_unavailable_selects_cpu_cwt(self):
        """When GPU is unavailable, CpuCWT should be constructed."""
        shader, mock_gpu_cwt, mock_cpu_cwt = _build_pipeline(gpu=False)
        mock_cpu_cwt.assert_called_once()
        mock_gpu_cwt.assert_not_called()

    def test_gpu_unavailable_logs_warning(self, caplog):
        """When GPU is unavailable, a warning should be logged."""
        import logging
        with caplog.at_level(logging.WARNING):
            _build_pipeline(gpu=False)
        assert any("GPU unavailable" in msg for msg in caplog.messages)
