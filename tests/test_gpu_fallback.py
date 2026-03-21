"""Tests for GPU detection and fallback behavior (PIPE-02, PIPE-03)."""

import pytest
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


class TestGpuFallback:
    """Test that SubShader selects correct wavelet class based on GPU availability."""

    @patch('subshader.__main__.gpu_available', return_value=True)
    @patch('subshader.__main__.ShaderPlot')
    @patch('subshader.__main__.AudioInput')
    def test_gpu_available_selects_cu_wavelet(self, mock_audio, mock_plotter, mock_gpu):
        """When GPU is available, CuWavelet should be selected."""
        mock_audio_instance = MagicMock()
        mock_audio_instance.get_sample_rate.return_value = 44100.0
        mock_audio_instance.get_chunk_size.return_value = 16384
        mock_audio.return_value = mock_audio_instance

        mock_wavelet = MagicMock()
        mock_wavelet.get_output_shape.return_value = (120, 256)

        from subshader.config import get_default_config
        config = get_default_config()

        with patch('subshader.__main__.CuWavelet', return_value=mock_wavelet) as mock_cu:
            with patch('subshader.__main__.NpWavelet') as mock_np:
                from subshader.__main__ import SubShader
                sub = SubShader(config)
                mock_cu.assert_called_once()
                mock_np.assert_not_called()
                sub.cleanup()

    @patch('subshader.__main__.gpu_available', return_value=False)
    @patch('subshader.__main__.ShaderPlot')
    @patch('subshader.__main__.AudioInput')
    def test_gpu_unavailable_selects_np_wavelet(self, mock_audio, mock_plotter, mock_gpu):
        """When GPU is unavailable, NpWavelet should be selected."""
        mock_audio_instance = MagicMock()
        mock_audio_instance.get_sample_rate.return_value = 44100.0
        mock_audio_instance.get_chunk_size.return_value = 16384
        mock_audio.return_value = mock_audio_instance

        mock_wavelet = MagicMock()
        mock_wavelet.get_output_shape.return_value = (120, 256)

        from subshader.config import get_default_config
        config = get_default_config()

        with patch('subshader.__main__.NpWavelet', return_value=mock_wavelet) as mock_np:
            with patch('subshader.__main__.CuWavelet') as mock_cu:
                from subshader.__main__ import SubShader
                sub = SubShader(config)
                mock_np.assert_called_once()
                mock_cu.assert_not_called()
                sub.cleanup()

    @patch('subshader.__main__.gpu_available', return_value=False)
    @patch('subshader.__main__.ShaderPlot')
    @patch('subshader.__main__.AudioInput')
    def test_gpu_unavailable_logs_warning(self, mock_audio, mock_plotter, mock_gpu, caplog):
        """When GPU is unavailable, a warning should be logged."""
        import logging
        mock_audio_instance = MagicMock()
        mock_audio_instance.get_sample_rate.return_value = 44100.0
        mock_audio_instance.get_chunk_size.return_value = 16384
        mock_audio.return_value = mock_audio_instance

        mock_wavelet = MagicMock()
        mock_wavelet.get_output_shape.return_value = (120, 256)

        from subshader.config import get_default_config
        config = get_default_config()

        with patch('subshader.__main__.NpWavelet', return_value=mock_wavelet):
            with caplog.at_level(logging.WARNING):
                from subshader.__main__ import SubShader
                sub = SubShader(config)
                assert any("GPU unavailable" in msg for msg in caplog.messages)
                sub.cleanup()
