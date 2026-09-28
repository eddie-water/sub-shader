"""Tests for AudioPlayer class (AUDIO-01, AUDIO-02)."""

import numpy as np
import pytest
import soundfile as sf
from unittest.mock import patch, MagicMock

from subshader.audio.player import AudioPlayer
from subshader.config import PipelineConfig
from subshader.exceptions import SubShaderException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mono_float64(n_samples: int = 44100, sample_rate: float = 44100.0):
    """Return a 1D float64 sine wave array at 440 Hz."""
    t = np.linspace(0, 1, n_samples, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float64), sample_rate


def _make_stereo_float64(n_samples: int = 44100, sample_rate: float = 44100.0):
    """Return a 2D stereo float64 array of shape (n_samples, 2)."""
    mono, sr = _make_mono_float64(n_samples, sample_rate)
    stereo = np.stack([mono, mono], axis=1)  # shape (n_samples, 2)
    return stereo, sr


def _make_player(tmp_path, data, sample_rate: float = 44100.0) -> AudioPlayer:
    """Write data to a temp WAV and construct an AudioPlayer from config."""
    wav_path = tmp_path / "test_tone.wav"
    sf.write(wav_path, data, int(sample_rate))
    config = PipelineConfig(file_path=str(wav_path), sample_rate=sample_rate)
    return AudioPlayer(config)


def _make_outdata(frames: int) -> np.ndarray:
    """Return a zeroed output buffer as sounddevice would provide (frames, 1)."""
    return np.zeros((frames, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_loads_audio_as_float32(tmp_path):
    """AudioPlayer stores audio data as np.float32 internally."""
    data, sr = _make_mono_float64()
    player = _make_player(tmp_path, data, sr)

    assert player._data.dtype == np.float32


def test_init_mono_conversion(tmp_path):
    """Stereo audio is converted to mono (1D) during initialization."""
    data, sr = _make_stereo_float64()
    player = _make_player(tmp_path, data, sr)

    assert player._data.ndim == 1


def test_get_playback_sample_initial(tmp_path):
    """get_playback_sample() returns 0 before start() is called."""
    data, sr = _make_mono_float64()
    player = _make_player(tmp_path, data, sr)

    assert player.get_playback_sample() == 0


def test_callback_advances_position(tmp_path):
    """Calling _callback with frames=1024 advances _current_frame by 1024."""
    data, sr = _make_mono_float64(n_samples=44100)
    player = _make_player(tmp_path, data, sr)

    frames = 1024
    outdata = _make_outdata(frames)
    player._callback(outdata, frames, time_info=None, status=None)

    assert player.get_playback_sample() == frames


def test_callback_loop_wraps_position(tmp_path):
    """When callback reaches end of buffer, _current_frame wraps to remainder."""
    n_samples = 44100
    data, sr = _make_mono_float64(n_samples=n_samples)
    player = _make_player(tmp_path, data, sr)

    frames = 1024
    # Position the frame pointer near the end - 512 samples before EOF
    player._current_frame = n_samples - 512

    outdata = _make_outdata(frames)
    player._callback(outdata, frames, time_info=None, status=None)

    # 1024 frames requested, 512 from end + 512 wrapped from beginning
    assert player.get_playback_sample() == 512


def test_callback_loop_sets_event(tmp_path):
    """When callback wraps, _loop_event is set."""
    n_samples = 44100
    data, sr = _make_mono_float64(n_samples=n_samples)
    player = _make_player(tmp_path, data, sr)

    frames = 1024
    player._current_frame = n_samples - 512

    outdata = _make_outdata(frames)
    player._callback(outdata, frames, time_info=None, status=None)

    assert player._loop_event.is_set()


def test_stop_closes_stream(tmp_path):
    """After stop(), _stream is set to None."""
    data, sr = _make_mono_float64()
    player = _make_player(tmp_path, data, sr)

    mock_stream = MagicMock()
    with patch("sounddevice.OutputStream", return_value=mock_stream):
        player.start()
        player.stop()

    assert player._stream is None


def test_invalid_empty_data_raises(tmp_path):
    """An audio file with zero frames raises SubShaderException."""
    wav_path = tmp_path / "empty.wav"
    sf.write(wav_path, np.array([], dtype=np.float32), 44100)
    config = PipelineConfig(file_path=str(wav_path), sample_rate=44100.0)

    with pytest.raises(SubShaderException):
        AudioPlayer(config)
