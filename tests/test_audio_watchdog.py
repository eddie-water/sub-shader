"""Tests for the AudioStream stall watchdog (E97).

A frozen audio clock (playback position stuck below the next chunk boundary)
must cause next_chunk() to raise AudioStreamStalledError instead of hanging
forever, while a live clock at/past the boundary returns the chunk normally.

Runs without conftest fixtures - invoke with --noconftest.
"""

import types

import pytest

from subshader.audio.audio_stream import AudioStream
from subshader.audio.player import AudioPlayer
from subshader.exceptions import AudioStreamStalledError

BOUNDARY = 57344
HOP_SIZE = 8192
FAST_TIMEOUT_S = 0.05

CHUNK_SENTINEL = object()


class FakeReader:
    """Minimal AudioReader stand-in with a mutable file_pos."""

    def __init__(self, file_pos: int, at_eof: bool = False) -> None:
        self.file_pos = file_pos
        self.at_eof = at_eof

    def get_chunk(self) -> object:
        return None if self.at_eof else CHUNK_SENTINEL


class FakePlayer:
    """Minimal AudioPlayer stand-in with a fixed playback position."""

    def __init__(self, playback_pos: int) -> None:
        self._playback_pos = playback_pos

    def get_playback_sample(self) -> int:
        return self._playback_pos

    def has_looped(self) -> bool:
        return False

    def clear_loop_event(self) -> None:
        return None

    def is_active(self) -> bool:
        return False


def _make_stream(playback_pos: int, at_eof: bool = False) -> AudioStream:
    """Build an AudioStream with injected fakes, bypassing __init__."""
    stream = AudioStream.__new__(AudioStream)
    stream._config = types.SimpleNamespace(hop_size=HOP_SIZE)
    stream._reader = FakeReader(file_pos=BOUNDARY, at_eof=at_eof)
    stream._player = FakePlayer(playback_pos=playback_pos)
    stream._stall_timeout_s = FAST_TIMEOUT_S
    return stream


def test_frozen_clock_raises_stalled_error():
    """A playback clock frozen below the boundary raises within the timeout."""
    stream = _make_stream(playback_pos=BOUNDARY - 154)

    with pytest.raises(AudioStreamStalledError):
        stream.next_chunk()


def test_live_clock_returns_chunk_without_raising():
    """A clock at/past the boundary returns the chunk with no false positive."""
    stream = _make_stream(playback_pos=BOUNDARY)

    chunk = stream.next_chunk()

    assert chunk is CHUNK_SENTINEL
    assert stream._reader.file_pos == (BOUNDARY // HOP_SIZE) * HOP_SIZE


def test_frozen_clock_at_eof_tail_raises_instead_of_spinning():
    """A frozen clock past the boundary but in the EOF tail (get_chunk None)
    must raise via the watchdog instead of busy-spinning forever.

    Regression: the original watchdog only guarded the pre-boundary wait
    branch; a clock frozen within chunk_size of EOF returned None instantly,
    making run() busy-spin with no sleep, no event pumping, and no watchdog.
    """
    stream = _make_stream(playback_pos=BOUNDARY + 100, at_eof=True)

    with pytest.raises(AudioStreamStalledError):
        stream.next_chunk()


class RecordingStream:
    """sd.OutputStream stand-in that records teardown calls."""

    def __init__(self) -> None:
        self.calls = []

    def abort(self, ignore_errors: bool = False) -> None:
        self.calls.append("abort")

    def stop(self, ignore_errors: bool = False) -> None:
        self.calls.append("stop")

    def close(self, ignore_errors: bool = True) -> None:
        self.calls.append("close")


def test_player_stop_aborts_instead_of_draining():
    """AudioPlayer.stop() must abort (discard pending buffers) rather than
    stop (wait for drain) - Pa_StopStream blocks unbounded on a stalled
    stream (observed 67 s teardown hang after AudioStreamStalledError).
    """
    player = AudioPlayer.__new__(AudioPlayer)
    fake_stream = RecordingStream()
    player._stream = fake_stream

    player.stop()

    assert "abort" in fake_stream.calls
    assert "stop" not in fake_stream.calls
    assert "close" in fake_stream.calls
    assert player._stream is None
