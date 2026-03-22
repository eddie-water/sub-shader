"""
Audio Player Module for SubShader.

Handles real-time audio playback via sounddevice OutputStream with a thread-safe
sample counter that serves as the timing reference for the render loop.
"""

import threading

import numpy as np
import sounddevice as sd

from subshader.utils.logging import get_logger
from subshader.exceptions import SubShaderException

log = get_logger(__name__)


class AudioPlayer:
    """Plays audio from an in-memory float32 array via sounddevice OutputStream.

    The audio device clock is the single source of truth for timing (D-06).
    The callback runs on a dedicated OS thread and increments _current_frame.
    The main thread reads get_playback_sample() to know which CWT chunk to render.
    """

    def __init__(self, audio_data: np.ndarray, sample_rate: float) -> None:
        """
        Initialize AudioPlayer with pre-loaded audio data.

        Args:
            audio_data: Audio samples, any float dtype, mono or stereo.
                        Stereo is converted to mono (first channel).
                        Stored internally as float32 (D-02, Pitfall 1).
            sample_rate: Sample rate in Hz (e.g. 44100.0).

        Raises:
            SubShaderException: If audio_data is empty.
        """
        if audio_data.size == 0:
            raise SubShaderException("AudioPlayer received empty audio data")

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # Store as float32 — PortAudio does not support float64 natively (Pitfall 1)
        self._data = audio_data.astype(np.float32)
        self._sample_rate = float(sample_rate)
        self._current_frame = 0
        self._lock = threading.Lock()
        self._loop_event = threading.Event()
        self._stream: sd.OutputStream | None = None

        log.info(f"AudioPlayer initialized: {len(self._data)} samples, {self._sample_rate} Hz")

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice OutputStream callback. Runs on a dedicated OS thread.

        Fills outdata from the in-memory audio array. When the end of the
        buffer is reached, wraps to the beginning for seamless looping (D-11).

        CRITICAL: No I/O, no logging, no contended locks in this function.
        The lock protecting _current_frame is held only for a single int read/write.
        """
        if status:
            # status reports xruns — cannot log here safely, just note it
            pass

        with self._lock:
            start = self._current_frame

        end = start + frames
        total = len(self._data)

        if end >= total:
            # Wrap: fill remainder from start of buffer (D-11 seamless loop)
            first_part = self._data[start:]
            remaining = frames - len(first_part)
            outdata[:len(first_part), 0] = first_part
            outdata[len(first_part):, 0] = self._data[:remaining]
            with self._lock:
                self._current_frame = remaining
            self._loop_event.set()
        else:
            outdata[:, 0] = self._data[start:end]
            with self._lock:
                self._current_frame = end

    def get_playback_sample(self) -> int:
        """Return the current playback position in samples.

        Thread-safe: called from the main render thread while the callback
        runs on the audio thread. Uses threading.Lock (Pitfall 2).
        """
        with self._lock:
            return self._current_frame

    def has_looped(self) -> bool:
        """Check whether the playback has crossed a loop boundary.

        Returns True if _loop_event is set (i.e. a wrap occurred since last
        call to clear_loop_event). The render loop uses this to reset
        AudioInput.file_pos (Pitfall 4).
        """
        return self._loop_event.is_set()

    def clear_loop_event(self) -> None:
        """Clear the loop event after the render loop has handled the reset."""
        self._loop_event.clear()

    def start(self) -> None:
        """Start audio playback.

        Opens a sounddevice OutputStream with blocksize=0 (optimal hardware
        buffer) and latency='low' (D-03, AUDIO-02).

        Raises:
            SubShaderException: If audio device is unavailable (Pitfall 6).
        """
        try:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype='float32',
                blocksize=0,
                latency='low',
                callback=self._callback,
            )
            self._stream.start()
            log.info("Audio playback started")
        except Exception as e:
            raise SubShaderException(
                f"Failed to start audio playback: {e}. "
                f"Check that an audio output device is available."
            ) from e

    def stop(self) -> None:
        """Stop audio playback and release the device (D-12).

        Safe to call multiple times. Called from SubShader.cleanup().
        """
        if self._stream is not None:
            try:
                self._stream.stop(ignore_errors=True)
                self._stream.close()
            except Exception:
                pass  # Best-effort cleanup
            finally:
                self._stream = None
            log.info("Audio playback stopped")
