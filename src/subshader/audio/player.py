"""
Audio Player Module for SubShader.

Handles real-time audio playback via sounddevice OutputStream with a thread-safe
sample counter that serves as the timing reference for the render loop.

Accepts a PipelineConfig — reads file_path and sample_rate from config after
AudioReader has written the discovered sample_rate back to config.
"""

import threading

import numpy as np
import soundfile as sf
import sounddevice as sd

from subshader.utils.logging import get_logger
from subshader.exceptions import SubShaderException, AudioFileNotFoundError
from subshader.config import PipelineConfig

log = get_logger(__name__)


class AudioPlayer:
    """Plays audio from an in-memory float32 array via sounddevice OutputStream.

    The audio device clock is the single source of truth for timing (D-06).
    The callback runs on a dedicated OS thread and increments _current_frame.
    The main thread reads get_playback_sample() to know which CWT chunk to render.

    Accepts a PipelineConfig. Reads file_path to load audio data and sample_rate
    for the output stream. sample_rate must have been written into config by
    AudioReader before AudioPlayer is constructed.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize AudioPlayer from pipeline config.

        Loads the full audio file into memory as float32 and configures the
        sounddevice OutputStream. Requires config.sample_rate to be set (written
        by AudioReader on construction).

        Args:
            config: Shared pipeline config. file_path and sample_rate are read.

        Raises:
            AudioFileNotFoundError: If config.file_path does not exist.
            SubShaderException: If the audio file is empty.
        """
        import os
        if not os.path.exists(config.file_path):
            raise AudioFileNotFoundError(f"Audio file not found: {config.file_path}")

        # Load full audio into memory for low-latency callback playback (D-02)
        with sf.SoundFile(config.file_path, 'r') as f:
            audio_data = f.read(dtype=np.float64)

        if audio_data.size == 0:
            raise SubShaderException("AudioPlayer received empty audio data")

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # Store as float32 — PortAudio does not support float64 natively (Pitfall 1)
        self._data = audio_data.astype(np.float32)
        self._sample_rate = float(config.sample_rate)
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

    def is_active(self) -> bool:
        """Report whether the OutputStream is currently running.

        Returns False if the stream never started or aborted (e.g. after an
        ALSA underrun). Cheap corroborating signal for the stall watchdog.
        """
        return self._stream is not None and self._stream.active

    def has_looped(self) -> bool:
        """Check whether the playback has crossed a loop boundary.

        Returns True if _loop_event is set (i.e. a wrap occurred since last
        call to clear_loop_event). The render loop uses this to reset
        file_pos (Pitfall 4).
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

        Safe to call multiple times. Called from AudioStream.cleanup().
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
