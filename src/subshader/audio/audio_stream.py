"""
AudioStream Facade Module for SubShader.

Wraps AudioReader (file I/O) and AudioPlayer (playback) behind a single
interface. AudioStream discovers sample_rate and total_samples from the
audio file on construction and writes them back into the shared PipelineConfig
so downstream stages (CWT, Renderer) see the correct runtime values.
"""

import time

from subshader.utils.logging import get_logger
from subshader.utils.timing import timed, timed_block
from subshader.config import PipelineConfig
from subshader.exceptions import AudioStreamStalledError
from .reader import AudioReader
from .player import AudioPlayer

log = get_logger(__name__)

# Seconds the playback clock may stay frozen on a chunk boundary before the
# stream is treated as stalled (e.g. ALSA underrun on the WSL2 audio bridge)
STREAM_STALL_TIMEOUT_S = 2.0


class AudioStream:
    """Facade wrapping audio file I/O and playback.

    Takes a PipelineConfig, discovers sample_rate and total_samples from the
    audio file (via AudioReader), and writes them back into config so downstream
    modules (CWT, Renderer) see the correct values.

    AudioReader is constructed first so that config.sample_rate is populated
    before AudioPlayer is constructed (AudioPlayer reads config.sample_rate for
    the OutputStream).
    """

    @timed
    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the audio stream from pipeline config.

        Constructs AudioReader (file open + config writeback) then AudioPlayer
        (uses the now-populated config.sample_rate). After __init__, config
        holds the correct sample_rate and total_samples for the full pipeline.

        Args:
            config: Shared pipeline config. file_path, chunk_size, and
                    overlap_factor are consumed. sample_rate and total_samples
                    are written back by AudioReader during construction.
        """
        self._config = config
        self._stall_timeout_s = STREAM_STALL_TIMEOUT_S
        # AudioReader opens the file and writes config.sample_rate + config.total_samples
        with timed_block(self, "audio_reader"):
            self._reader = AudioReader(config)
        # AudioPlayer reads the now-populated config.sample_rate
        with timed_block(self, "audio_player"):
            self._player = AudioPlayer(config)
        log.info("AudioStream initialized")

    def start(self) -> None:
        """Start audio playback."""
        self._player.start()

    @timed
    def get_chunk(self) -> object:
        """Return the next audio chunk at the current reader position.

        Non-blocking — returns None if there is no more audio data. The
        caller is responsible for checking None and handling end-of-file.

        Returns:
            np.ndarray[np.float64] or None: Mono audio chunk, or None at EOF.
        """
        return self._reader.get_chunk()

    @timed
    def next_chunk(self) -> object:
        """Block until the audio clock has advanced, then return the next chunk.

        Encapsulates the audio-clock sync logic that was previously in
        SubShader.loop(). The audio device clock (get_playback_sample()) is the
        single source of truth (D-06):

        - If the audio clock has not advanced past the next chunk boundary:
          yield 1ms and retry (avoids busy-wait, D-08).
        - If the render loop has fallen behind the audio clock: skip to the
          most recent chunk position (D-09, frame-skip).
        - If a loop wrap is detected: reset reader.file_pos to 0 and continue.
        - If no more audio data: return None (caller handles end-of-file).
        - If the audio clock stays frozen below the boundary for longer than
          self._stall_timeout_s: raise AudioStreamStalledError (stall watchdog).

        Returns:
            np.ndarray[np.float64] or None: Mono audio chunk aligned to the
                current audio clock position, or None at EOF.

        Raises:
            AudioStreamStalledError: If the playback clock has not advanced
                for self._stall_timeout_s seconds while waiting on a chunk
                boundary — typically an ALSA underrun on the WSL2 audio bridge.
        """
        hop_size = self._config.hop_size
        last_pos = self._player.get_playback_sample()
        stall_deadline = time.monotonic() + self._stall_timeout_s

        while True:
            # Handle loop wrap before any other checks (Pitfall 4)
            if self._player.has_looped():
                self._player.clear_loop_event()
                self._reader.file_pos = 0
                log.info("AudioStream: audio looped — reader reset to start")

            playback_pos = self._player.get_playback_sample()
            next_boundary = self._reader.file_pos

            if playback_pos < next_boundary:
                if playback_pos != last_pos:
                    # Clock is alive — reset the stall watchdog
                    last_pos = playback_pos
                    stall_deadline = time.monotonic() + self._stall_timeout_s
                elif time.monotonic() >= stall_deadline:
                    raise AudioStreamStalledError(
                        f"Audio stream stalled: audio clock frozen at sample "
                        f"{playback_pos} for {self._stall_timeout_s}s "
                        f"(stream active={self._player.is_active()}); likely an "
                        f"ALSA underrun on the WSL2 audio bridge — restart the "
                        f"app / audio device to recover"
                    )
                # Audio clock has not yet reached next chunk — yield briefly (D-08)
                time.sleep(0.001)
                continue

            # Audio has advanced: seek reader to match audio clock (D-06, D-09)
            # If multiple chunks were skipped, align to the most recent position
            target_sample = (playback_pos // hop_size) * hop_size
            self._reader.file_pos = target_sample

            chunk = self._reader.get_chunk()
            return chunk  # None signals EOF to caller

    def get_playback_sample(self) -> int:
        """Return the current playback position in samples."""
        return self._player.get_playback_sample()

    def has_looped(self) -> bool:
        """Return True if playback has crossed a loop boundary since last check.

        Note: next_chunk() clears the loop event internally. Callers using
        get_chunk() directly should call this and clear_loop_event() manually.
        """
        return self._player.has_looped()

    def clear_loop_event(self) -> None:
        """Clear the loop event after the caller has handled the loop reset."""
        self._player.clear_loop_event()

    def get_entire_audio(self) -> object:
        """Read the entire audio file as a single mono float64 array."""
        return self._reader.get_entire_audio()

    def cleanup(self) -> None:
        """Stop playback and close file handles. Safe to call multiple times."""
        if hasattr(self, '_player') and self._player is not None:
            try:
                self._player.stop()
            except Exception:
                pass

        if hasattr(self, '_reader') and self._reader is not None:
            try:
                self._reader.cleanup()
            except Exception:
                pass

        log.info("AudioStream cleaned up")
