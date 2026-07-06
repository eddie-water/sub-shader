"""
Audio Reader Module for SubShader.

Handles audio file I/O and overlapping chunk extraction for the pipeline.
Accepts a PipelineConfig and writes back discovered runtime values
(sample_rate, total_samples) so downstream stages see the correct values.
"""

import os

import numpy as np
import soundfile as sf

from subshader.utils.logging import get_logger
from subshader.exceptions import AudioFileNotFoundError
from subshader.config import PipelineConfig

log = get_logger(__name__)


class AudioReader:
    """Reads overlapping chunks from an audio file.

    Accepts a PipelineConfig, opens the audio file, and writes the discovered
    sample_rate and total_samples back to config so downstream stages (CWT,
    Renderer) see the correct runtime values without needing separate discovery.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize AudioReader from pipeline config.

        Opens the audio file, discovers sample_rate and total_samples, and
        writes them back into config. All subsequent reads use hop-based
        seeking for overlapping window extraction.

        Args:
            config: Shared pipeline config. file_path, chunk_size, and
                    overlap_factor are read. sample_rate and total_samples
                    are written back after file open.

        Raises:
            AudioFileNotFoundError: If config.file_path does not exist.
        """
        self._config = config
        self.file_path = config.file_path
        self.chunk_size = config.chunk_size
        self.overlap_factor = config.overlap_factor
        self.hop_size = config.hop_size

        if not os.path.exists(self.file_path):
            log.error(f"Audio file not found: {self.file_path}")
            raise AudioFileNotFoundError(f"Audio file not found: {self.file_path}")

        try:
            self._sf = sf.SoundFile(self.file_path, 'r')
            # Discover runtime values and write back to shared config (D-06)
            config.sample_rate = float(self._sf.samplerate)
            config.total_samples = self._sf.frames
            self.file_pos = 0
            log.info(
                f"Audio file loaded: {self.file_path} "
                f"({config.total_samples} frames, {config.sample_rate} Hz)"
            )
            log.info(
                f"Window size: {self.chunk_size}, "
                f"Overlap: {self.overlap_factor:.1%}, "
                f"Hop size: {self.hop_size}"
            )
        except AudioFileNotFoundError:
            raise
        except Exception as e:
            log.error(f"Failed to load audio file {self.file_path}: {e}")
            raise

    @property
    def sample_rate(self) -> float:
        """Sample rate of the audio file in Hz."""
        return self._config.sample_rate

    @property
    def total_samples(self) -> int:
        """Total number of samples in the audio file."""
        return self._config.total_samples

    def has_data(self) -> bool:
        """Return True if there is at least one full chunk remaining to read."""
        return self.file_pos + self.chunk_size <= self._config.total_samples

    def get_chunk(self) -> np.ndarray | None:
        """Retrieve the next overlapping chunk of audio samples.

        Returns None if fewer than chunk_size samples remain (end of file).
        Advances file_pos by hop_size on each call for overlapping windows.

        Returns:
            np.ndarray[np.float64]: Mono audio chunk of length chunk_size,
                                    or None if end of file reached.
        """
        if not self.has_data():
            return None

        self._sf.seek(self.file_pos)
        chunk = self._sf.read(self.chunk_size, dtype=np.float64)
        self.file_pos += self.hop_size

        # Convert stereo to mono (first channel)
        if chunk.ndim > 1:
            chunk = chunk[:, 0]

        return chunk

    def get_chunk_size(self) -> int:
        """Number of samples per chunk."""
        return self.chunk_size

    def get_sample_rate(self) -> float:
        """Sample rate in Hz as discovered from the audio file."""
        return self._config.sample_rate

    def get_total_samples(self) -> int:
        """Total number of samples in the audio file."""
        return self._config.total_samples

    def get_entire_audio(self) -> np.ndarray:
        """Read the entire audio file as a single mono float64 array.

        Returns:
            np.ndarray[np.float64]: Full audio buffer, mono.
        """
        self._sf.seek(0)
        audio_data = self._sf.read(dtype=np.float64)

        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        return audio_data

    def cleanup(self) -> None:
        """Close the audio file handle."""
        if hasattr(self, '_sf') and self._sf is not None:
            self._sf.close()

    def _report_file_info(self) -> None:
        """Log detailed information about the audio file."""
        with sf.SoundFile(self.file_path, 'r') as f:
            log.info(f"Audio file: {self.file_path}")
            log.debug(f"Mode: {f.mode}")
            log.debug(f"Sample rate: {f.samplerate} Hz")
            log.debug(f"Frames: {f.frames}")
            log.debug(f"Channels: {f.channels}")
            log.debug(f"Format: {f.format}")
            log.debug(f"Subtype: {f.subtype}")
            log.debug(f"Format info: {f.format_info}")
            log.debug(f"Extra info: {f.extra_info}")
            log.debug(f"Seekable: {f.seekable()}")
