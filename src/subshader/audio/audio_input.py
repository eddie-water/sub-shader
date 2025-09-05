"""
Audio Input Module for SubShader.

This module handles audio file processing and frame extraction for real-time
visualization:
 - Loads audio files using soundfile with format detection
 - Implements overlapping window extraction to reduce edge artifacts
 - Supports configurable chunk sizes and overlap factors
 - Provides graceful error handling for file operations
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os

import numpy as np
import soundfile as sf

from subshader.utils.logging import get_logger

# =============================================================================
# LOGGING
# =============================================================================

log = get_logger(__name__)

# =============================================================================
# EXCEPTIONS
# =============================================================================

class AudioFileNotFoundError(Exception):
    """Raised when the audio file cannot be found."""
    def __init__(self, message):
        super().__init__(message)
        self.log_level = "error"
        self.log_message = f"Audio file error: {message}"


class EndOfAudioException(Exception):
    """Raised when the audio file has been completely processed."""
    def __init__(self, message="Audio file processing complete"):
        super().__init__(message)
        self.log_level = "warning"
        self.log_message = f"Graceful exit: {message}"

# =============================================================================
# AUDIO INPUT CLASS
# =============================================================================

class AudioInput:
    def __init__(self, path: str, chunk_size: int, overlap_factor: float = 0.5) -> None:
        """
        Audio Input Initialization

        Args:
            path (str): Path to the audio file.
            chunk_size (int): Size of the audio frame in samples.
            overlap_factor (float): Overlap between consecutive windows (0.0 to 
                1.0). 0.5 means 50% overlap to reduce edge artifacts.
        """
        self.file_path = path
        self.chunk_size = chunk_size
        self.overlap_factor = overlap_factor
        self.hop_size = int(chunk_size * (1.0 - self.overlap_factor))
        
        # Check if file exists before trying to open
        if not os.path.exists(self.file_path):
            log.error(f"Audio file not found: {self.file_path}")
            raise AudioFileNotFoundError(f"Audio file not found: {self.file_path}")
        
        # Keep file handle open to avoid reopening it every time
        try:
            self.file_handle = sf.SoundFile(self.file_path, 'r')
            self.sample_rate = self.file_handle.samplerate
            self.total_frames = self.file_handle.frames
            self.pos = 0
            log.info(f"Audio file loaded: {self.file_path} ({self.total_frames} frames, {self.sample_rate} Hz)")
            log.info(f"Window size: {self.chunk_size}, Overlap: {self.overlap_factor:.1%}, Hop size: {self.hop_size}")
        except Exception as e:
            log.error(f"Failed to load audio file {self.file_path}: {e}")
            raise

    def get_chunk(self) -> np.ndarray:
        """
        Gets an overlapping frame of audio to reduce edge artifacts.
        
        Uses overlapping windows where each new frame advances by hop_size
        instead of the full window_size, providing better continuity for
        wavelet analysis and reducing cone of influence artifacts.

        Returns:
            np.ndarray: The next frame of audio data from the file.
                       None if EOF reached.
        """
        if self.pos + self.chunk_size > self.total_frames:
            return None  # Signal EOF
        
        # Seek and read (file stays open)
        self.file_handle.seek(self.pos)
        frame = self.file_handle.read(self.chunk_size)
        
        # Convert stereo to mono 
        if len(frame.shape) > 1:
            frame = frame[:, 0]
            
        # Advance by hop_size instead of full window_size for overlap
        self.pos += self.hop_size
        return frame

    def get_sample_rate(self) -> int:
        """
        Gets the sample rate of the audio file.

        Returns:
            int: Sample rate
        """
        return self.sample_rate

    def cleanup(self):
        """
        Audio File Cleanup
            Closes the file handle if it exists
        """
        if hasattr(self, 'file_handle'):
            self.file_handle.close()

    def _display_file_info(self) -> None:
        """
        Display File Information
            Logs information about the audio file
        """
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
