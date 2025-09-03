import numpy as np
import soundfile as sf
import os
from subshader.utils.logging import get_logger

log = get_logger(__name__)


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


class AudioInput:
    def __init__(self, path: str, audio_window_size: int, overlap_factor: float = 0.5) -> None:
        """
        Audio Input Initialization

        Args:
            path (str): Path to the audio file.
            audio_window_size (int): Size of the audio frame in samples.
            overlap_factor (float): Overlap between consecutive windows (0.0 to 1.0).
                                  0.5 means 50% overlap to reduce edge artifacts.
        """
        self.audio_file_path = path
        self.audio_window_size = audio_window_size
        self.overlap_factor = max(0.0, min(0.9, overlap_factor))  # Clamp to reasonable range
        self.hop_size = int(audio_window_size * (1.0 - self.overlap_factor))
        
        # Check if file exists before trying to open
        if not os.path.exists(self.audio_file_path):
            log.error(f"Audio file not found: {self.audio_file_path}")
            raise AudioFileNotFoundError(f"Audio file not found: {self.audio_file_path}")
        
        # Keep file handle open to avoid reopening it every time
        try:
            self.file_handle = sf.SoundFile(self.audio_file_path, 'r')
            self.sample_rate = self.file_handle.samplerate
            self.total_frames = self.file_handle.frames
            self.pos = 0
            log.info(f"Audio file loaded: {self.audio_file_path} ({self.total_frames} frames, {self.sample_rate} Hz)")
            log.info(f"Window size: {self.audio_window_size}, Overlap: {self.overlap_factor:.1%}, Hop size: {self.hop_size}")
        except Exception as e:
            log.error(f"Failed to load audio file {self.audio_file_path}: {e}")
            raise

    def get_frame(self) -> np.ndarray:
        """
        Gets an overlapping frame of audio to reduce edge artifacts.
        
        Uses overlapping windows where each new frame advances by hop_size
        instead of the full window_size, providing better continuity for
        wavelet analysis and reducing cone of influence artifacts.

        Returns:
            np.ndarray: The next frame of audio data from the file.
                       None if EOF reached.
        """
        if self.pos + self.audio_window_size > self.total_frames:
            return None  # Signal EOF
        
        # Seek and read (file stays open)
        self.file_handle.seek(self.pos)
        frame = self.file_handle.read(self.audio_window_size)
        
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
        with sf.SoundFile(self.audio_file_path, 'r') as f:
            log.info(f"Audio file: {self.audio_file_path}")
            log.debug(f"Mode: {f.mode}")
            log.debug(f"Sample rate: {f.samplerate} Hz")
            log.debug(f"Frames: {f.frames}")
            log.debug(f"Channels: {f.channels}")
            log.debug(f"Format: {f.format}")
            log.debug(f"Subtype: {f.subtype}")
            log.debug(f"Format info: {f.format_info}")
            log.debug(f"Extra info: {f.extra_info}")
            log.debug(f"Seekable: {f.seekable()}")
