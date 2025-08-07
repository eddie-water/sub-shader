import numpy as np
import soundfile as sf
from subshader.utils.logging import get_logger

log = get_logger(__name__)

# Frame overlap 
OVERLAP = 50.0

class AudioInput:
    def __init__(self, path: str, window_size: int) -> None:
        """
        Audio Input Initialization

        Args:
            path (str): Path to the audio file.
            window_size (int): Size of the audio frame in samples.
        """
        self.file_path = path
        self.window_size = window_size
        
        # Keep file handle open to avoid reopening it every time
        try:
            self.file_handle = sf.SoundFile(self.file_path, 'r')
            self.sample_rate = self.file_handle.samplerate
            self.total_frames = self.file_handle.frames
            self.pos = 0
            log.info(f"Audio file loaded: {self.file_path} ({self.total_frames} frames, {self.sample_rate} Hz)")
        except Exception as e:
            log.error(f"Failed to load audio file {self.file_path}: {e}")
            raise

    def get_frame(self) -> np.ndarray:
        """
        Gets a frame of audio the size of the window.

        Returns:
            np.ndarray: The next frame of audio data from the file.
        """
        if self.pos + self.window_size > self.total_frames:
            return None  # Signal EOF
        
        # Seek and read (file stays open)
        self.file_handle.seek(self.pos)
        frame = self.file_handle.read(self.window_size)
        
        # Convert stereo to mono if needed
        if len(frame.shape) > 1:
            frame = frame[:, 0]
            
        self.pos += self.window_size
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
