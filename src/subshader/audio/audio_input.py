import numpy as np
import soundfile as sf

# Frame overlap 
OVERLAP = 50.0

class AudioInput:
    def __init__(self, path: str, window_size: int) -> None:
        self.file_path = path
        self.window_size = window_size
        self.overlap = OVERLAP / 100.0
        self.slide_amount = int(self.window_size * self.overlap)
        
        # Keep file handle open to avoid reopening it every time
        self.file_handle = sf.SoundFile(self.file_path, 'r')
        self.sample_rate = self.file_handle.samplerate
        self.total_frames = self.file_handle.frames
        self.pos = 0

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
            
        self.pos += self.slide_amount
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
            Prints information about the audio file
        """
        with sf.SoundFile(self.file_path, 'r') as f:
            print("Information about the file:", self.file_path)
            print("mode", f.mode)
            print("samplerate", f.samplerate)
            print("frames", f.frames)
            print("channels", f.channels)
            print("format", f.format)
            print("subtype", f.subtype)
            print("format info", f.format_info)
            print("extra info", f.extra_info)
            print("seekable()", f.seekable())
