import numpy as np
import soundfile as sf

# Frane overlap 
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

    """
    Get Frame
        Returns: a block of audio whose size is specified by the window size
    """
    def get_frame(self) -> np.ndarray:
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

    """
    Get Entire Audio
        Returns: the entire audio file
    """
    def get_entire_audio(self) -> np.ndarray:
        with sf.SoundFile(self.file_path, 'r') as f:
            self.entire_file_size = f.frames

            f.seek(0)
            self.data = f.read(self.entire_file_size)
            self.data = self.data[:,0]
            return self.data        

    """
    Get Sample Rate
        Returns: the sample rate of the file
    """
    def get_sample_rate(self) -> int:
        return self.sample_rate

    """
    Audio File Cleanup
        Closes the file handle if it exists
    """
    def cleanup(self):
        if hasattr(self, 'file_handle'):
            self.file_handle.close()

    """
    Display File Information
        Prints information about the audio file
    """
    def _display_file_info(self) -> None:
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
