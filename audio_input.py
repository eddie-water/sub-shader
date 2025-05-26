import numpy as np
import soundfile as sf

# Percentage of frame overlap
OVERLAP = 50.0

class AudioInput:
    def __init__(self, path: str, window_size: int) -> None:
        # File attributes
        self.file_path = path
        self.pos = 0

        # Sliding frame attributes
        self.window_size = window_size
        self.overlap = OVERLAP / 100.0
        self.slide_amount = int(self.window_size * self.overlap)

    """
    Get Frame
        Returns: a block of audio whose size is specified by the window size
    """
    def get_frame(self) -> np.ndarray:
        with sf.SoundFile(self.file_path, 'r') as f:
            # Read one frame's worth of audio sample
            f.seek(self.pos)
            self.data = f.read(self.window_size)
            self.data = self.data[:, 0]

            # Slide the frame 
            self.pos = f.tell() - self.slide_amount

        return self.data

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
        with sf.SoundFile(self.file_path, 'r') as f:
            return f.samplerate

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
