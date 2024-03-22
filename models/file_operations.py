import numpy as np
import soundfile as sf

# TODO figure out where to maintain FRAME_SIZE and CHUNK_SIZE in the view
# TODO be consistent, let's pick what to call it frames, chunks, window, etc
FRAME_SIZE = 4096

# TODO HALF_FRAME when implementing sliding window

class FileOperations:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.pos = 0

    def display_file_info(self) -> None:
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

    def get_chunk(self) -> np.ndarray:
        with sf.SoundFile(self.file_path, 'r') as f:
            f.seek(self.pos)
            self.data = f.read(FRAME_SIZE)
            self.pos = f.tell()
            self.data = self.data[:, 0]

        return self.data
