import numpy as np

class FileOperations:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.sample_rate, self.data = wavfile.read(self.file_path)

        self.num_channels = self.data.shape[1]
        length = self.data.shape[0] / self.sample_rate

        time = np.linspace(0., length, self.data.shape[0])

        print("Reading", self.file_path) 
        print(self.file_path, "has", self.num_channels, "channels")

    def get_chunk(self) -> int:
        pass