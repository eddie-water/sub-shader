import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class FileOperations:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.sample_rate, self.data = wavfile.read(self.file_path)

        self.num_channels = self.data.shape[1]
        length = self.data.shape[0] / self.sample_rate

        time = np.linspace(0., length, self.data.shape[0])
        plt.plot(time, self.data[:, 0], label="Left channel")
        plt.plot(time, self.data[:, 1], label="Right channel")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

        print("Reading", self.file_path) 
        print(self.file_path, "has", self.num_channels, "channels")

    def get_chunk(self) -> int:
        pass