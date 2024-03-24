import numpy as np
from .fourier import Fourier

class FftMachine:
    def __init__(self, frame_size: int, sample_rate: int):
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        self.window = np.hanning(frame_size)
        self.fourier = Fourier(sample_rate = self.sample_rate)

    def compute_fft(self, data: np.ndarray) -> np.ndarray:
        return self.fourier.compute_amplitude(self.window * data)