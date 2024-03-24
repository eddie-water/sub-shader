# https://github.com/OmarAlkousa/Fourier-Analysis-as-Streamlit-Web-App/tree/main

import numpy as np
import scipy

class Fourier:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.time_step = 1.0 / self.sample_rate

    def compute_amplitude(self, signal: np.ndarray) -> np.ndarray:
        self.frequencies = scipy.fft.rfftfreq(
            len(signal), 
            d = self.time_step)

        self.fourier = scipy.fft.rfft(signal)
        self.amplitudes = 2*np.abs(self.fourier)/len(signal)

        size = self.amplitudes.size - 1
        return self.amplitudes[0:size]
