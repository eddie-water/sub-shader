# Number of samples per frame
FRAME_SIZE = 4096

from .audio_input import AudioInput
from .fft_machine import FftMachine
import numpy as np

class Model:
    def __init__(self) -> None:
        self.audio_file = "models/audio_files/c_octaves.wav"

        self.audio_input = AudioInput(
            path = self.audio_file,
            frame_size = FRAME_SIZE)

        self.fft_machine = FftMachine(
            frame_size = FRAME_SIZE,
            sample_rate = self.audio_input.sample_rate)

    # Sliding Discrete Fourier Transform
    def perform_sdft(self) -> np.ndarray:
        audio_data = self.audio_input.get_frame()
        return self.fft_machine.compute_fft(audio_data)
