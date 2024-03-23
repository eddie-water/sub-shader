# Number of samples per frame
FRAME_SIZE = 4096

# Percentage of frame overlap
OVERLAP = 50.0

from .audio_input import AudioInput
from .fft_machine import FftMachine
import numpy as np

class Model:
    def __init__(self) -> None:
        self.audio_file = "models/audio_files/c_octaves.wav"

        self.audio_input = AudioInput(
            path = self.audio_file,
            frame_size = FRAME_SIZE,
            overlap = OVERLAP)

        self.fft_machine = FftMachine()

    # Sliding Discrete Fourier Transform
    def sdft(self) -> np.ndarray:
        audio_data = self.audio_input.get_next_frame()
        return self.fft_machine.process(audio_data)
