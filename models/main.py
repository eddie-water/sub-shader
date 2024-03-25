from .audio_input import AudioInput
from .fft_machine import FftMachine
import numpy as np

class Model:
    def __init__(self, frame_size: int) -> None:
        self.frame_size = frame_size
        self.audio_file = "models/audio_files/zionsville.wav"

        self.audio_input = AudioInput(
            path = self.audio_file,
            frame_size = self.frame_size)

        self.fft_machine = FftMachine(
            frame_size = frame_size,
            sample_rate = self.audio_input.sample_rate)

    # Sliding Discrete Fourier Transform
    def perform_sdft(self) -> np.ndarray:
        audio_data = self.audio_input.get_frame()
        return self.fft_machine.compute_fft(audio_data)
