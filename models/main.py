from .audio_input import AudioInput
from .fft_machine import FftMachine
import numpy as np

FRAME_SIZE = 4096
FILE_PATH = "models/audio_files/zionsville.wav"

class Model:
    def __init__(self) -> None:
        self.frame_size = FRAME_SIZE
        self.audio_file = FILE_PATH

        self.audio_input = AudioInput(
            path = self.audio_file,
            frame_size = self.frame_size)

        self.sample_rate = self.audio_input.sample_rate

        self.fft_machine = FftMachine(
            frame_size = self.frame_size,
            sample_rate = self.sample_rate)

    def get_config_data(self) -> dict:
        return dict(
            song_name = self.audio_file.split("/")[-1].removesuffix(".wav"),
            frame_size = self.frame_size,
            sample_rate = self.sample_rate)

    # Sliding Discrete Fourier Transform
    def perform_sdft(self) -> np.ndarray:
        audio_data = self.audio_input.get_frame()
        return self.fft_machine.compute_fft(audio_data)
