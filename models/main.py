from .audio_input import AudioInput
import numpy as np

class Model:
    def __init__(self) -> None:
        self.audio_file = "models/audio_files/c_octaves.wav"
        self.audio_input = AudioInput(self.audio_file)

    def audio_task(self) -> np.ndarray:
        return self.audio_input.get()