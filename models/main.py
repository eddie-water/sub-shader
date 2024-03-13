from .audio_input import AudioInput
import numpy as np

class Model:
    def __init__(self) -> None:
        self.audio_file = "models/audio_files/aminor.wav"
        self.audio_input = AudioInput(self.audio_file)

    def audio_task(self) -> None:
        chunk = self.audio_input.get()
        # slide to the next chunk
        return chunk