from .audio_input import AudioInput

class Model:
    def __init__(self) -> None:
        self.audio_file = "audio_files/aminor.wav"
        self.audio_input = AudioInput(self.audio_file)

    def start(self) -> None:
        pass