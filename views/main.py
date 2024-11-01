from dataclasses import dataclass
from .plotter import Plotter

class View:
    def __init__(self, config_data: dict) -> None:
        self.data_shape = config_data.get("data_shape")
        self.sample_rate = config_data.get("sample_rate")
        self.song_name = config_data.get("song_name")

        self.plotter = Plotter(data_shape = self.data_shape,
                               sample_rate = self.sample_rate,
                               song_name = self.song_name)

    def plot_fft(self, data) -> None:
        self.plotter.update_stft(data)

    def plot_cwt(self, coefs, freqs, time) -> None:
        self.plotter.plot_scalogram(coefs, freqs, time)
