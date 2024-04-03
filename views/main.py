from dataclasses import dataclass
from .plotter import Plotter

class View:
    def __init__(self, config_data: dict) -> None:
        self.frame_size = config_data.get("frame_size")
        self.sample_rate = config_data.get("sample_rate")
        self.song_name = config_data.get("song_name")

        self.plotter = Plotter(
            frame_size = self.frame_size,
            sample_rate = self.sample_rate,
            song_name = self.song_name)

    def plot(self, data) -> None:
        self.plotter.update(data)
