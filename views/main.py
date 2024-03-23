from .plotter import Plotter

class View:
    def __init__(self, frame_size: int) -> None:
        self.plotter = Plotter(frame_size = frame_size)

    def plot(self, data) -> None:
        self.plotter.update(data)
