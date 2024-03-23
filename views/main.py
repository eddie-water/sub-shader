from .plotter import Plotter

class View:
    def __init__(self) -> None:
        self.plotter = Plotter()

    def plot(self, data) -> None:
        self.plotter.update(data)
