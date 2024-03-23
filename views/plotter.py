import numpy as np
import matplotlib.pyplot as plt
from .blit_manager import BlitManager

class Plotter:

    # TODO: pass in chunk size to plotter object
    CHUNK_SIZE = 4096 / 2

    def __init__(self) -> None:

        # Create figure and axis from plot
        self.figure, self.axis = plt.subplots(
            figsize = (7,5), 
            layout = 'constrained')

        # Stylize the plot and prevent it from hogging the program
        plt.style.use('_mpl-gallery')
        plt.axis([0, Plotter.CHUNK_SIZE, -0.5, 0.5])
        plt.show(block = False)
        plt.pause(0.1)

        # Retrieve line artist(s) to feed the blit manager 
        self.line, = self.axis.plot(
            0,
            0,
            animated = True)
        self.line.set_xdata(np.arange(0, Plotter.CHUNK_SIZE, 1))

        # TODO: explore the relationship between the canvas, renderer, and artist
        self.bm = BlitManager(self.figure.canvas, [self.line])

    def update(self, data: np.ndarray) -> None:
        self.line.set_ydata(data)
        self.bm.update()