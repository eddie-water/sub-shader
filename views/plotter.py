import numpy as np
import matplotlib.pyplot as plt
from .blit_manager import BlitManager

class Plotter:
    def __init__(self, frame_size: int) -> None:
        self.x_axis_length = frame_size / 2

        # Create figure and axis from plot
        self.figure, self.axis = plt.subplots(
            figsize = (7,5), 
            layout = 'constrained')

        # Stylize the plot and prevent it from hogging the program
        plt.style.use('_mpl-gallery')
        plt.axis([0, self.x_axis_length, 0, 0.25])
        plt.show(block = False)
        plt.pause(0.1)

        # Retrieve line artist(s) to feed the blit manager 
        self.line, = self.axis.plot(
            0,
            0,
            animated = True)
        self.line.set_xdata(np.arange(0, self.x_axis_length, 1))

        # TODO: explore the relationship between the canvas, renderer, and artist
        self.bm = BlitManager(self.figure.canvas, [self.line])

    def update(self, data: np.ndarray) -> None:
        self.line.set_ydata(data)
        self.bm.update()