import numpy as np
import matplotlib.pyplot as plt
from .blit_manager import BlitManager

class Plotter:
    def __init__(self) -> None:
        plt.style.use('_mpl-gallery')

        # TODO remove this
        self.x_data = np.arange(0,8)
        self.y_data = 2*self.x_data

        self.figure, self.axis = plt.subplots()

        # TODO figure out why do I need the ","
        (self.line,) = self.axis.plot(
            self.x_data,
            self.y_data,
            animated = True)

        plt.show(block = False)
        plt.pause(0.1)

        self.bm = BlitManager(self.figure.canvas, [self.line])

    def update(self) -> None:
        self.y_data = 1.001*self.y_data
        self.line.set_ydata(self.y_data)
        self.bm.update()