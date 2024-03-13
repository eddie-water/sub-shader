import numpy as np
import matplotlib.pyplot as plt
from .blit_manager import BlitManager

class Plotter:
    def __init__(self) -> None:
        plt.style.use('_mpl-gallery')

        self.figure, self.axis = plt.subplots()

        # TODO figure out why do I need the ","
        (self.line,) = self.axis.plot(
            0,
            0,
            animated = True)

        plt.show(block = False)
        plt.pause(0.1)

        self.bm = BlitManager(self.figure.canvas, [self.line])

    def update(self, data) -> None:
        self.line.set_ydata(data)
        self.bm.update()