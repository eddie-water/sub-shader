import scipy
import numpy as np
import matplotlib.pyplot as plt
from .blit_manager import BlitManager

# TODO: Plotter could benefit to know frame_size, sampling rate, file name

class Plotter:
    def __init__(self, frame_size: int) -> None:
        self.x_axis = scipy.fft.rfftfreq(
            n = frame_size,
            d = 1 / 44100.0
        )

        # Create figure and axis from plot
        self.figure, self.axis = plt.subplots(
            figsize = (10, 6), 
            layout = 'constrained')

        # Stylize the plot and prevent it from hogging the program
        self.figure.suptitle("Sliding FFT")
        self.axis.set_ylabel("Amplitude")
        self.axis.set_xlabel("Frequency")

        plt.style.use('_mpl-gallery')
        plt.xscale('log')
        plt.axis([10, 22050, 0, 0.15])
        plt.show(block = False)
        plt.pause(0.1)

        # Retrieve line artist(s) to feed the blit manager 
        self.line, = self.axis.plot(
            0,
            0,
            animated = True)
        self.line.set_xdata(self.x_axis)

        self.bm = BlitManager(self.figure.canvas, [self.line])

    def update(self, data: np.ndarray) -> None:
        self.line.set_ydata(data)
        self.bm.update()