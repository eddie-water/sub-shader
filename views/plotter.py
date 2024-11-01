import scipy
import numpy as np
import matplotlib.pyplot as plt
from .blit_manager import BlitManager

class Plotter:
    def __init__(self, data_shape: int, sample_frequency: int, plot_name: str) -> None:
        """Plot audio data 
        Continuously plot the FFT (STFT)
        Plot the CWT - TODO NOW currently figuring out how to do continuous plotting

        Args:
            data_shape: the dimensions of the data to be plotted
            sample_frequency: the sample rate of the time axis
            plot_name: the title of the plot
        """
        self.data_shape = data_shape
        self.sample_frequency = sample_frequency
        self.plot_name = plot_name

        self.__init_fft__()
        self.__init_cwt__()

    def __init_fft__(self):
        """Prepare the figure and axes for FFT data"""
        self.fig, self.ax = plt.subplots(
            figsize = (10, 6), 
            layout = 'constrained')

        # Stylize the plot and prevent it from hogging the program
        self.fig.suptitle("Sliding CWT of " + self.plot_name)
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlabel("Frequency")

        plt.style.use('_mpl-gallery')
        plt.xscale('log')
        plt.axis([10, (self.sample_frequency / 2), 0, 0.15])
        plt.show(block = False)
        plt.pause(0.1)

        self.x_axis = scipy.fft.rfftfreq(
            n = self.data_shape[1],
            d = (1 / self.sample_frequency))

        # Retrieve line artist(s) to feed the blit manager 
        self.line, = self.ax.plot(
            0,
            0,
            animated = True)

        # Blit only works if the artist obj has some kind of 'set_data' method
        self.line.set_xdata(self.x_axis)

        # Assign line artist to Blit Manager
        self.bm = BlitManager(self.fig.canvas, [self.line])

    def update_fft(self, data: np.ndarray) -> None:
        '''Update the plot with a new set of FFT data'''
        self.line.set_ydata(data)
        self.bm.update()

    def __init_cwt__(self):
        """Prepare the plot for CWT data"""
        self.y_dimension = self.data_shape[0]
        self.x_dimension = self.data_shape[1]

        # Create and initialize figure, axes parameters, and plot colormap
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle("Scalogram: Frequency vs Time")

        self.__init_pcm__()
        self.__init_img__()

    def __init_pcm__(self):
        self.cmap = plt.colormaps['magma']

        # Create time and frequecy axes
        self.pcm_x = np.arange(0, self.x_dimension) / self.sample_frequency
        a0 = 2**(1/12)

        i = np.arange(0, self.y_dimension, 1)
        s = a0**(i)
        f = 27.5*s
        self.pcm_y = f[f<22050]

    def update_cwt_pcm(self, coefs) -> None:
        """Update the plot with CWT data"""
        self.ax.set_xlabel("Time (s)")
        self.ax.set_yscale("log")
        self.ax.set_ylabel("Frequency (Hz)")

        """
        TODO Issue #10 

        Sharpen the higher frequency color's intensity/magnitude/activity with 
        a logarithmic color mapping function like in that one youtube video I 
        watched I think its called Mike X Cohen in the video where he first 
        graphed the scalogram and was showing the tradeoff between... etc

        I think it is something like: pcolormesh(norm= log)
        """

        self.pcm = self.ax.pcolormesh(self.pcm_x, 
                                      self.pcm_y, 
                                      coefs, 
                                      cmap= self.cmap)

        plt.show()

    def __init_img__(self):
        self.img_x = np.arange(0, self.x_dimension) / self.sample_frequency
        
        # # Initialize plot with empty TODO <insert name here>

        # """
        # TODO TESTING: ImShow (AxesImage)
        #         Problem - 
        #         Should Try - 
        # """

        # self.freq_coords, self.time_coords = np.meshgrid(self.freqs, self.time)

        # self.ax.set_xlabel("Time (s)")

        # # self.ax.set_ylim(self.freqs.min(), self.freqs.min()) # / sample_frequency)
        # # self.ax.set_yscale("log")
        # self.ax.set_ylabel("Frequency (Hz)")

        #  # note that the first draw comes before setting data
        # self.fig.canvas.draw()   

        # """
        # TODO TESTING
        #     Swapping in self.coef_img and self.img
        # """
        # # Initialize Blit Manager with the artist
        # self.bm = BlitManager(self.fig.canvas, [self.qmesh])

        # # Configure plot settings
        # # plt.style.use('_mpl-gallery')
        # # plt.pause(0.1)
        # plt.show(block = False)

    def update_cwt_img(self, coefs, freqs, time) -> None:
        pass
        # TODO look into contourf() it makes the pcolormesh really smooth

        """
        TODO TESTING
            Swapping 
                self.img.set_data(np.sin(self.X/3.+self.k)*np.cos(self.Y/3.+self.k))
            With 
                self.coef_img.set_data(coefs)
        """
        # self.qmesh.set_data(coefs)

        # self.coef_img.set_data(coefs)

        # temp_img_shape = self.img.get_shape()
        # temp_coef_img_shape = self.coef_img.get_shape()
        # self.k += 0.1

        # break_point = True

        # self.bm.update()
