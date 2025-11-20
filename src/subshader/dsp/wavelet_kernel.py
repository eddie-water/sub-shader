from typing import Optional, Final

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import cupy as cp

from subshader.dsp.gaussian import Gaussian
from subshader.utils.logging import get_logger

log = get_logger(__name__)  

PI: Final[np.float64] = np.pi

class WaveletKernel():
    def __init__(self,
                 f: np.float64,
                 sample_rate: int,
                 num_cycles: int,
                 num_fwhm_cycles: int,
                 input_n: int) -> tuple[list[np.ndarray[np.complex64]], list[np.ndarray[np.complex64]]]:
        """
        Constructs a wavelet kernel whose time support is defined by a desired 
        number of cycles of a specified center frequency. The wavelet is shaped 
        by a Gaussian bell curve whose Full Width at Half Maximum (FWHM) is 
        defined by a desired number of cycles of that time support.

        Args:
            f: The desired center frequency for the wavelet (Hz)
            sample_rate: The sample rate which specifies the spacing in time
                between each sample in the wavelet kernel (Hz)
            num_cycles: The number of cycles in the carrier sinusoid we define
                to be its time support aka how long in time the wavelet has
                meaningful energy since technically it could go on forever (num)
            num_fwhm_cycles: The number of cycles used to define the FWHM of the
                Gaussian (num)
            input_n: The number of samples in the signal we will analyze with 
                this wavelet kernel (num)
        """
        self.freq: np.float64 = f
        self.input_n: int = input_n

        # Time Support is the duration (s) over which the wavelet has meaningful 
        # energy, defined as the length of time needed to contain a given number
        # of cycles for a particular center frequency.
        time_support_s: np.float64 = num_cycles / self.freq

        # Convert to number of samples (s * samples / s)
        self.time_support_n: int = int(np.round(time_support_s * sample_rate))

        log.info(f"Wavelet {f:.2f} Hz | Time Support: {time_support_s:.2f} s, {self.time_support_n} samples")

        # Time vector centered at t = 0 with time support duration
        self.time_t: np.ndarray[np.float64] = (np.arange(self.time_support_n, dtype=np.float64) / sample_rate) - (time_support_s / 2)

        # Create Complex Morlet Wavelet by shaping a sinusoid with a Gaussian
        self.sinusoid: np.ndarray[np.complex64] = np.exp(1j * 2 * PI * self.freq * self.time_t)
        self.gaussian: np.ndarray[np.complex64] = Gaussian(self.time_t, self.freq, num_fwhm_cycles).gauss
        self.kernel_t: np.ndarray[np.complex64] = self.sinusoid * self.gaussian

        # Convolution length N
        self.conv_n: int = int(input_n + self.time_support_n - 1)

        # Transform the time domain wavelet kernel to the frequency domain
        self.kernel_f: np.ndarray[np.complex64] = fft(self.kernel_t, self.conv_n)

        # Create a slice object to later extract the valid portion of the convolution result
        half_width: int = int(self.time_support_n // 2)
        self.slice_start: int = half_width
        self.slice_end: int = half_width + input_n
        self.slice: slice = slice(self.slice_start, self.slice_end)

    def _plot_kernel(self) -> None:
        """
        Plot the wavelet kernel components.
        """
        # fig, ax = plt.subplots()
        # ax.set_title(f"Wavelet Kernel Components - Center Frequency: {self.freq:.1f} Hz")
        # ax.set_ylabel("Amplitude")
        # ax.set_xlabel("Time (s)")
        # ax.plot(self.time_t, self.sinusoid.real, label="Real Sin", color="orange", linewidth=2)
        # ax.plot(self.time_t, self.sinusoid.imag, label="Imag Sin", color="mediumslateblue")        
        # ax.plot(self.time_t, self.gaussian, label="Gaussian", color="firebrick")
        # ax.plot(self.time_t, self.kernel_t.real, label="Real CMW", color="black")
        # ax.legend(loc="upper right")

        # plt.show()
        pass

    def get_conv_n(self) -> int:
        """
        Get the convolution length N.
        """
        return self.conv_n
