from abc import ABC, abstractmethod
import numpy as np
import cupy as cp

import pywt
from numpy.fft import fft, ifft
from cupyx.scipy import fft as cp_fft

# Audio is typically sampled at 44.1 kHz
TYPICAL_SAMPLING_FREQ = 44100

# Musical Scale parameters 
NOTES_PER_OCTAVE = 12 
NUM_OCTAVES = 10
ROOT_NOTE_A0 = 27.5

# Math Constants
pi = np.pi

class Wavelet(ABC):
    def __init__(self, sample_rate: int, window_size: int):
        if sample_rate != TYPICAL_SAMPLING_FREQ:
            raise ValueError(f"Sampling Rate: {sample_rate},", 
                             f"is not {TYPICAL_SAMPLING_FREQ} Hz.",
                             f"The CWT may not work as expected.")

        if window_size <= 0:
            raise ValueError(f"Window Size: {window_size},",
                             f"must be greater than 0.")

        # Sampling Parameters
        self.sample_rate = sample_rate
        self.nyquist_freq = (sample_rate / 2.0)
        self.sampling_period = (1.0 / self.sample_rate)
        self.window_size = window_size

        # Frequency Axis that replicates the exponential step size of the musical scale
        self.scale_factor = 2**(1/NOTES_PER_OCTAVE)
        i = np.arange(0, NOTES_PER_OCTAVE*NUM_OCTAVES, 1)
        self.s = self.scale_factor**i
        self.freqs = ROOT_NOTE_A0*self.s

        # Discard frequencies that are unmeasurable
        self.freqs = self.freqs[self.freqs < self.nyquist_freq]
        self.num_freqs = len(self.freqs)

        self.result_shape = (self.num_freqs, self.window_size)

    """
    Computes the shape of the resultant CWT data

    Returns:
        Shape of the computed CWT data
    """
    def get_shape(self) -> np.ndarray.shape:
        return self.result_shape

    """
    Get the time resolution for the data

    Returns:
        Sampling period
    """
    def get_sample_period(self) -> float:
        return self.sampling_period

    """
    Get the number of frequencies in the used in the CWT

    Returns:
        Number of frequencies in the CWT
    """
    def get_num_freqs(self) -> int:
        return self.num_freqs
    
    """
    Performs the Continuous Wavelet Transform on raw audio and normalizes the 
    results

    Args:
        audio_data: raw audio signal data

    Returns:
        normalized CWT coefficients in the time-frequency domain

    """
    def compute_cwt(self, audio_data) -> np.ndarray:
        # Verify the audio data is valid 
        if len(audio_data) != self.window_size:
            raise ValueError(f"Audio data length {len(audio_data)}",
                             f"does not match window size {self.window_size}")
        # Increase precision
        data = audio_data.astype(np.float64)

        cwt_coefs = self.class_specific_cwt(data)

        return self.normalize_coefs(cwt_coefs)
    
    """
    Computes the CWT via the specific subclass implementation
    """
    @abstractmethod
    def class_specific_cwt(self, data) -> np.ndarray:
        pass

    """
    Cleans up the coef data for plotting
    - Takes the absolute values of the raw coefs to get the magnitude of the 
      resultant coefs
    - Normalizes the coefs against the scale to compensate for energy 
      accumulation bias in the CWT with higher scales (low frequencies)
    - Normalizes the coefs so the min and max map to 0 and 1
    - Downsamples the coefs to reduce plotting time
    - Transposes the coefs because the CWT swaps the axes for some reason

    Args:
        raw_coefs: raw CWT coefficients 

    Returns:
        coefs: normalized CWT coefficients
    """
    def normalize_coefs(self, raw_coefs) -> np.ndarray:
        # Absolute Value 
        coefs_abs = np.abs(raw_coefs)

        # Min-Max Normalization - squeeze data into the [0, 1] range
        coefs_min = np.min(coefs_abs)
        coefs_max = np.max(coefs_abs)
        coefs_norm = (coefs_abs - coefs_min) / (coefs_max - coefs_min)

        return coefs_norm
    
class PyWavelet(Wavelet):
    def __init__(self, sample_rate, window_size):
        super().__init__(sample_rate, window_size)

        # Wavelet info TODO LATER why 1.5-1.0?
        self.wavelet_name = "cmor1.5-1.0"

        # Scale array used to specify wavelet dilation amounts during CWT
        f_norm = (self.freqs / self.sample_rate)
        self.scales = pywt.frequency2scale(self.wavelet_name, f_norm)

    def class_specific_cwt(self, data):
        raw_coefs, _ = pywt.cwt(data = data,
                                scales = self.scales,
                                wavelet = self.wavelet_name,
                                sampling_period = self.sampling_period)
    
        # Scale-Based Normalization 
        coefs_scaled = raw_coefs / np.sqrt(self.scales[:, None])

        return self.normalize_coefs(coefs_scaled)

class AntsWavelet(Wavelet):
    def __init__(self, sample_rate, window_size):
        super().__init__(sample_rate, window_size)
        # Initialize the time-frequency matrix
        self.tf = np.zeros((self.num_freqs, self.window_size))

        # Create a centered time vector for the CMW
        cmw_t = np.arange(2*self.sample_rate) / self.sample_rate
        self.cmw_t = cmw_t - np.mean(cmw_t) 

        # N's of convolution
        self.data_n = self.window_size
        self.kern_n = len(cmw_t)
        self.conv_n = self.data_n + self.kern_n - 1
        self.half_kern_n = self.kern_n // 2

        # Full-Width Half Maximum - try out some different values
        fwhm = 0.3

        # Build a filter bank of frequency-domain wavelets
        # TODO NEXT Investigate the n = conv_n vs kern_n passed into the FFT - how does this affect the results of the CWT?
        self.wavelet_kernels = np.zeros((self.num_freqs, self.conv_n), dtype = cp.complex64)
        self.num_wavelets = self.wavelet_kernels.shape[0]

        for i, f in enumerate(self.freqs):
            # TODO SOON Determine the significance of the parameters of the guassian envelope - why -4?
            cmw_k = np.exp(1j*2*pi*f*self.cmw_t) * np.exp(-4*np.log(2)*self.cmw_t**2 / fwhm**2)
            
            # Normalize the wavelet kernel by 1/sqrt(scale) 
            cmw_k = np.sqrt(f) * cmw_k

            # TODO NEXT Investigate the n = conv_n vs kern_n passed into the FFT - how does this affect the results of the CWT?
            cmw_x = fft(cmw_k, self.conv_n)
            cmw_x = cmw_x / max(cmw_x)
            self.wavelet_kernels[i,:] = cmw_x 

    def class_specific_cwt(self, data) -> np.ndarray:
        pass

class NumpyWavelet(AntsWavelet):
    def __init__(self, sample_rate, window_size):
        super().__init__(sample_rate, window_size)

    def class_specific_cwt(self, data) -> np.ndarray:
        # Transform the Data time series into a spectrum
        data_x = fft(data, self.conv_n)

        # Perform the CWT with each wavelet 
        for i in range(self.num_wavelets):
            conv = ifft(data_x * self.wavelet_kernels[i,:])
            conv = conv[(self.half_kern_n):(-self.half_kern_n+1)]
            conv_pow = np.abs(conv)**2
            self.tf[i,:] = conv_pow

        # TODO ASP Clean up the boundary effects of the convolution

        return self.tf
    
class CupyWavelet(AntsWavelet):
    def __init__(self, sample_rate, window_size):
        super().__init__(sample_rate, window_size)
        self.tf_gpu = cp.zeros((self.num_freqs, self.window_size))

        # Move the wavelet kernels to the GPU
        self.wavelet_kernels = cp.asarray(self.wavelet_kernels)

    # TODO SOON Investigate writing a custom GPU kernel rather than using CuPy
    def class_specific_cwt(self, data) -> np.ndarray:
        # Transform the Data time series into a spectrum on the GPU
        # TODO NEXT Investigate how to minimize the CPU to GPU transfers
        data = cp.asarray(data, dtype=cp.complex64)
        data_x = cp_fft.fftn(data, self.conv_n)

        for i in range(self.num_wavelets):
            conv = cp_fft.ifft(data_x * self.wavelet_kernels[i,:])
            conv = conv[(self.half_kern_n):(-self.half_kern_n+1)]
            conv_pow = cp.abs(conv)**2
            self.tf_gpu[i,:] = conv_pow

        # Move the result back to the CPU
        # TODO NEXT Investigate how to minimize the GPU to CPU transfers
        self.tf = cp.asnumpy(self.tf_gpu)

        return self.tf
    
class ShadeWavelet(CupyWavelet):
    pass