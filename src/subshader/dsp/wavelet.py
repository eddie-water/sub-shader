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
    def __init__(self, sample_rate: int, window_size: int, ds_stride: int = 1):
        """
        Wavelet base class that all other wavelet classes are derived from.
        Uses a list of frequencies that follows the chromatic scale starting at
        A0 to specify which frequencies to look for in the audio data.

        Args:
            sample_rate (int): The rate the data was sampled in Hz
            window_size (int): The length of the data
        """
        if sample_rate != TYPICAL_SAMPLING_FREQ:
            raise ValueError(f"Sampling Rate: {sample_rate},", 
                             f"is not {TYPICAL_SAMPLING_FREQ} Hz.",
                             f"The CWT may not work as expected.")
        self.sample_rate = sample_rate

        if window_size <= 0:
            raise ValueError(f"Window Size: {window_size},",
                             f"must be greater than 0.")
        self.window_size = window_size

        if ds_stride <= 0:
            raise ValueError(f"Downsample stride: {ds_stride},",
                             f"must be greater than 0.")
        self.ds_stride = ds_stride

        # Sampling Parameters
        self.sample_rate = sample_rate
        self.nyquist_freq = (sample_rate / 2.0)
        self.sampling_period = (1.0 / self.sample_rate)

        # Frequency Axis that replicates the exponential step size of the musical scale
        self.scale_factor = 2**(1/NOTES_PER_OCTAVE)
        i = np.arange(0, NOTES_PER_OCTAVE*NUM_OCTAVES, 1)
        self.s = self.scale_factor**i
        self.freqs = ROOT_NOTE_A0*self.s

        # Discard frequencies that are unmeasurable
        self.freqs = self.freqs[self.freqs < self.nyquist_freq]
        self.num_freqs = len(self.freqs)

        # Resultant Shape of the CWT Data with Downsampling
        self.result_shape = (self.num_freqs, self.window_size // self.ds_stride)

    def get_shape(self) -> np.ndarray.shape:
        """
        Computes the shape of the resultant CWT data.

        Returns:
            np.ndarray.shape: Shape of the computed CWT data
        """
        return self.result_shape

    def get_num_freqs(self) -> int:
        """
        Get the number of frequencies in the used in the CWT

        Returns:
            int: Number of frequencies in the CWT
        """
        return self.num_freqs
    
    def compute_cwt(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Performs the Continuous Wavelet Transform (CWT) on raw audio data and 
        then normalizes the results

        Args:
            audio_data (np.ndarray): raw audio signal data

        Returns:
            np.ndarray: The normalized CWT coefficients in the scale-time

                domain
        """
        # Verify the audio data is valid 
        if len(audio_data) != self.window_size:
            raise ValueError(f"Audio data length {len(audio_data)}",
                             f"does not match window size {self.window_size}")

        # Increase precision
        data = audio_data.astype(np.float64)

        cwt_coefs = self.class_specific_cwt(data)

        return self.normalize_coefs(cwt_coefs)

    @abstractmethod
    def class_specific_cwt(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the subclass-specific implementation of the CWT

        Args:
            data (np.ndarray): The data to perform the CWT on

        Returns:
            np.ndarray: The CWT coefficients 
        """
        pass

    def normalize_coefs(self, raw_coefs: np.ndarray) -> np.ndarray:
        """
        Cleans up the raw CWT coefficients for plotting
          - Takes the absolute values of the raw coefs to get the magnitude of
          the resultant coefs
          - Normalizes the coefs so the min and max map to 0 and 1

        Args:
            raw_coefs (np.ndarray): raw CWT coefficients 

        Returns:
            np.ndarray: Normalized CWT coefficients
        """
        # Absolute Value 
        coefs_abs = np.abs(raw_coefs)

        # TODO ISSUE-36 See if we should do scale-based normalization for all 
        # the wavelet subclasses instead of only in the PyWavelet subclass

        # Min-Max Normalization - squeeze data into the [0, 1] range
        coefs_min = np.min(coefs_abs)
        coefs_max = np.max(coefs_abs)
        
        epsilon = 1e-10 # Prevents division by zero
        coefs_norm = (coefs_abs - coefs_min) / (coefs_max - coefs_min + epsilon)
        
        # Ensure output is in [0, 1] range
        coefs_norm = np.clip(coefs_norm, 0.0, 1.0)
        
        return coefs_norm
    
class PyWavelet(Wavelet):
    def __init__(self, sample_rate, window_size):
        """
        The PyWavelet implementation of the CWT

        Args:
            sample_rate (int): The rate the data was sampled in Hz
            window_size (int): The length of the data
        """
        super().__init__(sample_rate, window_size)

        # Wavelet info TODO ISSUE-36 why 1.5-1.0?
        self.wavelet_name = "cmor1.5-1.0"

        # Scale array used to specify wavelet dilation amounts during CWT
        f_norm = (self.freqs / self.sample_rate)
        self.scales = pywt.frequency2scale(self.wavelet_name, f_norm)

    def class_specific_cwt(self, data: np.ndarray) -> np.ndarray:
        """
        Produces the normalized CWT coefficients using PyWavelets. 

        Args:
            data (np.ndarray): The data to perform the CWT on

        Returns:
            np.ndarray: The scale-based normalized CWT coefficients 
        """
        coefs_raw, _ = pywt.cwt(data = data,
                                scales = self.scales,
                                wavelet = self.wavelet_name,
                                sampling_period = self.sampling_period)
    
        """
        Scale-Based Normalization 
        
        This account for the energy bias that occurs at higher scales which 
        PyWavelets does not do internally it seems. The wavelet equation is 
        this: 
        
            Psi_s(t) = 1/sqrt(s) * Psi(t-T/s)
        
        Where Psi is the wavelet at a scale s, and localized in time by T. We 
        need the '1/sqrt(s)' term to account for the energy bias that occurs at
        higher scales for 'Psi(t-T/s)'. The 's' term, since it's in the 
        denominator, stretches the wavelet horizontally when 's' is large, and 
        compresses it horizontally when 's' is small. Higher-scale wavelets 
        will have more area under their curves, and will seem to contribute 
        more energy to the inner product that is going on inside the CWT. To 
        account for this, we normalize the coefficients by dividing it by
        the square root of the scale. Not sure why PyWavelets doesn't just do
        this under the hood.
        """
        # TODO ISSUE-36 Why aren't I doing this to all the wavelet subclasses
        # in the parent class?
        coefs_scaled = coefs_raw / np.sqrt(self.scales[:, None])

        return coefs_scaled

class AntsWavelet(Wavelet):
    def __init__(self, sample_rate: int, window_size: int, ds_stride: int = 1):
        """
        This is a CWT implementation I got from Analyzing Neural Time Series 
        (ANTS) by Mike X Cohen. I had to translate it from Matlab. It manually 
        creates a bank of wavelet filters specified by the list of frequencies 
        and then manually performs the CWT on audio data.

        Args:
            sample_rate (int): The rate the data was sampled in Hz
            window_size (int): The length of the data
        """
        # TODO ISSUE-36 Use class inheritance effectively to avoid code 
        # duplication
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

        # TODO ISSUE-36 Full-Width Half Maximum - try out some different values
        fwhm = 0.3

        # Build a filter bank of frequency-domain wavelets
        # TODO ISSUE-36 Investigate the n = conv_n vs kern_n passed into the FFT - how does this affect the results of the CWT?
        self.wavelet_kernels = np.zeros((self.num_freqs, self.conv_n), dtype = cp.complex64)
        self.num_wavelets = self.wavelet_kernels.shape[0]

        for i, f in enumerate(self.freqs):
            # TODO ISSUE-36 Determine the significance of the parameters of the guassian envelope - why -4?
            cmw_k = np.exp(1j*2*pi*f*self.cmw_t) * np.exp(-4*np.log(2)*self.cmw_t**2 / fwhm**2)
            
            # TODO ISSUE-36 Figure out why the second band looks weaker than the first
            # Scale-Based Normalization: sqrt(f) = 1/sqrt(scale) 
            cmw_k = np.sqrt(f) * cmw_k

            # TODO ISSUE-36 Investigate the n = conv_n vs kern_n passed into the FFT - how does this affect the results of the CWT?
            cmw_x = fft(cmw_k, self.conv_n)
            cmw_x = cmw_x / max(cmw_x)
            self.wavelet_kernels[i,:] = cmw_x 

    def class_specific_cwt(self, data) -> np.ndarray:
        pass

class NumpyWavelet(AntsWavelet):
    def __init__(self, sample_rate, window_size, ds_stride):
        """
        This implements the ANTS CWT using NumPy.

        Args:
            sample_rate (int): The rate the data was sampled in Hz
            window_size (int): The length of the data
        """
        super().__init__(sample_rate, window_size)

        # Initialize the time-frequency matrix
        self.tf = np.zeros((self.num_freqs, self.window_size))

    def class_specific_cwt(self, data) -> np.ndarray:
        """
        This implements the ANTS CWT using NumPy.

        Args:
            data (np.ndarray): The data to perform the CWT on

        Returns:
            np.ndarray: The scale-based normalized CWT coefficients 
        """
        # Transform the Data time series into a spectrum
        data_x = fft(data, self.conv_n)

        # Perform the CWT with each wavelet 
        for i in range(self.num_wavelets):
            conv = ifft(data_x * self.wavelet_kernels[i,:])
            conv = conv[(self.half_kern_n):(-self.half_kern_n+1)]
            conv_pow = np.abs(conv)**2
            self.tf[i,:] = conv_pow

        # TODO ISSUE-36 Clean up the boundary effects of the convolution

        return self.tf
    
class CupyWavelet(AntsWavelet):
    def __init__(self, sample_rate, window_size, ds_stride):
        """
        This implements the ANTS CWT using CuPy to exploit the parallelizable
        aspects of the CWT by running on a GPU. Note - this won't work without
        an Nvidia GPU and bunch of CUDA dependencies.

        Args:
            sample_rate (int): The rate the data was sampled in Hz
            window_size (int): The length of the data
        """
        super().__init__(sample_rate, window_size, ds_stride)
        self.tf_gpu = cp.zeros((self.num_freqs, self.window_size))

        # Move the wavelet kernels to the GPU
        self.wavelet_kernels = cp.asarray(self.wavelet_kernels)

    # TODO ISSUE-36 Investigate writing a custom GPU kernel rather than using CuPy
    def class_specific_cwt(self, data) -> np.ndarray:
        """
        This implements the ANTS CWT using CuPy.

        Args:
            data (np.ndarray): The data to perform the CWT on

        Returns:
            np.ndarray: The CWT coefficients
        """
        # Transform the Data time series into a spectrum on the GPU
        # TODO ISSUE-33 Investigate how to minimize the CPU to GPU transfers
        data = cp.asarray(data, dtype=cp.complex64)
        data_x = cp_fft.fftn(data, self.conv_n)

        for i in range(self.num_wavelets):
            conv = cp_fft.ifft(data_x * self.wavelet_kernels[i,:])
            conv = conv[(self.half_kern_n):(-self.half_kern_n+1)]
            conv_pow = cp.abs(conv)**2
            self.tf_gpu[i,:] = conv_pow

        # Move the result back to the CPU (no downsampling)
        return cp.asnumpy(self.tf_gpu)
    
class ShadeWavelet(CupyWavelet):
    """
    This just kind of renames the class becuase it sounds ~cool~
    """
    pass