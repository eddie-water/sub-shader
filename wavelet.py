from abc import ABC, abstractmethod
import numpy as np
import pywt
from numpy.fft import fft, ifft

# Audio is typically sampled at 44.1 kHz
TYPICAL_SAMPLING_FREQ = 44100

# Musical Scale parameters 
NOTES_PER_OCTAVE = 12 
NUM_OCTAVES = 10
ROOT_NOTE_A0 = 27.5

pi = np.pi
# TODO NEXT begin creating the shade wavelet class that also inherits this class
class Wavelet(ABC):
    def __init__(self, sample_rate: int, frame_size: int, downsample_factor: int):
        if (sample_rate != TYPICAL_SAMPLING_FREQ):
            raise ValueError("Irregular Sampling Frequency")

        # Sampling Info
        self.sample_rate = sample_rate
        self.nyquist_freq = (sample_rate / 2.0)
        self.sampling_period = (1.0 / self.sample_rate)

        self.frame_size = frame_size

        self.downsample_factor = downsample_factor

        # Wavelet info TODO LATER why 1.5-1.0?
        self.wavelet_name = "cmor1.5-1.0"

        # Time Axis
        self.time = np.arange(0, frame_size) * self.sampling_period

        # Frequency Axis that mimics the variable step size of musical scales
        self.scale_factor = 2**(1/NOTES_PER_OCTAVE)
        i = np.arange(0, NOTES_PER_OCTAVE*NUM_OCTAVES, 1)
        self.s = self.scale_factor**i
        self.freqs = ROOT_NOTE_A0*self.s

        # Discard frequencies that are unmeasurable
        self.freqs = self.freqs[self.freqs < self.nyquist_freq]
        self.num_freqs = len(self.freqs)

        # Scale array used to specify wavelet dilation amounts during cwt
        f_norm = (self.freqs / self.sample_rate)
        self.scales = pywt.frequency2scale(self.wavelet_name, f_norm)

    """
    Computes the shape of the resultant CWT data

    Returns:
        Shape of the computed CWT data
    """
    def get_shape(self) -> np.ndarray.shape:
        return np.empty((self.freqs.size, self.time.size)).shape

    """
    Get the time resolution for the data
    Returns:
        Sampling period
    """
    def get_sample_period(self) -> float:
        return self.sampling_period
    
    """
    Gets the scale factor

    Returns:
        Scale factor
    """
    def get_scale_factor(self) -> float:
        return self.scale_factor

    """
    Performs the Continuous Wavelet Transform and normalizes the data
    - Computes the CWT on the raw audio data

    Args:
        audio_data: raw audio signal data

    Returns:
        raw_coefs: raw CWT coefficients

    """
    @abstractmethod
    def compute_cwt(self, audio_data) -> np.ndarray: 
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

        # Scale-Based Normalization 
        coefs_scaled = coefs_abs / np.sqrt(self.scales[:, None])

        # Min-Max Normalization 
        coefs_min = np.min(coefs_scaled)
        coefs_max = np.max(coefs_scaled)
        coefs_norm = (coefs_scaled - coefs_min) / (coefs_max - coefs_min)

        # Downsample TODO LATER remove - downsample shouldn't be needed if cwt is fast enough
        coefs = coefs_norm[::, ::(self.downsample_factor)]

        # Swap Axes
        # TODO SOON wait why am I transposing this if it's being transposed outside the compute_cwt method?
        coefs = np.transpose(coefs)
        return coefs
    
class PyWavelet(Wavelet):
    def __init__(self, sample_rate, frame_size, downsample_factor):
        super().__init__(sample_rate, frame_size, downsample_factor)

    def compute_cwt(self, audio_data):
        raw_coefs, _ = pywt.cwt(data = audio_data,
                                scales = self.scales,
                                wavelet = self.wavelet_name,
                                sampling_period = self.sampling_period)

        return self.normalize_coefs(raw_coefs)

class ShadeWavelet(Wavelet):
    def __init__(self, sample_rate, frame_size, downsample_factor):
        super().__init__(sample_rate, frame_size, downsample_factor)
    
    def compute_cwt(self, audio_data) -> np.ndarray:  
        data = audio_data.astype(np.float64)
        
        # Create a centered time vector for the CMW
        cmw_t = np.arange(2*self.sample_rate) / self.sample_rate
        cmw_t = cmw_t - np.mean(cmw_t) 

        # N's of convolution, note we're using the size of the reshaped data
        data_n = len(data)
        kern_n = len(cmw_t)
        conv_n = data_n + kern_n - 1
        half_kern_n = kern_n // 2

        t = np.arange(len(data)) * self.sample_rate

        # Transform the Data time series into a spectrum
        data_x = fft(data, conv_n)

        # Initialize the time-frequency matrix
        tf = np.zeros((self.num_freqs, len(t)))

        # Full-Width Half Maximum - try different out some different values
        s = 0.3 

        # TODO NEXT figure out why we create the wavelet on a different time vector
        for i in range(self.num_freqs):
            # TODO SOON Determine the significance of the parameters of the guassian envelope
            cmw_k = np.exp(1j*2*pi*self.freqs[i]*cmw_t) * np.exp(-4*np.log(2)*cmw_t**2 / s**2)
            cmw_x = fft(cmw_k, conv_n)
            cmw_x = cmw_x / max(cmw_x)

            conv = ifft(data_x * cmw_x)
            conv = conv[(half_kern_n):(-half_kern_n+1)]
            conv_pow = abs(conv)**2
            tf[i,:] = conv_pow

        return tf