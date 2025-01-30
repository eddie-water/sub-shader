import numpy as np
import pywt

# Most audio is sampled 44.1 kHz
TYPICAL_SAMPLING_FREQ = 44100

# Musical Scale parameters 
NOTES_PER_OCTAVE = 12 
NUM_OCTAVES = 10
ROOT_NOTE_A0 = 27.5

class Wavelet():
    def __init__(self, sampling_freq: int, frame_size: int, downsample_factor: int):
        if (sampling_freq != TYPICAL_SAMPLING_FREQ):
            raise ValueError("Irregular Sampling Frequency")

        # Sampling Info
        self.sampling_freq = sampling_freq
        self.nyquist_freq = (sampling_freq / 2.0)
        self.sampling_period = (1.0 / self.sampling_freq)

        self.frame_size = frame_size

        self.downsample_factor = downsample_factor

        # Wavelet info
        self.wavelet_name = "cmor1.5-1.0" # TODO LATER why 1.5-1.0?

        # Time Axis
        self.time = np.arange(0, frame_size) * self.sampling_period

        # Frequency Axis that mimics the variable step size of musical scales
        scale_factor = 2**(1/NOTES_PER_OCTAVE)
        i = np.arange(0, NOTES_PER_OCTAVE*NUM_OCTAVES, 1)
        s = scale_factor**i
        self.freq = ROOT_NOTE_A0*s

        # Discard frequencies that are unmeasurable
        self.freq = self.freq[self.freq < self.nyquist_freq]

        # Scale array used to specify wavelet dilation amounts during cwt
        f_norm = (self.freq / self.sampling_freq) # TODO comment why we do this
        self.scales = pywt.frequency2scale(self.wavelet_name, f_norm)

    """
    Computes the shape of the resultant CWT data

    Returns:
        Shape of the computed CWT data
    """
    def get_shape(self) -> np.ndarray.shape:
        return np.empty((self.freq.size, self.time.size)).shape

    """
    Performs the Continuous Wavelet Transform and normalizes the data
    - Computes the CWT on the raw audio data
    - Takes the absolute value to get the magnitude of the resultant coefs
    - Normalizes the coefs against the scale to compensate for energy 
      accumulation bias in the CWT with higher scales (low frequencies)
    - Normalizes the coefs so the min and max map to 0 and 1
    - Downsamples the coefs to reduce plotting time
    - Transposes the coefs because the CWT swaps the axes for some reason

    Args:
        audio_data: raw audio signal data

    Returns:
        coefs: the normalized CWT coefficients

    """
    def compute_cwt(self, audio_data): 
        coefs, _ = pywt.cwt(data = audio_data,
                            scales = self.scales,
                            wavelet = self.wavelet_name,
                            sampling_period = self.sampling_period)

        # Absolute Value 
        coefs_abs = np.abs(coefs)

        # Scale-Based Normalization 
        coefs_scaled = coefs_abs / np.sqrt(self.scales[:, None])

        # Min-Max Normalization 
        coefs_min = np.min(coefs_scaled)
        coefs_max = np.max(coefs_scaled)
        coefs_norm = (coefs_scaled - coefs_min) / (coefs_max - coefs_min)

        # Downsample 
        coefs = coefs_norm[::, ::(self.downsample_factor)]

        # Swap Axes
        coefs = np.transpose(coefs)
        return coefs