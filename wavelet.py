import numpy as np
import pywt

# Most audio is sampled 44.1 kHz
TYPICAL_SAMPLING_FREQ = 44100

# Musical Scale parameters 
NOTES_PER_OCTAVE = 12 
NUM_OCTAVES = 10
ROOT_NOTE_A0 = 27.5

class Wavelet():
    def __init__(self, sampling_freq: int, frame_size: int):
        if (sampling_freq != TYPICAL_SAMPLING_FREQ):
            raise ValueError("Irregular Sampling Frequency")

        # Sampling Info
        self.sampling_freq = sampling_freq
        self.nyquist_freq = (sampling_freq / 2.0)
        self.sampling_period = (1.0 / self.sampling_freq)
        self.frame_size = frame_size

        # Wavelet info
        self.wavelet_name = "cmor1.5-1.0" # TODO LATER why 1.5-1.0?

        # Configure time and frequency axes
        self.time = np.arange(0, frame_size) * self.sampling_period

        # Frequency array that mimics the variable step size of musical notes
        scale_factor = 2**(1/NOTES_PER_OCTAVE)
        i = np.arange(0, NOTES_PER_OCTAVE*NUM_OCTAVES, 1)
        s = scale_factor**i
        self.freq = ROOT_NOTE_A0*s

        # Discard frequencies that are unmeasurable
        self.freq = self.freq[self.freq < self.nyquist_freq]

        # Scale array used to specify wavelet dilation amounts during cwt
        f_norm = (self.freq / self.sampling_freq) # TODO comment why we do this
        self.scales = pywt.frequency2scale(self.wavelet_name, f_norm)

    def get_shape(self) -> np.ndarray.shape:
        """Computes the shape of the resultant CWT data
        
        Returns:
            Shape of the computed CWT data"""

        return np.empty((self.freq.size, self.time.size)).shape

    def compute_cwt(self, audio_data): 
        """Performs the CWT

        Args:
            audio_data: audio signal data

        Returns:
            abs_coefs: the absolute values of the CWT coefficients
        """
        coefs, freqs = pywt.cwt(data= audio_data,
                                scales= self.scales,
                                wavelet= self.wavelet_name,
                                sampling_period= self.sampling_period)

        # Important step - take abs value of the data and reduce axes by 1
        abs_coefs = np.abs(coefs[:-1, :-1])

        # TODO here is where we would use a log scale to sharpen the colors

        return abs_coefs
