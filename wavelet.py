import numpy as np
import pywt

# Signal Info
SAMPLING_FREQUENCY = 44100
NYQUIST_FREQUENCY = SAMPLING_FREQUENCY / 2
SAMPLING_PERIOD = 1.0 / SAMPLING_FREQUENCY
AUDIO_LENGTH_SECONDS = .1

# Calculate the set of scales we are interested in 
VOICES_PER_OCTAVE = 12 # TODO maybe call this notes/ per octave 
NUM_OCTAVES = 10
ROOT_NOTE_A0 = 27.5

class Wavelet():
    def __init__(self, frame_size: int):
        """Configuration for the Continuous Wavelet Transform"""
        self.frame_size = frame_size

        # Wavelet info
        self.wavelet_name = "cmor1.5-1.0" # TODO LATER why 1.5-1.0?
        self.sampling_period = SAMPLING_PERIOD

        # Configure time and frequency axes
        self.time = np.arange(0, frame_size) * self.sampling_period

        # Frequency array that mimic the variable step size of musical notes
        scale_factor = 2**(1/VOICES_PER_OCTAVE)
        i = np.arange(0, VOICES_PER_OCTAVE*NUM_OCTAVES, 1)
        s = scale_factor**i
        self.freq = ROOT_NOTE_A0*s

        # Discard frequencies that are unmeasurable
        self.freq = self.freq[self.freq < NYQUIST_FREQUENCY]

        # Scale array used to specify wavelet dilation amounts during cwt
        f_norm = self.freq / SAMPLING_FREQUENCY # TODO add comment why we do this
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

        # Important step - take abs value of the data
        abs_coefs = np.abs(coefs[:-1, :-1])
        return abs_coefs
