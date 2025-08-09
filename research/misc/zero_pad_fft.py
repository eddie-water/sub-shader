"""
This script generates an FFT of a given signal and compares it with the FFT of 
the same signal with zero-padding of different styles
"""
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import pyqtgraph as pg
import scipy.datasets

from audio_input import AudioInput
from wavelet import PyWavelet
from wavelet import AntsWavelet
from wavelet import CuWavelet

import numpy as np
import matplotlib.pyplot as plt

def generate_custom_wavelets(freqs, fs, target_len=4096, std_multiplier=5, fwhm=0.3):
    """
    Generate custom complex Morlet-like wavelets with 5-sigma time support,
    zero-padded to a common length.

    Parameters:
        freqs : array-like
            Center frequencies of the wavelets in Hz.
        fs : float
            Sampling rate in Hz.
        target_len : int
            Desired length of the padded wavelet (e.g., 4096).
        std_multiplier : float
            Number of standard deviations of time support.
        fwhm : float
            Full width at half max of the Gaussian envelope (in seconds).

    Returns:
        wavelets : np.ndarray
            Array of shape (len(freqs), target_len) with each padded wavelet.
    """
    wavelets = []

    # Convert FWHM to standard deviation
    std_t = fwhm / (2 * np.sqrt(2 * np.log(2)))  # ~ 0.127 for fwhm=0.3

    for f in freqs:
        t_max = std_multiplier * std_t
        t = np.arange(-t_max, t_max, 1/fs)

        # Custom Morlet-like wavelet: complex sine × Gaussian envelope
        kernel = np.exp(1j * 2 * np.pi * f * t) * np.exp(-4 * np.log(2) * t**2 / fwhm**2)

        # Center pad to target_len
        pad_total = target_len - len(kernel)
        if pad_total < 0:
            raise ValueError(f"Wavelet at {f} Hz is longer than target length {target_len}")
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded_kernel = np.pad(kernel, (pad_left, pad_right), mode='constant')

        wavelets.append(padded_kernel)

    return np.array(wavelets)

# Example usage
fs = 44100  # Sample rate
freqs = np.geomspace(50, 5000, num=10)  # 10 log-spaced frequencies between 50 Hz and 5 kHz

wavelet_bank = generate_custom_wavelets(freqs, fs, target_len=4096)

# Plot a few wavelets
plt.figure(figsize=(10, 4))
for i in [0, 3, 6, 9]:
    plt.plot(wavelet_bank[i].real, label=f"{freqs[i]:.1f} Hz")
plt.title("Real Part of Custom Complex Wavelets (Padded)")
plt.xlabel("Sample")
plt.legend()
plt.grid()
plt.show()



WINDOW_SIZE = 4096

DOWNSAMPLE_FACTOR = 1

FILE_PATH = "audio_files/c4_and_c7_4_arps.wav"

pi = np.pi

audio_input = AudioInput(path = FILE_PATH, window_size = WINDOW_SIZE)
audio_data = audio_input.get_frame()
sample_rate = audio_input.get_sample_rate() 



# create a wavelet kernel in the time domain
def create_wavelet_kernel(f, sample_rate, duration=2):
    t = np.arange(0, duration, 1/sample_rate)
    kernel = np.exp(1j * 2 * np.pi * f * t) * np.exp(-4 * np.log(2) * t**2 / (0.3**2))
    return kernel

F = 55 
wavelet_kernel = create_wavelet_kernel(F, sample_rate, duration=0.1)
N = len(wavelet_kernel)

# Right padding produces a true linear convolution of length N + M - 1
def right_pad(signal, pad_length):
    return np.pad(signal, (0, pad_length), mode='constant')

# Symmetric padding produces a center aligned filter 
def symmetric_pad(signal, pad_length):
    left_pad = pad_length // 2
    right_pad = pad_length - left_pad
    return np.pad(signal, (left_pad, right_pad), mode='constant')