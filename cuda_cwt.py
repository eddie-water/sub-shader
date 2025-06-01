import numpy as np
import pyqtgraph as pg
import scipy.datasets
import matplotlib.pyplot as plt

from audio_input import AudioInput
from wavelet import PyWavelet, NumpyWavelet, CupyWavelet

"""
Configurations
"""
# Size of audio sample block processed by CWT
WINDOW_SIZE = 4096

FILE_PATH = "audio_files/c4_and_c7_4_arps.wav"

# Audio Input, Audio Characteristics 
audio_input = AudioInput(path = FILE_PATH, window_size = WINDOW_SIZE)
audio_data = audio_input.get_frame()
sample_rate = audio_input.get_sample_rate() # 44.1 kHz

"""
Wavelet Objects
"""
# PyWavelet
py_wavelet = PyWavelet(sample_rate = sample_rate, 
                       window_size = WINDOW_SIZE)

coefs_py_wavelet = py_wavelet.compute_cwt(audio_data)

# NumPy ANTS Wavelet
np_wavelet = NumpyWavelet(sample_rate = sample_rate, 
                          window_size = WINDOW_SIZE)

coefs_np_wavelet = np_wavelet.compute_cwt(audio_data)

# Shade Wavelet 
cp_wavelet = CupyWavelet(sample_rate = sample_rate, 
                         window_size = WINDOW_SIZE)

coefs_cp_wavelet = cp_wavelet.compute_cwt(audio_data)

"""
Plotting
"""
fig, axes = plt.subplots(4, 1, figsize=(10,5))

# TODO LATER Fix the axes so they display freqs, not scales
axes[0].set_title("Test Signal Time Series: C4 + C7")
axes[0].plot(audio_data)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Amplitude")
axes[0].margins(x=0, y=0)

axes[1].set_title("PyWavelet CWT")
axes[1].imshow(coefs_py_wavelet, cmap = "magma", aspect = "auto")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Scale")

axes[2].set_title("NumPy CWT")
axes[2].imshow(coefs_np_wavelet, cmap = "magma", aspect = "auto")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Scale")

axes[3].set_title("CuPy CWT")
axes[3].imshow(coefs_cp_wavelet, cmap = "magma", aspect = "auto")
axes[3].set_xlabel("Time")
axes[3].set_ylabel("Scale")

plt.tight_layout()
plt.show()

while(True):
    pass
