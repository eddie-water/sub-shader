import numpy as np
import pyqtgraph as pg
import scipy.datasets
import matplotlib.pyplot as plt

from audio_input import AudioInput
from wavelet import Wavelet
from plotter import Plotter

from pycudwt import Wavelets
from wavelets_pytorch.transform import WaveletTransformTorch

# Constants: 
#   Ideally have the biggest frame size and the smallest downsample factor 
FRAME_SIZE = 1024
DOWNSAMPLE_FACTOR = 8

FILE_PATH = "audio_files/c4_and_c7_4_arps.wav"

# Audio Input, Audio Characteristics 
audio_input = AudioInput(path = FILE_PATH, frame_size = FRAME_SIZE)
audio_data = audio_input.get_frame()
sampling_freq = audio_input.get_sample_rate() # 44.1 kHz

# PyWavelet
wavelet_pywavelet = Wavelet(sampling_freq = sampling_freq, 
                            frame_size = FRAME_SIZE,
                            downsample_factor = DOWNSAMPLE_FACTOR)
coefs_pywavelet = wavelet_pywavelet.compute_cwt(audio_data)
coefs_pywavelet = np.transpose(coefs_pywavelet)

# PyTorchWavelet
dt = wavelet_pywavelet.get_sample_period()
dj = wavelet_pywavelet.get_scale_factor()
wavelet_torch = WaveletTransformTorch(dt = dt, dj = dj, unbias = False)
coefs_torch = wavelet_torch.power(audio_data)

# PyCuDwt (doesn't have CWT unforunately)
wavelet_pcudwt = Wavelets(audio_data, "db2", 3)
wavelet_pcudwt.forward()
wavelet_pcudwt.soft_threshold(10)
wavelet_pcudwt.inverse()
coefs_pycudwt = wavelet_pcudwt.image

# Plot
fig, axes = plt.subplots(4, 1, figsize=(10,5))

axes[0].set_title("Signal Time Series")
axes[0].plot(audio_data)
axes[0].axis('off')

axes[1].set_title("PyWavelet")
axes[1].imshow(coefs_pywavelet, cmap = "magma", aspect = "auto")
axes[1].axis('off')

axes[2].set_title("PyTorchWavelet")
axes[2].imshow(coefs_torch, cmap = "magma", aspect = "auto")
axes[2].axis('off')

axes[3].set_title("PyCuDwt")
axes[3].imshow(coefs_pycudwt, cmap = "magma", aspect="auto")
axes[3].axis('off')

plt.show()

while(True):
    pass
