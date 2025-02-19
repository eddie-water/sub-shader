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
wavelet = Wavelet(sampling_freq = sampling_freq, 
                  frame_size = FRAME_SIZE,
                  downsample_factor = DOWNSAMPLE_FACTOR)
coefs = wavelet.compute_cwt(audio_data)
coefs = np.transpose(coefs)

# PyTorchWavelet
dt = wavelet.get_sample_period()
dj = wavelet.get_scale_factor()
torch_wavelet = WaveletTransformTorch(dt = dt, dj = dj, unbias = False)
torch_coefs = torch_wavelet.power(audio_data)

# PyCuDwt (doesn't have CWT unforunately)
W = Wavelets(audio_data, "db2", 3)
W.forward()
W.soft_threshold(10)
W.inverse()

# Plot
fig, axes = plt.subplots(4, 1, figsize=(10,5))

axes[0].set_title("Signal Time Series")
axes[0].plot(audio_data)
axes[0].axis('off')

axes[1].set_title("PyWavelet")
axes[1].imshow(coefs, cmap = "magma", aspect = "auto")
axes[1].axis('off')

axes[2].set_title("PyTorchWavelet")
axes[2].imshow(torch_coefs, cmap = "magma", aspect = "auto")
axes[2].axis('off')

axes[3].set_title("PyCuDwt")
axes[3].imshow(W.image, cmap = "magma", aspect="auto")
axes[3].axis('off')

plt.show()

while(True):
    pass
