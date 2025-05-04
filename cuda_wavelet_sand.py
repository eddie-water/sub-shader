import numpy as np
import pyqtgraph as pg
import scipy.datasets
import matplotlib.pyplot as plt

from audio_input import AudioInput
from plotter import Plotter
from wavelet import PyWavelet
from wavelet import ShadeWavelet

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
sample_rate = audio_input.get_sample_rate() # 44.1 kHz

# PyWavelet
pywavelet = PyWavelet(sample_rate = sample_rate, 
                      frame_size = FRAME_SIZE,
                      downsample_factor = DOWNSAMPLE_FACTOR)

# TODO SOON wait why am I transposing this if it's being transposed in the compute_cwt method?
coefs_pywavelet = pywavelet.compute_cwt(audio_data)
coefs_pywavelet = np.transpose(coefs_pywavelet)

# Non Accelerated Manual CWT # TODO LATER find a better name for this object class
shade_wavelet = ShadeWavelet(sample_rate = sample_rate, 
                             frame_size = FRAME_SIZE,
                             downsample_factor = DOWNSAMPLE_FACTOR)

coefs_shade_wavelet = shade_wavelet.compute_cwt(audio_data)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10,5))

axes[0].set_title("Signal Time Series")
axes[0].plot(audio_data)
axes[0].axis('off')

axes[1].set_title("PyWavelet")
axes[1].imshow(coefs_pywavelet, cmap = "magma", aspect = "auto")
axes[1].axis('off')

axes[2].set_title("Shade Wavelet")
axes[2].imshow(coefs_shade_wavelet, cmap = "magma", aspect = "auto")
axes[2].axis('off')

plt.tight_layout()
plt.show()

while(True):
    pass
