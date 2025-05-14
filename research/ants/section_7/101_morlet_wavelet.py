import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

"""
Plot Init
"""
fig, axes = plt.subplots(4, 1)

"""
Morlet Wavelet - Time Domain
"""
SAMPLE_RATE = 1000
step = 1/SAMPLE_RATE
t = np.arange(-1, 1, step)

f_wavelet = 5 # Hz

sine_wave = np.cos(2*np.pi*f_wavelet*t)

fwhm = 0.5
gaus_win = np.exp((-4 * np.log(2) * t**2) / (fwhm**2))

morlet_wavelet = sine_wave * gaus_win

axes[0].plot(t, sine_wave, color = 'blue')
axes[0].set_title(str(f_wavelet) + 'Hz Sinusoid')

axes[1].plot(t, gaus_win, color = 'orange')
axes[1].set_title('Guassian Curve')

axes[2].plot(t, sine_wave, color = 'blue')
axes[2].plot(t, gaus_win, color = 'orange')
axes[2].plot(t, morlet_wavelet, color = 'black')
axes[2].set_title('Morlet Wavelet - Time Domain')

"""
Morlet Wavelet - Frequency Domain
"""
num_points = len(t)
mwX = np.abs(fft(morlet_wavelet) / num_points)
hz = np.linspace(0, SAMPLE_RATE, num_points)

axes[3].plot(hz, mwX, color = 'black')
axes[3].set_title('Morlet Wavelet - Frequency Domain')

"""
Show Plots
"""
plt.tight_layout()
plt.show()