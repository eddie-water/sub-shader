import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(5,5))

SAMPLE_RATE = 1000
step = 1/SAMPLE_RATE
t = np.arange(-1, 1, step)

# why is the frequency 2pi? aka like 6 Hz lol
f_wavelet = 2*np.pi 

sine_wave = np.cos(2*np.pi*f_wavelet*t)

fwhm = 0.5
gaus_win = np.exp((-4 * np.log(2) * t**2) / (fwhm**2))

morlet_wavelet = sine_wave * gaus_win

axes[0].plot(t, sine_wave, color = 'dodgerblue')
axes[1].plot(t, gaus_win, color = 'orange')

axes[2].plot(t, sine_wave, color = 'dodgerblue')
axes[2].plot(t, gaus_win, color = 'orange')
axes[2].plot(t, morlet_wavelet, lw = 3, color = 'black')

plt.tight_layout()
plt.show()