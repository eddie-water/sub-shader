import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

sample_rate = 1000
t = np.arange(-1, 1, (1 / sample_rate))
freq = 2*pi # This is a dumb frequency to show case

sine_w = np.exp(1j*2*pi*freq*t)

plt.plot(sine_w.real)
plt.plot(sine_w.imag)
plt.show()

fwhm = 0.5
gaus_window = np.exp(-4)