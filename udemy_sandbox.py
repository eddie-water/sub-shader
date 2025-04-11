import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 1000
step = 1/SAMPLE_RATE
t = np.arange(-1, 1, step)

# why is the frequency 2pi? aka like 6 Hz lol
f_wavelet = 2*np.pi 

sine_wave = np.cos(2*np.pi*f_wavelet*t)