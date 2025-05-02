import os
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
import matplotlib.pyplot as plt

pi = np.pi

file_data = loadmat("./udemy/section7/ANTS_matlab/v1_laminar.mat")
csd = file_data["csd"]
data = csd[5,:,9] 
data = data.flatten()

# Not part of the Matlab code, but the .mat file defines this when it gets 
# loaded in Matlab the script. Need to do it explicitly myself 
sample_rate = 762.9395

# Create a centered time vector for the CMW
cmw_t = np.arange(2*sample_rate) / sample_rate
cmw_t = cmw_t - np.mean(cmw_t) 

# 'Center Frequency' of the Wavelet (I think)
freq = 45

# Create a guassian window using the 'num-cycles' formula. Don'cmw_t know what that is?
s = 7 / (2*pi*freq)

# Create a Complex Morlet Wavelet
cmw = np.exp(1j*2*pi*freq*cmw_t) * np.exp((-cmw_t**2) / (2*s**2))

# N's of Convolution
t = file_data['timevec']
t = t.flatten()

data_n = len(t)
kern_n = len(cmw)
conv_n = data_n + kern_n - 1    # convolution length
half_kern_n = kern_n // 2       # this equivalent to floor(kern_n / 2)

'''
I like to use the '_x' suffix to signify that this variable represents frequency
domain data / spectrum data
'''

# FFT the signal and the kernel, normalize the kernel
data_x = fft(data, conv_n)
kern_x = fft(cmw, conv_n)
kern_x = kern_x / max(kern_x)

# Need to plot these separately bc they have such a large magnitude discrepancy
plt.plot(abs(data_x))
plt.show()
plt.show()

plt.plot(abs(kern_x))
plt.show()

# Multiply the spectra
conv_x = data_x * kern_x

# These plots are off by a little: compare this plot to Lesson 110 @ 8:13 
plt.plot(abs(data_x))
plt.plot(abs(conv_x))
plt.show()

# IFFT result back to the time domain and cut off wings
conv = ifft(conv_x)
conv = conv[half_kern_n : -half_kern_n+1]

# I don't trust this guy's x-axes... but here goes it
hz = np.linspace(0, (sample_rate / 2), (conv_n//2) + 1)

# Plot all the signals
fig, axes = plt.subplots(2)

# Frequency Domain
axes[0].plot(hz, abs(data_x[0:len(hz)]), color = 'orange', label = "Input Signal")
axes[0].plot(hz, abs(kern_x[0:len(hz)]) * max(abs(data_x))/2, color = 'blue', label = '45 Hz Wavelet Kernel')
axes[0].plot(hz, abs(conv_x[0:len(hz)]), color = 'black', linewidth = 2, label = 'Ouput Signal')

axes[0].set_title('Frequency Domain Plot')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_xlim(0, 2*freq)
axes[0].set_ylabel('Amplitude (A. U.)')
axes[0].set_ylim(0, 3e5)
axes[0].legend(loc = 'upper right')

# Time Domain
axes[1].plot(t, data, color = 'orange', label = "Input Signal")
axes[1].plot(t, conv.real, color = 'black', linewidth = 2, label = 'Ouput Signal')

axes[1].set_title('Time Domain Plot')
axes[1].set_xlabel('Time (ms)')
axes[1].set_xlim(-0.1, 1.3)
axes[1].set_ylabel('Amplitude (uV)')
axes[1].set_ylim(-2000, 2000)
axes[1].legend(loc = 'upper right')

plt.tight_layout()
plt.show()