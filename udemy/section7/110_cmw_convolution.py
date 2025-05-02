import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
import matplotlib.pyplot as plt

pi = np.pi

file_data = loadmat("./ANTS_matlab/v1_laminar.mat")
csd = file_data["csd"]
data = csd[5,:,9]
print("data shape =", data.shape)
data = data.flatten()
print("data shape flattened =", data.shape)

sample_rate = 1000

# Create a centered time vector for the CMW
cmw_t = np.arange(2*sample_rate) / sample_rate
cmw_t = cmw_t - np.mean(cmw_t) 

# 'Center Frequency' of the Wavelet (I think)
freq = 45

# Create a guassian window using the 'num-cycles' formula. Don'cmw_t know what that is?
s = 7 / (2*pi*freq)

# Create a Complex Morlet Wavelet
cmw = np.exp(1j*2*pi*freq*cmw_t) * np.exp((-cmw_t**2) / ((2*s)**2))

# N's of Convolution
t = file_data['timevec']
print("t shape =", data.shape)
t = t.flatten()
print("t shape flattened =", data.shape)

data_n = len(t)
kern_n = len(cmw)
conv_n = data_n + kern_n - 1 # convolution length
half_kern_n = kern_n // 2

'''
I like to use the '_x' suffix to signify that this variable represents frequency
domain data / spectrum
'''

# FFT the signal and the kernel, normalize the kernel
data_x = fft(data, conv_n)
kern_x = fft(cmw, conv_n)
kern_x = kern_x / max(kern_x)

# Multiply the spectra, IFFT result back to the time domain and cut off wings
conv_x = data_x * kern_x
conv = ifft(conv_x)
conv = conv[half_kern_n : -half_kern_n+1]

# I don't trust this guy's x-axes... but here goes it
hz = np.linspace(0, sample_rate / 2, (conv_n//2+1))

# Result of a convolution of a CMW centered at 45 Hz and 
plt.plot(t, data, color = 'orange', label = "LFP data")
plt.plot(t, abs(conv), color = 'black', linewidth = 2, label = 'Convolved Data')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (uV)')
plt.show()