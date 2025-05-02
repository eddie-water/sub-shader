# Following this:
# https://github.com/dxganta/Neural-Signal-Processing-Analysis/blob/main/Section%207/uANTS_timefreq.ipynb

import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat("./ANTS_matlab/v1_laminar.mat")

sample_rate = data['srate']
t = data['timevec'].T

# Signal will be ERP data from channel 7
signal = np.mean(data['csd'][6, :, :], axis=1)

# Create a Gaussian Curve and normalize it
h = .05; # FWHM (seconds)
gaus_t =  np.arange(-1,1,1/sample_rate)
gaus = np.exp(-4 * np.log(2) * gaus_t**2 / h**2 )
gaus = gaus / sum(gaus) 

# Step 1: N's of Convolution
signal_len = len(signal)
kernel_len = len(gaus)
conv_len = signal_len + kernel_len - 1 # convolution length
half_kern_len = kernel_len // 2

# Step 2: FFT the Signal and Kernel
signal_x = fft(signal, n=conv_len)
kernel_x = fft(gaus, n=conv_len)

# plt.plot(signal_x)
# plt.plot(kernel_x)
# plt.show()

# Step 3: Multiply the Spectra
conv_x = signal_x * kernel_x

# Step 4: IFFT the Spectrum back to Time Domain
conv = ifft(conv_x)

# Step 5: Cut off the "Wings"
conv = conv[half_kern_len : -half_kern_len + 1]
plt.plot(t, signal, label='Original ERP')
plt.plot(t, conv.real,'r',linewidth=2, label='Gaussian-convolved')

plt.xlim([-.1, 1.4])
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Activity (uV)')
plt.show()