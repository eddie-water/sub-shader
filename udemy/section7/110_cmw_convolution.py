import os
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
import matplotlib.pyplot as plt

pi = np.pi

'''
Load the data from a file and prepare for the convolution
'''
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

# Guassian window using the 'num-cycles' formula. Don't know what that is...?
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
Perform the Convolution
  - The '_x' suffix signifies variables that are in the frequency domain and 
    represent spectrum data
'''
# FFT the signal and the kernel 
data_x = fft(data, conv_n)
kern_x = fft(cmw, conv_n)

# Normalize the kernel
kern_x = kern_x / max(kern_x)

# Multiply the spectra (this is equivalent to time domain convolution)
conv_x = data_x * kern_x

# IFFT result back to the time domain and cut off wings
conv = ifft(conv_x)
conv = conv[half_kern_n : -half_kern_n+1]

'''
Plot each signal/wavelet to see what they look like in freq vs time domain
'''
fig, axes = plt.subplots(5)

fig.canvas.manager.set_window_title('Convolution of Signal and Complex Morlet Wavelet @ 45 Hz')

axes[0].plot(data, color = 'orange')
axes[0].set_title('Input Signal (Time Domain)')

axes[1].plot(abs(data_x), color = 'orange')
axes[1].set_title('Input Signal (Frequency Domain)')

axes[2].plot(cmw, color = 'blue')
axes[2].set_title('45 Hz CMW (Time Domain)')

axes[3].plot(abs(kern_x), color = 'blue')
axes[3].set_title('45 Hz CMW (Frequency Domain)')

axes[4].plot(abs(data_x), color = 'orange')
axes[4].plot(abs(conv_x), color = 'black')
axes[4].set_title('Input and Output Signal (Frequency Domain)')

plt.tight_layout()
plt.show()

'''
Plotting the Frequency and Time Domain of the Input, Wavelet, and Output 
'''
# I don't trust this guy's x-axes... but here goes it
hz = np.linspace(0, (sample_rate / 2), (conv_n//2) + 1)

fig, axes = plt.subplots(2)

fig.canvas.manager.set_window_title('Filtering a Signal with a Complex Morlet Wavelet @ 45 Hz')

# Frequency Domain
axes[0].plot(hz, abs(data_x[0:len(hz)]), color = 'orange', label = "Input Signal")
axes[0].plot(hz, abs(kern_x[0:len(hz)]) * max(abs(data_x))/2, color = 'blue', label = 'Wavelet Spectrum')
axes[0].plot(hz, abs(conv_x[0:len(hz)]), color = 'black', linewidth = 2, label = 'Ouput Signal')

axes[0].set_title('Frequency Domain Plot')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_xlim(0, 2*freq)
axes[0].set_ylabel('Amplitude (A. U.)')
axes[0].set_ylim(0, 3e5)
axes[0].legend(loc = 'upper right')

# Time Domain
axes[1].plot(t, data, color = 'orange', label = "Input Signal")
axes[1].plot(t, conv.real, color = 'black', linewidth = 2, label = 'Output Signal [Real]')

axes[1].set_title('Time Domain Plot')
axes[1].set_xlabel('Time (ms)')
axes[1].set_xlim(-0.1, 1.3)
axes[1].set_ylabel('Amplitude (uV)')
axes[1].set_ylim(-2000, 2000)
axes[1].legend(loc = 'upper right')

plt.tight_layout()
plt.show()

'''
Analyzing the Filtered Signal (Output Signal) 
  - Power and phase plots
'''
fig, axes = plt.subplots(3)
fig.canvas.manager.set_window_title('Analyzing the Filtered Signal')

axes[0].plot(t, conv.real, color = 'black', linewidth = 2, label = 'Output Signal [Real]')
axes[0].set_title('Real Part of the Filtered Signal')
axes[0].set_xlabel('Time (ms)')
axes[0].set_xlim(-0.1, 1.4)
axes[0].set_ylabel('Amplitude (uV)')

axes[1].plot(t, abs(conv), color = 'indianred', linewidth = 2, label = 'Output Signal [Real]')
axes[1].set_title('Power Plot')
axes[1].set_xlabel('Time (ms)')
axes[1].set_xlim(-0.1, 1.4)
axes[1].set_ylabel('Amplitude (uV^2)')

'''
Special Note
  - Looking at around 0.7 ms, there is a "Phase Slip" in the Phase
    Plot. This is the because the overall power approached 0, which apparently 
    makes it very difficult to determine the phase. To me it makes sense when I 
    try thinking about it in terms of the polar plot. Obviously we are in a 
    digital computing environment which is inherently discrete. So when the 
    power magnitude gets close to 0, the angle resolution probably sucks, so it 
    they just get binned together, causing a disruption in the actual flow of 
    the phase. That is all to say, the 'blip' in phase is not necessarily from 
    the physical signal itself, but is an issue with the analysis of points in 
    the power plot that are close to 0.
'''
axes[2].plot(t, np.angle(conv), color = 'mediumslateblue', linewidth = 2, label = 'Output Signal [Real]')
axes[2].set_title('Phase Plot')
axes[2].set_xlabel('Time (ms)')
axes[2].set_xlim(-0.1, 1.4)
axes[2].set_ylabel('Phase (rad)')

plt.tight_layout()
plt.show()