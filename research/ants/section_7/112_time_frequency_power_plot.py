"""
Full Time-Frequency Power Plot

Analyzing Neural Time Series (ANTS) by Mike X Cohen, Lessons 111 and 112

This script diverges from the lesson and example code becuase this script 
implements the Continuous Wavelet Transform on a single trial of example 
neurosignal data, rather than 200 trials. Regarding the application I'm working 
towards, I won't have multiple trials to conduct multiple convolutional runs 
on. I'm just going to have a signal that comes in and needs to be analyzed 
accurately as fast as possible. So I don't see where running multiple trials of 
anything will play a part in it. 
"""
import os
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
import matplotlib.pyplot as plt

pi = np.pi

"""
Load the data and prepare for the convolutional loop

The sample rate needed to be defined explicitly here because in the video
lesson, it is automatically defined when loading the .mat file in Matlab
"""
# Grab channel 5, all the time samples (1527), and trial #100
file_data = loadmat("./udemy/section7/ANTS_matlab/v1_laminar.mat")
csd = file_data["csd"]
data = np.squeeze(csd[5,:,100])
data = data.astype(np.float64) # double precision

sample_rate = 762.9395

# Create a centered time vector for the CMW
cmw_t = np.arange(2*sample_rate) / sample_rate
cmw_t = cmw_t - np.mean(cmw_t) 

# N's of convolution, note we're using the size of the reshaped data
data_n = len(data)
kern_n = len(cmw_t)
conv_n = data_n + kern_n - 1
half_kern_n = kern_n // 2

t = file_data['timevec']
t = t.flatten()

# Transform the Data time series into a spectrum
data_x = fft(data, conv_n)

# Frequency parameters in Hz
min_freq = 5 
max_freq = 90
num_freq = 30

# List of frequencies / frequency range
freqs = np.linspace(min_freq, max_freq, num_freq)

# Initialize TF matrix
tf = np.zeros((num_freq, len(t)))

"""
Convolutional Loop:

Creates a wavelet centered around a particular frequency from the list of 
frequencies. Transform the wavelet kernel into a spectrum and normalize it. 
Multiply the data (that was already transformed before the loop) and wavelet 
spectra to produce a filtered output signal (convolution result) back into the 
time domain, trim the wings and reshape back into the original data shape. 
Extract the power of the filtered signal. Average the power of all the trials 
and store it in a matrix to easily plot in time-frequency plot.
"""
# TODO ISSUE-36 Full-Width Half Maximum - try different out some different values
s = 0.3 

# TODO ISSUE-36 figure out why we create the wavelet on a different time vector
for i in range(num_freq):
    # TODO ISSUE-36 Determine the significance of the parameters of the guassian envelope
    cmw_k = np.exp(1j*2*pi*freqs[i]*cmw_t) * np.exp(-4*np.log(2)*cmw_t**2 / s**2)
    cmw_x = fft(cmw_k, conv_n)
    cmw_x = cmw_x / max(cmw_x)

    conv = ifft(data_x * cmw_x)
    conv = conv[(half_kern_n):(-half_kern_n+1)]
    conv_pow = abs(conv)**2
    tf[i,:] = conv_pow

fig, ax = plt.subplots()

ax.contourf(np.squeeze(t), freqs, tf, 40, vmin = 0, vmax = 10000, cmap = "jet")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()