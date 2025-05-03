import os
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import loadmat
import matplotlib.pyplot as plt

pi = np.pi

'''
Load the data from a file and prepare for the convolution

Note: the sample rate needs to be defined explicitly here because in the video
lesson, it is automatically defined when loading the .mat file in Matlab
'''
# Grab channel 5, all the time samples (1527), all the trials (200)
file_data = loadmat("./udemy/section7/ANTS_matlab/v1_laminar.mat")
csd = file_data["csd"]
data = np.squeeze(csd[5,:,:])

# Reshape the data - TODO NEXT Why do we need to reshape it?
data_r = np.reshape(data, -1, order='F')

sample_rate = 762.9395

# Create a centered time vector for the CMW
cmw_t = np.arange(2*sample_rate) / sample_rate
cmw_t = cmw_t - np.mean(cmw_t) 

# N's of convolution, note we're using the size of the reshaped data
data_n = len(data_r)
kern_n = len(cmw_t)
conv_n = data_n + kern_n - 1
half_kern_n = kern_n // 2

t = file_data['timevec']
t = t.flatten()

# Transform the Data time series into a spectrum
data_x = fft(data_r, conv_n)

# Frequency parameters in Hz
min_freq = 5 
max_freq = 90
num_freq = 30

# List of frequencies / frequency range
freqs = np.linspace(min_freq, max_freq, num_freq)

# Initialize TF matrix
tf = np.zeros((num_freq, len(t)))

'''
TODO NOW

Create a for loop that:
Creates a wavelet centered around a particular frequency from our list of 
frequencies. Transform the wavelet kernel into a spectrum and normalize it. 
Multiply the data and wavelet spectra to produce a filtered output signal. 
Transform back into the time domain, trim the wings and reshape back into the 
original data shape. Extract the power of the filtered signal. Average the 
power of all the trials and store it in a matrix to easily plot the '3D' result.
'''