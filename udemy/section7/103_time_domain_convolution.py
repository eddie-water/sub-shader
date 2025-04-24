import numpy as np
import matplotlib.pyplot as plt
from plot_time_domain_conv import plot_time_domain_conv

"""
Kernels and Convolutions in the Time Domain 1

What is a kernel and what's its purpose?
    - A sequence of numbers that acts like a template or filter for processing
      signals or data
    - Tells you how to weight the affect of nearby points in the data you are
      analyzing
    - Can be used to modify, boost, or reduce parts of the data or detect 
      features in the data when used in convolution (or correlation)
"""
fig, axes = plt.subplots(3, 3)

t = np.linspace(-2, 2, 20)

# Smoothing Kernel
kernel = np.exp(-1*t**2)
kernel = kernel / np.sum(kernel)

arrays = [ np.zeros(30), 
           np.ones(2), 
           np.zeros(20), 
           np.ones(30), 
           2*np.ones(10), 
           np.zeros(30), 
           -1*np.ones(10), 
           np.zeros(40) ]

signal = np.concatenate(arrays)

axes[0, 0].plot(signal, color = 'black')
axes[0, 0].set_title('Input Signal')

axes[1, 0].plot(kernel)
axes[1, 0].set_title('Smoothing Kernel')

result = np.convolve(signal, kernel, mode = 'same')
axes[2, 0].plot(signal, color = 'black')
axes[2, 0].plot(result, color = 'red')
axes[2, 0].set_title('Output Signal - Smoothed')

# Inverted Smoothing Kernel
inverted_kernel = -1*kernel
inverted_result = np.convolve(signal, inverted_kernel, mode = 'same')

axes[0, 1].plot(signal, color = 'black')
axes[0, 1].set_title('Input Signal')

axes[1, 1].plot(inverted_kernel)
axes[1, 1].set_title('Inverted Smoothing Kernel')

axes[2, 1].plot(signal, color = 'black')
axes[2, 1].plot(inverted_result, color = 'red')
axes[2, 1].set_title('Output Signal - Inverted Smoothed')

# Signed Edge Detection Kernel
arrays = [np.zeros(9), np.ones(1), -1*np.ones(1), np.zeros(9)]
edge_kernel = np.concatenate(arrays)
edge_result = np.convolve(signal, edge_kernel, mode = 'same')

axes[0, 2].plot(signal, color = 'black')
axes[0, 2].set_title('Input Signal')

axes[1, 2].plot(edge_kernel)
axes[1, 2].set_title('Edge Detection Kernel')

axes[2, 2].plot(signal, color = 'black')
axes[2, 2].plot(edge_result, color = 'red')
axes[2, 2].set_title('Output Signal - Edge Detection')

"""
Kernels and Convolutions in the Time Domain 2

What is convolution and why do we need to flip the kernel when doing it?
    - Convolution is a model of causality aka cause-and-effect: at the current 
      time 't', what is the total accumulated effect of all the past inputs, 
      weighted by the system's response to those inputs
    - Time domain convolution occurs when you flip the kernel, overlap it with
      the input signal, perform the dot (inner) product, store that result in the
      output signal, slide the kernel over by one, and then repeat until the 
      kernel has slid over the entire input 
    - Since we want to see how past events affect the current one, we need to 
      flip the kernel because we are saying time increases from left to right 
    - If you don't flip the kernel, you would be applying the right most sample
      of the kernel to left most sample of the signal, which is basically saying:
      how does a future moment (which hasn't happened yet) affect the current 
      moment (which doesn't make sense for a cause-and-effect model)
"""
fig, axes = plt.subplots(3)

# Input Signal
arrays = [np.zeros(7), np.ones(8), np.zeros(6)]
signal = np.concatenate(arrays)

# Linearly Decreasing Kernel
kernel = np.array([1, .8, .6, .4, .2])

# Convolution using NumPy
numpy_result = np.convolve(signal, kernel, mode = 'same')

axes[0].plot(signal, color = 'black')
axes[0].set_title('Input Signal')

axes[1].plot(kernel)
axes[1].set_title('Linearly Decreasing Kernel')

axes[2].plot(signal, color = 'black')
axes[2].plot(numpy_result, color = 'red')
axes[2].set_title('Output Signal')

# Plot still figures
plt.tight_layout()
plt.show()

# Turn on interactive plotting for the next few examples
plt.ion

# Infinite loop of a time domain convolution animation
try:
    plot_time_domain_conv(input_signal = signal, 
                          kernel = kernel, 
                          y_min = -1, 
                          y_max = 4,
                          delay = 0.5)
except Exception as e:
    print("Caught exception:", e)
except KeyboardInterrupt as e:
    print("Caught keyboard exception")

'''
Kernels and Convolutions in the Time Domain 3

What is the point of mean centering the kernel?
    - TODO EVENTUALLY
'''
# Mean center the kernel
kernel = kernel - np.mean(kernel)

try:
    plot_time_domain_conv(input_signal = signal, 
                          kernel = kernel, 
                          y_min = -2, 
                          y_max = 2,
                          delay = 0.5)

except Exception as e:
    print("Caught exception:", e)
except KeyboardInterrupt as e:
    print("Caught keyboard exception")

# Time domain convolution using the morlet wavelet as the kernel
SAMPLE_RATE = 50 # NumPy and Matplotlib can't handle too many points
step = 1 / SAMPLE_RATE

t = np.arange(-1, 1, step)

f_wavelet = 5 # Hz
sine_wave = np.cos(2*np.pi*f_wavelet*t)

fwhm = 0.5 # Full wave half maximum
gaus_win = np.exp((-4 * np.log(2) * t**2) / (fwhm**2))


morlet_wavelet = sine_wave * gaus_win

t = np.arange(0, 2*np.pi, step)
sine_wave = np.cos(2*np.pi*f_wavelet*t)

try:
    plot_time_domain_conv(input_signal = sine_wave, 
                          kernel = morlet_wavelet, 
                          y_min = -20, 
                          y_max = 20,
                          delay = 0.1)
except Exception as e:
    print("Caught exception:", e)
except KeyboardInterrupt as e:
    print("Caught keyboard exception")

plt.ioff