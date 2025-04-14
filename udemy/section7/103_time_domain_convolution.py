import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

"""
Kernels and Convolutions in the Time Domain 1

What is a kernel and what's their purpose?
    - A sequence of numbers that acts like a template or filter for processing
    signals or data
    - Tells you how to weight the affect of nearby points in the data you are
    analyzing
    - Can be used to modify the data or detect features in the data when used
    in convolution (or correlation)
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
axes[1, 0].set_title('Kernel')

result = np.convolve(signal, kernel, mode = 'same')
axes[2, 0].plot(signal, color = 'black')
axes[2, 0].plot(result, color = 'red')
axes[2, 0].set_title('Output Signal')

# Inverted Smoothing Kernel
inverted_kernel = -1*kernel
inverted_result = np.convolve(signal, inverted_kernel, mode = 'same')

axes[0, 1].plot(signal, color = 'black')
axes[0, 1].set_title('Input Signal')

axes[1, 1].plot(inverted_kernel)
axes[1, 1].set_title('Inverted Kernel')

axes[2, 1].plot(signal, color = 'black')
axes[2, 1].plot(inverted_result, color = 'red')
axes[2, 1].set_title('Output Signal')

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
axes[2, 2].set_title('Output Signal')

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
fig, axes = plt.subplots(3, 2)

arrays = [np.zeros(8), np.ones(7), np.zeros(5)]
signal = np.concatenate(arrays)

# Linearly Decreasing Kernel
kernel = np.array([1, .8, .6, .4, .2])

signal_length = len(signal)
kernel_length = len(kernel)
conv_length = signal_length + kernel_length - 1

axes[0, 0].plot(signal, color = 'black')
axes[0, 0].set_title('Input Signal')

axes[1, 0].plot(kernel)
axes[1, 0].set_title('Linearly Decreasing Kernel')

result = np.convolve(signal, kernel, mode = 'same')
axes[2, 0].plot(signal, color = 'black')
axes[2, 0].plot(result, color = 'red')
axes[2, 0].set_title('Output Signal')

# Reverse the kernel and shift it down by the kernels average value
mean_centered_kernel = kernel[::-1] - np.mean(kernel)

"""
Why do we reverse the kernel?


"""

axes[0, 1].plot(signal, color = 'black')
axes[0, 1].set_title('Input Signal')

axes[1, 1].plot(mean_centered_kernel)
axes[1, 1].set_title('Mean Centered Kernel')

result = np.convolve(signal, mean_centered_kernel, mode = 'same')
axes[2, 1].plot(signal, color = 'black')
axes[2, 1].plot(result, color = 'red')
axes[2, 1].set_title('Output Signal')

"""
Display Plot
"""
plt.tight_layout()
plt.show()