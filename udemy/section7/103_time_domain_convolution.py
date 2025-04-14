import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

"""
Kernel Play 1

What is a convolution?
    - When you convolve a signal and a kernel (filter) you are effectively 
    asking at the current time 't', what is the the total accumulated effect
    of all the past inputs, weighted by the system's response to those inputs.
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
Kernel Play 2

Why do you flip the kernel when convolving in the time domain?
    - Convolution in the time domain is taking the kernel, overlapping it with
    the input signal, performing the dot (inner) product, storing that result in
    the output signal, sliding the kernel over, and then repeat until the kernel 
    has been slid over the entire input signal.
    - Since we are saying time increases from left to right in our sequences, 

    So if you didn't flip the kernel, and you took the ker

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