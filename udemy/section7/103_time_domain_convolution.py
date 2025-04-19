import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

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

# Convolution using Old Skool methods
import time

plt.ion()

fig, ax = plt.subplots()

signal_len = len(signal)
kernel_len = len(kernel)
result_len = signal_len + kernel_len - 1

half_kern_length = int(np.floor(kernel_len / 2))

ax.set_ylim(-1, 3)

# Vertical Dashed Lines
ax.axvline(x = half_kern_length, color = 'gray', linestyle='--')
ax.axvline(x = (half_kern_length + signal_len - 1), color = 'gray', linestyle='--')

# Reverse ze kernel (nifty way of instantiating empty 2D array)
flipped_kernel = kernel[::-1] 

# Pad the signal with zeros - half the size of the kernel on each side
arrays = [np.zeros(half_kern_length), signal, np.zeros(half_kern_length)]
padded_signal = np.concatenate(arrays)

signal_plot_line, = ax.plot(padded_signal, '-o', color = 'black', zorder = 1)
kernel_plot_line, = ax.plot([], [], '-o', color = 'red', zorder = 2)

# Destination array for convolution 
result = np.zeros(result_len)
result_plot_line, = ax.plot([], [], '-o', color = 'mediumslateblue')

# Do the time domain convolution by hand
i_start = half_kern_length
i_end = result_len - half_kern_length

for i in range(i_start, i_end):
    signal_slice = padded_signal[(i - half_kern_length):(i + half_kern_length + 1)]
    result[i - i_start] = np.sum(signal_slice * flipped_kernel)
    result_plot_line.set_data(np.arange(i + 1), result[:i + 1])

    kernel_x_vals = np.arange(i - half_kern_length, i - half_kern_length + kernel_len)
    kernel_y_vals = flipped_kernel
    kernel_plot_line.set_data(kernel_x_vals, kernel_y_vals)

    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(1)

plt.style.use('dark_background')
plt.tight_layout()
plt.show()

while(True):
    pass