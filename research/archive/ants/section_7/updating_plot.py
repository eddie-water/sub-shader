import numpy as np
import matplotlib.pyplot as plt
import time

# Signal and kernel
signal = np.array([0, 1, 0, 2, 1, 3, 0, 1, 0, 2])
kernel = np.array([1, 2, 3, 4, 5])
kernel_len = len(kernel)
output_len = len(signal) - kernel_len + 1
output = np.zeros(output_len)

# Setup plot
plt.ion()
fig, ax = plt.subplots()
ax.set_ylim(0, 50)
ax.set_xlim(0, len(signal))

# Plot elements
signal_line, = ax.plot(signal, label='Signal', color='blue')
kernel_line, = ax.plot([], [], label='Kernel (Flipped)', color='red')
output_line, = ax.plot([], [], label='Convolution Output', color='green', marker='o')
dot_text = ax.text(0.5, 9, '', fontsize=12)
connect_line = None

# Flip the kernel for convolution
flipped_kernel = kernel[::-1]

for i in range(output_len):
    # Get segment and compute dot product
    segment = signal[i:i + kernel_len]
    result = np.dot(segment, flipped_kernel)
    output[i] = result

    # Update kernel position
    x_vals = np.arange(i, i + kernel_len)
    kernel_line.set_data(x_vals, flipped_kernel)

    # Update convolution output line
    output_line.set_data(np.arange(i + 1), output[:i + 1])

    # Add connecting dashed line for visualizing contribution
    if connect_line:
        connect_line.remove()
    connect_line = ax.plot([i + kernel_len // 2, i + kernel_len // 2],
                           [0, result],
                           linestyle='--', color='gray')[0]

    dot_text.set_text(f'Dot product = {result:.1f}')
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.5)

plt.ioff()
plt.legend()
plt.show()
