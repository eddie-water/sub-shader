import time
import numpy as np
import matplotlib.pyplot as plt

# TODO LATER move this in to the plotting utilities

def plot_time_domain_conv(input_signal, kernel):
    plt.ion()
    # plt.legend(title = "Signals", 
    #            fontsize = 12, 
    #            title_fontsize = 10)

    fig, ax = plt.subplots()

    signal_len = len(input_signal)
    kernel_len = len(kernel)
    half_kern_len = int(np.floor(kernel_len / 2))
    result_len = signal_len

    # Reverse ze kernel 
    flipped_kernel = kernel[::-1] 

    # Pad the input signal with zeros - half the size of the kernel on each side
    arrays = [np.zeros(half_kern_len), input_signal, np.zeros(half_kern_len)]
    padded_signal = np.concatenate(arrays)

    while(True):
        # Configure and populate elements on figure
        ax.set_ylim(-1, 4)
        ax.set_title("Time Domain Convolution")
        ax.legend()
        ax.axvline(x = half_kern_len, color = 'gray', linestyle='--')
        ax.axvline(x = (half_kern_len + signal_len - 1), color = 'gray', linestyle='--')

        # Initial x and y values for initially drawn kernel
        kernel_x_vals = np.arange(kernel_len)
        kernel_y_vals = np.array(flipped_kernel).flatten()

        # Destination array for convolution result 
        result = np.zeros(result_len)

        # Instantiate plot lines (result gets plotted but its empty)
        signal_plot_line, = ax.plot(padded_signal, 
                                    label = 'Input Signal', 
                                    marker = 'o', 
                                    linestyle = '-',
                                    color = 'black', zorder = 1)

        kernel_plot_line, = ax.plot(kernel_x_vals, 
                                    kernel_y_vals, 
                                    label = 'Kernel',
                                    marker = 'o',
                                    linestyle = '-', 
                                    color = 'red', 
                                    zorder = 2)

        result_plot_line, = ax.plot([], # empty x values
                                    [], # empty y values
                                    label = 'Output Signal', 
                                    marker = 'o',
                                    linestyle = '-',
                                    color = 'mediumslateblue', 
                                    zorder = 3)

        # Do the time domain convolution by hand
        for i in range(0, result_len):
            # Grab one kernel-length of input signal
            signal_slice = padded_signal[i : i + kernel_len]

            # Compute dot product on it (note the flipped kernel)
            result[i] = np.sum(signal_slice * flipped_kernel)

            # Update x and y values of the plots (this is just a matplot lib thing)
            result_x_vals = np.arange(half_kern_len, half_kern_len + i + 1)
            result_y_vals = result[0 : i + 1]
            result_plot_line.set_data(result_x_vals, result_y_vals)

            kernel_x_vals = np.arange(i, i + kernel_len)
            kernel_y_vals = flipped_kernel
            kernel_plot_line.set_data(kernel_x_vals, kernel_y_vals)

            # Refresh figure and slow it down for the viewer
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(.75)

        ax.cla()
