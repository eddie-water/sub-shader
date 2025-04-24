import time
import numpy as np
import matplotlib.pyplot as plt

# TODO LATER move this in to the plotting utilities

def plot_time_domain_conv(input_signal, kernel, y_min, y_max, delay):
    try:
        # Delay must be > 0 or else the interactive plot silently breaks by 
        # starving the GUI event loop, force it to a non-zero value
        if (delay == 0):
            delay == 0.001

        fig, ax = plt.subplots(figsize = (12, 8))

        # Get signal params for configuring buffer lengths, normalizaiton, and stuff
        signal_len = len(input_signal)
        kernel_len = len(kernel)
        # TODO LATER what happens when the kernel happens to be even? How would that
        # affect the padding and loop indeces. Try and see if it breaks and if it 
        # does, fix it or just try to avoid that.
        half_kern_len = int(np.floor(kernel_len / 2))
        result_len = signal_len

        # Pad the signal with zeros (half kernel length on each side)
        arrays = [np.zeros(half_kern_len), input_signal, np.zeros(half_kern_len)]
        padded_signal = np.concatenate(arrays)

        # Reverse ze kernel (must be done in time domain to preserve 'causality')
        flipped_kernel = kernel[::-1] 

        # Only plot things if the figure isn't closed
        while(plt.fignum_exists(fig.number)):
            # Flush Matplotlib GUI event loop
            plt.pause(0.001)

            # Configure and populate initial elements on the figure
            ax.set_title("Time Domain Convolution")
            ax.axvline(x = half_kern_len, color = 'gray', linestyle='--')
            ax.axvline(x = (half_kern_len + signal_len - 1), color = 'gray', linestyle='--')

            # Auto resizing in interactive mode may break things, so do it explicitly 
            ax.set_xlim(-1, len(padded_signal))
            ax.set_ylim(y_min, y_max)

            # Initial x and y values for initially drawn kernel
            kernel_x_vals = np.arange(kernel_len)
            kernel_y_vals = np.array(flipped_kernel).flatten()

            # Destination array for convolution result 
            result = np.zeros(result_len)

            # Instantiate plot lines (result gets plotted but its empty)
            signal_plot, = ax.plot(padded_signal, 
                                label = 'Input (Padded)', 
                                marker = 'o', 
                                linestyle = '-',
                                color = 'black', zorder = 1)

            kernel_plot, = ax.plot(kernel_x_vals, 
                                kernel_y_vals, 
                                label = 'Kernel (Flipped)',
                                marker = 'o',
                                linestyle = '-', 
                                color = 'mediumslateblue', 
                                zorder = 2)

            result_plot, = ax.plot([], # empty x values
                                [], # empty y values
                                label = 'Output', 
                                marker = 's',
                                linestyle = '-',
                                color = 'darkorange', 
                                zorder = 3)
            
            # Update the legend after plotting all the lines
            ax.legend(loc = 'upper right')

            # Do the time domain convolution by hand
            for i in range(0, result_len):
                # Grab one kernel-length of input signal
                signal_slice = padded_signal[i : i + kernel_len]

                # Compute dot product on it (note the flipped kernel)
                result[i] = np.sum(signal_slice * flipped_kernel)

                # Update x and y values of the plots (this is just a matplot lib thing)
                result_x_vals = np.arange(half_kern_len, half_kern_len + i + 1)
                result_y_vals = result[0 : i + 1]
                result_plot.set_data(result_x_vals, result_y_vals)

                kernel_x_vals = np.arange(i, i + kernel_len)
                kernel_y_vals = flipped_kernel
                kernel_plot.set_data(kernel_x_vals, kernel_y_vals)

                # Plot the normalized version of the result
                if i == (result_len - 1):
                    normalized_result = result / np.max(result)
                    norm_x_vals = np.arange(half_kern_len, half_kern_len + i + 1)
                    norm_y_vals = normalized_result[0 : i + 1]
                    normalized_plot , = ax.plot(norm_x_vals,
                                                norm_y_vals, 
                                                label = 'Output (Normalized)', 
                                                marker = 's', 
                                                linestyle = '-',
                                                color = 'firebrick', 
                                                zorder = 4)
                    ax.legend(loc = 'upper right')

                # Refresh figure and slow it down for the viewer
                if plt.fignum_exists(fig.number):
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(delay)

            # Display the final results and then clear the plots before restarting
            if plt.fignum_exists(fig.number):
                plt.pause(5)
                ax.cla()

    except Exception as e:
        print("Caught exception:", e)
    except KeyboardInterrupt as e:
        print("Caught keyboard exception")

    # The User closed the window, but still explicitly close it
    plt.close(fig)