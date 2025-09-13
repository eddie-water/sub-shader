"""
TODO 36 - Rename this to bench or performance or something funny
Benchmark Module for SubShader Performance Analysis.

This module provides comprehensive performance benchmarking for all SubShader
components:
 - Measures audio input processing performance
 - Compares different wavelet transform implementations
 - Benchmarks visualization rendering performance
 - Generates comparative analysis and visualizations
"""

# =============================================================================
# IMPORTS
# =============================================================================

# TODO 36 - Go through every file and make sure the imports go through the pattern of stdlib, third party, first party, sibling, etc

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from subshader.config import get_default_config
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet, CuPyWavelet
from subshader.viz.plotter import PyQtPlotter, ShaderPlot

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load default configuration
config = get_default_config()

# Override benchmark-specific configs
config.audio.file_path = "assets/audio/daw/a2a3_a4_minor_scale.wav"
config.audio.chunk_size = 1024  # 1 << 10
config.viz.num_frames = 256

# =============================================================================
# TEST PARAMS
# =============================================================================

NUM_ITERATIONS = 500

# =============================================================================
# BENCHMARK CLASS
# =============================================================================

class Benchmark():
    def __init__(self) -> None:
        
        # Audio Input
        self.audio_input = AudioInput(
            path=config.audio.file_path, 
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor
        )

        self.audio_data = self.audio_input.get_chunk()

        self.sample_rate = self.audio_input.get_sample_rate()

        # Wavelet Implementations
        self.py_wavelet = PyWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size,
            config=config.wavelet
        )

        self.coefs_py_wavelet = self.py_wavelet.compute_cwt(self.audio_data)

        self.np_wavelet = NumPyWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size,
            config=config.wavelet
        )

        self.coefs_np_wavelet = self.np_wavelet.compute_cwt(self.audio_data)

        self.cp_wavelet = CuPyWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size,
            config=config.wavelet
        )

        self.coefs_cp_wavelet = self.cp_wavelet.compute_cwt(self.audio_data)

        # Plotter Implementations
        self.plot_shape = self.py_wavelet.get_output_shape()

        self.pyqtg = PyQtPlotter(
            file_path=config.audio.file_path,
            frame_shape=self.plot_shape
        )

        self.shader = ShaderPlot(
            file_path=config.audio.file_path,
            frame_shape=self.plot_shape,
            num_frames=config.viz.num_frames,
            gamma=config.viz.gamma
        )

        # Function List and Dummy Arguments 
        self.func_list = [
            (self.audio_input.get_chunk,         ()),
            (self.py_wavelet.compute_cwt,        (self.audio_data,)),
            (self.np_wavelet.compute_cwt,        (self.audio_data,)),
            (self.cp_wavelet.compute_cwt,        (self.audio_data,)),
            (self.pyqtg.update_plot,             (self.coefs_py_wavelet,)),
            (self.shader.update_plot,            (self.coefs_cp_wavelet,))
        ]

        # Tracks the run time of each function
        self.func_times = np.zeros(len(self.func_list))

    def static_wavelet_kernel_analysis(self):
        """
        Plot all the wavelet kernels in the time and frequency domain using a
        slider to control which frequency-centered kernel is currently being
        plotted. The plots have three different time and frequency ranges to
        account for the wavelet shape dilation.
        """
        # Plot visual characteristics
        REAL_PART_COLOR = 'darkorange'
        IMAG_PART_COLOR = 'mediumslateblue'
        MAG_COLOR = 'black'
        GRID_ALPHA = 0.25
        LINE_WIDTH = 2

        # Ranges for time (s) and frequency (Hz) axes
        time_ranges_s = [
            0.5, 
            0.05, 
            0.005, 
        ]

        freq_range_hz = [
            (20.0, 200.0),
            (200.0, 2000.0),
            (2000.0, 20000.0),
        ]

        # Left column for time domain, right column for frequency domain
        fig = plt.figure(figsize=(16, 9))
        fig.subplots_adjust(bottom=0.15, top=0.92, left=0.06, right=0.98, wspace=0.12, hspace=0.4)
        ax_ts = [fig.add_subplot(len(time_ranges_s), 2, 1 + i*2) for i in range(len(time_ranges_s))]
        ax_fs = [fig.add_subplot(len(freq_range_hz), 2, 2 + i*2) for i in range(len(freq_range_hz))]

        # Get the time and frequency domain kernels and frequency list
        kernels_t = self.np_wavelet.get_wavelet_kernels('time')
        kernels_f = self.np_wavelet.get_wavelet_kernels('freq')
        freqs = np.asarray(self.np_wavelet.freqs)
        num_wavelets = len(kernels_t)

        def plot_kernel(i: int):
            # Clear axes
            for ax in ax_ts: ax.cla()
            for ax in ax_fs: ax.cla()

            '''Time series plots'''
            # Grab the kernel time series and create a time axis centered at zero
            k_t = kernels_t[i]
            t = np.arange(len(k_t)) / self.sample_rate
            t = t - t[len(t)//2]

            for ax, t_range in zip(ax_ts, time_ranges_s):
                # Plot all signal components
                ax.plot(t, np.real(k_t), REAL_PART_COLOR, label='Real', lw=LINE_WIDTH)
                ax.plot(t, np.imag(k_t), IMAG_PART_COLOR, label='Imag', lw=LINE_WIDTH)
                ax.plot(t, np.abs(k_t), MAG_COLOR, label='Mag', lw=LINE_WIDTH)
                
                # Format time range string for tidy title
                x_lim = t_range/2
                if x_lim >= 0.1:
                    time_range = f'{x_lim} s'
                else:
                    time_range = f'{x_lim*1000:.1f} ms'

                # Decorate plot 
                ax.set_title(f'Time Domain ±{time_range}')
                ax.set_xlim(-x_lim, x_lim)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=GRID_ALPHA)

            # Color legend only on first time series plot
            ax_ts[0].legend(loc='upper right', frameon=False)

            '''Frequency domain plots'''
            # Grab the fft'd kernel (frequency domain) and create x axis
            k_f = kernels_f[i]
            n_f = len(k_f)
            f_axis = np.fft.fftfreq(n_f, d=1/self.sample_rate)
            
            # Create x axis and compute magnitudes just for "positive" frequencies
            pos_freq_slice = slice(0, n_f//2)
            f_axis = f_axis[pos_freq_slice]
            k_f_mag = np.abs(k_f[pos_freq_slice])

            # Plot the frequency series for each decade
            for ax, (f_lo, f_hi) in zip(ax_fs, freq_range_hz):
                ax.plot(f_axis, k_f_mag, MAG_COLOR, lw=LINE_WIDTH)

                # Frequency range fstring for tidy title
                if f_hi >= 1000:
                    f_range = f'{f_lo:.0f}–{f_hi/1000:.1f}k Hz'
                else:
                    f_range = f'{f_lo:.0f}–{f_hi:.0f} Hz'

                # Decorate plot
                ax.set_title(f'Frequency Domain {f_range}')
                ax.set_xlim(f_lo, f_hi)
                ax.set_xlabel('Freq (Hz)')
                ax.set_xscale('log')
                ax.set_ylabel('Magnitude')
                ax.grid(True, which='both', alpha=GRID_ALPHA)

            fig.suptitle(f'Wavelet Kernel Visualization - Center Frequency {freqs[i]:.1f} Hz ({i+1}/{num_wavelets})', fontsize=12)

        # Slider, buttons, controls and callbacks
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        kernel_slider = Slider(ax_slider, 'Kernel', 0, num_wavelets-1, valinit=0, valfmt='%d')

        def update(val):
            plot_kernel(int(kernel_slider.val))
            fig.canvas.draw()

        kernel_slider.on_changed(update)

        # Simple buttons
        def prev(_): 
            new_val = (int(kernel_slider.val) - 1) % num_wavelets
            kernel_slider.set_val(new_val)

        def next(_): 
            new_val = (int(kernel_slider.val) + 1) % num_wavelets
            kernel_slider.set_val(new_val)

        prev_button = Button(fig.add_axes([0.1, 0.05, 0.06, 0.03]), 'Prev')
        next_button = Button(fig.add_axes([0.84, 0.05, 0.06, 0.03]), 'Next')
        prev_button.on_clicked(prev)
        next_button.on_clicked(next)

        # Initial draw
        plot_kernel(int(kernel_slider.val))
        plt.show()
    
    def static_plot_analysis(self):
        # Single window with time series on left, CWTs stacked on right
        fig = plt.figure(constrained_layout=False)  # Disable constrained_layout to use subplots_adjust
        fig.canvas.manager.set_window_title(f"Time Series vs CWT {os.path.basename(config.audio.file_path)}")
        
        # Create a 2x2 grid and use different subplot positions
        ax_ts = fig.add_subplot(1, 2, 1)  # Left column, spans full height
        ax_py = fig.add_subplot(2, 2, 2)  # Right column, top
        ax_cp = fig.add_subplot(2, 2, 4)  # Right column, bottom
        
        # Add padding between the left and right plots, minimize edge padding
        plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07, wspace=0.1, hspace=0.2)

        # Time series on the left
        ax_ts.set_title("Test Signal Time Series")
        ax_ts.plot(self.audio_data)
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Amplitude")
        ax_ts.margins(x=0, y=0)

        # PyWavelet CWT on top right
        ax_py.set_title("PyWavelet CWT")
        ax_py.imshow(self.coefs_py_wavelet, cmap="magma", aspect="auto", origin='lower')
        ax_py.set_xlabel("Time")
        ax_py.set_ylabel("Freq Bin")

        # CuPy CWT on bottom right
        ax_cp.set_title("CuPy CWT")
        ax_cp.imshow(self.coefs_cp_wavelet, cmap="magma", aspect="auto", origin='lower')
        ax_cp.set_xlabel("Time")
        ax_cp.set_ylabel("Freq Bin")

        # Maximize the window
        try:
            mng = fig.canvas.manager
            if hasattr(mng, 'window') and hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()
        except Exception:
            pass

        plt.show()

    def dynamic_plot_analysis(self):
        print()
        print("Starting Timing Analysis...\n")

        for _ in range(NUM_ITERATIONS):
            # Initialize variables to store intermediate results
            fresh_audio_data = None
            py_cwt_result = None
            np_cwt_result = None  
            cp_cwt_result = None
            
            for i, item in enumerate(self.func_list):
                # Grab the function and its arg(s)
                func = item[0]
                base_args = item[1] if len(item) > 1 else ()
                kwargs = item[2] if len(item) > 2 else {}

                # TODO 36 - I don't like this it looks so gross
                # Modify args based on function type and available data
                if func.__name__ == 'get_chunk':
                    # Audio input - use original args
                    args = base_args
                elif func.__name__ == 'compute_cwt':
                    # CWT functions - use fresh audio data if available
                    if fresh_audio_data is not None:
                        args = (fresh_audio_data,)
                    else:
                        args = base_args
                elif func.__name__ == 'update_plot':
                    # Plotting functions - use fresh CWT results if available
                    if 'PyQt' in func.__self__.__class__.__name__:
                        # PyQtPlotter gets PyWavelet results
                        if py_cwt_result is not None:
                            args = (py_cwt_result,)
                        else:
                            args = base_args
                    elif 'Shader' in func.__self__.__class__.__name__:
                        # ShaderPlot gets CuPyWavelet results  
                        if cp_cwt_result is not None:
                            args = (cp_cwt_result,)
                        else:
                            args = base_args
                    else:
                        args = base_args
                else:
                    args = base_args

                # Time the runtime of the function
                t_start = time.perf_counter()
                result = func(*args, **kwargs)
                t_end = time.perf_counter()

                # Store results for next functions in pipeline
                if func.__name__ == 'get_chunk':
                    fresh_audio_data = result
                elif func.__name__ == 'compute_cwt':
                    if 'PyWavelet' in func.__self__.__class__.__name__:
                        py_cwt_result = result
                    elif 'AntsWavelet' in func.__self__.__class__.__name__:
                        np_cwt_result = result  # Not used in plotting but stored
                    elif 'CuPyWavelet' in func.__self__.__class__.__name__:
                        cp_cwt_result = result

                t_delta = t_end - t_start
                self.func_times[i] += t_delta

        # Average the runtimes
        self.avg_func_times = self.func_times / int(NUM_ITERATIONS)
        print(f"Results:")

        for i, item in enumerate(self.func_list):
            func = item[0]
            time_ms = self.avg_func_times[i] * 1000  # Convert to milliseconds
            print(f"-> {func.__self__.__class__.__name__}.{func.__name__}()\t{time_ms:7.3f} ms")

        print()
        print(f"Timing Analysis Complete - every function averaged over {NUM_ITERATIONS} iterations\n")
 
        pass  

    def run_tests(self):
        self.static_wavelet_kernel_analysis()
        self.static_plot_analysis()


if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.run_tests()