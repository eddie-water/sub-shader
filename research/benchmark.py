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
# CONSTANTS
# =============================================================================

NUM_ITERATIONS = 500

# =============================================================================
# BENCHMARK CLASS
# =============================================================================

class Benchmark():
    def __init__(self) -> None:
        
        # =====================================================================
        #  Audio Input
        # =====================================================================
        self.audio_input = AudioInput(
            path=config.audio.file_path, 
            chunk_size=config.audio.chunk_size,
            overlap_factor=config.audio.overlap_factor
        )

        self.audio_data = self.audio_input.get_chunk()

        self.sample_rate = self.audio_input.get_sample_rate()

        # =====================================================================
        #  Wavelet Implementations
        # =====================================================================
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

        # =====================================================================
        #  Plotter Implementations
        # =====================================================================
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

    def static_plot_analysis(self):
        """
        Visualize all wavelet kernels in time and frequency domains.
        Creates an interactive matplotlib plot with slider navigation.
        """
        # Get wavelet kernels from AntsWavelet implementation
        kernels_t = self.np_wavelet.get_wavelet_kernels('time')
        kernels_f = self.np_wavelet.get_wavelet_kernels('freq')
        freqs = self.np_wavelet.freqs
        
        if not kernels_t or not kernels_f:
            print("No wavelet kernels available for visualization")
            return
            
        # Create interactive plot with slider
        fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(15, 6))
        plt.subplots_adjust(bottom=0.2)
        
        # Initial kernel index
        kernel_idx = 0
        
        def plot_kernel(idx):
            """Plot kernel at given index"""
            ax_t.clear()
            ax_f.clear()
            
            # Time domain plot
            kernel_t = kernels_t[idx]
            time_samples = len(kernel_t)
            sample_rate = self.sample_rate
            time_axis = np.arange(time_samples) / sample_rate
            time_axis = time_axis - time_axis[len(time_axis)//2]  # Center at 0
            
            ax_t.plot(time_axis, np.real(kernel_t), 'b-', label='Real', alpha=0.7)
            ax_t.plot(time_axis, np.imag(kernel_t), 'r-', label='Imag', alpha=0.7)
            ax_t.plot(time_axis, np.abs(kernel_t), 'k-', label='Magnitude', linewidth=2)
            ax_t.set_title(f'Time Domain - Freq: {freqs[idx]:.1f} Hz')
            ax_t.set_xlabel('Time (s)')
            ax_t.set_ylabel('Amplitude')
            ax_t.legend()
            ax_t.grid(True, alpha=0.3)
            
            # Frequency domain plot
            kernel_f = kernels_f[idx]
            freq_samples = len(kernel_f)
            freq_axis = np.fft.fftfreq(freq_samples, 1/sample_rate)
            
            # Only plot positive frequencies for clarity
            pos_freqs = freq_axis[:freq_samples//2]
            pos_kernel = kernel_f[:freq_samples//2]
            
            ax_f.plot(pos_freqs, np.abs(pos_kernel), 'g-', linewidth=2)
            ax_f.set_title(f'Frequency Domain - Freq: {freqs[idx]:.1f} Hz')
            ax_f.set_xlabel('Frequency (Hz)')
            ax_f.set_ylabel('Magnitude')
            ax_f.set_xlim(0, min(2000, sample_rate//2))  # Limit to 2kHz for visibility
            ax_f.grid(True, alpha=0.3)
            
            fig.suptitle(f'Wavelet Kernel {idx+1}/{len(kernels_t)} - {freqs[idx]:.1f} Hz', fontsize=14)
            
        # Create slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = plt.Slider(ax_slider, 'Kernel', 0, len(kernels_t)-1, 
                           valinit=kernel_idx, valfmt='%d')
        
        def update_plot(val):
            idx = int(slider.val)
            plot_kernel(idx)
            plt.draw()
            
        slider.on_changed(update_plot)
        
        # Plot initial kernel
        plot_kernel(kernel_idx)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.1, 0.05, 0.05, 0.04])
        ax_next = plt.axes([0.85, 0.05, 0.05, 0.04])
        btn_prev = plt.Button(ax_prev, 'Prev')
        btn_next = plt.Button(ax_next, 'Next')
        
        def prev_kernel(event):
            current_val = max(0, slider.val - 1)
            slider.set_val(current_val)
            
        def next_kernel(event):
            current_val = min(len(kernels_t)-1, slider.val + 1)
            slider.set_val(current_val)
            
        btn_prev.on_clicked(prev_kernel)
        btn_next.on_clicked(next_kernel)
        
        plt.show()  

    def dynamic_plot_analysis(self):
        # TODO 36 - This is where we bench mark the real time plotting performance
        # and time all the functions in the plotting pipeline 
        pass  

    def run_tests(self):
        self.static_plot_analysis()

        # self.dynamic_plot_analysis()

        # # TODO 36 - Move the below into the dynamic plot analysis function
        # print()
        # print("Starting Timing Analysis...\n")

        # for _ in range(NUM_ITERATIONS):
        #     # Initialize variables to store intermediate results
        #     fresh_audio_data = None
        #     py_cwt_result = None
        #     np_cwt_result = None  
        #     cp_cwt_result = None
            
        #     for i, item in enumerate(self.func_list):
        #         # Grab the function and its arg(s)
        #         func = item[0]
        #         base_args = item[1] if len(item) > 1 else ()
        #         kwargs = item[2] if len(item) > 2 else {}

        #         # TODO 36 - I don't like this it looks so gross
        #         # Modify args based on function type and available data
        #         if func.__name__ == 'get_chunk':
        #             # Audio input - use original args
        #             args = base_args
        #         elif func.__name__ == 'compute_cwt':
        #             # CWT functions - use fresh audio data if available
        #             if fresh_audio_data is not None:
        #                 args = (fresh_audio_data,)
        #             else:
        #                 args = base_args
        #         elif func.__name__ == 'update_plot':
        #             # Plotting functions - use fresh CWT results if available
        #             if 'PyQt' in func.__self__.__class__.__name__:
        #                 # PyQtPlotter gets PyWavelet results
        #                 if py_cwt_result is not None:
        #                     args = (py_cwt_result,)
        #                 else:
        #                     args = base_args
        #             elif 'Shader' in func.__self__.__class__.__name__:
        #                 # ShaderPlot gets CuPyWavelet results  
        #                 if cp_cwt_result is not None:
        #                     args = (cp_cwt_result,)
        #                 else:
        #                     args = base_args
        #             else:
        #                 args = base_args
        #         else:
        #             args = base_args

        #         # Time the runtime of the function
        #         t_start = time.perf_counter()
        #         result = func(*args, **kwargs)
        #         t_end = time.perf_counter()

        #         # Store results for next functions in pipeline
        #         if func.__name__ == 'get_chunk':
        #             fresh_audio_data = result
        #         elif func.__name__ == 'compute_cwt':
        #             if 'PyWavelet' in func.__self__.__class__.__name__:
        #                 py_cwt_result = result
        #             elif 'AntsWavelet' in func.__self__.__class__.__name__:
        #                 np_cwt_result = result  # Not used in plotting but stored
        #             elif 'CuPyWavelet' in func.__self__.__class__.__name__:
        #                 cp_cwt_result = result

        #         t_delta = t_end - t_start
        #         self.func_times[i] += t_delta

        # # Average the runtimes
        # self.avg_func_times = self.func_times / int(NUM_ITERATIONS)
        # print(f"Results:")

        # for i, item in enumerate(self.func_list):
        #     func = item[0]
        #     time_ms = self.avg_func_times[i] * 1000  # Convert to milliseconds
        #     print(f"-> {func.__self__.__class__.__name__}.{func.__name__}()\t{time_ms:7.3f} ms")

        # print()
        # print(f"Timing Analysis Complete - every function averaged over {NUM_ITERATIONS} iterations\n")

        # # Static Plots
        # # Single window with time series on left, CWTs stacked on right
        # fig = plt.figure(constrained_layout=False)  # Disable constrained_layout to use subplots_adjust
        # fig.canvas.manager.set_window_title(f"Time Series vs CWT {os.path.basename(config.audio.file_path)}")
        
        # # Create a 2x2 grid and use different subplot positions
        # ax_ts = fig.add_subplot(1, 2, 1)  # Left column, spans full height
        # ax_py = fig.add_subplot(2, 2, 2)  # Right column, top
        # ax_cp = fig.add_subplot(2, 2, 4)  # Right column, bottom
        
        # # Add padding between the left and right plots, minimize edge padding
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07, wspace=0.1, hspace=0.2)

        # # Time series on the left
        # ax_ts.set_title("Test Signal Time Series")
        # ax_ts.plot(self.audio_data)
        # ax_ts.set_xlabel("Time")
        # ax_ts.set_ylabel("Amplitude")
        # ax_ts.margins(x=0, y=0)

        # # PyWavelet CWT on top right
        # ax_py.set_title("PyWavelet CWT")
        # ax_py.imshow(self.coefs_py_wavelet, cmap="magma", aspect="auto", origin='lower')
        # ax_py.set_xlabel("Time")
        # ax_py.set_ylabel("Freq Bin")

        # # CuPy CWT on bottom right
        # ax_cp.set_title("CuPy CWT")
        # ax_cp.imshow(self.coefs_cp_wavelet, cmap="magma", aspect="auto", origin='lower')
        # ax_cp.set_xlabel("Time")
        # ax_cp.set_ylabel("Freq Bin")

        # # Maximize the window
        # try:
        #     mng = fig.canvas.manager
        #     if hasattr(mng, 'window') and hasattr(mng.window, 'showMaximized'):
        #         mng.window.showMaximized()
        # except Exception:
        #     pass

        plt.show()

if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.run_tests()