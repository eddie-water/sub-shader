"""
WARNING: This benchmark script may not work with the current codebase.

This script was designed for earlier versions of the project and may have
compatibility issues with the current implementation due to:
- Recent refactoring of the SubShader module structure
- Changes to class names and interfaces (e.g., Shader class changes)
- Updated import paths and method signatures
- New modular architecture

Use this script as reference only. It may need significant updates to work
with the current codebase.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumpyWavelet, CupyWavelet
from subshader.viz.plotter import PyQtPlotter, ShaderPlot

'''
Constants
'''
NUM_ITERATIONS = 100

WINDOW_SIZE = 4096

FILE_PATH = "assets/audio/daw/a2_stuttered_a4_230ms.wav"

class Benchmark():
    def __init__(self) -> None:
        
        '''
        Audio Input
        '''
        audio_input = AudioInput(path = FILE_PATH, window_size = WINDOW_SIZE)
        self.audio_data = audio_input.get_frame()
        sample_rate = audio_input.get_sample_rate() # 44.1 kHz

        '''
        Wavelet Implementations
        '''
        # PyWavelet 
        py_wavelet = PyWavelet(sample_rate = sample_rate, 
                               window_size = WINDOW_SIZE)
        
        self.coefs_py_wavelet = py_wavelet.compute_cwt(self.audio_data)

        # NumPy ANTS Wavelet
        np_wavelet = NumpyWavelet(sample_rate = sample_rate, 
                                  window_size = WINDOW_SIZE)

        self.coefs_np_wavelet = np_wavelet.compute_cwt(self.audio_data)

        # CuPy ANTS Wavelet 
        cp_wavelet = CupyWavelet(sample_rate = sample_rate, 
                                 window_size = WINDOW_SIZE)

        self.coefs_cp_wavelet = cp_wavelet.compute_cwt(self.audio_data)

        '''
        Plotter Implementations
        '''
        # Plot shapes
        self.plot_shape = py_wavelet.get_shape()
        self.plot_shape_downsampled = py_wavelet.get_downsampled_result_shape()

        # PyQtGraph Plotter
        pyqtg = PyQtPlotter(file_path = FILE_PATH,
                            frame_shape = self.plot_shape_downsampled)

        # Shader Plotter
        shader = ShaderPlot(file_path = FILE_PATH,
                            frame_shape = self.plot_shape_downsampled)

        '''
        Function List and Dummy Arguments 
            - note special python ',' syntax
        '''
        self.func_list = [
            (audio_input.get_frame,         ()),
            (py_wavelet.compute_cwt,        (self.audio_data,)),
            (np_wavelet.compute_cwt,        (self.audio_data,)),
            (cp_wavelet.compute_cwt,        (self.audio_data,)),
            (pyqtg.update_plot,             (self.coefs_py_wavelet,)),
            (shader.update_plot,            (self.coefs_cp_wavelet,))
        ]

        # Tracks the run time of each function
        self.func_times = np.zeros(len(self.func_list))

    def main(self):
        print()
        print("Starting Timing Analysis...\n")

        for _ in range(NUM_ITERATIONS):
            for i, item in enumerate(self.func_list):
                # Grab the function and its arg(s)
                func = item[0]
                args = item[1] if len(item) > 1 else ()
                kwargs = item[2] if len(item) > 2 else {}

                # Time the runtime of the function
                t_start = time.perf_counter()
                _ = func(*args, **kwargs)
                t_end = time.perf_counter()

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

        '''
        Static Plots
        '''
        # Single window with time series on left, CWTs stacked on right
        fig = plt.figure(constrained_layout=False)  # Disable constrained_layout to use subplots_adjust
        fig.canvas.manager.set_window_title(f"Time Series vs CWT {os.path.basename(FILE_PATH)}")
        
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
        ax_py.set_ylabel("Scale")

        # CuPy CWT on bottom right
        ax_cp.set_title("CuPy CWT")
        ax_cp.imshow(self.coefs_cp_wavelet, cmap="magma", aspect="auto", origin='lower')
        ax_cp.set_xlabel("Time")
        ax_cp.set_ylabel("Scale")

        # Maximize the window
        try:
            mng = fig.canvas.manager
            if hasattr(mng, 'window') and hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()
        except Exception:
            pass

        plt.show()

if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.main()