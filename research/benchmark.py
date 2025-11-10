"""
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
from subshader.viz.comparison_navigator import KernelNavigator, DspStageNavigator, TransformNavigator

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load default configuration
config = get_default_config()

# Override benchmark-specific configs
config.audio.file_path = "assets/audio/daw/a2a3_a4_minor_scale.wav"
config.audio.chunk_size = 1024  # 1 << 10
config.viz.num_frames = 256
config.wavelet.target_width = config.audio.chunk_size 

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

        self.py_coefs = self.py_wavelet.cwt(self.audio_data)

        self.np_wavelet = NumPyWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size,
            config=config.wavelet
        )

        self.np_coefs = self.np_wavelet.cwt(self.audio_data)

        self.cp_wavelet = CuPyWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size,
            config=config.wavelet
        )

        self.cp_coefs = self.cp_wavelet.cwt(self.audio_data)

        # Plotter Implementations
        self.plot_shape = self.py_wavelet.get_output_shape()

        self.pyqtg = PyQtPlotter(
            file_path=config.audio.file_path,
            frame_shape=self.plot_shape
        )

        self.shader = ShaderPlot(
            file_path=config.audio.file_path,
            frame_shape=self.plot_shape,
            config=config.viz
        )

        # Function List and Dummy Arguments 
        self.func_list = [
            (self.audio_input.get_chunk,    ()),
            (self.py_wavelet.cwt,  (self.audio_data,)),
            (self.np_wavelet.cwt,  (self.audio_data,)),
            (self.cp_wavelet.cwt,  (self.audio_data,)),
            (self.pyqtg.update_plot,        (self.py_coefs,)),
            (self.shader.update_plot,       (self.cp_coefs,))
        ]

        # Tracks the run time of each function
        self.func_times = np.zeros(len(self.func_list))

    def static_plot_analysis(self):
        """
        Static Plot Analysis and Comparisons
        """
        # Static Wavelet Kernel Analysis: Time vs Frequency Domain
        # TODO 36 - it would be nice to have a way to see the construction of 
        # these wavelets - original sine wave and its time support and the 
        # gaussian bell curve used to create the wavelet

        

        # KernelNavigator(
        #     np_wavelet=self.np_wavelet,
        #     title="Static Wavelet Kernel Analysis"
        # )

        # DspStageNavigator(
        #     audio_input=self.audio_input,
        #     wavelet=self.np_wavelet,
        #     title="Static DSP Stage Analysis"
        # )

        # # Static CWT Analysis: Audio Time Series vs CWT Time-Freq Coefficients
        # TransformNavigator(
        #     audio_input=self.audio_input,
        #     py_wavelet=self.py_wavelet,
        #     cp_wavelet=self.cp_wavelet,
        #     cwt_function=lambda wavelet, data: wavelet.class_specific_cwt(data),
        #     title=f"Static Class-Specific CWT Plot Analysis and Comparison — {os.path.basename(config.audio.file_path)}"
        # )

        # Then do a global normalization plot analysis where we try different 
        # types of normalizaition params, be able to turn off the warm up to isolate

        # Then do a static plot analysis for the coi where we try out different
        # coi region lengths - see what's obnoxious vs what's needed

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
                elif func.__name__ == 'cwt':
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
                elif func.__name__ == 'cwt':
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
            time_ms = self.avg_func_times[i] * 1000  # ms
            print(f"-> {func.__self__.__class__.__name__}.{func.__name__}()\t{time_ms:7.3f} ms")

        print()
        print(f"Timing Analysis Complete - every function averaged over {NUM_ITERATIONS} iterations\n")
 
        pass  

    def run_tests(self):
        self.static_plot_analysis()
        # self.dynamic_plot_analysis()

if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.run_tests()