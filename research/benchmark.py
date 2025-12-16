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

# Force TkAgg backend for WSL compatibility (must be before pyplot import)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from subshader.config import get_default_config
from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import PyWavelet, NumPyWavelet, CuPyWavelet
from subshader.viz.plotter import PyQtPlotter, ShaderPlot
from subshader.viz.comparison_navigator import AudioNavigator, KernelNavigator, DspStageNavigator, TransformNavigator, TopLevelComparisonNavigator

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load default configuration
config = get_default_config()

# Override benchmark-specific configs (ensure string, not Path)
# config.audio.file_path = str("assets/audio/daw/a2a3_a4_minor_scale.wav")
config.audio.file_path = str("assets/audio/daw/a3b3c4d4_a5arp_16bars.wav")

# =============================================================================
# TEST PARAMS
# =============================================================================

NUM_ITERATIONS = 500

# =============================================================================
# BENCHMARK CLASS
# =============================================================================

class Benchmark():
    # =========================================================================
    # INIT - INITIALIZE EVERYTHING
    # =========================================================================
    
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

        self.pywt = PyWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size,
            config=config.wavelet
        )

        self.npwt = NumPyWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size,
            config=config.wavelet
        )

        self.npwt_coefs = self.npwt.cwt(self.audio_data)

        self.cpwt = CuPyWavelet(
            sample_rate=self.sample_rate, 
            input_n=config.audio.chunk_size,
            config=config.wavelet
        )

        self.cpwt_coefs = self.cpwt.cwt(self.audio_data)

        # Plotter Implementations
        self.plot_shape = self.npwt.get_output_shape()

        # self.pyqtg = PyQtPlotter(
        #     file_path=config.audio.file_path,
        #     frame_shape=self.plot_shape
        # )

        # self.shader = ShaderPlot(
        #     file_path=config.audio.file_path,
        #     frame_shape=self.plot_shape,
        #     config=config.viz
        # )

        # # Function List and Dummy Args 
        # self.func_list = [
        #     (self.audio_input.get_chunk,    ()),
        #     (self.npwt.cwt,           (self.audio_data,)),
        #     (self.cpwt.cwt,           (self.audio_data,)),
        #     (self.shader.update_plot,       (self.cpwt_coefs,))
        # ]

        # # Tracks the run time of each function
        # self.func_times = np.zeros(len(self.func_list))

    # =========================================================================
    # STATIC PLOT ANALYSIS - PLOT INTERMEDIATE STEPS OF CWT
    # =========================================================================
    
    def static_plot_analysis(self):
        """
        Static Plot Analysis and Comparisons
        """
        try:
            TopLevelComparisonNavigator(
                audio_input=self.audio_input,
                pywt=self.pywt,
                cpwt=self.cpwt,
                title="Side by Side Comparison",
                midi_img_path="assets/images/a3b3c4d4_a5arp_13p25_bars.png"
            )
        except Exception as e:
            print(f"[ERROR] Navigator failed: {e}")
            import traceback
            traceback.print_exc()

        # Static Wavelet Kernel Analysis: Time vs Frequency Domain
        # AudioNavigator(
        #     audio_input=self.audio_input,
        #     title="Static Audio Overlap Analysis"
        # )

        # KernelNavigator(
        #     wavelet=self.npwt,
        #     title="Static Wavelet Kernel Analysis"
        # )

        # DspStageNavigator(
        #     audio_input=self.audio_input,
        #     wavelet=self.npwt,
        #     title="Static DSP Stage Analysis"
        # )

        # # Static CWT Analysis: Audio Time Series vs CWT Time-Freq Coefficients
        # TransformNavigator(
        #     audio_input=self.audio_input,
        #     py_wavelet=self.pywt,
        #     cp_wavelet=self.cpwt,
        #     cwt_function=lambda wavelet, data: wavelet.class_specific_cwt(data),
        #     title=f"Static Class-Specific CWT Plot Analysis and Comparison — {os.path.basename(config.audio.file_path)}"
        # )

        # Then do a global normalization plot analysis where we try different 
        # types of normalizaition params, be able to turn off the warm up to isolate

        # Then do a static plot analysis for the coi where we try out different
        # coi region lengths - see what's obnoxious vs what's needed

    # =========================================================================
    # DYNAMIC PLOT ANALYSIS - MEASURE RUNTIME OF PIPELINE
    # =========================================================================
    
    def dynamic_plot_analysis(self):
        print()
        print("Starting Timing Analysis...\n")

        for _ in range(NUM_ITERATIONS):
            for i, item in enumerate(self.func_list):
                func = item[0]
                args = item[1] if len(item) > 1 else ()
                kwargs = item[2] if len(item) > 2 else {}

                t_start = time.perf_counter()
                func(*args, **kwargs)
                t_end = time.perf_counter()

                self.func_times[i] += t_end - t_start

        self._print_results()
 
    def plot_side_by_side(self):
        TopLevelComparisonNavigator(
            audio_input=self.audio_input,
            wavelet=self.npwt,
            title="Side by Side Comparison"
        )

    def _print_results(self):
        # Average the runtimes
        self.avg_func_times = self.func_times / int(NUM_ITERATIONS)
        print(f"Results:")

        for i, item in enumerate(self.func_list):
            func = item[0]
            time_ms = self.avg_func_times[i] * 1000  # ms
            func_name = f"{func.__self__.__class__.__name__}.{func.__name__}()"
            print(f"-> {func_name:<40}{time_ms:7.3f} ms")

        print()
        print(f"Timing Analysis Complete - every function averaged over {NUM_ITERATIONS} iterations\n")
 
        pass  

    def run_tests(self):
        self.static_plot_analysis()
        # self.dynamic_plot_analysis()
        input("Press Enter to exit...")

if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.run_tests()