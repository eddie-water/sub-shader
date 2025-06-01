import time
import numpy as np
import matplotlib.pyplot as plt

from audio_input import AudioInput
from wavelet import PyWavelet, NumpyWavelet, CupyWavelet
from plotter import Plotter 

NUM_ITERATIONS = 10

# TODO SOON make a list of frame sizes and downsample factors to see
# which combo gets the best performance
WINDOW_SIZE = 4096

FILE_PATH = "audio_files/c4_and_c7_4_arps.wav"

# TODO NOW Consolidate this benchmark code with the cuda_cwt.py
class Benchmark():
    def __init__(self) -> None:
        # Audio Input
        audio_input = AudioInput(path = FILE_PATH, window_size = WINDOW_SIZE)
        self.audio_data = audio_input.get_frame()
        sample_rate = audio_input.get_sample_rate() # 44.1 kHz

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

        # Plotter Object
        plotter = Plotter(file_path = FILE_PATH)

        # Function List and Dummy Arguments (note special python ',' syntax)
        self.func_list = [
            (audio_input.get_frame,         ()),
            (py_wavelet.compute_cwt,        (self.audio_data,)),
            (np_wavelet.compute_cwt,        (self.audio_data,)),
            (cp_wavelet.compute_cwt,        (self.audio_data,)),
            (plotter.update_plot,           (self.coefs_py_wavelet,))
        ]

        # Tracks the run time of each function
        self.func_times = np.zeros(len(self.func_list))

    def main(self):
        print("Timing Analysis")

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
        print(f"Function runtimes averaged over {NUM_ITERATIONS} iterations")

        for i, item in enumerate(self.func_list):
            func = item[0]
            print(f"-> {func.__self__.__class__.__name__}.{func.__name__}()\t{self.avg_func_times[i]:6f} sec")

        """
        Static Plots
        """
        fig, axes = plt.subplots(4, 1, figsize=(10,5))

        # TODO LATER Fix the axes so they display freqs, not scales
        axes[0].set_title("Test Signal Time Series: C4 + C7")
        axes[0].plot(self.audio_data)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Amplitude")
        axes[0].margins(x=0, y=0)

        axes[1].set_title("PyWavelet CWT")
        axes[1].imshow(self.coefs_py_wavelet, cmap = "magma", aspect = "auto")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Scale")

        axes[2].set_title("NumPy CWT")
        axes[2].imshow(self.coefs_np_wavelet, cmap = "magma", aspect = "auto")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Scale")

        axes[3].set_title("CuPy CWT")
        axes[3].imshow(self.coefs_cp_wavelet, cmap = "magma", aspect = "auto")
        axes[3].set_xlabel("Time")
        axes[3].set_ylabel("Scale")

        plt.tight_layout()
        plt.show()

        while(True):
            pass

if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.main()