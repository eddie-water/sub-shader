import time
import numpy as np

from audio_input import AudioInput
from wavelet import PyWavelet, AntsWavelet
from plotter import Plotter 

NUM_ITERATIONS = 10

# TODO SOON make a list of frame sizes and downsample factors to see
# which combo gets the best performance
WINDOW_SIZE = 4096
DOWNSAMPLE_FACTOR = 1

FILE_PATH = "audio_files/c_4_arps.wav"

class Benchtest():
    def __init__(self) -> None:
        # Audio Input
        audio_input = AudioInput(path = FILE_PATH, window_size = WINDOW_SIZE)

        # TODO NEXT create 
        dummy_audio = audio_input.get_frame()

        sample_rate = audio_input.get_sample_rate() # 44.1 kHz

        # PyWavelet Object
        py_wavelet = PyWavelet(sample_rate = sample_rate, 
                               window_size = WINDOW_SIZE,
                               downsample_factor = DOWNSAMPLE_FACTOR)

        py_coefs = py_wavelet.compute_cwt(dummy_audio)

        # Shade Wavelet Object
        ants_wavelet = AntsWavelet(sample_rate = sample_rate, 
                                     window_size = WINDOW_SIZE,
                                     downsample_factor = DOWNSAMPLE_FACTOR)

        shade_coefs = ants_wavelet.compute_cwt(dummy_audio)

        # Plotter Object
        plotter = Plotter(file_path = FILE_PATH)

        # Function List and Dummy Arguments (note special python ',' syntax)
        self.func_list = [
            (audio_input.get_frame,         ()),
            (py_wavelet.compute_cwt,        (dummy_audio,)),
            (ants_wavelet.compute_cwt,     (dummy_audio,)),
            (plotter.update_plot,           (py_coefs,))
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

if __name__ == '__main__':
    benchtest = Benchtest()
    benchtest.main()