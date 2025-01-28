import time
import numpy as np

from audio_input import AudioInput
from wavelet import Wavelet
from plotter import Plotter 

NUM_ITERATIONS = 100

# TODO NEXT make a list of frame sizes and downsample factors to see
# which combo gets the best performance
FRAME_SIZE = 256
DOWNSAMPLE_FACTOR = 8

# TODO NEXT Create a wav for bench testing
FILE_PATH = "audio_files/zionsville.wav"

class Benchtest():
    def __init__(self) -> None:
        # Audio Input
        audio_input = AudioInput(path = FILE_PATH, frame_size = FRAME_SIZE)

        dummy_audio = audio_input.get_frame()

        sampling_freq = audio_input.get_sample_rate() # 44.1 kHz

        # Wavelet Object
        wavelet = Wavelet(sampling_freq = sampling_freq, 
                          frame_size = FRAME_SIZE,
                          downsample_factor = DOWNSAMPLE_FACTOR)

        dummy_coefs = wavelet.compute_cwt(dummy_audio)

        # Plotter Object
        plotter = Plotter(file_path = FILE_PATH)

        # Function List AN
        self.func_list = [
            (audio_input.get_frame, ()),
            (wavelet.compute_cwt,   (dummy_audio,)),
            (plotter.update_plot,   (dummy_coefs,))
        ]

        self.func_times = np.zeros(len(self.func_list))

    def main(self):
        print("Timing Analysis")

        for i in range(NUM_ITERATIONS):
            # print(f"Loop {i}:")

            for i, item in enumerate(self.func_list):
                # Grab the function
                func = item[0]
                args = item[1] if len(item) > 1 else ()
                kwargs = item[2] if len(item) > 2 else {}

                # Time the function
                t_start = time.perf_counter()
                _ = func(*args, **kwargs)
                t_end = time.perf_counter()
                t_delta = t_end - t_start

                self.func_times[i] += t_delta

        # Average the times
        self.avg_func_times = self.func_times / int(NUM_ITERATIONS)

        print(f"Time to run each function averaged at {NUM_ITERATIONS} times")

        for i, item in enumerate(self.func_list):
            func = item[0]
            print(f"-> {func.__name__}\t{self.avg_func_times[i]:6f} sec")

if __name__ == '__main__':
    benchtest = Benchtest()
    benchtest.main()