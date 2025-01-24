import time
import numpy as np

from audio_input import AudioInput
from wavelet import Wavelet
from plotter import Plotter 

NUM_FUNCTIONS = 3
NUM_ITERATIONS = 10

# TODO NEXT make a list of frame sizes and downsample factors to see
# which combo gets the best performance
FRAME_SIZE = 256
DOWNSAMPLE_FACTOR = 8

# TODO NEXT Create a wav for bench testing
FILE_PATH = "audio_files/zionsville.wav"

# TODO SOON figure out a way to take the return value a method, and insert it
# into the list for the next function

# TODO SOON maybe that's not the point of bench test, maybe it should all
# just be dummy values

class Benchtest():
    def __init__(self) -> None:
        # Audio Input
        audio_input = AudioInput(path = FILE_PATH, frame_size = FRAME_SIZE)

        sampling_freq = audio_input.get_sample_rate() # 44.1 kHz

        # Wavelet Object
        wavelet = Wavelet(sampling_freq = sampling_freq, 
                          frame_size = FRAME_SIZE,
                          downsample_factor = DOWNSAMPLE_FACTOR)

        # Plotter Object
        plotter = Plotter(file_path = FILE_PATH)

    def main(self):
        """
        Audio Data Acquisition
        """
        t_start = time.perf_counter()
        audio_data = self.audio_input.get_frame()
        t_delta = time.perf_counter() - t_start
        print(f"Audio Data Retrieval:               {t_delta:.6f} seconds")

        self.t_total += t_delta

        """
        CWT Computation 
        """
        t_start = time.perf_counter()
        coefs = self.wavelet.compute_cwt(audio_data)
        t_delta = time.perf_counter() - t_start
        print(f"CWT Computation:                    {t_delta:.6f} seconds")

        self.t_total += t_delta

        """
        Plotting CWT Data
        """
        t_start = time.perf_counter()
        self.plotter.update_cwt_pcm(coefs= coefs)
        t_delta = time.perf_counter() - t_start
        print(f"Plotting CWT Data:                  {t_delta:.6f} seconds")

        self.t_total += t_delta

        """
        Calculating Time Metrics
        """
        print("\n")
        print(f"Total Average Function Time:        {self.t_total:.6f} seconds")
        print(f"Total Average Function Time Goal:  <{self.t_total_max:.6f} seconds")

        # Assess
        percent_error = 100 * (((self.t_total_max - self.t_total) / self.t_total_max))

        print("\n")
        if (percent_error < 0):
            print(f"Fail. Percent Error: {percent_error:.2f} %")
        else:
            print(f"Success. Percent Error: {percent_error:.2f} %")
        print("\n")

if __name__ == '__main__':
    benchtest = Benchtest()
    benchtest.main()