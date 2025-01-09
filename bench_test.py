import time
import numpy as np

# pyqtgraph dependencies
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from .utils import FrameCounter # TODO SOON mvdir somewhere better

from models.audio_input import AudioInput
from models.wavelet import Wavelet
from views.plotter import Plotter 

NUM_FUNCTIONS = 3
NUM_ITERATIONS = 10

FILE_PATH = "models/audio_files/zionsville.wav"

# TODO NOW > NEXT > SOON > LATER

class Benchtest():
    def __init__(self) -> None:
        self.sampling_rate = 44100.0 # Hz
        self.sample_period = (1 / self.sampling_rate)

        self.frame_size = 4096

        self.t_total_max = self.sample_period * self.frame_size

        self.t_total = 0

        # Setup
        self.audio_input = AudioInput(
            path= FILE_PATH,
            frame_size= self.frame_size)

        self.wavelet = Wavelet(frame_size= self.frame_size)

        self.plotter = Plotter(
            data_shape= self.wavelet.get_shape(),
            sampling_rate= self.sampling_rate,
            plot_name= "Bench Testing Zionsville")

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