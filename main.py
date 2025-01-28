import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter

from audio_input import AudioInput
from wavelet import Wavelet
from plotter import Plotter

import time

# Constants: 
#   Ideally have the biggest frame size and the smallest downsample factor 
FRAME_SIZE = 256
DOWNSAMPLE_FACTOR = 8

FILE_PATH = "audio_files/zionsville.wav"

"""
Start Up
"""
# Start Up Timing 
t_start_up_start = time.perf_counter()

# Audio Input, Audio Characteristics 
audio_input = AudioInput(path = FILE_PATH, frame_size = FRAME_SIZE)

sampling_freq = audio_input.get_sample_rate() # 44.1 kHz

# Wavelet Object
wavelet = Wavelet(sampling_freq = sampling_freq, 
                  frame_size = FRAME_SIZE,
                  downsample_factor = DOWNSAMPLE_FACTOR)

# Plotter Object
plotter = Plotter(file_path = FILE_PATH)

def main_loop():
    # Grab a frame of audio
    audio_data = audio_input.get_frame()

    # Compute CWT on that frame
    coefs = wavelet.compute_cwt(audio_data)

    # Update plot
    plotter.update_plot(coefs)

    # Update FPS Count
    fps_counter.update()

timer = QtCore.QTimer()
timer.timeout.connect(main_loop)
timer.start()

# FPS Counter (stolen from PyQtGraph)
fps_counter = FrameCounter()
fps_counter.sigFpsUpdate.connect(lambda fps: plotter.update_fps(fps))

# Main 
if __name__ == '__main__':
    pg.exec()
