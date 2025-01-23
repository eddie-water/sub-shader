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
sampling_period = (1.0 / sampling_freq)

# Wavelet Object
wavelet = Wavelet(sampling_freq = sampling_freq, 
                  frame_size = FRAME_SIZE,
                  downsample_factor = DOWNSAMPLE_FACTOR)

data_shape = wavelet.get_shape()

# Plotter Object
plotter = Plotter(file_path = FILE_PATH)

# Print Start Up Timing Analysis
t_start_up_delta = time.perf_counter() - t_start_up_start
print("Start Up Timing Analysis:", t_start_up_delta)
print('\n')

"""
Runtime Loop
"""
# Loop Timing
print("Runtime Loop Timing Analysis:")
loop_count = 0

def main_loop():
    # Grab the global loop variable
    global loop_count

    # Grab a frame of audio
    t_start = time.perf_counter()
    audio_data = audio_input.get_frame()
    t_audio = time.perf_counter() - t_start

    # Compute CWT on that frame
    t_start = time.perf_counter()
    coefs = wavelet.compute_cwt(audio_data)
    t_cwt = time.perf_counter() - t_start

    # Update plot
    t_start = time.perf_counter()
    plotter.update_plot(coefs)
    t_plot = time.perf_counter() - t_start

    # Update FPS Count
    t_start = time.perf_counter()
    fps_counter.update()
    t_fps = time.perf_counter() - t_start

    # Print Runtime Loop Timing Analysis
    t_loop_total = t_audio + t_cwt + t_plot + t_fps
    print("Loop", loop_count, ":", t_loop_total)

    print("Audio :", t_audio)
    print("CWT   :", t_cwt)
    print("Plot  :", t_plot)
    print("FPS   :", t_fps)
    print("\n")

    loop_count += 1

timer = QtCore.QTimer()
timer.timeout.connect(main_loop)
timer.start()

# FPS Counter (stolen from PyQtGraph)
fps_counter = FrameCounter()
fps_counter.sigFpsUpdate.connect(lambda fps: plotter.update_fps(fps))

# Main 
if __name__ == '__main__':
    pg.exec()
