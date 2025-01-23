import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter

from audio_input import AudioInput
from wavelet import Wavelet
from plotter import Plotter

"""
Constants
    Ideally we have the biggest frame size and the smallest downsample possible
"""
FRAME_SIZE = 256
DOWNSAMPLE_FACTOR = 8

FILE_PATH = "audio_files/zionsville.wav"

"""
Audio Input, Audio Characteristics 
"""
audio_input = AudioInput(path = FILE_PATH, frame_size = FRAME_SIZE)

sampling_freq = audio_input.get_sample_rate() # 44.1 kHz
sampling_period = (1.0 / sampling_freq)

"""
Wavelet Object
"""
wavelet = Wavelet(sampling_freq = sampling_freq, 
                  frame_size = FRAME_SIZE,
                  downsample_factor = DOWNSAMPLE_FACTOR)

data_shape = wavelet.get_shape()

"""
Plotter Object
"""
plotter = Plotter(file_path = FILE_PATH)

"""
Main Loop
"""
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

"""
FPS Counter
    Stolen from PyQtGraph
"""
fps_counter = FrameCounter()
fps_counter.sigFpsUpdate.connect(lambda fps: plotter.update_fps(fps))

"""
Main 
"""
if __name__ == '__main__':
    pg.exec()
