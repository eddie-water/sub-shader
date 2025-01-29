import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utils import FrameCounter

from audio_input import AudioInput
from wavelet import Wavelet
from plotter import Plotter

# Constants: 
#   Ideally have the biggest frame size and the smallest downsample factor 
FRAME_SIZE = 4096
DOWNSAMPLE_FACTOR = 8

FILE_PATH = "audio_files/c4_and_c7_4_arps.wav"

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
    # TODO Do not merge this is just for testing and debugging
    # coefs = wavelet.compute_cwt(audio_data)
    coefs = wavelet.compute_cwt_norm(audio_data)

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
