# src/subshader/__main__.py
import time

from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import ShadeWavelet
from subshader.viz.plotter import Shader

WINDOW_SIZE = 4096
DOWNSAMLPLE_STRIDE = 4
FILE_PATH = "assets/audio/c4_and_c7_4_arps.wav"

# Audio Input, Audio Characteristics 
audio_input = AudioInput(path = FILE_PATH, window_size = WINDOW_SIZE)

sample_rate = audio_input.get_sample_rate() # 44.1 kHz

# Wavelet Object
wavelet = ShadeWavelet(sample_rate = sample_rate, 
                       window_size = WINDOW_SIZE,
                       ds_stride = DOWNSAMLPLE_STRIDE)

# Plotter Object
plot_shape = wavelet.get_shape()
plotter = Shader(file_path = FILE_PATH,
                 shape = plot_shape)

fps_timer = time.perf_counter()
frame_times = []

def main_loop():
    global fps_timer

    while plotter.window and not plotter.should_window_close():

        frame_start = time.perf_counter()

        # Grab a frame of audio
        audio_data = audio_input.get_frame()

        # Compute CWT on that frame
        coefs = wavelet.compute_cwt(audio_data)

        # Update plot
        plotter.update_plot(coefs)

        # TODO ISSUE-33 Put this in the plotter class
        frame_end = time.perf_counter()
        frame_times.append(frame_end - frame_start)

        # TODO ISSUE-33 Is this a weird way to measure FPS?
        if time.time() - fps_timer > 1.0 and len(frame_times) > 0:
            avg_frame = sum(frame_times) / len(frame_times)
            print(f"FPS: {1.0 / avg_frame:.2f}")
            frame_times.clear()
            fps_timer = time.time()

# Main entry point
if __name__ == '__main__':
    main_loop()
