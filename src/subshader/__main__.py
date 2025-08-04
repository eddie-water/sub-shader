# src/subshader/__main__.py
import time

from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import ShadeWavelet
from subshader.viz.plotter import Shader

WINDOW_SIZE = 2 << 11 # 4k
DOWNSAMLPLE_STRIDE = 1
# # DO NOT MERGE
FILE_PATH = "assets/audio/daw/chirp_beat.wav"
# FILE_PATH = "assets/audio/songs/beltran_soundcloud.wav"
# FILE_PATH = "assets/audio/songs/zionsville.wav"
# FILE_PATH = "assets/audio/daw/c4_and_c7_4_arps.wav"

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
                 shape = plot_shape,
                 num_frames = 128)

fps_timer = time.perf_counter()
frame_times = []

def main_loop():
    global fps_timer

    while not plotter.should_window_close():
        # FPS
        frame_start = time.perf_counter()

        # Grab a frame of audio
        audio_data = audio_input.get_frame()

        if audio_data is None:
            print("End of audio reached")
            break

        # Compute CWT on that frame
        coefs = wavelet.compute_cwt(audio_data)

        # Update plot
        plotter.update_plot(coefs)

        # FPS 
        frame_end = time.perf_counter()
        frame_times.append(frame_end - frame_start)
        if time.time() - fps_timer > 1.0 and len(frame_times) > 0:
            avg_frame = sum(frame_times) / len(frame_times)
            print(f"FPS: {1.0 / avg_frame:.2f}")
            frame_times.clear()
            fps_timer = time.time()

    # Clean shutdown
    plotter.cleanup()

# Main entry point
if __name__ == '__main__':
    main_loop()