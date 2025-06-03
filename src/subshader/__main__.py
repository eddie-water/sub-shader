# src/subshader/__main__.py

from subshader.audio.audio_input import AudioInput
from subshader.dsp.wavelet import ShadeWavelet
from subshader.viz.plotter import Shader

WINDOW_SIZE = 4096
FILE_PATH = "assets/audio/c4_and_c7_4_arps.wav"

# Audio Input, Audio Characteristics 
audio_input = AudioInput(path = FILE_PATH, window_size = WINDOW_SIZE)

sample_rate = audio_input.get_sample_rate() # 44.1 kHz

# Wavelet Object
wavelet = ShadeWavelet(sample_rate = sample_rate, 
                       window_size = WINDOW_SIZE)

# Plotter Object
plot_shape = wavelet.get_shape()
plotter = Shader(file_path = FILE_PATH,
                 shape = plot_shape)

def main_loop():
    while plotter.window and not plotter.should_window_close():
        # Grab a frame of audio
        audio_data = audio_input.get_frame()

        # Compute CWT on that frame
        coefs = wavelet.compute_cwt(audio_data)

        # Update plot
        plotter.update_plot(coefs)
        
        # TODO ISSUE-33 Reinstate FPS tracker logic - just pass it in the update_plot no?

# Main entry point
if __name__ == '__main__':
    main_loop()
