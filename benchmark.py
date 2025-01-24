import time

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

        dummy_audio = audio_input.get_frame()

        sampling_freq = audio_input.get_sample_rate() # 44.1 kHz

        # Wavelet Object
        wavelet = Wavelet(sampling_freq = sampling_freq, 
                          frame_size = FRAME_SIZE,
                          downsample_factor = DOWNSAMPLE_FACTOR)

        dummy_coefs = wavelet.compute_cwt(dummy_audio)

        # Plotter Object
        plotter = Plotter(file_path = FILE_PATH)

        # Function List
        self.function_list = [
            (audio_input.get_frame, ()),
            (wavelet.compute_cwt,   (dummy_audio,)),
            (plotter.update_plot,   (dummy_coefs,))
        ]

    def main(self):
        print("Timing Analysis")

        for item in self.function_list:
            func = item[0]
            args = item[1] if len(item) > 1 else ()
            kwargs = item[2] if len(item) > 2 else {}

            t_start = time.perf_counter()
            _ = func(*args, **kwargs)
            t_end = time.perf_counter()
            t_delta = t_end - t_start

            print(f"{func.__name__}:\n -> {t_delta:6f} s")

if __name__ == '__main__':
    benchtest = Benchtest()
    benchtest.main()