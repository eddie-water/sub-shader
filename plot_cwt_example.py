import numpy as np
import matplotlib.pyplot as plt
import pywt
from models.audio_input import AudioInput

FRAME_SIZE = 4096
FILE_PATH = "models/audio_files/c_octaves.wav"


def gaussian(x, x0, sigma):
    return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)

def make_chirp(t, t0, a):
    frequency = (a * (t + t0)) ** 2
    chirp = np.sin(2 * np.pi * frequency * t)
    return chirp, frequency

if __name__ == "__main__":
    frame_size = FRAME_SIZE
    audio_file = FILE_PATH

    window_size = 100000

    audio_input = AudioInput(path = audio_file,
                             frame_size = frame_size)

    signal = audio_input.get_entire_audio()
    print("The entire signal shape is", signal.shape)
    print("The entire signal type is", type(signal))

    # generate signal
    time = np.arange(0, len(signal))
    time = time[:window_size]
    print("The time shape is", time.shape, "and time is", time)

    chirp1, frequency1 = make_chirp(time, 0.2, 9)
    chirp2, frequency2 = make_chirp(time, 0.1, 5)
    chirp = chirp1 + 0.6 * chirp2
    chirp *= gaussian(time, 0.5, 0.2)

    # SIGNAL INFO
    # signal = chirp
    # print("Signal Length:", len(signal), "Type:", type(signal))

    # WAVELET INFO
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    # print("axs:", axs, "\n", "type(axs)", type(axs))

    wavelet_name = "cmor1.5-1.0"
    sampling_period = (1.0 / 44100)


    print("ax:", ax, "type(ax)", type(ax))

    signal = signal[:window_size]
    print("The windowed signal shape is", signal.shape)

    print("***")
    print("BEGIN CWT")
    print("***")

    # PLOT WAVELET
    widths = np.geomspace(1, 1024, num=75)
    cwtmatr, freqs = pywt.cwt(data = signal, 
                              scales = widths, 
                              wavelet = wavelet_name, 
                              sampling_period = sampling_period
    )

    print("***")
    print("DING! CWT COMPLETE")
    print("***")

    print("cwtmatr is of type", type(cwtmatr), "and of shape", cwtmatr.shape)#, "cwtmatr:", cwtmatr)
    print("Now taking abs value of cwtmar")
    cwtmatr = np.abs(cwtmatr[:-1, :-1])
    print("cwtmatr is of type", type(cwtmatr), "and of shape", cwtmatr.shape)#, "cwtmatr:", cwtmatr)
    print("freqs is of type", type(freqs), "and freqs is of shape", freqs.shape)#, "freqs:", freqs)

    pcm = ax.pcolormesh(time, freqs, cwtmatr)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("CWT Plotting Example")
    plt.colorbar(pcm, ax=ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
