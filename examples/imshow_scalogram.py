from pylab import *
import pywt
import scipy.io.wavfile as wavfile

FILE_PATH = "models/audio_files/zionsville.wav"


# Find the highest power of two less than or equal to the input.
def lepow2(x):
    return 2 ** floor(log2(x))

# Make a scalogram given an MRA tree.
def scalogram(data):
    bottom = 0

    vmin = min(map(lambda x: min(abs(x)), data))
    vmax = max(map(lambda x: max(abs(x)), data))

    gca().set_autoscale_on(False)

    for row in range(0, len(data)):
        scale = 2.0 ** (row - len(data))

        imshow(
            array([abs(data[row])]),
            interpolation = 'nearest',
            vmin = vmin,
            vmax = vmax,
            extent = [0, 1, bottom, bottom + scale])

        bottom += scale

# Load the signal, take the first channel, limit length to a power of 2 for simplicity.
rate, signal = wavfile.read(FILE_PATH)
signal = signal[0:lepow2(len(signal)),0]
tree = pywt.wavedec(signal, 'db5')

# Plotting.
gray()
scalogram(tree)
show()
