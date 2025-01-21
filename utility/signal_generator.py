import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator():
    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = sample_rate
        self.sample_period = 1.0 / self.sample_rate
        pass

    def create_mixed_signal(time: np.ndarray, sinusoid_list) -> np.ndarray:
        """
        Keyword Argument:
            time: the 
        """
        num_samples = time.size

        signal = np.zeros(num_samples)

        for start, end, freq in sinusoid_list:
            n = time

            # Create impulse using start and end %
            start = n[int((start/100.0)*num_samples)]
            end = n[int((end/100.0)*num_samples) - 1]

            # Creates a pulse mask
            mask = []
            for j in n:
                if ((j >= start) and (j <= end)):
                    mask.append(1)
                else:
                    mask.append(0)
            mask = np.array(mask)

            # Apply mask
            n = n * mask

            signal += np.sin(2*np.pi*freq*n)

        return signal

# TODO moves this to example file
if __name__ == "__main__":

    # The audio was sampled at 44100 Hz
    SAMPLING_FREQUENCY = 44100
    SAMPLING_PERIOD = 1.0 / SAMPLING_FREQUENCY
    # print("The Sampling Frequency:", SAMPLING_FREQUENCY, "Hz and Sampling Period", SAMPLING_PERIOD, "seconds")

    # How long the audio is in seconds
    AUDIO_LENGTH_SECONDS = .1

    # When using numpy.linspace, you need to know number of steps
    # ta = np.linspace(0, AUDIO_LENGTH_SECONDS, AUDIO_LENGTH_SECONDS*SAMPLING_FREQUENCY)
    # ta_sample_period = np.diff(ta).mean()

    # When using numpy.arange, you need to know step_size
    tb = np.arange(0, AUDIO_LENGTH_SECONDS, SAMPLING_PERIOD)
    tb_sample_period = np.diff(tb).mean()

    # print("Linspace: The average sampling period is:", ta_sample_period)
    # print("Linspace: Error:", abs(SAMPLING_PERIOD - ta_sample_period))
    # print("Arange: The average sampling period is:", tb_sample_period)
    # print("Arange: Error:", abs(SAMPLING_PERIOD - tb_sample_period))

    # 220 Hz    0.25 >= t < 1.00
    # 440 Hz    0.50 >= t < 1.00
    # 1760 Hz   0.50 >= t < 0.75
    FREQ_A3 = 220
    FREQ_A4 = 440
    FREQ_A6 = 1760

    # START %, END %, FREQ
    list_of_signals = [(0, 100, FREQ_A3),
                       (50, 100, FREQ_A4),
                       (25, 75, FREQ_A6)]

    t = np.arange(0, AUDIO_LENGTH_SECONDS, SAMPLING_PERIOD)

    signal = create_mixed_signal(t, list_of_signals)

    fig, ax = plt.subplots()
    ax.plot(t, signal)
    ax.grid()
    plt.show()
