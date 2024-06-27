import numpy as np
import matplotlib.pyplot as plt
import pywt

def create_mixed_signal(time, sinusoid_list, length, sample_rate):
    sample_period = 1.0 / sample_rate

    num_samples = time.size
    num_sinusoids = len(sinusoid_list)

    print("num_sinuspoids", num_sinusoids)
    signal = np.zeros(shape = (num_sinusoids, num_samples))
    print(signal)

    # TODO delete this 
    i = 0 
    for start, end, freq in sinusoid_list:

        print(start, end, freq)

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
        print("mask:", mask)

        # Apply mask
        n = n * mask

        signal[i] += np.sin(2*np.pi*freq*n)

        fig, ax = plt.subplots()
        ax.set_title("Mask of " + str(freq) + " Hz Signal")
        ax.plot(t, signal[i])

        i += 1

    return (signal[0] + signal[1] + signal[2])

# TODO moves this to example file
if __name__ == "__main__":

    # The audio was sampled at 44100 Hz
    SAMPLING_FREQUENCY = 44100
    SAMPLING_PERIOD = 1.0 / SAMPLING_FREQUENCY
    print("The Sampling Frequency:", SAMPLING_FREQUENCY, "Hz and Sampling Period", SAMPLING_PERIOD, "seconds")

    # How long the audio is in seconds
    AUDIO_LENGTH_SECONDS = 1

    # When using numpy.linspace, you need to know number of steps
    ta = np.linspace(0, AUDIO_LENGTH_SECONDS, AUDIO_LENGTH_SECONDS*SAMPLING_FREQUENCY)
    ta_sample_period = np.diff(ta).mean()

    # When using numpy.arange, you need to know step_size
    tb = np.arange(0, AUDIO_LENGTH_SECONDS, SAMPLING_PERIOD)
    tb_sample_period = np.diff(tb).mean()

    # print("Linspace: The average sampling period is:", ta_sample_period)
    # print("Linspace: Error:", abs(SAMPLING_PERIOD - ta_sample_period))
    # print("Arange: The average sampling period is:", tb_sample_period)
    # print("Arange: Error:", abs(SAMPLING_PERIOD - tb_sample_period))
    
    # Using numpy.arange is better
    t = tb

    # Piano note's frequencies in Hz
    FREQ_A3 = 220
    FREQ_A4 = 440
    FREQ_A6 = 1760

    # START %, END %, FREQ
    list_of_signals = [(0, 100, FREQ_A3),
                       (50, 100, FREQ_A4),
                       (25, 75, FREQ_A6)]

    # TODO revert
    # t = np.arange(0, AUDIO_LENGTH_SECONDS*SAMPLING_FREQUENCY, AUDIO_LENGTH_SECONDS)
    # t = np.arange(0, (8*SAMPLING_PERIOD), SAMPLING_PERIOD)
    t = np.arange(0, AUDIO_LENGTH_SECONDS, SAMPLING_PERIOD)
    # t = np.arange(0, 0.8, 0.1)

    signal = create_mixed_signal(t, list_of_signals, AUDIO_LENGTH_SECONDS, SAMPLING_FREQUENCY)

    # 220 Hz 0 >= t < 2
    signal_a3 = np.sin(2*np.pi*FREQ_A3*t)

    # 440 Hz 1 >= t < 2
    # t4 = np.pad(t, (50 / 100 *t.size, 100 / 100), 'constant')[(50 / 100)*t.size:(100 / 100)*t.size]
    # signal_a4 = np.sin(2*np.pi*FREQ_A4*t4)

    # 1760 Hz 0.5 >= t < 1.5
    
    # t4 = np.pad(t, (shift, 0), 'constant')[:t.size]
    # signal_a4 = np.sin(2*np.pi*FREQ_A4*t4)


    fig, ax = plt.subplots()
    # ax.spines['bottom'].set_position('bottom')
    ax.plot(t, signal)
    ax.grid()
    plt.show()

    while(True):
        x = 1