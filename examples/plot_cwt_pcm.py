import numpy as np
import matplotlib.pyplot as plt
import pywt

def create_mixed_signal(time, sinusoid_list):

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

# TODO turn this into an example that shows the short comings of the STFT
if __name__ == "__main__":
    try:
        # Signal Info
        SAMPLING_FREQUENCY = 44100
        SAMPLING_PERIOD = 1.0 / SAMPLING_FREQUENCY
        AUDIO_LENGTH_SECONDS = .1
        FREQ_A3 = 220
        FREQ_A4 = 440
        FREQ_A6 = 1760

        # START %, END %, FREQ
        signal_a3 = [(0, 100, FREQ_A3)]
        signal_a4 = [(50, 100, FREQ_A4)]
        signal_a6 = [(25, 75, FREQ_A6)]

        list_of_signals = [signal_a3[0], 
                           signal_a4[0],
                           signal_a6[0]]

        # Time axis
        t = np.arange(0, AUDIO_LENGTH_SECONDS, SAMPLING_PERIOD)

        # Generate 3 signals of different frequencies and time-supprt
        signal_a3 = create_mixed_signal(t, signal_a3)
        signal_a4 = create_mixed_signal(t, signal_a4)
        signal_a6 = create_mixed_signal(t, signal_a6)

        # Add all them together to create a nonstationary mixed signal
        signal_all = signal_a3 + signal_a4 + signal_a6

        signals = [signal_a3,
                signal_a4,
                signal_a6,
                signal_all]

        plot_titles = ["Signal A3: 220 Hz, 0% - 100% Time Support",
                       "Signal A4: 440 Hz, 50% - 100% Time Support",
                       "Signal A6: 1760 Hz, 25% - 75% Time Support",
                       "Signal A3 + Signal A4 + Signal A6"]

        # Wavelet Info
        wavelet_name = "cmor1.5-1.0"
        sampling_period = SAMPLING_PERIOD


        # Calculate the set of scales we are interested in 
        VOICES_PER_OCTAVE = 12
        NUM_OCTAVES = 7

        a0 = 2**(1/VOICES_PER_OCTAVE)
        i = np.arange(0, VOICES_PER_OCTAVE*NUM_OCTAVES, 1)
        s = a0**(i)
        f = 27.5*s
        f_norm = f / SAMPLING_FREQUENCY
        scales = pywt.frequency2scale(wavelet_name, f_norm)

        # Plot the signal ingre
        for i, signal in enumerate(signals):
            cwtmatr, freqs = pywt.cwt(data = signal, 
                                      scales = scales, 
                                      wavelet = wavelet_name, 
                                      sampling_period = sampling_period)

            # TODO sharpen the higher transient activity with a log function 
            # like in that one youtube video I watched

            # VERY IMPORTANT STEP HERE
            cwtmatr = np.abs(cwtmatr[:-1, :-1])

            # Plot 
            fig, axes = plt.subplots(2, 1)
            cmap = plt.colormaps['magma']

            # Plot signal amplitude vs time
            axes[0].set_title(plot_titles[i])
            axes[0].set_xlabel("Time (s)")
            axes[0].set_xlim(0, .1)
            axes[0].set_ylim(-2.5, 2.5)
            axes[0].set_ylabel("Magnitude")
            axes[0].plot(t, signal)

            # Plot signal scalogram frequency vs time
            axes[1].set_xlim(0, .1)
            axes[1].set_xlabel("Time (s)")
            axes[1].set_yscale("log")
            axes[1].set_ylabel("Frequency (Hz)")
            pcm = axes[1].pcolormesh(t, freqs, cwtmatr, cmap = cmap)

            """
            Everything below here is trying to find a faster alternative to 
            pcolormesh, I was going to try and use the img obj, but never got
            around to finishing that
            """
            # default_coefs = np.zeros(cwtmatr.shape)

            # # temp = pcm.get_array()

            # # ------> pcm.set_array(coefs) <-----------
            # axes[2].set_xlim(0, .1)
            # axes[2].set_xlabel("Time (s)")
            # axes[2].set_yscale("log")
            # axes[2].set_ylabel("Frequency (Hz)")
            # pcm2 = axes[2].pcolormesh(t, freqs, default_coefs)

        # Plot all the scalograms
        plt.show()

    except KeyboardInterrupt:
        print("'Ctrl' + 'C' to force quit the program")