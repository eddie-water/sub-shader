# https://github.com/OmarAlkousa/Fourier-Analysis-as-Streamlit-Web-App/tree/main

import numpy as np
import scipy

class Fourier:
    """
    Apply the Discrete Fourier Transform (DFT) on the signal using the Fast Fourier 
    Transform (FFT) from the scipy package.

    Example:
      fourier = Fourier(signal, sampling_rate=2000.0)
    """

    def __init__(self, signal, sampling_rate):
        """
        Initialize the Fourier class.

        Args:
            signal (np.ndarray): The samples of the signal
            sampling_rate (float): The sampling per second of the signal

        Additional parameters,which are required to generate Fourier calculations, are
        calculated and defined to be initialized here too:
            time_step (float): 1.0/sampling_rate
            time_axis (np.ndarray): Generate the time axis from the duration and
                                  the time_step of the signal. The time axis is
                                  for better representation of the signal.
            duration (float): The duration of the signal in seconds.
            frequencies (numpy.ndarray): The frequency axis to generate the spectrum.
            fourier (numpy.ndarray): The DFT using rfft from the scipy package.
        """
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.time_step = 1.0/self.sampling_rate
        self.duration = len(self.signal)/self.sampling_rate
        self.time_axis = np.arange(0, self.duration, self.time_step)
        self.frequencies = scipy.fft.rfftfreq(
            len(self.signal), d=self.time_step)
        self.fourier = scipy.fft.rfft(self.signal)

    # Generate the actual amplitudes of the spectrum
    def amplitude(self):
        """
        Method of Fourier

        Returns:
            numpy.ndarray of the actual amplitudes of the sinusoids.
        """
        return 2*np.abs(self.fourier)/len(self.signal)

    # Generate the phase information from the output of rfft
    def phase(self, degree=False):
        """
        Method of Fourier

        Args:
            degree: To choose the type of phase representation (Radian, Degree).
                    By default, it's in radian. 

        Returns:
            numpy.ndarray of the phase information of the Fourier output.
        """
        return np.angle(self.fourier, deg=degree)

    # Plot the Signal and the Spectrum interactively
    def plot_time(self,
                  ylabel="Amplitude",
                  title="Time Domain",
                  line_color='#00FF00'):
        """
        Plot the Signal in Time Domain using plotly.

        Args:
            ylabel (String): Label of the y-axis in Time-Domain
            title (String): Title of the Time-Domain plot
            line_color (String): The color of the line chart (HTML Code)

        Returns:
            One figure: the time-domain.
        """
        # Time Domain
        self.fig = px.line(x=self.time_axis, y=self.signal)
        self.fig.update_layout({"title": {"text": title,
                                          "font": {"size": 30, "family": "Times New Roman, bold"},
                                          "x": 0.5,
                                          "xanchor": "center",
                                          "yanchor": "top"},
                                "xaxis": {"title": "Time [sec]"},
                                "yaxis": {"title": ylabel},
                                "hovermode": "x unified"
                                })
        self.fig.update_traces(line_color=line_color, line_width=1,
                               hovertemplate="Time= %{x}<br>Amplitude= %{y}")
        return self.fig

    def plot_frequency(self,
                       ylabel="Amplitude",
                       title="Frequency Domain",
                       line_color='#FF0000'):
        """
        Plot the Signal in Frequency Domain using plotly.

        Args:
            ylabel (String): Label of the y-axis in Frequency-Domain
            title (String): Title of the frequency-Domain plot
            line_color (String): The color of the line chart (HTML Code)

        Returns:
            One figure: the frequency-domain.
        """
        # Frequency Domain
        self.fig = px.line(x=self.frequencies, y=self.amplitude())
        self.fig.update_layout({"title": {"text": title,
                                          "font": {"size": 30, "family": "Times New Roman, bold"},
                                          "x": 0.5,
                                          "xanchor": "center",
                                          "yanchor": "top"},
                                "xaxis": {"title": "Frequency [Hz]"},
                                "yaxis": {"title": ylabel},
                                "hovermode": "x unified"
                                })
        self.fig.update_traces(line_color=line_color, line_width=1,
                               hovertemplate="Time= %{x}<br>Amplitude= %{y}")
        return self.fig


###################################
######### Streamlit Code ##########
###################################

# Set a title of the app
st.markdown("<h1 style='text-align: center; color: grey;'>Fourier Analysis</h1>",
            unsafe_allow_html=True)
# Explanation of the web app
st.markdown("Digital signal processing (DSP) is the computation of mathematical methods used to manipulate signal data. \
            One of the most important tools in digital signal processing is the Discrete Fourier Transform (DFT). \
            It is usually used to produce a signal's frequency-domain (spectral) representation.")

st.markdown("The **Fast Fourier Transform (FFT)** is the practical implementation of the Fourier Transform on Digital Signals. \
    FFT is considered one of the [**top 10 algorithms**](https://doi.ieeecomputersociety.org/10.1109/MCISE.2000.814652) \
    with the greatest impact on science and engineering in the 20th century.")

st.markdown("For more information about Fourier Transform, check out this\
            [post](https://medium.com/towards-data-science/learn-discrete-fourier-transform-dft-9f7a2df4bfe9)\
             on Towards Data Science and see the how it works mathematically. Also, if you are interested in how to implement \
            the FFT algorithm in Python, follow this [post](https://towardsdatascience.com/fourier-transform-the-practical-python-implementation-acdd32f1b96a).")

st.markdown("This web app allows you to decompose your signals or time series using FFT and gives the opportunity to \
            interactively investigate the signal and its spectrum using the advantage of **Plotly** package. \
            All you have to do is to upload the file of the signal **(.csv)** and specify the **sampling rate**. \
            Additional properties (optional) can be edited like the title of each time and freuquency figures, the labels of the y-axes, \
            and the the color line of each time and frequency data.")

st.markdown("#### Note:")

st.markdown("Make sure to specify the exact sampling rate of the signal, otherwise you might end up with the wrong frequency resolution \
            (or even worse...👻 **Aliasing** 👻).")

st.markdown(
    "Get the code of this web app following the [**GitHub link**](https://github.com/OmarAlkousa/Fourier-Analysis-as-Streamlit-Web-App.git).")

# Horizontal Separator Line
st.markdown("""---""")

# Fast Example of the app
st.markdown("### Import the signal file (.csv)")
st.markdown("If you want a fast try of this app and you don't have any signal file, you can download example file that is in the same \
            [**GitHub repository**](https://github.com/OmarAlkousa/Fourier-Analysis-as-Streamlit-Web-App/blob/main/example.csv) of the app. \
            The sampling rate of our example signal is **360.0** sample per second.")

# Upload the file of the signal (only .csv)
uploaded_file = st.file_uploader(
    label="Import the file of the signal (.csv)", type='csv')

# If the file is uploaded
if uploaded_file is not None:

    # Caching the data for faster implementation
    @st.cache_data
    def load_signal():
        sig = np.loadtxt(uploaded_file.name, delimiter=',')
        return sig

    # Load the Data
    signal = load_signal()

    # Input the sampling rate of the signal
    sampling_rate = st.number_input(label='Sampling Rate [samples per second]',
                                    min_value=0.01,
                                    help='Specify the exact sampling rate of the signal')

    # Optional Configuration
    with st.expander("Optional Configuration"):

        # Configuration of the time domain
        t_title = st.text_input(label='Specify the title of the time domain:',
                                value='Time Domain', placeholder="'Time Domain' By default")
        t_ylabel = st.text_input(label='Specify the label of y-axis of the time domain plot:',
                                 value='Amplitude', placeholder="'Amplitude' By default")
        t_line_color = st.color_picker(
            label='Specify the color of the line chart of the time domain:', value='#0000FF')

        # Configuration of the frequency domain
        f_title = st.text_input(label='Specify the title of the frequency domain:',
                                value='Frequency Domain', placeholder="'Frequency Domain' By default")
        f_ylabel = st.text_input(label='Specify the label of y-axis of the frequency domain plot:',
                                 value='Amplitude', placeholder="'Amplitude' By default")
        f_line_color = st.color_picker(
            label='Specify the color of the line chart of the frequency domain:', value='#FF0000')

    # DFT using the class Fourier
    signal_spectrum = Fourier(signal=signal, sampling_rate=sampling_rate)

    # Plot the time domain of the signal
    fig1 = signal_spectrum.plot_time(
        title=t_title, ylabel=t_ylabel, line_color=t_line_color)

    # Plot the frequency domain of the signal
    fig2 = signal_spectrum.plot_frequency(
        title=f_title, ylabel=f_ylabel, line_color=f_line_color)

    # Streamlit the plotly figures
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
