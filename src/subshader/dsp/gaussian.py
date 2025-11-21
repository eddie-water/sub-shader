import numpy as np

class Gaussian:
    def __init__(self, 
                 t: np.ndarray[np.float64],
                 f: np.float64,
                 num_fwhm_cycles: int = 3) -> None:
        """
        Constructs a Gaussian bell curve. The bell curve is shaped using a the
        Full Width at Half Maximum (FWHM), where we define this width to contain
        a given number of cycles a particular frequency.

        The FWHM are the points of the curve where its value is half of its
        maximum value. So if the maximum value is 1, the FWHM is the width
        between the points on the curve where the value is 0.5. This
        characterizes the how steep the curve's rolloff is  and controls how much energy of the
        carrier signal is retained.

        When the FWHM is wide, the curve will roll off slowly and more energy
        will be retained. When the FWHM is narrow, the curve will roll off
        quickly and less energy will be preserved.

        Args:
            t: The time array (s)
            f: The center frequency (Hz)
            num_fwhm_cycles: The number of cycles of the center frequency to 
                contain in the FWHM (num)
        """
        # FWHM-based Gaussian
        self.fwhm_support_s: np.float64 = np.float64(num_fwhm_cycles / f)
        self.gauss_fwhm: np.ndarray[np.complex64] = np.exp(-4 * np.log(2) * (t ** 2) / self.fwhm_support_s ** 2).astype(np.complex64)

        # Standard deviation-based Gaussian
        self.num_std_deviations: int = 3
        self.gauss_std: np.ndarray[np.complex64] = np.exp(-(t ** 2) / (2 * self.num_std_deviations ** 2)).astype(np.complex64)

        # Use the FWHM-based Gaussian by defualt
        self.gauss_t = self.gauss_fwhm 