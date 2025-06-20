import matplotlib.pyplot as plt

class QuickPlot():
    def __init__(self, data):
        """
        Creates a quick plot of the provided data using matplotlib.
        
        Args:
            data (list or np.ndarray): The data to be plotted.
        """
        self.fig, self.axes = plt.subplots()
        self.axes.plot(data)
