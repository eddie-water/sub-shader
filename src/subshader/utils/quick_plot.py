import matplotlib.pyplot as plt

class QuickPlot():
    def __init__(self, data):
        self.fig, self.axes = plt.subplots()

        self.axes.plot(data)
