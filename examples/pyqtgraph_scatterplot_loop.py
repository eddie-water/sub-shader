#!/usr/bin/env python

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

import time

NUM_POINTS = 10000

class MyWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        # Configure Qt Timer at 100 ms
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.onNewData)
        self.timer.start()

        # Add a plot to MyWidget... and then plot an empty array?

        # TODO B implement something like pyqtgraph/examples/PColorMeshItem.py
        self.plotItem = self.addPlot(title=f"{NUM_POINTS} Random Points")
        self.plotDataItem = self.plotItem.plot([], 
                                               pen=None, 
                                               symbolBrush=(255,0,0), 
                                               symbolSize=5, 
                                               symbolPen=None)

        # TODO A
        # Performance can be significantly improved in 
        # this example if you also disable autoscaling of the axes. Just set a 
        # fixed scale for each. I just discovered that today, by chance

        self.t_now = 0
        self.t_then = 0

    def setData(self, x, y):
        self.plotDataItem.setData(x, y)

    def onNewData(self):
        t_start = time.perf_counter()

        x = np.random.normal(size=NUM_POINTS)
        y = np.random.normal(size=NUM_POINTS)

        self.setData(x, y)

        t_delta = time.perf_counter() - t_start 
        print(f"plotDataItem.setData:               {t_delta:.6f} seconds")

def main():
    app = QtWidgets.QApplication([])

    pg.setConfigOptions(antialias=False) # True seems to work as well

    win = MyWidget()
    win.show()
    win.resize(800,600) 
    win.raise_()
    app.exec_()

if __name__ == "__main__":
    main()
