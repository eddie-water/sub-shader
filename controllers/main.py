import time
from models.main import Model
from views.main import View

import numpy as np

class Controller:
    def __init__(self, model: Model, view: View) -> None:
        self.model = model
        self.view = view

    def run(self) -> None:
        try:
            # TODO the passing of args between these two modules is clunky
            cwt_data = self.model.cwt()
            self.view.plot_cwt(coefs = cwt_data[0],
                               freqs = cwt_data[1],
                               time = cwt_data[2])

        except KeyboardInterrupt:
                print("Keyboard Interrupt")
