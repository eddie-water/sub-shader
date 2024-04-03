import time
from models.main import Model
from views.main import View

class Controller:
    def __init__(self, model: Model, view: View) -> None:
        self.model = model
        self.view = view

    def run(self) -> None:
        try:
            while(1):
                fft_data = self.model.sliding_fft()
                self.view.plot(fft_data)
                time.sleep(.05)

        except Exception as e:
                print("Exception: ", e)
