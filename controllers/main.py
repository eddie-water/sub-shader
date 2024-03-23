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
                # TODO: Implement some kind of timing bench mark around these
                # two operations
                fft_data = self.model.perform_sdft()
                self.view.plot(fft_data)
                time.sleep(.05)

        except Exception as e:
                print("Exception: ", e)
