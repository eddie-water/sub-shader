from models.main import Model
from views.main import View

class Controller:
    def __init__(self, model: Model, view: View) -> None:
        self.model = model
        self.view = view

    def run(self) -> None:
        try:
            while(1):
                data = self.model.audio_task()
                self.view.plot_task(data)

        except Exception as e:
                print("Exception: ", e)
