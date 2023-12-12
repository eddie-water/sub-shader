from models.main import Model
from views.main import View

class Controller:
    def __init__(self, model: Model, view: View) -> None:
        print("Hello controller")
        self.model = model
        self.view = view

    def start(self) -> None:
        print("Controller start")