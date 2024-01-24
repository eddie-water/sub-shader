from .plotter import Plotter

class View:
    def __init__(self) -> None:
        self.plotter = Plotter()

    def start(self) -> None:
        try:
            print("Starting view tasks\n")

            while(1):
                self.plotter.update()

        except KeyboardInterrupt:
            print("Stopping because of Keyboard Interrupt")