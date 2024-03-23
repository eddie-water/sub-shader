from views.main import View
from models.main import Model
from controllers.main import Controller

FRAME_SIZE = 4096

def main():
    model = Model(FRAME_SIZE)
    view = View(FRAME_SIZE)
    controller = Controller(model, view)
    controller.run()

if __name__ == "__main__":
     main()
