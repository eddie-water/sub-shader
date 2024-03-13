from views.main import View
from models.main import Model
from controllers.main import Controller

def main():
    model = Model()
    view = View()
    controller = Controller(model, view)
    controller.run()

if __name__ == "__main__":
     main()
