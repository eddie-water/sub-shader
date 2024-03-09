import os

class FileOperations:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

        if (os.path.isfile(self.file_path)):
            return self.open(file_path)

    def open(self, file_path: str) -> bool:
        with open(file_path) as f:
            print("Opening file:", file_path)
        return True