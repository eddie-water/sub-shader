from .file_operations import FileOperations

class AudioInput:
    def __init__(self, path) -> None:
        self.file_operations = FileOperations(path)
    
    def get(self) -> int:
        return self.file_operations.get_chunk()
