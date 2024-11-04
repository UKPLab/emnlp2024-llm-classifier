from pathlib import Path
import importlib

class TaskDataPreprocessor:
    def __init__(self, task_name:str, method:str) -> None:
        '''
        '''
        pck = importlib.import_module(f"tasks.{task_name}.{method}")
        print(f'import tasks.{task_name}.{method}')
        data_preprocessor = getattr(pck, 'DataPreprocessor')
        self.task_name = task_name
        self.method = method
        self.data_preprocessor = data_preprocessor()

