from pathlib import Path
import importlib

class TaskEvaluater:
    def __init__(self, task_name:str, method:str) -> None:
        '''
        '''
        pck = importlib.import_module(f"tasks.{task_name}.{method}")
        evaluater = getattr(pck, 'Evaluater')
        self.task_name = task_name
        self.method = method
        self.evaluater = evaluater()
