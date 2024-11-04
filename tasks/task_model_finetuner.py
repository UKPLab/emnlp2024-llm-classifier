from pathlib import Path
import importlib

class TaskModelFinetuner:
    def __init__(self, task_name:str, method:str) -> None:
        '''
        '''
        pck = importlib.import_module(f"tasks.{task_name}.{method}")
        model_finetuner = getattr(pck, 'ModelFinetuner')
        self.task_name = task_name
        self.method = method
        self.model_finetuner = model_finetuner()
    
