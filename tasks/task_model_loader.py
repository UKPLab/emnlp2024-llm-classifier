from pathlib import Path
import importlib

class TaskModelLoader:
    def __init__(self, task_name:str, method:str) -> None:
        '''
        '''
        pck = importlib.import_module(f"tasks.{task_name}.{method}")
        model_loader = getattr(pck, 'ModelLoader')
        self.task_name = task_name
        self.method = method
        self.model_loader = model_loader()
    
