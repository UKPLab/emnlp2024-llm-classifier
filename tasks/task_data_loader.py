from pathlib import Path
from datasets import load_dataset

class TaskDataLoader:
    def __init__(self, task_name:str, train_type:str, val_type:str, test_type:str=None) -> None:
        '''
        params:task_name: str: name of the task
        params:train_type: str: name of the training data 
        params:val_type: str: name of the validation data 
        params:test_type: str: name of the test data 
        '''
        self.task_name = task_name
        self.train_type = train_type
        self.test_type = test_type
        self.val_type = val_type
        self.task_data_dir = Path('data/Re3-Sci/tasks') / task_name
        data_files = {}
        for i in self.task_data_dir.iterdir():
            if i.is_file() and i.name.endswith('.csv'):
                data_files[i.stem] = str(i)
        self.data_files = data_files
        self.dataset = load_dataset("csv", data_files = data_files, keep_default_na=False)
        label_names = sorted(set(label for label in self.dataset["train"]["label"]))
        self.labels = label_names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            label2id[label] = i
            id2label[i] = label
        self.label2id = label2id
        self.id2label = id2label
        

    def load_train(self):
        return self.dataset[self.train_type]
    def load_val(self):
        return self.dataset[self.val_type]
    def load_test(self):
        if self.test_type is None:
            return None
        return self.dataset[self.test_type]
    def load_data(self):
        return self.load_train(), self.load_val(), self.load_test()
    def get_labels(self):
        return self.labels, self.label2id, self.id2label
        

