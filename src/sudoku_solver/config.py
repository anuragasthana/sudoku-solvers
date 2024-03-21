from pydantic import BaseModel
from os.path import exists
import json


class Hyperparams(BaseModel):
    
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 64
    
    early_stopping: bool = True
    patience: int = 5

    samples: int = 10000
    # dataset: str = '3m'
    min_difficulty: float = 0
    max_difficulty: float = 1
    
    datasplit: list = [0.8, 0.1, 0.1]
    model: str = 'CNN'
    curriculum: bool = False
    num_mini_batches: int = 1000
    
    # etc. etc. etc.
    
    def to_name(self):
        return f"{self.model}_{self.lr}_{self.epochs}_{self.batch_size}_{self.samples}_{self.min_difficulty}_{self.max_difficulty}"

    def to_data_filename(self):
        return f"data_{self.samples}_{self.min_difficulty}_{self.max_difficulty}.npz"

def check_config(filename):
    if exists(filename) and filename[-5:] == '.json':
        file = open(filename)
        data = json.load(file)
        file.close()
        if type(data) is not dict:
            print("Invalid JSON Configuration")
            exit(1)
        return data
    else:
        print("Provided file is not a JSON object file or does not exist.")
        exit(1)