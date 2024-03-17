from pydantic import BaseModel
from os.path import exists
import json


class Hyperparams(BaseModel):
    
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 64
    
    early_stopping: bool = True
    patience: int = 5

    samples: int = 100000
    min_difficulty: int = 0
    max_difficulty: int = 1
    
    datasplit: list = [0.8, 0.1, 0.1]
    model: str = 'CNN'
    
    # etc. etc. etc.

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
