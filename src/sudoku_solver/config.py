from pydantic import BaseModel


class Hyperparams(BaseModel):
    
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 64
    
    early_stopping: bool = True
    patience: int = 5

    samples: int = 10000
    min_difficulty: int = 0
    max_difficulty: int = 1
    
    # etc. etc. etc.