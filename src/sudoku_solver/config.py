from pydantic import BaseModel


class Hyperparams(BaseModel):
    
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 64
    
    early_stopping: bool = True
    patience: int = 5
    
    # etc. etc. etc.