from torch.utils.data import Dataset, DataLoader
from torch import nn

class SudokuCNN(nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        
        # Create sample layers
        # Ref: Github Copilot (autogen)
        
        # TODO: Note that this ref model outputs 81 values! It is NOT one hot softmax!
        
        layers = []
        layers.append(nn.Conv2d(1, 16, 3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Conv2d(16, 32, 3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Conv2d(32, 64, 3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(64*3*3, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 81))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.model(x)