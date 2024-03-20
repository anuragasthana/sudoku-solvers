import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import ViTConfig, ViTModel
from torch_geometric.nn import GCNConv
import torch.nn as nn

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
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 81*9))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        
        # Set x to its own channel
        x = x.unsqueeze(1)
        
        x = self.model(x)
        
        # Convert back to 81x9
        x = x.view(-1, 81, 9)
        
        return x
    
class SudokuTransformer(nn.Module):
    def __init__(self):
        super(SudokuTransformer, self).__init__()
        config = ViTConfig(hidden_size=768, 
                           num_hidden_layers=12, 
                           num_attention_heads=12, 
                           num_channels=1,
                           patch_size=1, 
                           image_size=9)
        self.encoder = ViTModel(config)
        self.decoder = nn.Sequential(
            nn.Conv1d(82, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 81*9)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)[0]
        x = self.decoder(x)
        x = x.view(-1, 81, 9)
        return x
    
class SudokuGNN(nn.Module):
    def __init__(self, num_node_features):
        super(SudokuGNN, self).__init__()
        layers = []
        layers.append(GCNConv(num_node_features, 16))  # Intermediate layer
        layers.append(GCNConv(16, 32))
        layers.append(GCNConv(32, 64))
        layers.append(GCNConv(64, 128))
        layers.append(nn.ReLU())
        layers.append(GCNConv(128, 64))
        layers.append(GCNConv(64, 32))
        layers.append(GCNConv(32, 16))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 81*9))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.model(x)
        x = x.view(-1, 81, 9)
        
        return x
