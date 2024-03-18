import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import ViTConfig, ViTModel
import torch
import torch.nn.functional as F


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
        print(x.shape)
        # Set x to its own channel
        x = x.unsqueeze(1)
        print(x.shape)
        x = self.model(x)
        print(x.shape)
        # Convert back to 81x9
        x = x.view(-1, 81, 9)
        print(x.shape)
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
    
class SudokuRNN(nn.Module):
    def __init__(self, model_type = "RNN", hidden_size=300, num_layers=10):
        super(SudokuRNN, self).__init__()

        self.input_size = 9
        self.hidden_size = hidden_size
        self.output_size = 81*9
        self.num_layers = num_layers
        self.model_type = model_type
        
        # Define encoder
        if (self.model_type == "LSTM"):
            self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif (self.model_type == "GRU"):
            self.encoder = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.encoder = nn.RNN(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Linear layer to reshape decoder output
        self.linear = nn.Linear(hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        _, encoder_hidden = self.encoder(input_seq.view(batch_size, -1, self.input_size))

        if (self.model_type == "LSTM"):
            encoder_hidden, cell_state = encoder_hidden
        
        encoder_hidden = encoder_hidden.view(self.num_layers, batch_size, -1)
        
        # Take the hidden state from the last layer
        last_layer_hidden = encoder_hidden[-1]
        
        # Apply linear layer to reshape decoder output
        outputs = self.linear(last_layer_hidden)
        outputs = outputs.view(-1, 81, 9)
        
        return outputs
