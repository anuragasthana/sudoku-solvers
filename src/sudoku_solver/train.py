from .config import Hyperparams
from .data import SudokuDataloaders
from .model import SudokuCNN

from torch import nn, optim
from torch.utils.data import DataLoader as Dataloader
import torch

# GH Copilot autogen
class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, loss: float):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

def train(data: SudokuDataloaders, params: Hyperparams):
    print("Training with hyperparameters:")
    print(params)
    
    # Create model
    model = SudokuCNN()
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    
    criterion = nn.CrossEntropyLoss() #TODO: Warning, assuming CEL for one-hot categories but model outputs 81 scalar values (1->9)
    early_stopper = EarlyStopper(params.patience)

    # Iterate over epochs
    for epoch in range(params.epochs):
        print(f"Epoch {epoch+1}/{params.epochs}")
        
        # Iterate over batches
        for inputs, labels in data.train:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Print statistics
            print(f"Loss: {loss.item()}")
        
        # Get validation accuracy and loss
        # GH Copilot autogen
        percent_correct, ave_val_loss = get_model_performance(data.validation, model, criterion)
        print(f"Validation accuracy: {percent_correct}%")
        print(f"Validation loss: {ave_val_loss}")
        
        # Use early stopping
        early_stopper(ave_val_loss)
        if early_stopper.early_stop:
            print("Early stopping")
            break
        
        # Save model weights after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        
    return model

def test(data: SudokuDataloaders, model: nn.Module):
    
    percent_correct, ave_test_loss = get_model_performance(data.test, model, nn.CrossEntropyLoss())
    print(f"Test accuracy: {percent_correct}%")
    print(f"Test loss: {ave_test_loss}")

def get_model_performance(dataloader: Dataloader, model: nn.Module, criterion: nn.Module):
    # Get validation accuracy and loss
    # GH Copilot autogen
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Get loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    ave_val_loss = val_loss/len(dataloader)
    percent_correct = 100 * correct / total
    
    return percent_correct, ave_val_loss
    
