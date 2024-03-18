from sudoku_solver.validation import validate_board
from .config import Hyperparams
from .data import SudokuDataloaders
from .model import SudokuCNN, SudokuTransformer, SudokuRNN
from .backtrack_solver import check_board_solved

from torch import nn, optim
from torch.utils.data import DataLoader as Dataloader
import torch
from tqdm import tqdm as progress_bar


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

def train(data: SudokuDataloaders, params: Hyperparams, model: nn.Module = None):
    print("Training with hyperparameters:")
    print(params)
    
    if params.model == 'CNN':
        model = SudokuCNN()
    elif params.model == 'RNN':
        model = SudokuRNN()
    else:
        model = SudokuTransformer()
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(params.patience)

    # Iterate over epochs
    for epoch in range(params.epochs):
        
        # GH Copilot autogen
        print(f"Epoch {epoch+1}/{params.epochs}")
        
        # Iterate over batches
        cum_loss = 0
        for _, (inputs, labels) in progress_bar(enumerate(data.train), total=len(data.train)):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Combine x and y dims (1, 2) and swap classes and length (2, 1)
            loss = get_loss(criterion, labels, outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            cum_loss += loss.item()
        
        print(f"Average training loss: {cum_loss/len(data.train)}")
        
        # Get validation accuracy and loss
        # GH Copilot autogen
        percent_puzzles_solved, percent_correct, ave_val_loss = get_model_performance(data.validation, model, criterion)
        print(f"Validation accuracy: {percent_correct}%")
        print(f"Validation loss: {ave_val_loss}")
        print(f"Percent puzzles solved: {percent_puzzles_solved}%")
        
        print(f"---------------------------------------")
        
        # Use early stopping
        early_stopper(ave_val_loss)
        if early_stopper.early_stop:
            print("Early stopping")
            break
        
        # Save model weights after each epoch
        torch.save(model.state_dict(), f"artifacts/models/model_epoch_{epoch}.pth")
        
    return model

# Involves weird reshaping
def get_loss(criterion, labels, outputs):
    outputs = outputs.permute(0, 2, 1)
    loss = criterion(outputs, labels)
    return loss

def test(data: SudokuDataloaders, model: nn.Module):
    
    percent_puzzles_solved, percent_correct, ave_test_loss = get_model_performance(data.test, model, nn.CrossEntropyLoss())
    print(f"Test accuracy: {percent_correct}%")
    print(f"Test loss: {ave_test_loss}")
    print(f"Percent puzzles solved: {percent_puzzles_solved}%")

def get_model_performance(dataloader: Dataloader, model: nn.Module, criterion: nn.Module):
    # Get validation accuracy and loss
    # GH Copilot autogen
    cells_correct = 0
    puzzles_solved = 0
    total_puzzles = 0
    val_loss = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 2)
            
            total_puzzles += labels.size(0)
            cells_correct += (predicted == labels).sum().item()
            
            # Unpack board shape into 9x9
            predicted_unpacked = (predicted+1).view(-1, 9, 9)
            puzzles_solved += validate_board(predicted_unpacked, inputs.to(torch.int64))
            
            # Get loss
            loss = get_loss(criterion, labels, outputs)
            val_loss += loss.item()
            
    ave_val_loss = val_loss/len(dataloader)
    percent_correct = 100 * cells_correct / total_puzzles / 81
    percent_puzzles_solved = 100 * puzzles_solved / total_puzzles
    
    return percent_puzzles_solved, percent_correct, ave_val_loss
    
