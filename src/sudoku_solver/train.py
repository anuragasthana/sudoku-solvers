import os
from sudoku_solver.plot import EpochResults, Results, TestResult
from sudoku_solver.validation import validate_board
from .config import Hyperparams
from .data import SudokuDataloaders, sudoku_to_graph
from .model import SudokuCNN, SudokuTransformer, SudokuRNN, SudokuGNN
from .backtrack_solver import check_board_solved
from .curriculum import Curriculum

from torch import nn, optim
from torch.utils.data import DataLoader as Dataloader
import numpy as np
import torch
from tqdm import tqdm as progress_bar
from torch_geometric.data import Data, Batch


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

def train_with_curriculum(data: SudokuDataloaders, params: Hyperparams, device, model: nn.Module = None):
    print("Training with hyperparameters:")
    print(params)
    
    if params.model == 'CNN':
        model = SudokuCNN()
    elif params.model == 'RNN':
        model = SudokuRNN()
    elif params.model == 'RNNLSTM':
        model = SudokuRNN(model_type="LSTM")
    elif params.model == "RNNGRU":
        model = SudokuRNN(model_type="GRU")
    elif params.model == "GNN":
        model = SudokuGNN()
    else:
        model = SudokuTransformer()
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(params.patience)
    
    results: Results = Results(params=params, epochs_output=[], test_output=None)
    
    # Get curriculum learning batches
    curriculum = Curriculum(data.train) 
    curriculum_batches = curriculum.curriculum_learning_batches(params.num_mini_batches)
    
    # Iterate over epochs
    for epoch in range(params.epochs):
        print(f"Epoch {epoch+1}/{params.epochs}")
        
        # Iterate over curriculum learning batches
        cum_loss = 0
        for minibatch in curriculum_batches:

            # Replace None with 0 in inputs
            inputs = minibatch['inputs'].copy()

            # Preprocess inputs to ensure consistency
            inputs = [[[0 if item is None else item for item in row] for row in grid] for grid in inputs]

            # Convert processed inputs to NumPy array
            inputs = torch.tensor(np.array(inputs, dtype=np.float32))
        
            labels = torch.tensor(np.array(minibatch['labels'], dtype=np.int64))
        
            difficulties = torch.tensor(np.array(minibatch['difficulties'], dtype=np.float32))

            #graphs = minibatch['graphs']
            graph_list = []

            # Iterate over each graph dictionary
            for graph_data in minibatch['graphs']:
                x = torch.tensor(graph_data['x'], dtype=torch.float32)
                edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
                y = torch.tensor(graph_data['y'], dtype=torch.long)
                
                # Combine the tensors into a single data object
                data = Data(x=x, edge_index=edge_index, y=y)
                
                # Add the data object to a list
                graph_list.append(data)

            # Batch the list of data objects
            graphs = Batch.from_data_list(graph_list).to(device)
            
            #graphs = torch.tensor(np.array(minibatch['graphs']))
    
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            inputs = inputs.to(device)
            labels = labels.to(device)
            difficulties = difficulties.to(device)
            graphs = graphs.to(device)
            
            if (params.model == "GNN"):
                outputs = model(graphs)
            else:
                outputs = model(inputs)
            
            # Calculate loss
            loss = get_loss(criterion, labels, outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            cum_loss += loss.item()
        
        # Calculate average training loss
        training_loss = cum_loss / len(curriculum_batches)
        print(f"Average training loss: {training_loss}")
        
        # Get validation accuracy and loss
        val_output = get_model_performance(data.validation, params, model, criterion, device)
        print(f"Validation cell accuracy: {val_output.percent_cells_correct}%")
        print(f"Validation loss: {val_output.ave_loss}")
        print(f"Validation % boards solved: {val_output.percent_boards_solved}%")
        print(f"---------------------------------------")
        
        results.epochs_output.append(EpochResults(test_result=val_output, training_loss=training_loss))
        
        # Use early stopping
        early_stopper(val_output.ave_loss)
        if early_stopper.early_stop:
            print("Early stopping")
            break
        
        # Save model weights after each epoch
        model_subdir = f"artifacts/models/{params.to_name()}"
        if not os.path.exists(model_subdir):
            os.makedirs(model_subdir)
        torch.save(model.state_dict(), f"{model_subdir}/model_epoch_{epoch}.pth")
        
    return model, results

def train(data: SudokuDataloaders, params: Hyperparams, device, model: nn.Module = None):
    print("Training with hyperparameters:")
    print(params)
    
    if params.model == 'CNN':
        model = SudokuCNN()
    elif params.model == 'RNN':
        model = SudokuRNN()
    elif params.model == 'RNNLSTM':
        model = SudokuRNN(model_type="LSTM")
    elif params.model == "RNNGRU":
        model = SudokuRNN(model_type="GRU")
    elif params.model == "GNN":
        model = SudokuGNN()
    else:
        model = SudokuTransformer()
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(params.patience)
    
    results: Results = Results(params=params, epochs_output=[], test_output=None)
    # Iterate over epochs
    for epoch in range(params.epochs):
        
        # GH Copilot autogen
        print(f"Epoch {epoch+1}/{params.epochs}")
        
        # Iterate over batches
        cum_loss = 0
        for _, (inputs, labels, difficulties, graphs) in progress_bar(enumerate(data.train), total=len(data.train)):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            inputs = inputs.to(device)
            labels = labels.to(device)
            difficulties = difficulties.to(device)

            #TODO: Move Graphs to Device - Some Starter Code Below - Doesn't fully work yet
            print("TODO: Move Graphs to Device - Some Starter Code Below - Doesn't fully work yet")
            raise NotImplementedError
            # graph_list = []

            # Iterate over each graph dictionary
            # for graph_data in graphs:
            #     print(graph_data)
            #     exit()
            #     x = torch.tensor(graph_data['x'], dtype=torch.float32)
            #     edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
            #     y = torch.tensor(graph_data['y'], dtype=torch.long)
                
            #     # Combine the tensors into a single data object
            #     data = Data(x=x, edge_index=edge_index, y=y)
                
            #     # Add the data object to a list
            #     graph_list.append(data)

            # Batch the list of data objects
            #graphs = Batch.from_data_list(graph_list).to(device)

            if (params.model == "GNN"):
                outputs = model(graphs)
            else:
                outputs = model(inputs)
            
            # Combine x and y dims (1, 2) and swap classes and length (2, 1)
            loss = get_loss(criterion, labels, outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            cum_loss += loss.item()
        
        training_loss = cum_loss/len(data.train)
        print(f"Average training loss: {training_loss}")
        
        # Get validation accuracy and loss
        # GH Copilot autogen
        val_output = get_model_performance(data.validation, params, model, criterion, device)
        print(f"Validation cell accuracy: {val_output.percent_cells_correct}%")
        print(f"Validation loss: {val_output.ave_loss}")
        print(f"Validation % boards solved: {val_output.percent_boards_solved}%")
        
        print(f"---------------------------------------")
        
        results.epochs_output.append(EpochResults(test_result=val_output, training_loss=training_loss))
        
        # Use early stopping
        early_stopper(val_output.ave_loss)
        if early_stopper.early_stop:
            print("Early stopping")
            break
        
        # Save model weights after each epoch
        
        model_subdir = f"artifacts/models/{params.to_name()}"
        
        # Check if this directory exists
        if not os.path.exists(model_subdir):
            os.makedirs(model_subdir)
        
        torch.save(model.state_dict(), f"{model_subdir}/model_epoch_{epoch}.pth")
        
    return model, results

# Involves weird reshaping
def get_loss(criterion, labels, outputs):
    outputs = outputs.permute(0, 2, 1)
    loss = criterion(outputs, labels)
    return loss

def test(data: SudokuDataloaders, params: Hyperparams, model: nn.Module, device: torch.device, results: Results):
    
    test_output = get_model_performance(data.test, params, model, nn.CrossEntropyLoss(), device)
    print(f"Test cell accuracy: {test_output.percent_cells_correct}%")
    print(f"Test loss: {test_output.ave_loss}")
    print(f"Test % boards solved: {test_output.percent_boards_solved}%")
    
    if results is not None:
        results.test_output = test_output

def get_model_performance(dataloader: Dataloader, params: Hyperparams, model: nn.Module, criterion: nn.Module, device: torch.device) -> TestResult:
    # Get validation accuracy and loss
    # GH Copilot autogen
    cells_correct = 0
    puzzles_solved = 0
    total_puzzles = 0
    val_loss = 0
    
    with torch.no_grad():
        for inputs, labels, difficulties, graphs in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if (params.model == "GNN"):
                outputs = model(graphs)
            else:
                outputs = model(inputs)
            predicted = torch.argmax(outputs.data, 2)
            
            total_puzzles += labels.size(0)
            cells_correct += (predicted == labels).sum().item()
            
            # Unpack board shape into 9x9
            predicted_unpacked = (predicted+1).view(-1, 9, 9)
            puzzles_solved += validate_board(predicted_unpacked, inputs.to(torch.int64))
            
            # Get loss
            loss = get_loss(criterion, labels, outputs)
            val_loss += loss.item()
            
    ave_val_loss = val_loss/len(dataloader)
    percent_cells_correct = 100 * cells_correct / total_puzzles / 81
    percent_puzzles_solved = 100 * puzzles_solved / total_puzzles

    return TestResult(ave_loss=ave_val_loss, percent_cells_correct=percent_cells_correct, percent_boards_solved=percent_puzzles_solved)
    # return percent_puzzles_solved, percent_cells_correct, ave_val_loss
    
