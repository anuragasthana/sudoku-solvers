import os
import numpy as np
import torch

from sudoku_solver.board import board_to_string
from sudoku_solver.config import Hyperparams
from sudoku_solver.data import SudokuDataloaders
from sudoku_solver.model import SudokuCNN
from sudoku_solver.backtrack_solver import check_board_solved
from sudoku_solver.train import get_model_performance

def load_puzzle_from_file(path):
    return torch.tensor([[int(c) for c in line.strip()] for line in open(path).readlines()])

def test_hardest_puzzle():
    
    for f in os.listdir('artifacts/puzzles'):
        if f.endswith('.txt'):
            puzzle = load_puzzle_from_file(f'artifacts/puzzles/{f}')
            puzzle = puzzle.float().unsqueeze(0)
            
            model = load_model()
            
            solved = False
            n = 10000
            
            for _ in range(n):
                output = model(puzzle)            
                output = output.argmax(2).view(9, 9).numpy() + 1
                if check_board_solved(output):
                    with open(f'artifacts/puzzles/{f}_solution.o', 'w') as f:
                        f.write(board_to_string(output))
                        
                    solved = True
                    break
            
            assert solved, f"Model failed to solve {f} in {n} iterations"

def load_model():
    model = SudokuCNN()
    model.load_state_dict(torch.load('artifacts/models/model_epoch_19.pth'))
    return model

def test_kaggle_3m():
    
    with open('artifacts/puzzles/sudoku-3m.csv') as f:
        puzzles = f.readlines()[1:]
        
        # Convert into labels and inputs
        
        inputs = []
        labels = []
        
        for line in puzzles[:1000]:
            line = line.strip().split(',')
            
            # Convert to 9x9
            input = np.array([0 if c == '.' else int(c) for c in line[1]]).reshape(9,9)
            label = np.array([int(c) for c in line[2]]).reshape(9,9)
            
            inputs.append(input)
            labels.append(label)
        
        data = {'inputs': np.array(inputs), 'labels': np.array(labels)}
        
        # Add into dataloader
        dataloader = SudokuDataloaders(Hyperparams(datasplit=[0.1, 0.8, 0.1]), data=data, batch_size=32)
        model = load_model()
        criterion = torch.nn.CrossEntropyLoss()
        
        solve_rate, cell_acc, loss = get_model_performance(dataloader.test, model, criterion)
        
        print(f"Solve rate: {solve_rate}")
        print(f"Cell accuracy: {cell_acc}")
        print(f"Loss: {loss}")