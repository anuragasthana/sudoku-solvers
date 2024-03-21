import torch

from random import randrange
from sys import maxsize
from pydantic import BaseModel
from sudoku import Sudoku#, _SudokuSolver, UnsolvableSudoku
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm as progress_bar

from sudoku_solver.config import Hyperparams
from typing import List, Tuple, Optional

from typing import List, Optional

from torch_geometric.data import Data

class SudokuDifficulty:
    _empty_cell_value = None

    #From SudokuSolver clas in Sudoku import
    def __init__(self, width: int = 3, height: Optional[int] = None, board: Optional[List[List[Optional[int]]]] = None):
        self.width = width
        self.height = height if height else width
        self.size = self.width * self.height
        self.board = board

    #From SudokuSolver clas in Sudoku import
    def __get_blanks(self) -> List[List[int]]:
        blanks = []
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                if cell == Sudoku._empty_cell_value:
                    blanks.append((i, j))
        return blanks

    #From SudokuSolver class in Sudoku import
    def __calculate_blank_cell_fillers(self, blanks: List[List[int]]) -> List[List[List[bool]]]:
        valid_fillers = [[[True for _ in range(self.size)] for _ in range(self.size)] for _ in range(self.size)]
        for row, col in blanks:
            for i in range(self.size):
                same_row = self.board[row][i]
                same_col = self.board[i][col]
                if same_row and i != col:
                    valid_fillers[row][col][same_row - 1] = False
                if same_col and i != row:
                    valid_fillers[row][col][same_col - 1] = False
            grid_row, grid_col = row // self.height, col // self.width
            grid_row_start = grid_row * self.height
            grid_col_start = grid_col * self.width
            for y_offset in range(self.height):
                for x_offset in range(self.width):
                    if grid_row_start + y_offset == row and grid_col_start + x_offset == col:
                        continue
                    cell = self.board[grid_row_start + y_offset][grid_col_start + x_offset]
                    if cell:
                        valid_fillers[row][col][cell - 1] = False
        return valid_fillers

    #ChatGPT autogenerated method using Sudoku import to calculate the difficulty of puzzle
    #Prompt: From this code, can you write a function that takes in a Sudoku puzzle
    #and calculates difficulty without using the _SudokuSolver class: [insert sudoku.py code from Sudoku import]
    def calculate_difficulty(self) -> float:
        blanks = self.__get_blanks()
        blank_fillers = self.__calculate_blank_cell_fillers(blanks)
        blank_count = len(blanks)
        total_cells = self.size * self.size
        return blank_count / total_cells if total_cells > 0 else 0


# Returns label one hot encoded
class SudokuDataset(Dataset):
    def __init__(self, data):
        assert len(data['inputs']) == len(data['labels']), "Inputs and labels must be the same length"
        assert isinstance(data['inputs'], np.ndarray), "Inputs must be a numpy array"
        assert isinstance(data['labels'], np.ndarray), "Labels must be a numpy array"
        
        assert data['inputs'].shape[1:] == (9, 9), "Inputs must be 9x9"
        assert data['labels'].shape[1:] == (9, 9), "Labels must be 9x9"
        assert data['graphs'][0]['edge_index'].shape == (2, 1944), "Edge Index Shape must be 2x1944"

        # Replace None with 0 in inputs
        inputs = data['inputs'].copy()
        inputs[inputs == None] = 0
        inputs = inputs.astype(np.float32)
        
        labels = data['labels'].astype(np.int64)
        
        difficulties = data['difficulties']
        graphs = data['graphs']
        
        self.data = data
        self.inputs = inputs
        self.labels = labels
        self.difficulties = difficulties
        self.graphs = graphs
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        labels = self.labels[idx]-1
        labels = labels.reshape(-1)
        
        inputs = self.inputs[idx]
        difficulties = self.difficulties[idx]
        graphs = self.graphs[idx]
        return inputs, labels, difficulties, graphs

# https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
def one_hot_encode(a, n_cat = 9):
    return (np.arange(n_cat) == a[...,None]-1).astype(int)

def one_hot_decode(a):
    return (np.argmax(a, axis=-1) + 1)
    

class SudokuDataloaders():
    
    def __init__(self, params: Hyperparams, batch_size = 32):
        # put in dataloader to send to main
        
        #if params.dataset == '3m':
        kaggle_data = load_kaggle_data(params)
        generated_data = check_data(params=params)
        data = {
            'inputs': np.concatenate([kaggle_data['inputs'], generated_data['inputs']]),
            'labels': np.concatenate([kaggle_data['labels'], generated_data['labels']]),
            'difficulties': np.concatenate([kaggle_data['difficulties'], generated_data['difficulties']]),
            'graphs': np.concatenate([kaggle_data['graphs'], generated_data['graphs']]),
        }
        #elif params.dataset == 'generated':
        #    data = check_data(params=params)
        #else:
        #    raise ValueError(f"Invalid datasource {params.dataset}")       
        split = split_data(data, split=params.datasplit)
        
        train = DataLoader(SudokuDataset(split['train']), batch_size=batch_size, shuffle=True)
        test = DataLoader(SudokuDataset(split['test']), batch_size=batch_size, shuffle=True)
        validation = DataLoader(SudokuDataset(split['validation']), batch_size=batch_size, shuffle=True)
        
        # Print sizes
        print(f"Train size: {len(train.dataset)}")
        print(f"Test size: {len(test.dataset)}")
        print(f"Validation size: {len(validation.dataset)}")
        
        self.train = train
        self.test = test
        self.validation = validation


def check_data(params):
    path = f"artifacts/puzzles/{params.to_data_filename()}"
    if not os.path.exists(path):
        generate(params, path)
    # put in dataloader to send to main
    return np.load(path, allow_pickle=True)
    

def generate(params: Hyperparams, savepath: str):
    inputs = []
    labels = []
    difficulties = []
    graphs = []
    
    for _, _ in progress_bar(enumerate(range(params.samples)), total=params.samples, desc="Generating data"):
        puzzle = Sudoku(3).difficulty(np.random.uniform(params.min_difficulty, params.max_difficulty))
        inputs.append(np.array(puzzle.board))
        
        solution = puzzle.solve()
        labels.append(np.array(solution.board))

        #Calculate curriculum difficulty through py-sudoku (replicating 3m difficulty values is difficult)
        sudoku_difficulty = SudokuDifficulty(width=3, height=3, board=np.array(puzzle.board))
        difficulty = sudoku_difficulty.calculate_difficulty()
        difficulties.append(difficulty)

        graph = sudoku_to_graph(puzzle.board, solution.board)
        graphs.append(graph)

    # Convert graphs to a format suitable for saving
    graph_data = []
    for graph in graphs:
        graph_dict = {
            'x': graph.x.numpy(),
            'edge_index': graph.edge_index.numpy(),
            'y': graph.y
        }
        graph_data.append(graph_dict)

    np.savez(savepath, inputs=inputs, labels=labels, difficulties=difficulties, graphs=graph_data)
    print("Data saved")

# Train, test, validate split
def split_data(data, split=[0.8, 0.1, 0.1]):
    assert len(split) == 3, "Split must be a list of 3 values"
    assert sum(split) == 1, "Split values must sum to 1"
    
    # Split into train, test, and validation sets
    o = {}
    total = len(data['inputs'])
    train = int(total * split[0])
    test = int(total * split[1])
    
    o['train'] = {
        'inputs': data['inputs'][:train],
        'labels': data['labels'][:train],
        'difficulties': data['difficulties'][:train],
        'graphs': data['graphs'][:train]
    }
    
    o['test'] = {
        'inputs': data['inputs'][train:train+test],
        'labels': data['labels'][train:train+test],
        'difficulties': data['difficulties'][train:train+test],
        'graphs': data['graphs'][train:train+test]
    }
    
    o['validation'] = {
        'inputs': data['inputs'][train+test:],
        'labels': data['labels'][train+test:],
        'difficulties': data['difficulties'][train+test:],
        'graphs': data['graphs'][train+test:]
    }
    
    return o


def load_kaggle_data(params: Hyperparams):
    with open('artifacts/puzzles/sudoku-3m.csv') as f:
        puzzles = f.readlines()[1:]
        
        # Convert into labels and inputs
        
        inputs = []
        labels = []
        difficulties = []
        graphs = []
        
        for line in puzzles[:params.samples]:
            line = line.strip().split(',')
            
            # Convert to 9x9
            input = np.array([None if c == '.' else int(c) for c in line[1]]).reshape(9,9)
            label = np.array([int(c) for c in line[2]]).reshape(9,9)

            #Calculate curriculum difficulty through py-sudoku (replicating 3m difficulty values is difficult)
            sudoku_difficulty = SudokuDifficulty(width=3, height=3, board=input)
            difficulty = sudoku_difficulty.calculate_difficulty()
            graph = sudoku_to_graph(input, label)

            # Convert graph to a dictionary
            graph_dict = {
                'x': graph.x.numpy(),
                'edge_index': graph.edge_index.numpy(),
                'y': graph.y
            }
            inputs.append(input)
            labels.append(label)
            difficulties.append(difficulty)
            graphs.append(graph_dict)
        
        data = {'inputs': np.array(inputs), 'labels': np.array(labels), 'difficulties': np.array(difficulties), 'graphs': graphs}
        
        return data

    
# GPT generated    
def sudoku_to_graph(puzzle, solution):
# Assuming puzzle and solution are 9x9 numpy arrays or similar
    
    # Node features: One-hot encode the puzzle digits
    node_features = torch.zeros((81, 10), dtype=torch.float)
    for index, value in np.ndenumerate(puzzle):
        node_id = index[0] * 9 + index[1]
        node_features[node_id, value] = 1  # value is 0 for empty cells
        
    # Labels: Flatten the solution to a vector
    #print(solution)
    labels = solution#.flatten()
       
    # Define edges based on Sudoku rules (rows, columns, subgrids)
    edges = []
    for i in range(9):
        for j in range(9):
            # Row and column connections
            for k in range(9):
                if k != j:
                    edges.append((i*9+j, i*9+k))  # Row
                if k != i:
                    edges.append((i*9+j, k*9+j))  # Column
            
            # Subgrid connections
            subgrid_row, subgrid_col = 3 * (i // 3), 3 * (j // 3)
            for m in range(subgrid_row, subgrid_row + 3):
                for n in range(subgrid_col, subgrid_col + 3):
                    if m != i or n != j:
                        edges.append((i*9+j, m*9+n))
                            
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  
    # Create Data object
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    return data
