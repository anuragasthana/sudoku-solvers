from pydantic import BaseModel
from sudoku import Sudoku
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm as progress_bar

from sudoku_solver.config import Hyperparams

# Returns label one hot encoded
class SudokuDataset(Dataset):
    def __init__(self, data):
        assert len(data['inputs']) == len(data['labels']), "Inputs and labels must be the same length"
        assert isinstance(data['inputs'], np.ndarray), "Inputs must be a numpy array"
        assert isinstance(data['labels'], np.ndarray), "Labels must be a numpy array"
        
        assert data['inputs'].shape[1:] == (9, 9), "Inputs must be 9x9"
        assert data['labels'].shape[1:] == (9, 9), "Labels must be 9x9"
        
        # Replace None with 0 in inputs
        inputs = data['inputs'].copy()
        inputs[inputs == None] = 0
        inputs = inputs.astype(np.float32)
        
        labels = data['labels'].astype(np.int64)
        # labels = one_hot_encode(labels)
        
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        labels = self.labels[idx]-1
        labels = labels.reshape(-1)
        
        return self.inputs[idx], labels

# https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
def one_hot_encode(a, n_cat = 9):
    return (np.arange(n_cat) == a[...,None]-1).astype(int)

def one_hot_decode(a):
    return (np.argmax(a, axis=-1) + 1)
    

class SudokuDataloaders():
    
    def __init__(self, params: Hyperparams, batch_size = 32):
        # put in dataloader to send to main
        
        if params.datasource == '3m':
            data = load_kaggle_data(params)
        elif params.datasource == 'generated':
            data = check_data(params=params)
        else:
            raise ValueError(f"Invalid datasource {params.datasource}")
        
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
    
    for _, _ in progress_bar(enumerate(range(params.samples)), total=params.samples, desc="Generating data"):
        puzzle = Sudoku(3).difficulty(np.random.uniform(params.min_difficulty, params.max_difficulty))
        inputs.append(np.array(puzzle.board))
        
        solution = puzzle.solve()
        labels.append(np.array(solution.board))
    np.savez(savepath, inputs=inputs, labels=labels)
    print("Data saved")

# Train, test, validate split
def split_data(data, split = [0.8, 0.1, 0.1]):
    
    assert len(split) == 3, "Split must be a list of 3 values"
    assert sum(split) == 1, "Split values must sum to 1"
    
    # Split into train, test, and validation sets
    o = {}
    
    total = len(data['inputs'])
    train = int(total * split[0])
    test = int(total * split[1])
    
    o['train'] = {'inputs': data['inputs'][:train], 'labels': data['labels'][:train]}
    o['test'] = {'inputs': data['inputs'][train:train+test], 'labels': data['labels'][train:train+test]}
    o['validation'] = {'inputs': data['inputs'][train+test:], 'labels': data['labels'][train+test:]}
    
    return o

def load_kaggle_data(params: Hyperparams):
    with open('artifacts/puzzles/sudoku-3m.csv') as f:
        puzzles = f.readlines()[1:]
        
        # Convert into labels and inputs
        
        inputs = []
        labels = []
        
        for line in puzzles[:params.samples]:
            line = line.strip().split(',')
            
            # Convert to 9x9
            input = np.array([0 if c == '.' else int(c) for c in line[1]]).reshape(9,9)
            label = np.array([int(c) for c in line[2]]).reshape(9,9)
            
            inputs.append(input)
            labels.append(label)
        
        data = {'inputs': np.array(inputs), 'labels': np.array(labels)}
        
        return data
