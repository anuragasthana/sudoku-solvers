from pydantic import BaseModel
from sudoku import Sudoku
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm as progress_bar

class SudokuDataset(Dataset):
    def __init__(self, data):
        self.inputs = data['inputs']
        self.labels = data['labels']
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        
        print(self.inputs[idx])
        print(self.labels[idx])
        print("----")
        
        return self.inputs[idx], self.labels[idx]

class SudokuDataloaders():
    def __init__(self, train: DataLoader, test: DataLoader, validation: DataLoader):
        self.train = train
        self.test = test
        self.validation = validation

def get_dataloaders(datafile='data.npz', batch_size = 32):
    
    # put in dataloader to send to main
    data = check_data(datafile)
    split = split_data(data)
    
    train = DataLoader(SudokuDataset(split['train']), batch_size=batch_size, shuffle=True)
    test = DataLoader(SudokuDataset(split['test']), batch_size=batch_size, shuffle=True)
    validation = DataLoader(SudokuDataset(split['validation']), batch_size=batch_size, shuffle=True)
    
    # Print sizes
    print(f"Train size: {len(train.dataset)}")
    print(f"Test size: {len(test.dataset)}")
    print(f"Validation size: {len(validation.dataset)}")
    
    return SudokuDataloaders(train=train, test=test, validation=validation)


def check_data(path):
    if not os.path.exists(path):
        generate()
    # put in dataloader to send to main
    return np.load(path, allow_pickle=True)
    

def generate(nan_value = 0.0):
    inputs = []
    labels = []
    for _, _ in progress_bar(enumerate(range(100000)), total=100000):
        puzzle = Sudoku(3).difficulty(np.random.uniform(.25, .75))
        
        # Make sure to save the puzzle with the nan_val instead of NaNs
        puzzle_board_with_zeroes = np.nan_to_num(puzzle.board, nan=nan_value)
        inputs.append(np.array(puzzle_board_with_zeroes))
        
        solution = puzzle.solve()
        labels.append(np.array(solution.board))
    np.savez('data.npz', inputs=inputs, labels=labels)
    print("Data saved")


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