from sudoku import Sudoku
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm as progress_bar

def check_data():
    if not os.path.exists('data.npz'):
        generate()
    # put in dataloader to send to main
    return np.load('data.npz', allow_pickle=True)
    

def generate():
    inputs = []
    labels = []
    for _, _ in progress_bar(enumerate(range(100000)), total=100000):
        puzzle = Sudoku(3).difficulty(np.random.uniform(.25, .75))
        inputs.append(np.array(puzzle.board))
        solution = puzzle.solve()
        labels.append(np.array(solution.board))
    np.savez('data.npz', inputs=inputs, labels=labels)
    print("Data saved")