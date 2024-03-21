from sudoku_solver.plot import create_plots
from sudoku_solver.train import train, test, get_model_performance
from sudoku_solver.config import Hyperparams, check_config
from sudoku_solver.data import SudokuDataloaders, load_kaggle_data
from sudoku_solver.backtrack_solver import solve_sudoku
import numpy as np
import os
import sys
import torch
import time
import pandas as pd
import json
from tqdm import tqdm as progress_bar


def go(device, params, comp = False):
    
    gen_data = SudokuDataloaders(params)
    beg = time.time()
    model, results = train(gen_data, params, device)
    end = time.time()
    training_time = end-beg
    print(f"Training time for {params.model}: {training_time} seconds")
    beg = time.time()
    test(gen_data, model, device, results)
    end = time.time()
    avg_inference_time = (end-beg)/len(gen_data.test.dataset)
    print(f"Average Inference time for {params.model}: {avg_inference_time} seconds")
    
    create_plots(results)

    if comp:
        return training_time, results.test_output.percent_boards_solved, avg_inference_time
    
def run_manifest(device):
    path = None
    for root, dirs, files in os.walk('.'):
        if 'manifest.json' in files:
            path = os.path.abspath(os.path.join(root, 'manifest.json'))
            break
    if path is None:
        print("manifest.json not found")
        exit(1)

    file = open(path)
    config_files = json.load(file)
    file.close()
    if type(config_files) is not list:
        print("Invalid JSON Configuration")
        exit(1)

    comps = pd.DataFrame(columns=["Config", 
                          "Model", 
                          "Training Time (s)", 
                          "Inference % Solved", 
                          "Average Inference Time (s)"])
    
    # Running Backtracking

    print("\n\nBacktracking\n\n")
    
    diffs = {"min_difficulty": 0.21, "max_difficulty": 0.99}
    backtrack_params = Hyperparams(**diffs)
    dataloader = SudokuDataloaders(backtrack_params)

    inference = dataloader.test
    total_time = 0

    for _, (inputs, _) in progress_bar(enumerate(inference), total=len(inference), desc="Running Backtracking"):
        for i in range(inputs.shape[0]):
            board = np.array(inputs[i])
            beg = time.time()
            solve_sudoku(board)
            end = time.time()
            total_time += (end-beg)

    avg_inference_time = total_time/len(inference.dataset)

    comps.loc[len(comps.index)] = ["N/A", "Backtracking", 0, 100, avg_inference_time]

    # Running rest of the manifest

    configs = []

    for config_file in config_files:
        kwargs = check_config("src/sudoku_solver/configs"+config_file)
        params = Hyperparams(**kwargs)

        configs.append(params.model_dump())

        print(f"\n\n{params.model}: {config_file}\n\n")

        training_time, inference_percent_solved, avg_inference_time = go(device, params, True)

        comps.loc[len(comps.index)] = [config_file, 
                                       params.model, 
                                       training_time, 
                                       inference_percent_solved, 
                                       avg_inference_time]

    pd.DataFrame(configs).to_csv("artifacts/comps/configs.csv")
    comps.to_csv("artifacts/comps/model_comps.csv")
    
    
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Running on GPU", torch.cuda.get_device_name(device))
    else:
        print("Running on CPU")
    
    if len(sys.argv) < 2:
        print("No configuration file provided, using default parameters")
        go(device, Hyperparams())
    elif sys.argv[1] == '-a' or sys.argv[1] == '--all':
        run_manifest(device)
    else:
        kwargs = check_config(sys.argv[1])
        params = Hyperparams(**kwargs)
        go(device, params)
