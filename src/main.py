import os
from sudoku_solver.plot import create_plots
from sudoku_solver.train import train, test, get_model_performance
from sudoku_solver.config import Hyperparams, check_config
import sys
from sudoku_solver.data import SudokuDataloaders, load_kaggle_data
import torch

def go(device, params):
    
    gen_data = SudokuDataloaders(params)
    model, results = train(gen_data, params, device)
    test(gen_data, model, device, results)
    
    create_plots(results)
    
def run_all_configs(config_path, device):
    
    if os.path.isfile(config_path):
        if config_path[-5:] != '.json':
            return
        
        kwargs = check_config(config_path)
        params = Hyperparams(**kwargs)
        
        go(device, params)
    else:
        files = os.listdir(config_path)
        for f in files:
            run_all_configs(os.path.join(config_path, f), device)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Running on GPU", torch.cuda.get_device_name(device))
    else:
        print("Running on CPU")
    
    if len(sys.argv) < 2:
        print("No configuration file provided, using default parameters")
        go(device, Hyperparams())
    else:
        run_all_configs(sys.argv[1], device)
