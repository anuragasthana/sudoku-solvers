from sudoku_solver.train import train, test, get_model_performance
from sudoku_solver.config import Hyperparams, check_config
from sudoku_solver.data import get_dataloaders
import sys

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Invalid number of arguments provided")
        exit(1)

    args = check_config(sys.argv[1])
    
    params = Hyperparams(**args)
    data = get_dataloaders(params)
    
    model = train(data, params)
    
    test(data, model)