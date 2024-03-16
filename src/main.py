from sudoku_solver.train import train, test, get_model_performance
from sudoku_solver.config import Hyperparams, check_config
import sys
from sudoku_solver.data import SudokuDataloaders, load_kaggle_data

if __name__ == "__main__":

    params = Hyperparams()
    if len(sys.argv) < 2:
        print("No configuration file provided, using default parameters")
    else:
        kwargs = check_config(sys.argv[1])
        params = Hyperparams(**kwargs)
    
    gen_data = SudokuDataloaders(params)
    model = train(gen_data, params)
    test(gen_data, model)
    
    # k_data = SudokuDataloaders(params, data=load_kaggle_data(params))
    # model = train(k_data, params, model=model) 
    # test(k_data, model)