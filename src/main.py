from sudoku_solver.train import train, test, get_model_performance
from sudoku_solver.config import Hyperparams
from sudoku_solver.data import get_dataloaders

if __name__ == "__main__":
    
    params = Hyperparams()
    data = get_dataloaders()
    
    model = train(data, params)
    
    test(data, model)