from pydantic import BaseModel

from sudoku_solver.config import Hyperparams

class TestOutput(BaseModel):
    ave_loss: float
    percent_cells_correct: float
    percent_boards_solved: float

class Results(BaseModel):
    params: Hyperparams
    epochs_output: list[TestOutput]
    test_output: TestOutput