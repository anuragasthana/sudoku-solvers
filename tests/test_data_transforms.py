
import numpy as np
import torch
from sudoku_solver.backtrack_solver import check_board_valid, check_cell_valid
from sudoku_solver.data import check_data
from sudoku_solver.board import board_to_string
from sudoku_solver.data import one_hot_encode, one_hot_decode
from sudoku_solver.validation import torch_givens_unchanged

def test_one_hot_encode():
    
    loaded_sudoku_data = check_data()
    loaded_sudoku_puzzles = loaded_sudoku_data['inputs']
    loaded_sudoku_solutions = loaded_sudoku_data['labels']
    
    for i in range(10):
        # NOTE: one_hot does not handle the 0 category! It expects values to start from 1.
        
        # assert (one_hot_decode(one_hot_encode(loaded_sudoku_puzzles[i])) == loaded_sudoku_puzzles[i]).all(), "Failed for puzzle " + str(i)
        # print("Test passed for puzzle ", i)
        
        assert (one_hot_decode(one_hot_encode(loaded_sudoku_solutions[i])) == loaded_sudoku_solutions[i]).all(), "Failed for solution " + str(i)
        print("Test passed for solution ", i)

def test_givens_unchanged():
    board = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(9)])
    
    board = torch.tensor(board)
    
    assert torch_givens_unchanged(board, board, 0), "Failed for unchanged board"
    
    board_copy = board.clone()
    board_copy[0][0] = 0
    assert torch_givens_unchanged(board, board_copy, 0), "Failed for board with one zero added"
    
    board_copy[0][0] = 5
    assert not torch_givens_unchanged(board, board_copy, 0), "Failed for board with one value changed to 5"