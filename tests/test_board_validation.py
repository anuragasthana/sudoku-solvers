import random
from sudoku_solver.backtrack_solver import check_board_valid, check_cell_valid
from sudoku_solver.config import Hyperparams
from sudoku_solver.data import check_data
from sudoku_solver.board import board_to_string

def test_validate_board():
    print("Testing board validation")
    
    loaded_sudoku_data = check_data()
    loaded_sudoku_puzzles = loaded_sudoku_data['inputs']
    loaded_sudoku_solutions = loaded_sudoku_data['labels']
    
    for i in range(10):
        assert check_board_valid(loaded_sudoku_puzzles[i]) == True, "Puzzle " + str(i) + " is invalid: \n" + board_to_string(loaded_sudoku_puzzles[i], True)
        print("Test passed for puzzle ", i)
        
        assert check_board_valid(loaded_sudoku_solutions[i]) == True, "Solution " + str(i) + " is invalid: \n" + board_to_string(loaded_sudoku_puzzles[i], True)
        print("Test passed for solution ", i)

def test_validate_cell():
    print("Testing cell validation")
    
    loaded_sudoku_data = check_data()
    loaded_sudoku_puzzles = loaded_sudoku_data['inputs']
    
    p = loaded_sudoku_puzzles[0]
    
    for i in range(9):
        for j in range(9):
            if p[i][j] == 0 or p[i][j] == None:
                continue
            
            assert check_cell_valid(p, i, j, p[i][j]) == True, "Cell " + str(i) + ", " + str(j) + " is invalid: " + str(p[i][j])

def test_errors_detected():
    print("Testing board validation detects errors")
    
    loaded_sudoku_data = check_data(Hyperparams())
    loaded_sudoku_solutions = loaded_sudoku_data['labels']

    for i in range(100):
        p = loaded_sudoku_solutions[i].copy()
        
        # introduce an error
        x, y = random.randint(0, 8), random.randint(0, 8)
        p[x][y] = (p[x][y] + 1) % 9 + 1
        
        assert check_board_valid(p) == False, "Board " + str(i) + " evaded detection: \n" + board_to_string(p, True)
        print("Test passed for invalid board ", i)