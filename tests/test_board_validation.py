from sudoku_solver.backtrack_solver import check_board_valid, check_cell_valid
from sudoku_solver.data import check_data

def board_to_string(puzzle, replace_none=False):
    o=""
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == None:
                if replace_none:
                    o += "0 "
                else:
                    o += "None "
            else:
                o += str(puzzle[i][j]) + " "
        
        o += "\n"
    
    return o

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