from sudoku_solver.backtrack_solver import check_correct
from sudoku_solver.data_generate import check_data


if __name__ == "__main__":
    loaded_sudoku_data = check_data()
    loaded_sudoku_puzzles = loaded_sudoku_data['inputs']
    loaded_sudoku_solutions = loaded_sudoku_data['labels']

    for i in range(100):
        puzzle = loaded_sudoku_puzzles[i]
        solution = loaded_sudoku_solutions[i]
        
        check_correct(sudoku_board=puzzle, solution=solution)