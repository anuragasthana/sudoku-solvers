from .data import check_data
import numpy as np

def check_cell_valid(sudoku_board, row, column, num):
    # Check row and column
    # Skips over the cell we are checking
    for i in range(9):
        if (sudoku_board[row][i] == num and i != column):
            return False

        if (sudoku_board[i][column] == num and i != row):
            return False
    
    #Check 3 by 3 box
    startRow = row - row%3
    startColumn = column - column%3
    for i in range(3):
        for j in range(3):
            r, c = i + startRow, j + startColumn
            
            # Skip over the cell we are checking
            if r == row and c == column:
                continue
            
            if (sudoku_board[r][c] == num):
                return False
            
    return True

def check_board_valid(sudoku_board):
    for i in range(9):
        for j in range(9):
            if (sudoku_board[i][j] == 0 or sudoku_board[i][j] == None):
                continue

            if (check_cell_valid(sudoku_board, i, j, sudoku_board[i][j]) == False):
                return False
    return True

def solve_sudoku(sudoku_board):

    emptySquare = False

    for i in range(9):
        for j in range(9):
            #Assuming 0 is the "empty" value
            if ((sudoku_board[i][j] == 0) or (sudoku_board[i][j] == None)):
                emptySquare = True
                rowEmpty = i
                columnEmpty = j
                break
        if (emptySquare):
            break

    if (emptySquare == False):
        return True
    
    for num in range(1,10):
        #Place numbers on the board
        if (check_cell_valid(sudoku_board, rowEmpty, columnEmpty, num)):
            sudoku_board[rowEmpty][columnEmpty] = num

            #Apply recursion until sudoku is solved when emptySquare == False indicating no more empty squares left
            if (solve_sudoku(sudoku_board)):
                return True
            
            #Backtracking step in case we don't reach a solution this way
            sudoku_board[rowEmpty][columnEmpty] = None

def check_correct(sudoku_board, solution):
    if (solve_sudoku(sudoku_board=sudoku_board)):
        if (np.array_equal(sudoku_board, solution)):
            print("Sudoku Solved Correctly")
        else:
            print("Sudoku Solved Incorrectly")

            print("Sudoku Board: ")
            print(sudoku_board)
            
            print("Solution: ")
            print(solution)
            
            our_solution_valid = check_board_valid(sudoku_board)
            if (our_solution_valid):
                print("Our solution is valid")
            else:
                print("Our solution is invalid")
            
    else:
        print("No solution exists - Should be impossible by assumption")

#Example of a sudoku board that can be used for testing
# sudoku_board = [
#       [5, 3, 0, 0, 7, 0, 0, 0, 0],
#       [6, 0, 0, 1, 9, 5, 0, 0, 0],
#       [0, 9, 8, 0, 0, 0, 0, 6, 0],
#       [8, 0, 0, 0, 6, 0, 0, 0, 0],
#       [0, 0, 0, 8, 0, 3, 0, 0, 1],
#       [7, 0, 0, 0, 2, 0, 0, 0, 6],
#       [0, 6, 0, 0, 0, 0, 2, 8, 0],
#       [0, 0, 0, 4, 1, 9, 0, 0, 5],
#       [0, 0, 0, 0, 8, 0, 0, 7, 9]
# ]
# Solution:
# [[5, 3, 4, 6, 7, 8, 9, 1, 2],
#  [6, 7, 2, 1, 9, 5, 3, 4, 8],
#  [1, 9, 8, 3, 4, 2, 5, 6, 7],
#  [8, 5, 9, 7, 6, 1, 4, 2, 3],
#  [4, 2, 6, 8, 5, 3, 7, 9, 1],
#  [7, 1, 3, 9, 2, 4, 8, 5, 6],
#  [9, 6, 1, 5, 3, 7, 2, 8, 4],
#  [2, 8, 7, 4, 1, 9, 6, 3, 5], 
#  [3, 4, 5, 2, 8, 6, 1, 7, 9]]    

