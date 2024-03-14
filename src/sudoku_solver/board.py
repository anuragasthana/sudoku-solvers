
import torch


def board_to_string(puzzle, replace_none=False):
    
    assert puzzle.shape == (9, 9), "Puzzle must be a 9x9 array"
    
    if type(puzzle) == torch.Tensor:
        puzzle = puzzle.numpy()
    
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