from .backtrack_solver import check_board_solved
from .board import board_to_string
import torch

# TODO: Do analysis of bad predictions
def validate_board(predicted, inputs):
    puzzles_solved = 0
    
    for i in range(predicted.shape[0]):
        
        digits_unchanged = torch_givens_unchanged(predicted, inputs, i)
        
        if check_board_solved(predicted[i].numpy()) and digits_unchanged:
            puzzles_solved += 1
    
    return puzzles_solved

def torch_givens_unchanged(predicted, inputs, i):
    return torch.all((predicted[i] == inputs[i]) | (inputs[i] == 0)).item()