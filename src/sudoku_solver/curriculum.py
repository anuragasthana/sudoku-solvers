import torch
from pydantic import BaseModel
from sudoku import Sudoku
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm as progress_bar

from sudoku_solver.config import Hyperparams
from data import *


class Curriculum:
    def __init__(self, data, scoring_func, pacing_func):
        self.data = data
        self.scoring_func = scoring_func
        self.pacing_func = pacing_func
        
    def score(self, data):
        