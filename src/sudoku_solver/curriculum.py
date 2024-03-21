import torch
from pydantic import BaseModel
from sudoku import Sudoku
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import math
import random
from tqdm import tqdm as progress_bar

from sudoku_solver.config import Hyperparams
from .data import *



class Curriculum:
    def __init__(self, training_data: DataLoader):
        self.training_data = training_data
        
    # def pacing(self, i, step=20, step_length=1944, increase=1.5, starting_percent=0.1):
    #     exponent = math.floor(i / step_length)
    #     value = min(starting_percent * (increase ** exponent), 1) * step
    #     return value
    
    def curriculum_learning_batches(self, num_mini_batches):
        # Assuming you have train_loader.train.dataset.data
        data = self.training_data.dataset.data
        # First, zip the data together so you can sort it based on 'difficulties'
        zipped_data = zip(data['inputs'], data['labels'], data['difficulties'], data['graphs'])
        # Sort the zipped data based on 'difficulties' (index 2 in the zipped tuple)
        sorted_data = sorted(zipped_data, key=lambda x: x[2])
        # Unzip the sorted data
        sorted_inputs, sorted_labels, sorted_difficulties, sorted_graphs = zip(*sorted_data)
        # Convert the sorted data back into a dictionary
        sorted_dataset = {
            'inputs': np.array(sorted_inputs),
            'labels': np.array(sorted_labels),
            'difficulties': np.array(sorted_difficulties),
            'graphs': np.array(sorted_graphs)
        }

        result = []
        for i in range(1, num_mini_batches+1):
            size = num_mini_batches*200  # Convert size to integer
            first_size_entrysets = {
                'inputs': sorted_dataset['inputs'][:size],
                'labels': sorted_dataset['labels'][:size],
                'difficulties': sorted_dataset['difficulties'][:size],
                'graphs': sorted_dataset['graphs'][:size]
            }
            mini_batch = self.sampler(first_size_entrysets)
            result.append(mini_batch)
        #Returns a sequence of minibatches for training procedure
        return result

    def sampler(self, dataset_dict):
        # Get the total number of entrysets in the dictionary
        total_entrysets = len(dataset_dict['inputs'])
        print(total_entrysets)
        # Generate a random sample of N indices without replacement
        sample_indices = random.sample(range(total_entrysets), 200)
        
        # # Create a new dictionary containing the sampled entrysets
        sampled_data = {
            'inputs': [dataset_dict['inputs'][i] for i in sample_indices],
            'labels': [dataset_dict['labels'][i] for i in sample_indices],
            'difficulties': [dataset_dict['difficulties'][i] for i in sample_indices],
            'graphs': [dataset_dict['graphs'][i] for i in sample_indices]
        }

        return sampled_data

        