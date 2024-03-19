import os
from typing import Optional
from pydantic import BaseModel

from sudoku_solver.config import Hyperparams
import matplotlib.pyplot as plt
import numpy as np

class TestResult(BaseModel):
    ave_loss: float
    percent_cells_correct: float
    percent_boards_solved: float

class EpochResults(BaseModel):
    test_result: TestResult
    training_loss: float

class Results(BaseModel):
    params: Hyperparams
    epochs_output: list[EpochResults]
    test_output: Optional[TestResult] = None

def save_plot(filename):
    plt.savefig(os.path.join("artifacts", "plots", filename))

def create_plots(results: Results):
    
    # GH Copilot Autogen -------------------------------------
    # Plot training loss
    training_loss = [epoch.training_loss for epoch in results.epochs_output]
    plt.plot(training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Per Epoch")
    save_plot("training_loss.png")
    
    # Plot validation loss
    val_loss = [epoch.test_result.ave_loss for epoch in results.epochs_output]
    plt.plot(val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Per Epoch")
    save_plot("validation_loss.png")
    
    # Plot validation accuracy
    val_acc = [epoch.test_result.percent_cells_correct for epoch in results.epochs_output]
    plt.plot(val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Per Epoch")
    save_plot("validation_accuracy.png")
    
    # Plot % boards solved
    boards_solved = [epoch.test_result.percent_boards_solved for epoch in results.epochs_output]
    plt.plot(boards_solved)
    plt.xlabel("Epoch")
    plt.ylabel("Percent Boards Solved")
    plt.title("Percent Boards Solved Per Epoch")
    save_plot("boards_solved.png")
    
    # Plot test loss
    test_loss = results.test_output.ave_loss
    plt.bar(["Test Loss"], [test_loss])
    plt.ylabel("Loss")
    plt.title("Test Loss")
    save_plot("test_loss.png")
    
    # Plot test accuracy
    test_acc = results.test_output.percent_cells_correct
    plt.bar(["Test Accuracy"], [test_acc])
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    save_plot("test_accuracy.png")
    
    # Plot % boards solved
    test_boards_solved = results.test_output.percent_boards_solved
    plt.bar(["Percent Boards Solved"], [test_boards_solved])
    plt.ylabel("Percent Boards Solved")
    plt.title("Percent Boards Solved")
    save_plot("test_boards_solved.png")
    
    return None
