import json

from matplotlib import pyplot as plt
from sudoku_solver.plot import Results
import os


def main():
    
    # Load results from JSON
    results: list[Results] = []
    
    for root, dirs, files in os.walk('artifacts/results'):
        res = [os.path.join(root, file) for file in files if file[-5:] == '.json']
        
        for file in res:
            with open(file) as f:
                raw = json.load(f)
                
                # Convert JSON to Results object
                results.append(Results.model_validate(raw))
    
    # Print results
    for res in results:
        print(res)
    
    # Plot results
    
    # Relevant plots:
    # Validation accuracy vs. Epoch
    
    for res in results:
        val_acc = [epoch.test_result.percent_cells_correct for epoch in res.epochs_output]
        plt.plot(val_acc, label=res.params.model)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    
    # Save artifacts/plots/validation_accuracy.png
    plt.savefig('artifacts/plots/validation_accuracy.png')
    plt.clf()
        
    
    

if __name__ == '__main__':
    main()