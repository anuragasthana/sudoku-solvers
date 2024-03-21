# sudoku-solvers

This repository contains a collection of sudoku solvers written in different programming languages. The goal is to compare the performance of different programming languages and algorithms.

## Solvers

- CNN: Convolutional Neural Network
- RNN: Recurrent Neural Network
- Transformers
- Backtracking
- GNN: Graph Neural Network

## How to run

### Run with defaults
```bash
python src/main.py
```

### Run specific configuration
```bash
python src/main.py path_to_config_file
```

### Run manifest of configurations
```bash
python src/main.py -a path_to_manifest_file
```

## How to plot many results
```bash
python src/plot_many.py
```

`plot_many.py` will plot all the results that have been collected in the `artifacts/results` folder.

## Artifacts generated
- `artifacts/results`: Contains the results of the different solvers
- `artifacts/plots`: Contains the plots of the results. Each model run will generate its individual plots, and `plot_many.py` will generate a plot with all the results.
- `artifacts/models`: Contains the trained models
- `artifacts/comps`: Contains the comparison between the different solvers
- `artifacts/puzzles`: Contains the puzzles used for the experiments. Generated puzzles are cached here.

## GH Copilot generated code

### plot_many.py

**Case 1**

prompt:
```python
# Relevant plots:
# Validation accuracy vs. Epoch
```

output:
```python
for res in results:
    val_acc = [epoch.test_result.percent_cells_correct for epoch in res.epochs_output]
    plt.plot(val_acc, label=res.params.model)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
```

**Case 2**

prompt:
```python
# Plot training time
```

output:
```python
models = [res.params.model for res in results]
training_times = [res.training_time for res in results]
plt.bar(models, training_times)

plt.xlabel('Model')
plt.ylabel('Training Time (s)')
```

**Case 3**

prompt:
```python
# Plot average inference time
```

output:
```python
avg_inference_times = [res.avg_inference_time for res in results]
plt.bar(models, avg_inference_times)

plt.xlabel('Model')
plt.ylabel('Average Inference Time (s)')models = [res.params.model for res in results]
```

### model.py

prompt:

```python
# Create sample layers
```

output:
```python
layers = []
layers.append(nn.Conv2d(1, 16, 3, padding=1))
layers.append(nn.ReLU())
layers.append(nn.MaxPool2d(2))
layers.append(nn.Conv2d(16, 32, 3, padding=1))
layers.append(nn.ReLU())
layers.append(nn.MaxPool2d(2))
layers.append(nn.Conv2d(32, 64, 3, padding=1))
layers.append(nn.ReLU())
layers.append(nn.MaxPool2d(2))
layers.append(nn.Flatten())
layers.append(nn.Linear(64, 128))
layers.append(nn.ReLU())
layers.append(nn.Linear(128, 81*9))
self.model = nn.Sequential(*layers)
```

### train.py

**Case 1**

prompt:
```python
class EarlyStopper:
```

output:
```python

class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, loss: float):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
```

**Case 2**

prompt:
```python
for epoch in range(params.epochs):
```

output:
```python
print(f"Epoch {epoch+1}/{params.epochs}")
```

**Case 3**

prompt:
```python
def get_model_performance(dataloader: Dataloader, model: nn.Module, criterion: nn.Module, device: torch.device) -> TestResult:
    # Get validation accuracy and loss
```

output:
```python
with torch.no_grad():
        for inputs, labels in dataloader:
```