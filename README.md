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