# Task 02: Create Baseline `solution.py` and `test_solution.py`

## Problem Setup

- **Dataset**: CSV with columns `x`, `y` (numeric features) and `target` (numeric target)
- **Train**: `/Users/tony/datasets/three_peaks/train_dataset.csv`
- **Test**: `/Users/tony/datasets/three_peaks/test_dataset.csv`
- **Objective**: Minimize MSE on test set; keep model size small
- **Model API**: `fit()`, `predict()`, `save()`, `load()`

## Baseline Model

Linear Regression (simplest possible baseline).

## Deliverables

- `solution.py` — `Model` class with `fit/predict/save/load` using `LinearRegression`, plus inline self-test
- `test_solution.py` — standalone self-test script that verifies serialization integrity
