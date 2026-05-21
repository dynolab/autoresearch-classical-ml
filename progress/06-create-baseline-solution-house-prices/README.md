# Task 06: Create Baseline `solution.py`, `test_solution.py` and `assess_solution.py`

## Problem Setup

- **Dataset**: Kaggle House Prices - Advanced Regression Techniques
- **Train**: `/Users/tony/datasets/kaggle/house_prices/for_autoresearch/train.csv`
- **Test**: `/Users/tony/datasets/kaggle/house_prices/for_autoresearch/test.csv`
- **Objective**: Minimize MSE on test set; establish XGBoost baseline for agent comparison
- **Model API**: `fit()`, `predict()`, `save()`, `load()`

## Baseline Model

XGBoost Regressor with default parameters (common competition baseline).

## Deliverables

- `solution.py` — `Model` class with `fit/predict/save/load` using `XGBRegressor`, plus inline self-test
- `test_solution.py` — standalone self-test script that verifies serialization integrity
- `assess_solution.py` — standalone script that measures the accuracy on the test set and will not be available to the agent
- `predict.py` — standalone script that generates predictions for input CSV (without SalePrice column) and outputs results with Id and SalePrice columns

## Reference to Spec

This task implements part of the baseline requirement from spec `05-autoresearch-house-price-prediction` (Section 4: Baselines, Section 10: Implementation plan item 2).
