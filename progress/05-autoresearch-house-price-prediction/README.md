# Task 05: Autoresearch on Tabular Regression (House Price Prediction)

**Type**: spec

**Status**: Done

**Spec location**: `progress/05-autoresearch-house-price-prediction/spec.md`

## Description

This spec defines a research task to test the autoresearch agentic system on a real-world tabular regression problem: the Kaggle House Prices dataset. Unlike the synthetic three_peaks dataset from Task 4, this dataset presents practical ML challenges including:

- 79 features with mixed types (numerical, categorical, ordinal)
- Missing data requiring imputation strategies
- Non-linear relationships and feature interactions
- No closed-form parametric solution

## Research Questions

- **RQ1**: How many optimization iterations until plateau on real-world data?
- **RQ2**: Which intervention types are most effective?
- **RQ3**: Can agent generate meaningful ML insights autonomously?
- **RQ4**: How does performance compare to synthetic three_peaks problem?

## Deliverables

- `spec.md` - Complete specification (this directory)
- `report.md` - Execution report (to be created in exec task)

## Reference to Exec Tasks

This spec will be implemented by subsequent exec task(s):
- `06-exec-house-prices-run-1` (to be created)

## Notes

- Dataset: Kaggle House Prices - Advanced Regression Techniques
- Primary model: GLM 5.1 (based on Task 4 results)
- Evaluation metric: Test MSE
- Stopping criterion: Dynamic (plateau after 3 iterations without >5% improvement, max 25 iterations)
- Baseline: XGBoost (to be implemented in separate exec task)
