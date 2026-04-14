# Task 01: Autoresearch MSE Minimization

## Specification

First autoresearch instance: Given a train-test split, an agentic system is tasked with fitting any classical ML model to minimize MSE on the test set while keeping the model as small as possible.

## Problem Setup

- **Input**: Train-test split of a dataset (format TBD)
- **Objective**: Minimize MSE on test set
- **Constraint**: Model size must be minimized (interpretability/compression angle)
- **Allowed approaches**: Any classical ML technique (linear models, trees, ensembles, etc.)

## Deliverables

- `spec.md`: Detailed specification of the autoresearch setup (agent prompts, stopping criteria, data format, evaluation protocol)
