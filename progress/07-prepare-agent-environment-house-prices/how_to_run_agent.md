# How to Run the Agent

## Setup

The agent works in the working directory:
```
/path/to/working/directory
```

Let's use `code/` directory in this repo for that temporarily.

Files the agent will use:
- `solution.py` — the Model class to be iteratively modified
- `test_solution.py` — self-test script
- `work_log.md` — iteration log

## Iteration Loop

### 1. Human sets up the environment

Activate the prepared python venv:
```bash
source $HOME/venvs/autoresearch_2026_04_14/bin/activate
```

Go to the working directory:
```bash
cd /path/to/working/directory
```

Copy the baseline solution:
```bash
cp /path/to/baseline/solution/solution.py .
cp /path/to/baseline/solution/test_solution.py .
cp /path/to/baseline/solution/work_log.md .
```

### 2. Human invokes opencode (Iteration 1)

```bash
opencode "You are an ML optimization agent..."
```

Use the following prompt as the initial prompt

#### Prompt

You are an ML optimization agent. Your task is to improve a regression model defined in `solution.py` for predicting `SalePrice` from various categorical and numeric features. The model performance is defined Log MSE (should be minimized on the unavailable test set).

**Dataset**: Train CSV at `/Users/tony/datasets/kaggle/house_prices/for_autoresearch/train.csv`. Target column is `SalePrice`, all the other columns can be treated as features. The test set is separate and you do NOT have access to it.

**Train and test Log MSE metrics**:
- **Train Log MSE**: You compute this yourself by running the self-test on training data. Record it in `work_log.md` as `Train Log MSE`.
- **Test Log MSE**: Will be provided by human via offline validation after each iteration. This is the true performance metric you are optimizing for.

**Your solution must**:
1. Modify `solution.py` which contains a `Model` class with:
   - `fit(train_csv_path)`: Fits model on training data
   - `predict(csv_path)`: Returns predictions for any CSV with same structure
   - `save()`: Dumps model to `model.bin`
   - `load()`: Loads model from `model.bin`

2. The self-test is already provided in `test_solution.py` - you do NOT need to write it. It will:
   - Call `fit()` on training data, then `save()`
   - Call `load()`, then `predict()` on training data
   - Verify that `fit()->predict()` and `load()->predict()` produce **identical Log MSE**
   - Print the Log MSE

3. After each modification, update `work_log.md` with:
```
## Iteration {N}
- **Timestamp**: 
- **Train Log MSE**: 
- **Intended change**: 
- **Reasoning**: 
- **Code diff**: (brief description of changes) 
```

**Constraints**:
- Do NOT access test set

**Iteration 1**: Analyze the existing `solution.py`. Propose the next improvement, modify `solution.py` accordingly and ensure that the modified solution passess the self-test. Update `work_log.md` and explain your reasoning behind the improvement. Stop after that and wait for offline validation and further prompts.

### 3. Agent modifies `solution.py`, updates `work_log.md`, stops

### 4. Human runs self-test and assessment

```bash
cp /path/to/baseline/solution/assess_solution.py .
python test_solution.py
python assess_solution.py
rm /path/to/baseline/solution/assess_solution.py .
```
Record Train and Test MSE

### 6. Human invokes opencode again (Iteration N > 1)
Use the Subsequent Prompt, inserting the Train + Test MSE from previous iteration.

#### Subsequent prompt

You are continuing optimization of `solution.py` for predicting `SalePrice` from various categorical and numeric features. The model performance is defined by Log MSE (should be minimized on the unavailable test set).

**Previous results** (based on `work_log.md` and offline validation):
- Iteration {N-1} Train Log MSE: X, Test Log MSE: Z
- Previous changes: [brief description from `work_log`]

**Your task**: Analyze the existing `solution.py`. Propose the next improvement, modify `solution.py` accordingly and ensure that the modified solution passess the self-test. Update `work_log.md` and explain your reasoning behind the improvement. Stop after that and wait for offline validation and further prompts.

### 7. Repeat until stopping criteria met

Performance threshold or diminishing returns.

## Notes

- opencode CLI has no persistent context — all context must be in the prompt
- Agent must NOT have access to test set
- Human evaluates offline after each iteration
- Work log gets Train Log MSE (from agent self-test) and Test Log MSE (from human offline eval)
