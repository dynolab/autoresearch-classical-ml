# Autoresearch Spec: MSE Minimization with Model Size Constraint

## 0. Research Questions Addressed

- **RQ1**: How many optimization iterations can the agentic system support performance improvement until reaching a plateau?
- **RQ2**: What types of interventions (hyperparameter tuning, feature engineering, model selection, pipeline restructuring) are most effective for agentic improvement?
- **RQ3**: What theoretical and practical ML analysis tools can the agentic system come up with without external guidance?

## 1. Problem Setup

### Dataset
- **Source**: Google Drive
- **Train**: https://drive.google.com/file/d/1hyp4EvWKsz2TLQJSnedktqOTHowhMxw7/view?usp=drive_link
- **Test**: https://drive.google.com/file/d/1_NpRRQ4nBe1VxsCJqSJT6rCZ8_dj6L8a/view?usp=drive_link
- **Features**: `x`, `y` (columns in CSV)
- **Target**: `target` (column in CSV)
- **Baseline performance**: TBD (to be filled after baseline solution)

### Objectives
- Primary: Minimize MSE on test set
- Secondary: Minimize model size (interpretability/compression angle)

## 2. Agent System

### Agent Scaffolding
**opencode CLI** - The agent is opencode itself running in CLI mode. Since CLI mode has no persistent context between turns, the initial prompt must contain all necessary information.

Iteration control:
1. Human invokes opencode with the initial prompt
2. Agent writes/changes `solution.py` and updates `work_log.md`
3. Human runs the solution, evaluates MSE, reports back to agent
4. Human invokes opencode again with updated context (new MSE, observations)
5. Repeat until stopping criteria met

### Backend LLM
- **Model**: MiniMax 2.7
- **Provider**: GPTunnel
- **Parameters**: default

### Environment
- **Python version**: 3.12
- **Initial packages**: numpy, scipy, matplotlib, pandas, scikit-learn, joblib, pytest
- **Constraint**: No installing new packages on top of the initial environment
- **Execution mode**: local

### Solution Structure
Human writes the **baseline solution** (`solution.py`) before agent iterations begin.

`solution.py` contains a `Model` class with the following API:
- `fit(train_csv_path)`: Fits the model on training data from the specified CSV path
- `predict(csv_path)`: Returns predictions for any CSV with the same structure as the training set
- `save()`: Dumps the model to `model.bin` using as few bytes as possible
- `load()`: Loads the model from `model.bin`

**Self-test**: The script must include a self-test that:
1. Calls `fit()` on training data, then `save()`
2. Calls `load()`, then `predict()` on training data
3. Compares (`fit() -> predict()`) vs (`load() -> predict()`) - they must produce identical MSE

This ensures the model is actually serialized correctly and prevents cheating (e.g., saving a noop with zero bytes).

**No test set access**: The agent must not have access to the test set. The human evaluates the final model on the test set separately.

Agent then iteratively modifies `solution.py` to improve MSE (measured via self-test on train) while keeping model size small.

### Prompt Template

#### Initial Prompt (Iteration 1)
You are an ML optimization agent. Your task is to improve a regression model defined in `solution.py` for predicting `target` from features `x` and `y`. The model performance is defined by two metrics: MSE (should be minimized on the unavailable test set) and the disk space occupied by the model dump (should be minimized). Thus, by "improvement", we understand minimization of these two metrics.

**Dataset**: Train CSV at `/Users/tony/datasets/abc/train_dataset.csv` with columns `x`, `y`, `target`. The test set is separate and you do NOT have access to it.

**Train and test MSE metrics**:
- **Train MSE**: You compute this yourself by running the self-test on training data. Record it in `work_log.md` as `Train MSE`.
- **Test MSE**: Will be provided by human via offline validation after each iteration. This is the true performance metric you are optimizing for.

**Your solution must**:
1. Modify `solution.py` which contains a `Model` class with:
   - `fit(train_csv_path)`: Fits model on training data
   - `predict(csv_path)`: Returns predictions for any CSV with same structure
   - `save()`: Dumps model to `model.bin` (as small as possible)
   - `load()`: Loads model from `model.bin`

2. The self-test is already provided in `test_solution.py` - you do NOT need to write it. It will:
   - Call `fit()` on training data, then `save()`
   - Call `load()`, then `predict()` on training data
   - Verify that `fit()->predict()` and `load()->predict()` produce **identical MSE**
   - Print the MSE

3. After each modification, update `work_log.md` with:
```
## Iteration {N}
- **Timestamp**: 
- **Train MSE**: 
- **Model size (bytes)**: 
- **Intended change**: 
- **Reasoning**: 
- **Code diff**: (brief description of changes) 
```

**Constraints**:
- Python 3.12 with: numpy, scipy, matplotlib, pandas, scikit-learn, joblib, pytest
- NO installing new packages
- Do NOT access test set

**Iteration 1**: Analyze the existing `solution.py`. Propose the next improvement, modify `solution.py` accordingly and ensure that the modified solution passess the self-test. Update `work_log.md` and explain your reasoning behind the improvement. Stop after that and wait for offline validation and further prompts.

#### Subsequent Prompts (Iteration N > 1)
You are continuing optimization of `solution.py` for predicting `target` from features `x` and `y`. The model performance is defined by two metrics: MSE (should be minimized on the unavailable test set) and the disk space occupied by the model dump (should be minimized). Thus, by "improvement", we understand minimization of these two metrics.

**Previous results** (based on `work_log.md` and offline validation):
- Iteration {N-1} Train MSE: X, Test MSE: Z, Model size: Y bytes
- Previous changes: [brief description from `work_log`]

**Your task**: Analyze the existing `solution.py`. Propose the next improvement, modify `solution.py` accordingly and ensure that the modified solution passess the self-test. Update `work_log.md` and explain your reasoning behind the improvement. Stop after that and wait for offline validation and further prompts.

### Work Log
At each optimization iteration, the agent appends to `work_log.md`:

```markdown
## Iteration {N}
- **Timestamp**: 
- **Train MSE**: 
- **Model size (bytes)**: 
- **Intended change**: 
- **Reasoning**: 
- **Code diff**: (brief description of changes)
```

Note: Test MSE is added to the prompt by the human after offline evaluation.

### Intervention Taxonomy (for RQ2)
The agent's actions will be classified into:
- Hyperparameter tuning
- Feature engineering
- Model selection
- Pipeline restructuring

We do not to enforce these types of actions explicitly. Instead, we will classify them post factum based on `work_log.md`.

## 3. Evaluation Protocol

### Stopping Criteria
- Maximum iterations: 100
- Time limit: 10 minutes per iteration
- Performance threshold: **TBD**
- Diminishing returns threshold: **TBD**

### Metrics (mapped to RQs)
- **RQ1**: Iteration count until plateau, performance curve
- **RQ2**: Intervention type tracking, MSE gain per intervention type
- **RQ3**: ML analysis tools/code produced by agent (qualitative)

### Logging
- `work_log.md`: Agent's iteration-by-iteration work log (as structured above)
- Code versions per iteration: after each iteration, `solution.py` should be copied to `solution_iter_{N}.py`

## 4. Expected Outcomes
- Baseline model train MSE: 0.013833
- Baseline model test MSE: 0.006492
- Expected improvement (test MSE): 0.001
