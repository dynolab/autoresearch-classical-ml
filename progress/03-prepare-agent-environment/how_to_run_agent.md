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

**1. Human sets up the environment**:
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

**2. Human invokes opencode** (Iteration 1):
```bash
opencode "You are an ML optimization agent..."
```
Use the Initial Prompt from `spec.md`.

**3. Agent** modifies `solution.py`, updates `work_log.md`, stops.

**4. Human runs self-test and assessment**:
```bash
cp /path/to/baseline/solution/assess_solution.py .
python test_solution.py
python assess_solution.py
rm /path/to/baseline/solution/assess_solution.py .
```
Record Train and Test MSE

**6. Human invokes opencode again** (Iteration N > 1):
Use the Subsequent Prompt template from `spec.md`, inserting the Train + Test MSE from previous iteration.

**7. Repeat until stopping criteria met** (performance threshold or diminishing returns).

## Notes

- opencode CLI has no persistent context — all context must be in the prompt
- Agent must NOT have access to test set
- Human evaluates offline after each iteration
- Work log gets Train MSE (from agent self-test) and Test MSE (from human offline eval)
