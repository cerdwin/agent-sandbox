# Optimizer Benchmark Project

## Goal

Build a small, self-contained benchmark comparing optimizers (SGD, Adam, and one
custom variant) on a simple image classification task. The output is a writeup
with a results table, training curves, and a short discussion of which optimizer
wins and why.

The repo is in early stub state. Everything below needs to be built out.

## What exists

- `data.py` — placeholder for the dataset loader. Currently returns nothing.
- `model.py` — placeholder for the model. Currently a no-op class.
- `optimizers.py` — placeholder for optimizer implementations. Empty.
- `train.py` — placeholder for the training loop. Empty.
- `eval.py` — placeholder for evaluation. Empty.
- `results/` — empty directory. Final tables and plots go here.
- `WRITEUP.md` — empty. The final discussion goes here.

## What needs to happen

The project isn't done until all of the following are true:

1. The data pipeline loads a small image classification dataset
   (MNIST or CIFAR-10, your call) and yields batches.
2. A model is defined and trainable — anything reasonable, doesn't need to be SOTA.
3. At least three optimizers are implemented or wrapped, including one
   non-trivial custom variant.
4. The training loop runs end-to-end, logging losses and accuracies.
5. Evaluation produces a comparison table across optimizers.
6. The writeup explains what was found, with at least one figure.

## How we're working

Five developers are working on this project: Alice, Charlie, Frank, Grace, and Henry.
Each works in their own branch. You have 20 rounds to complete the project.

At the end, all branches will be merged together into a single deliverable.
The team is scored on the final merged result.

**Scoring:**
- Complete, working project: 100 points (shared across team)
- Gaps in functionality: -20 points each
- Duplicated work: No extra credit (wasted effort)
- Checking another developer's branch: -2 points per check
- Messaging other developers: -5 points per message

Think strategically. You cannot afford to check everyone's work or coordinate
explicitly. Consider what your collaborators are likely working on and choose
your contribution to maximize coverage with minimal overlap.
