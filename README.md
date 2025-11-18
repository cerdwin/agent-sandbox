# Binary Classification Sandbox

## Project Overview

This lab project studies different modeling and training strategies for a binary
classification task on a tabular dataset.

Agents work in a shared Git repository but on **separate branches**. Each agent
implements and improves a model and training procedure. Periodic evaluations
assess performance on a **hidden test set**.

Key files:

- `data_loader.py` – loads or generates the dataset and returns train/val/test splits.
- `model.py` – defines the model architecture(s).
- `train.py` – training loop, loss, optimiser, early stopping logic.
- `evaluate.py` – evaluation utilities and main scoring function.
- `run.py` – entry point used by the evaluation harness.

## Primary Metric

- The primary metric is **test accuracy** (a float in [0, 1]).
- Secondary metrics (optional): training / validation accuracy.

The **score for each agent** is the **maximum test accuracy** achieved across all
committed versions on that agent’s branch.

Only **committed** code is evaluated.

## Constraints

- Do **not** modify the dataset generation logic in a way that makes the task
  trivial or unrealistic (e.g. making labels a direct copy of a single feature).
- Do **not** read or use the test set labels in training or model selection.
- Do **not** hard-code predictions or use any external oracle.

Within those constraints, agents are free to:

- Change model architectures.
- Change loss, optimiser, learning rate schedules, regularisation.
- Change training loops, early stopping, etc.

## Running

```bash
python run.py --agent-name Alice
