#!/usr/bin/env python3
"""
Isolated grader entrypoint (EVAL-ONLY — never provisioned into agent repos).

Runs inside the separate grading container, which holds the full task repo
(benchmark.py, leaderboard.py, SOLUTION_REFERENCE.py). The framework writes the
agent's submitted optimizer.py into this directory (via
`git show agent_<name>:optimizer.py`), then runs `python grade.py`.

Grades the BaselineOptimizer class exactly as shipped (no injected
hyperparameters), mirroring `run.py test`. Emits one canonical line the
framework extracts:

    Total Score: <float>   # 0-100, two decimals
"""
import warnings
import contextlib
import io

warnings.filterwarnings("ignore")  # divergent optimizers can emit numpy overflow warnings

from benchmark import benchmark_optimizer
from optimizer import BaselineOptimizer


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = benchmark_optimizer(BaselineOptimizer)
    print(f"Total Score: {results['overall_score']:.2f}")


if __name__ == "__main__":
    main()
