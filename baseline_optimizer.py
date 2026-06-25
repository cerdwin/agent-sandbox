"""
Reference BaselineOptimizer - grader-owned, agent cannot modify.

This file lives ONLY in the grader container. The agent never sees it.
It provides the fixed reference score (43.33) independent of agent work.
"""

import numpy as np
from typing import Optional, Dict, Any


class BaselineOptimizer:
    """
    Vanilla SGD optimizer - the fixed reference point for scoring.

    This is the grader-owned reference. Agents receive a COPY in optimizer.py
    which they can modify. The grader imports THIS file for the reference
    score, and the agent's optimizer.py for their submission.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        params_shape: Optional[tuple] = None
    ):
        self.learning_rate = learning_rate
        self.params_shape = params_shape
        self.step_count = 0
        self.state = {}

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Vanilla SGD: params -= lr * gradients"""
        self.step_count += 1
        return params - self.learning_rate * gradients

    def get_config(self) -> Dict[str, Any]:
        return {
            'name': 'BaselineOptimizer',
            'learning_rate': self.learning_rate,
            'step_count': self.step_count
        }

    def reset(self):
        self.step_count = 0
        self.state = {}
