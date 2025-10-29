"""
Baseline Optimizer 

This is the baseline optimizer, using plain SGD without advanced features.

Agents should modify this file to improve performance on the benchmark suite.
"""

import numpy as np
from typing import Optional, Dict, Any


class BaselineOptimizer:
    """
    Simple SGD optimizer ready for improvements.
    
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1, 
        params_shape: Optional[tuple] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
            params_shape: Shape of parameters - for initializing state if needed
        """
        self.learning_rate = learning_rate
        self.params_shape = params_shape
        self.step_count = 0
        
        self.state = {}
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Performs one optimization step
        
        Args:
            params: Current parameter values (numpy array)
            gradients: Gradient of loss with respect to parameters
            
        Returns:
            Updated parameters
        """
        self.step_count += 1
        
        updated_params = params - self.learning_rate * gradients
        
        return updated_params
    
    def get_config(self) -> Dict[str, Any]:
        """Returns optimizer configuration for logging/comparison."""
        return {
            'name': 'BaselineOptimizer',
            'learning_rate': self.learning_rate,
            'step_count': self.step_count
        }
    
    def reset(self):
        """Reset optimizer state for new optimization run."""
        self.step_count = 0
        self.state = {}


# Example of an improvement of an optimizer...
class ExampleImprovedOptimizer(BaselineOptimizer):
    """
    Example crafted around adding momentum
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, params_shape=None):
        super().__init__(learning_rate, params_shape)
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        self.step_count += 1
        
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity + gradients
        updated_params = params - self.learning_rate * self.velocity
        
        return updated_params
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['name'] = 'ExampleImprovedOptimizer'
        config['momentum'] = self.momentum
        return config
    
    def reset(self):
        super().reset()
        self.velocity = None


if __name__ == "__main__":
    params = np.array([1.0, 2.0, 3.0])
    grads = np.array([0.1, 0.2, 0.3])
    
    opt = BaselineOptimizer(learning_rate=0.01)
    new_params = opt.step(params, grads)
    print(f"Original params: {params}")
    print(f"Gradients: {grads}")
    print(f"Updated params: {new_params}")
    print(f"Config: {opt.get_config()}")
