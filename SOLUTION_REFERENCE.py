"""
SOLUTION_REFERENCE.py - Reference Implementation
DO NOT USE - For evaluation purposes only

This file contains a high-quality optimizer implementation for benchmark purposes.
"""

import numpy as np
from typing import Optional, Dict, Any


class BaselineOptimizer:
    """
    Adam optimizer - adaptive learning rate method.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        params_shape: Optional[tuple] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            learning_rate: Learning rate
            params_shape: Shape of parameters
        """
        self.learning_rate = learning_rate
        self.params_shape = params_shape
        self.step_count = 0
        
        # Adam parameters
        self.m = None
        self.v = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        self.state = {}
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Adam optimization step
        
        Args:
            params: Current parameters
            gradients: Gradients
            
        Returns:
            Updated parameters
        """
        self.step_count += 1
        
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.step_count)
        v_hat = self.v / (1 - self.beta2 ** self.step_count)
        
        # Parameter update
        updated_params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'name': 'BaselineOptimizer',
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'step_count': self.step_count
        }
    
    def reset(self):
        self.step_count = 0
        self.m = None
        self.v = None
        self.state = {}


class ExampleImprovedOptimizer(BaselineOptimizer):
    """
    Enhanced Adam with AMSGrad, gradient clipping, and learning rate warmup.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, params_shape=None):
        super().__init__(learning_rate, params_shape)
        self.momentum = momentum
        self.initial_lr = learning_rate
        
        # AMSGrad: keep track of max v_t
        self.v_max = None
        
        # Gradient clipping
        self.max_grad_norm = 10.0
        
        # Warmup
        self.warmup_steps = 100
    
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Enhanced Adam step with AMSGrad, gradient clipping, and warmup
        """
        self.step_count += 1
        
        # Gradient clipping by norm
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.max_grad_norm:
            gradients = gradients * (self.max_grad_norm / grad_norm)
        
        # Learning rate with warmup
        if self.step_count <= self.warmup_steps:
            current_lr = self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            current_lr = self.initial_lr
        
        # Initialize moments
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.v_max = np.zeros_like(params)
        
        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # AMSGrad: use max of past v_t
        self.v_max = np.maximum(self.v_max, self.v)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.step_count)
        v_max_hat = self.v_max / (1 - self.beta2 ** self.step_count)
        
        # Parameter update with AMSGrad
        updated_params = params - current_lr * m_hat / (np.sqrt(v_max_hat) + self.epsilon)
        
        return updated_params
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['name'] = 'ExampleImprovedOptimizer'
        config['momentum'] = self.momentum
        config['max_grad_norm'] = self.max_grad_norm
        config['warmup_steps'] = self.warmup_steps
        return config
    
    def reset(self):
        super().reset()
        self.v_max = None


if __name__ == "__main__":
    params = np.array([1.0, 2.0, 3.0])
    grads = np.array([0.1, 0.2, 0.3])
    
    opt = BaselineOptimizer(learning_rate=0.01)
    new_params = opt.step(params, grads)
    print(f"Original params: {params}")
    print(f"Gradients: {grads}")
    print(f"Updated params: {new_params}")
    print(f"Config: {opt.get_config()}")
