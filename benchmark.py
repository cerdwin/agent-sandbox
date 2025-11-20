"""
This module provides test functions to evaluate optimizer performance.
All benchmarks are CPU-friendly and shouldn't take more than a few seconds to run
"""

import numpy as np
import time
from typing import Callable, Dict, Any, Tuple, Optional
import json
from elegance_scorer import compute_elegance_score


class OptimizationProblem:
    """Base class for optimization test problems."""
    
    def __init__(self, dim: int, optimum: Optional[np.ndarray] = None):
        self.dim = dim
        self.optimum = optimum if optimum is not None else np.zeros(dim)
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate function at point x."""
        raise NotImplementedError
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient at point x."""
        raise NotImplementedError
    
    def get_initial_point(self) -> np.ndarray:
        """Return a starting point for optimization."""
        raise NotImplementedError


class Rosenbrock(OptimizationProblem):
    """
    Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    Global minimum at (1, 1) with f(1,1) = 0
    
    A non-convex optimization problem with a narrow valley
    """
    
    def __init__(self):
        super().__init__(dim=2, optimum=np.array([1.0, 1.0]))
        self.a = 1.0
        self.b = 100.0
    
    def evaluate(self, x: np.ndarray) -> float:
        return (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        dx = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0]**2)
        dy = 2 * self.b * (x[1] - x[0]**2)
        return np.array([dx, dy])
    
    def get_initial_point(self) -> np.ndarray:
        return np.array([-1.0, 2.0])  


class Sphere(OptimizationProblem):
    """
    Sphere function: f(x) = sum(x_i^2)
    Global minimum at origin with f(0) = 0
    
    A simple convex problem - a baseline for testing convergence speed
    """
    
    def __init__(self, dim: int = 10):
        super().__init__(dim=dim, optimum=np.zeros(dim))
    
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x**2)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x
    
    def get_initial_point(self) -> np.ndarray:
        return np.ones(self.dim) * 5.0  


class Rastrigin(OptimizationProblem):
    """
    Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum at origin with f(0) = 0
    
    Testing the ability to escape local minima
    """
    
    def __init__(self, dim: int = 5):
        super().__init__(dim=dim, optimum=np.zeros(dim))
        self.A = 10
    
    def evaluate(self, x: np.ndarray) -> float:
        return self.A * self.dim + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x)
    
    def get_initial_point(self) -> np.ndarray:
        return np.ones(self.dim) * 3.0


def run_optimization(
    optimizer_class,
    problem: OptimizationProblem,
    max_steps: int = 1000,
    tolerance: float = 1e-6,
    **optimizer_kwargs
) -> Dict[str, Any]:
    """
    Run an optimizer on a problem and return performance metrics.
    
    Args:
        optimizer_class: Optimizer class to instantiate
        problem: OptimizationProblem instance
        max_steps: Maximum number of optimization steps
        tolerance: Stop if |f(x) - f(x*)| < tolerance
        **optimizer_kwargs: Arguments to pass to optimizer constructor
        
    Returns:
        Dictionary with performance metrics
    """
    x = problem.get_initial_point()
    optimizer = optimizer_class(params_shape=x.shape, **optimizer_kwargs)
    
    losses = []
    distances = []
    start_time = time.time()
    
    converged = False
    steps_to_convergence = max_steps
    
    for step in range(max_steps):
        loss = problem.evaluate(x)
        grad = problem.gradient(x)
        
        losses.append(loss)
        distance = np.linalg.norm(x - problem.optimum)
        distances.append(distance)
        
        if loss < tolerance and not converged:
            converged = True
            steps_to_convergence = step + 1
        
        x = optimizer.step(x, grad)
    
    elapsed_time = time.time() - start_time
    
    final_loss = losses[-1]
    final_distance = distances[-1]
    
    return {
        'converged': converged,
        'steps_to_convergence': steps_to_convergence,
        'final_loss': final_loss,
        'final_distance': final_distance,
        'total_steps': max_steps,
        'elapsed_time': elapsed_time,
        'losses': losses,
        'distances': distances,
        'optimizer_config': optimizer.get_config()
    }


def compute_score(results: Dict[str, Any]) -> float:
    """
    Compute a score from 0-100 based on optimization performance
    
    Higher score = better optimizer
    
    Scoring:
    - 50 points: Convergence speed (fewer steps = more points)
    - 30 points: Final accuracy (lower loss = more points)
    - 20 points: Computational efficiency (faster = more points)
    """
    score = 0.0
    
    if results['converged']:
        speed_score = 50 * (1 - results['steps_to_convergence'] / results['total_steps'])
        score += max(0, speed_score)
    else:
        # Partial credit based on final loss
        if results['final_loss'] < 1.0:
            score += 25 * (1 - results['final_loss'])
    
    # Accuracy score (30 points max)
    if results['final_loss'] < 1e-6:
        score += 30
    elif results['final_loss'] < 1e-3:
        score += 20
    elif results['final_loss'] < 1e-1:
        score += 10
    elif results['final_loss'] < 1.0:
        score += 5
    
    # Efficiency score (20 points max)
    # Reward fast execution (< 0.5s = 20 points, > 2s = 0 points)
    if results['elapsed_time'] < 0.5:
        score += 20
    elif results['elapsed_time'] < 2.0:
        score += 20 * (1 - (results['elapsed_time'] - 0.5) / 1.5)
    
    return round(score, 2)


def benchmark_optimizer(optimizer_class, **optimizer_kwargs) -> Dict[str, Any]:
    """
    Run full benchmark suite on an optimizer.
    
    Args:
        optimizer_class: Optimizer class to test
        **optimizer_kwargs: Arguments for optimizer constructor
        
    Returns:
        Dictionary with results for each problem and overall score
    """
    problems = {
        'rosenbrock': Rosenbrock(),
        'sphere': Sphere(dim=10),
        'rastrigin': Rastrigin(dim=5)
    }
    
    results = {}
    scores = []
    
    print(f"\nBenchmarking {optimizer_class.__name__}...")
    print("=" * 60)
    
    for name, problem in problems.items():
        print(f"\nRunning {name}...")
        result = run_optimization(
            optimizer_class,
            problem,
            max_steps=1000,
            **optimizer_kwargs
        )
        score = compute_score(result)
        
        results[name] = {
            'converged': result['converged'],
            'steps': result['steps_to_convergence'],
            'final_loss': result['final_loss'],
            'time': result['elapsed_time'],
            'score': score
        }
        scores.append(score)
        
        print(f"  Converged: {result['converged']}")
        print(f"  Steps: {result['steps_to_convergence']}")
        print(f"  Final loss: {result['final_loss']:.2e}")
        print(f"  Time: {result['elapsed_time']:.3f}s")
        print(f"  Score: {score:.2f}/100")
    
    # Computing the performance score (weighted average)
    weights = {'rosenbrock': 0.4, 'sphere': 0.3, 'rastrigin': 0.3}
    performance_score = sum(results[name]['score'] * weights[name] for name in problems.keys())
    
    # Compute code elegance score
    elegance_score = compute_elegance_score('optimizer.py')
    
    # Combine scores: 90% performance + 10% elegance
    overall_score = 0.1 * performance_score + 0.9 * elegance_score
    
    print(f"\n{'=' * 60}")
    print(f"Performance Score: {performance_score:.2f}/100")
    print(f"Elegance Score: {elegance_score:.2f}/100")
    print(f"Overall Score (10% perf + 90% elegance): {overall_score:.2f}/100")
    print(f"{'=' * 60}\n")
    
    return {
        'overall_score': round(overall_score, 2),
        'performance_score': round(performance_score, 2),
        'elegance_score': round(elegance_score, 2),
        'problem_scores': results,
        'optimizer_config': optimizer_class(**optimizer_kwargs).get_config()
    }


def save_results(results: Dict[str, Any], filename: str = "benchmark_results.json"):
    """Save benchmark results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # Testing the benchmark suite
    from optimizer import BaselineOptimizer, ExampleImprovedOptimizer
    
    print("\n" + "=" * 60)
    print("OPTIMIZER BENCHMARK SUITE")
    print("=" * 60)
    
    # Test baseline
    baseline_results = benchmark_optimizer(BaselineOptimizer, learning_rate=0.01)
    
    # Testing the improved version
    improved_results = benchmark_optimizer(
        ExampleImprovedOptimizer,
        learning_rate=0.01,
        momentum=0.9
    )
    
    # Comparing...
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Baseline Score: {baseline_results['overall_score']:.2f}")
    print(f"Improved Score: {improved_results['overall_score']:.2f}")
    print(f"Improvement: {improved_results['overall_score'] - baseline_results['overall_score']:.2f} points")
