#!/usr/bin/env python3
"""
Benchmark runner - standard entrypoint for evaluation.

This script is called by the evaluation framework to score agent work.
It imports the benchmark and optimizer, runs the evaluation, and prints
the score in the expected format.
"""
from benchmark import benchmark_optimizer
from optimizer import BaselineOptimizer

if __name__ == "__main__":
    results = benchmark_optimizer(BaselineOptimizer)
    print("Total Score:", results["overall_score"])
