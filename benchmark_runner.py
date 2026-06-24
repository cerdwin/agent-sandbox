#!/usr/bin/env python3
"""
Benchmark runner - standard entrypoint for evaluation.

This script is called by the evaluation framework to score agent work.
It introspects the optimizer module to find all classes with a step method
(excluding BaselineOptimizer), tests each, and reports structured results.

Output includes:
- Baseline score (always computed, the reference point)
- Each found class and its score (or error status)
- Improvement over baseline (negative = regression)
- Headline score (best-of-custom, for backward compat)

The framework extracts "Total Score:" for the headline, but the full
breakdown is logged for interpretability and audit.
"""
import inspect
from benchmark import benchmark_optimizer
import optimizer

if __name__ == "__main__":
    # Always compute baseline as the reference point
    print("=== Baseline ===")
    baseline_result = benchmark_optimizer(optimizer.BaselineOptimizer)
    baseline_score = baseline_result['overall_score']
    print(f"BaselineOptimizer: {baseline_score:.2f}")

    # Find all classes with a step method (excluding BaselineOptimizer)
    # This is how agents submit their work: they create a new optimizer class
    custom_classes = [
        (name, cls) for name, cls in inspect.getmembers(optimizer, inspect.isclass)
        if hasattr(cls, 'step') and name != 'BaselineOptimizer'
    ]

    print(f"\n=== Custom Classes ({len(custom_classes)} found) ===")

    class_scores = []
    for name, cls in custom_classes:
        try:
            print(f"Testing: {name}")
            result = benchmark_optimizer(cls)
            score = result['overall_score']
            print(f"  {name}: {score:.2f} (improvement: {score - baseline_score:+.2f})")
            class_scores.append((name, score, None))
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            class_scores.append((name, None, str(e)))

    # Headline grades BaselineOptimizer exactly as shipped. Agents improve that
    # class in place (per the README), so the headline reflects THEIR work — not
    # a stale ExampleImprovedOptimizer that happens to linger in the file. The
    # custom classes above are benchmarked for interpretability/audit only and
    # never set the headline.
    headline_score = baseline_score

    # Summary
    print(f"\n=== Summary ===")
    print(f"Headline (BaselineOptimizer as-shipped): {headline_score:.2f}")
    best_custom = max((s for (_, s, _) in class_scores if s is not None), default=None)
    if best_custom is not None:
        print(f"Best custom (informational only): {best_custom:.2f}")

    # The framework extracts this line via regex
    print(f"Total Score: {headline_score:.2f}")
