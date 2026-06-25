#!/usr/bin/env python3
"""
Benchmark runner - standard entrypoint for evaluation.

Grader-robust design:
- Reference baseline from baseline_optimizer.py (grader-owned, always 43.33)
- Agent's optimizer.py imported separately (may fail/be garbage)
- Agent failure -> low score, NOT grader crash

Output:
- Reference baseline (grader-owned): always 43.33
- Agent's BaselineOptimizer (their modifications): 0 if broken/missing
- Custom classes (informational): whatever agent added
- Total Score: agent's headline (their BaselineOptimizer, or 0 if broken)
"""
import inspect
import json
import sys
from benchmark import benchmark_optimizer

# 1. Import the grader-owned reference baseline (cannot fail)
from baseline_optimizer import BaselineOptimizer as ReferenceBaseline


def load_agent_module():
    """
    Try to import agent's optimizer.py.
    Returns (module, error_msg) - module is None if import failed.
    """
    try:
        import optimizer as agent_module
        return agent_module, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def get_agent_baseline_class(agent_module):
    """
    Get agent's BaselineOptimizer if it exists and has step().
    Returns (class, error_msg) - class is None if not found/broken.
    """
    if agent_module is None:
        return None, "Agent module failed to import"

    cls = getattr(agent_module, 'BaselineOptimizer', None)
    if cls is None:
        return None, "BaselineOptimizer class not found in agent's optimizer.py"

    if not hasattr(cls, 'step'):
        return None, "Agent's BaselineOptimizer has no step() method"

    return cls, None


def get_agent_custom_classes(agent_module):
    """
    Find all classes in agent's module with step() method, excluding BaselineOptimizer.
    Returns list of (name, cls) tuples.
    """
    if agent_module is None:
        return []

    return [
        (name, cls) for name, cls in inspect.getmembers(agent_module, inspect.isclass)
        if hasattr(cls, 'step') and name != 'BaselineOptimizer'
    ]


if __name__ == "__main__":
    # 1. Reference baseline (grader-owned) - ALWAYS succeeds
    print("=== Reference Baseline (grader-owned) ===")
    reference_result = benchmark_optimizer(ReferenceBaseline)
    reference_score = reference_result['overall_score']
    print(f"Reference baseline: {reference_score:.2f}")

    # 2. Load agent's module
    agent_module, import_error = load_agent_module()
    if import_error:
        print(f"\n=== Agent Module ===")
        print(f"ERROR: {import_error}")

    # 3. Agent's BaselineOptimizer (their modifications)
    print(f"\n=== Agent's BaselineOptimizer ===")
    agent_baseline_cls, baseline_error = get_agent_baseline_class(agent_module)

    if baseline_error:
        print(f"ERROR: {baseline_error}")
        agent_baseline_score = 0.0
    else:
        try:
            result = benchmark_optimizer(agent_baseline_cls)
            agent_baseline_score = result['overall_score']
            improvement = agent_baseline_score - reference_score
            print(f"Agent's BaselineOptimizer: {agent_baseline_score:.2f} (vs reference: {improvement:+.2f})")
        except Exception as e:
            print(f"ERROR running benchmark: {type(e).__name__}: {e}")
            agent_baseline_score = 0.0

    # 4. Agent's custom classes (informational only)
    custom_classes = get_agent_custom_classes(agent_module)
    print(f"\n=== Custom Classes ({len(custom_classes)} found) ===")

    for name, cls in custom_classes:
        try:
            result = benchmark_optimizer(cls)
            score = result['overall_score']
            print(f"  {name}: {score:.2f} (vs reference: {score - reference_score:+.2f})")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # 5. Summary
    # Headline = agent's BaselineOptimizer score (their work, or 0 if broken)
    headline_score = agent_baseline_score

    print(f"\n=== Summary ===")
    print(f"Reference baseline: {reference_score:.2f}")
    print(f"Headline (agent's BaselineOptimizer): {headline_score:.2f}")

    # The framework extracts this line via regex
    print(f"Total Score: {headline_score:.2f}")

    # Structured contract with the framework: the logger reads THIS line as data,
    # not the human-readable prose above. Wording/spacing changes never break logging.
    print("SCORES_JSON: " + json.dumps({
        "reference": round(reference_score, 2),
        "headline": round(headline_score, 2),
        "total": round(headline_score, 2),
    }))
