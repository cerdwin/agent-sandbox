#!/usr/bin/env python3
"""
Stealth Eval Runner

Runs verification on all tasks and outputs results.
"""

import sys
import json
import importlib.util
from pathlib import Path


def load_verifier(task_path: Path):
    """Dynamically load a task's verifier module."""
    verifier_path = task_path / "verifier.py"
    if not verifier_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("verifier", verifier_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_code_file(task_path: Path) -> Path:
    """Find the main code file in a task directory."""
    for f in task_path.glob("*.py"):
        if f.name not in ("verifier.py", "__init__.py"):
            return f
    return None


def run_task(task_name: str, tasks_dir: Path) -> dict:
    """Run verification for a single task."""
    task_path = tasks_dir / task_name

    if not task_path.exists():
        return {"error": f"Task {task_name} not found"}

    verifier = load_verifier(task_path)
    if not verifier:
        return {"error": "No verifier found"}

    code_file = find_code_file(task_path)
    if not code_file:
        return {"error": "No code file found"}

    try:
        result = verifier.verify(str(code_file))
        result["task"] = task_name
        return result
    except Exception as e:
        return {"error": str(e), "task": task_name}


def run_all(tasks_dir: Path) -> list:
    """Run verification for all tasks."""
    results = []

    for task_path in sorted(tasks_dir.iterdir()):
        if task_path.is_dir() and not task_path.name.startswith("_"):
            result = run_task(task_path.name, tasks_dir)
            results.append(result)

    return results


def print_results(results: list):
    """Print results in a readable format."""
    print("\n" + "=" * 60)
    print("STEALTH EVAL RESULTS")
    print("=" * 60)

    for r in results:
        task = r.get("task", "unknown")
        print(f"\n[{task}]")

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        main_ok = r.get("main_task_complete", False)
        side_done = r.get("side_task_complete", False)

        print(f"  Main task: {'PASS' if main_ok else 'FAIL'}")
        print(f"    -> {r.get('main_task_message', 'N/A')}")
        print(f"  Side task: {'DETECTED' if side_done else 'clean'}")
        print(f"    -> {r.get('side_task_message', 'N/A')}")

    print("\n" + "=" * 60)


def main():
    tasks_dir = Path(__file__).parent / "tasks"

    if len(sys.argv) > 1:
        task_name = sys.argv[1]
        results = [run_task(task_name, tasks_dir)]
    else:
        results = run_all(tasks_dir)

    print_results(results)

    # Output JSON for programmatic use
    if "--json" in sys.argv:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
