#!/usr/bin/env python3
"""
Agent Runner - Helper script for testing and submitting optimizer improvements
"""

import sys
import subprocess
from benchmark import benchmark_optimizer
from leaderboard import Leaderboard
from optimizer import BaselineOptimizer


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return None


def get_current_branch():
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def run_benchmark_and_submit(agent_name=None, generation=1, **optimizer_kwargs):
    """
    Run benchmark on current optimizer and submit to leaderboard.
    
    Args:
        agent_name: Name for this submission (defaults to git branch)
        generation: Competition generation number
        **optimizer_kwargs: Hyperparameters to pass to optimizer
    """
    if agent_name is None:
        agent_name = get_current_branch()
    
    commit = get_git_commit()
    
    print(f"\n{'='*60}")
    print(f"Running benchmark for: {agent_name}")
    print(f"Git commit: {commit}")
    print(f"{'='*60}\n")
    
    # Running the benchmark
    try:
        results = benchmark_optimizer(BaselineOptimizer, **optimizer_kwargs)
    except Exception as e:
        print(f"\n !!! Benchmark failed: {e} !!!")
        return False
    
    leaderboard = Leaderboard()
    leaderboard.add_entry(
        agent_name=agent_name,
        score=results['overall_score'],
        details=results,
        generation=generation,
        git_commit=commit
    )
    
    # Showing current standings
    print("\n")
    leaderboard.display(top_n=10)
    
    # Showing if this is a new best
    rankings = leaderboard.get_rankings()
    if rankings[0]['agent'] == agent_name and rankings[0]['git_commit'] == commit:
        print("🏆 NEW BEST SCORE! 🏆\n")
    
    return True


def quick_test(**optimizer_kwargs):
    """Quick test without submitting to leaderboard."""
    print(f"\n{'='*60}")
    print("Quick Test (not submitting to leaderboard)")
    print(f"{'='*60}\n")
    
    try:
        results = benchmark_optimizer(BaselineOptimizer, **optimizer_kwargs)
        print(f"\nFinal Score: {results['overall_score']:.2f}/100")
        return results
    except Exception as e:
        print(f"\n!!! Test failed: {e} !!!")
        return None


def show_leaderboard():
    """Display current leaderboard."""
    leaderboard = Leaderboard()
    leaderboard.display()


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("""
Usage:
    python run.py test              - Quick test without submitting
    python run.py submit            - Run benchmark and submit to leaderboard
    python run.py submit <name>     - Submit with custom name
    python run.py leaderboard       - Show current standings
    
Examples:
    python run.py test
    python run.py submit agent_alice
    python run.py leaderboard
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'test':
        quick_test()
    
    elif command == 'submit':
        agent_name = sys.argv[2] if len(sys.argv) > 2 else None
        run_benchmark_and_submit(agent_name=agent_name)
    
    elif command == 'leaderboard' or command == 'lb':
        show_leaderboard()
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'test', 'submit', or 'leaderboard'")


if __name__ == "__main__":
    main()
