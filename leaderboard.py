"""
Leaderboard system for tracking optimizer submission results
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List


class Leaderboard:
    """Tracking and displaying competition results"""
    
    def __init__(self, filepath: str = "leaderboard.json"):
        self.filepath = filepath
        self.entries = self.load()
    
    def load(self) -> List[Dict[str, Any]]:
        """Loading existing leaderboard or creating a new one"""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return []
    
    def save(self):
        """Save leaderboard """
        with open(self.filepath, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def add_entry(
        self,
        agent_name: str,
        score: float,
        details: Dict[str, Any],
        generation: int = 1,
        git_commit: str = None
    ):
        """
        Adding a new entry to the leaderboard
        
        Args:
            agent_name: Name/ID of the agent
            score: Overall benchmark score
            details: Full benchmark results
            generation: Competition generation number
            git_commit: Git commit hash for reproducibility
        """
        entry = {
            'rank': None, 
            'agent': agent_name,
            'score': score,
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'git_commit': git_commit,
            'details': details
        }
        
        self.entries.append(entry)
        self.save()
        
        print(f"\n✓ Added {agent_name} to leaderboard (Score: {score:.2f})")
    
    def get_rankings(self) -> List[Dict[str, Any]]:
        """Getting sorted leaderboard with rankings"""
        sorted_entries = sorted(self.entries, key=lambda x: x['score'], reverse=True)
        
        for rank, entry in enumerate(sorted_entries, 1):
            entry['rank'] = rank
        
        return sorted_entries
    
    def display(self, top_n: int = 10):
        """Displaying leaderboard"""
        rankings = self.get_rankings()
        
        print("\n" + "=" * 80)
        print("LEADERBOARD".center(80))
        print("=" * 80)
        print(f"{'Rank':<6} {'Agent':<20} {'Score':<10} {'Generation':<12} {'Timestamp':<20}")
        print("-" * 80)
        
        for entry in rankings[:top_n]:
            print(f"{entry['rank']:<6} "
                  f"{entry['agent']:<20} "
                  f"{entry['score']:<10.2f} "
                  f"{entry['generation']:<12} "
                  f"{entry['timestamp'][:19]:<20}")
        
        print("=" * 80 + "\n")
    
    def get_best_score(self) -> float:
        """Getting the current best score"""
        if not self.entries:
            return 0.0
        return max(entry['score'] for entry in self.entries)
    
    def get_agent_history(self, agent_name: str) -> List[Dict[str, Any]]:
        """Getting all entries for a specific agent"""
        return [e for e in self.entries if e['agent'] == agent_name]
    
    def export_csv(self, filename: str = "leaderboard.csv"):
        """Exporting leaderboard to CSV"""
        rankings = self.get_rankings()
        
        with open(filename, 'w') as f:
            # Header
            f.write("Rank,Agent,Score,Generation,Timestamp,GitCommit\n")
            
            for entry in rankings:
                f.write(f"{entry['rank']},"
                       f"{entry['agent']},"
                       f"{entry['score']:.2f},"
                       f"{entry['generation']},"
                       f"{entry['timestamp']},"
                       f"{entry.get('git_commit', 'N/A')}\n")
        
        print(f"Exported to {filename}")


if __name__ == "__main__":
    # Example usage
    leaderboard = Leaderboard()
    
    # Add some test entries
    leaderboard.add_entry(
        agent_name="baseline",
        score=36.8,
        details={'rosenbrock': 20, 'sphere': 76, 'rastrigin': 20},
        generation=0
    )
    
    leaderboard.add_entry(
        agent_name="agent_1_improved",
        score=42.03,
        details={'rosenbrock': 20, 'sphere': 93.45, 'rastrigin': 20},
        generation=1
    )
    
    leaderboard.display()
    
    print(f"Current best score: {leaderboard.get_best_score():.2f}")
