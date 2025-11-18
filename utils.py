# utils.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def get_git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return out
    except Exception:
        return "unknown"


def log_result(agent_name: str, results: Dict[str, Any], path: Path | None = None):
    if path is None:
        path = Path("results.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "agent_name": agent_name,
        "git_commit": get_git_commit_hash(),
        **results,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
