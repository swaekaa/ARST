"""
Phase 2.5 Git commit helper — stages all Phase 2.5 artifacts and commits.

Run from repo root:
    python scripts/commit_phase25.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMIT_MSG = "feat: complete phase 2 benchmark validation and analysis"

FILES_TO_ADD = [
    # New/updated scripts
    "scripts/run_phase25_full.py",
    "scripts/run_phase25.ps1",
    "scripts/train_baseline.py",
    "scripts/commit_phase25.py",
    "scripts/generate_benchmark_report.py",
    # Reports
    "reports/baseline_benchmark_results.md",
    "reports/model_efficiency_report.md",
    "reports/baseline_error_analysis.md",
    "reports/phase2_findings.md",
    "reports/phase3_readiness.md",
    # Updated files
    "README.md",
    # Directories
    "outputs/benchmarks/",
]


def main() -> None:
    print("=" * 60)
    print("ARST Phase 2.5 — Git Commit")
    print("=" * 60)

    # Stage all changes
    result = subprocess.run(
        ["git", "add", "-A"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git add failed: {result.stderr}")
        sys.exit(1)
    print("✅ git add -A complete")

    # Show what's staged
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    print("\nStaged changes:")
    print(result.stdout)

    # Commit
    result = subprocess.run(
        ["git", "commit", "-m", COMMIT_MSG],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: git commit returned {result.returncode}")
        print(result.stderr)
        print("Try: git commit -m 'feat: complete phase 2 benchmark validation and analysis'")
    else:
        print(f"✅ Committed: {COMMIT_MSG}")
        print(result.stdout)


if __name__ == "__main__":
    main()
