"""
Phase 2.7 - Part 1: Verify Class Distributions
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from arst.data.dataloader import build_csv_loaders

def run_class_distribution_analysis():
    csv_path = Path("data/raw/train.csv")
    report_path = Path("reports/class_distribution_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # 1. Raw rows distribution
    print("Loading raw CSV...")
    df = pd.read_csv(csv_path, usecols=["label"])
    raw_counts = df["label"].value_counts().sort_index()
    raw_dist = raw_counts / len(df) * 100

    # 2. Windowed distribution
    print("Building DataLoaders to inspect windowed distribution...")
    train_loader, val_loader, test_loader, data_info = build_csv_loaders(
        csv_path=csv_path,
        window_size=64,
        batch_size=32,
        val_fraction=0.15,
        test_fraction=0.15,
        num_workers=0,
        seed=42
    )

    def get_loader_counts(loader):
        all_labels = []
        for batch in loader:
            all_labels.extend(batch["label"].numpy().tolist())
        counts = pd.Series(all_labels).value_counts().sort_index()
        dist = counts / len(all_labels) * 100 if len(all_labels) > 0 else counts
        return counts, dist

    print("Counting Train windows...")
    train_counts, train_dist = get_loader_counts(train_loader)
    
    print("Counting Val windows...")
    val_counts, val_dist = get_loader_counts(val_loader)
    
    print("Counting Test windows...")
    test_counts, test_dist = get_loader_counts(test_loader)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2.7 - Class Distribution Report\n\n")
        
        f.write("## 1. Raw Rows Distribution\n")
        f.write("| Class | Count | Percentage |\n|---|---|---|\n")
        for cls in raw_counts.index:
            f.write(f"| {cls} | {raw_counts[cls]} | {raw_dist[cls]:.2f}% |\n")
        
        f.write("\n## 2. Train Windows Distribution\n")
        f.write("| Class | Count | Percentage |\n|---|---|---|\n")
        for cls in train_counts.index:
            f.write(f"| {cls} | {train_counts.get(cls, 0)} | {train_dist.get(cls, 0):.2f}% |\n")

        f.write("\n## 3. Val Windows Distribution\n")
        f.write("| Class | Count | Percentage |\n|---|---|---|\n")
        for cls in val_counts.index:
            f.write(f"| {cls} | {val_counts.get(cls, 0)} | {val_dist.get(cls, 0):.2f}% |\n")

        f.write("\n## 4. Test Windows Distribution\n")
        f.write("| Class | Count | Percentage |\n|---|---|---|\n")
        for cls in test_counts.index:
            f.write(f"| {cls} | {test_counts.get(cls, 0)} | {test_dist.get(cls, 0):.2f}% |\n")

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_class_distribution_analysis()
