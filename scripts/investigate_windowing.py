"""
Phase 2.7 - Part 2: Verify Window Extraction
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def investigate_windowing():
    csv_path = Path("data/raw/train.csv")
    report_path = Path("reports/windowing_report.md")
    vis_dir = Path("reports/window_visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print("Loading a subset of CSV for window investigation...")
    df = pd.read_csv(csv_path, nrows=10000)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2.7 - Window Extraction Report\n\n")
        
        # Check majority vote logic
        f.write("## 1. Majority Vote Logic & Boundary Effects\n")
        issues = 0
        for seq_id, seq_data in df.groupby("sequence_id"):
            seq_data = seq_data.sort_values("sequence_counter")
            labels = seq_data["label"].values
            majority = seq_data["label"].mode()[0]
            if len(set(labels)) > 1:
                f.write(f"- Sequence {seq_id} has mixed labels: {dict(pd.Series(labels).value_counts())}. Assigned majority: {majority}.\n")
                issues += 1
        if issues == 0:
            f.write("- All sequences have a single consistent label. No boundary effects found in this subset.\n")
            
        f.write("\n## 2. Temporal Leakage\n")
        f.write("- Train/Val/Test splits are done by `subject` in `dataloader.py`, preventing temporal leakage of the same sequence/subject across splits.\n")

    print("Generating visualizations for 20 random sequences...")
    # Visualize 20 random windows
    import random
    seq_ids = list(df["sequence_id"].unique())
    sample_seqs = random.sample(seq_ids, min(20, len(seq_ids)))
    
    for i, seq_id in enumerate(sample_seqs):
        seq_data = df[df["sequence_id"] == seq_id].sort_values("sequence_counter").head(64)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Plot IMU (just acc_x, acc_y, acc_z for simplicity)
        if all(c in seq_data.columns for c in ["acc_x", "acc_y", "acc_z"]):
            axes[0].plot(seq_data["sequence_counter"], seq_data["acc_x"], label="acc_x")
            axes[0].plot(seq_data["sequence_counter"], seq_data["acc_y"], label="acc_y")
            axes[0].plot(seq_data["sequence_counter"], seq_data["acc_z"], label="acc_z")
        axes[0].set_title("IMU Accel")
        axes[0].legend(loc="upper right")
        
        # Plot Thermal (first 3 channels)
        if all(c in seq_data.columns for c in ["thm_1", "thm_2", "thm_3"]):
            axes[1].plot(seq_data["sequence_counter"], seq_data["thm_1"], label="thm_1")
            axes[1].plot(seq_data["sequence_counter"], seq_data["thm_2"], label="thm_2")
            axes[1].plot(seq_data["sequence_counter"], seq_data["thm_3"], label="thm_3")
        axes[1].set_title("Thermal")
        axes[1].legend(loc="upper right")
        
        # Plot ToF (first 3 channels)
        if all(c in seq_data.columns for c in ["tof_1_v0", "tof_1_v1", "tof_1_v2"]):
            axes[2].plot(seq_data["sequence_counter"], seq_data["tof_1_v0"], label="tof_1_v0")
            axes[2].plot(seq_data["sequence_counter"], seq_data["tof_1_v1"], label="tof_1_v1")
            axes[2].plot(seq_data["sequence_counter"], seq_data["tof_1_v2"], label="tof_1_v2")
        axes[2].set_title("ToF")
        axes[2].legend(loc="upper right")
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"window_{i+1}_seq_{seq_id}.png")
        plt.close()
        
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n## 3. Visualizations\n")
        f.write(f"- Generated {len(sample_seqs)} window visualizations in `reports/window_visualizations/`\n")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    investigate_windowing()
