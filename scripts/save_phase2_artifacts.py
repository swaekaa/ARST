import os
import shutil
from pathlib import Path

def save_phase2_artifacts():
    base_dir = Path("results/phase2")
    models = ["mlp", "cnn", "lstm", "transformer"]
    
    # Create directories
    for model in models:
        (base_dir / model).mkdir(parents=True, exist_ok=True)
        
    outputs_dir = Path("outputs/evaluation")
    benchmarks_dir = Path("outputs/benchmarks/confusion_matrices")
    
    print("Created results/phase2/ directories.")
    print("Please manually copy your best run artifacts into these folders:")
    print("- test_metrics.json")
    print("- per_class_f1.csv")
    print("- confusion_matrix.npy / .png")
    print("- training curves")
    print("- best checkpoint information (source run IDs)")

if __name__ == "__main__":
    save_phase2_artifacts()
