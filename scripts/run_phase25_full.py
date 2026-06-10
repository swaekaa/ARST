"""
ARST Phase 2.5 — Full Baseline Benchmark Pipeline (Python driver).

Trains CNN, LSTM, and Transformer baselines sequentially, then generates
all benchmark reports and visualizations.

Usage (from repo root)::

    python scripts/run_phase25_full.py
    python scripts/run_phase25_full.py --models cnn lstm         # subset
    python scripts/run_phase25_full.py --skip-training           # reports only
    python scripts/run_phase25_full.py --models transformer      # one model

Design:
    - Delegates to train.py via subprocess so Hydra config merges correctly.
    - Streams train.py output live so epoch progress is visible.
    - Hard-fails on any training failure (non-zero exit code).
    - Runs generate_benchmark_report.py after all training is done.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase25_pipeline")

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd: list[str], cwd: Path, label: str) -> None:
    """Run a command, streaming output live, raising on failure."""
    logger.info("=" * 60)
    logger.info("Running: %s", " ".join(cmd))
    logger.info("=" * 60)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        logger.error("%s FAILED (exit code %d) after %.1fs", label, result.returncode, elapsed)
        sys.exit(result.returncode)

    logger.info("%s COMPLETE in %.1fs (%.1f min)", label, elapsed, elapsed / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="ARST Phase 2.5 benchmark pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn", "lstm", "transformer"],
        choices=["cnn", "lstm", "transformer"],
        help="Which models to train (default: all three)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only generate reports (all models must already be trained)",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip report generation (only train)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ARST Phase 2.5 — Baseline Benchmark Pipeline")
    logger.info("=" * 60)
    logger.info("Models to train: %s", args.models if not args.skip_training else "SKIPPED")
    logger.info("Repo root: %s", REPO_ROOT)

    python = sys.executable

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.skip_training:
        model_labels = {
            "cnn": "CNN Baseline",
            "lstm": "BiLSTM Baseline",
            "transformer": "Transformer Baseline",
        }

        for i, model_name in enumerate(args.models, 1):
            logger.info(
                "\n[%d/%d] Training %s...",
                i,
                len(args.models),
                model_labels.get(model_name, model_name),
            )
            run_command(
                [python, "train.py", f"model={model_name}"],
                cwd=REPO_ROOT,
                label=model_labels.get(model_name, model_name),
            )

        logger.info("\n✅ All baselines trained successfully!")

    # ── Report generation ─────────────────────────────────────────────────────
    if not args.skip_reports:
        logger.info("\n[Final] Generating benchmark reports and visualizations...")
        run_command(
            [python, "scripts/generate_benchmark_report.py"],
            cwd=REPO_ROOT,
            label="Benchmark report generation",
        )
        logger.info("\n✅ Benchmark reports generated!")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2.5 Complete!")
    logger.info("=" * 60)
    logger.info("Outputs:")
    logger.info("  outputs/benchmarks/benchmark_accuracy.png")
    logger.info("  outputs/benchmarks/benchmark_macro_f1.png")
    logger.info("  outputs/benchmarks/benchmark_vram.png")
    logger.info("  outputs/benchmarks/benchmark_training_time.png")
    logger.info("  outputs/benchmarks/confusion_matrices/cnn_confusion_matrix.png")
    logger.info("  outputs/benchmarks/confusion_matrices/lstm_confusion_matrix.png")
    logger.info("  outputs/benchmarks/confusion_matrices/transformer_confusion_matrix.png")
    logger.info("  reports/baseline_benchmark_results.md")
    logger.info("  reports/model_efficiency_report.md")
    logger.info("  reports/baseline_error_analysis.md")
    logger.info("  reports/phase2_findings.md")
    logger.info("  reports/phase3_readiness.md")
    logger.info("  README.md (updated)")


if __name__ == "__main__":
    main()
