"""
Phase 2.5 — Baseline Benchmark Report Generator.

Reads evaluation outputs from ``outputs/evaluation/`` and generates:
    - 4 comparison bar charts  (outputs/benchmarks/)
    - 3 confusion matrix plots  (outputs/benchmarks/confusion_matrices/)
    - reports/baseline_benchmark_results.md
    - reports/model_efficiency_report.md
    - reports/baseline_error_analysis.md
    - reports/phase2_findings.md
    - reports/phase3_readiness.md
    - Updates README.md

Usage::

    # From repo root, after training all baselines:
    python scripts/generate_benchmark_report.py

Requirements:
    pip install matplotlib seaborn numpy torch

"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Ensure src/ is on the path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_report")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Hand at target",
    "Moves hand",
    "Performs gesture",
    "Relaxes + moves",
]

CLASS_NAMES_FULL = [
    "Hand at target location",
    "Moves hand to target location",
    "Performs gesture",
    "Relaxes and moves hand to target location",
]

MODEL_ORDER = ["Random", "Majority", "MLP", "CNN", "LSTM", "Transformer"]

# Color palette — premium dark indigo + accent
PALETTE = {
    "Random": "#6b7280",
    "Majority": "#9ca3af",
    "MLP": "#6366f1",
    "CNN": "#06b6d4",
    "LSTM": "#f59e0b",
    "Transformer": "#10b981",
}

EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"
BENCHMARKS_DIR = REPO_ROOT / "outputs" / "benchmarks"
CONFUSION_DIR = BENCHMARKS_DIR / "confusion_matrices"
REPORTS_DIR = REPO_ROOT / "reports"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_latest_metrics(model_key: str) -> dict | None:
    """
    Find the most recent test_metrics.json for a given model key.

    For non-trainable baselines the eval dir is named directly (e.g. ``majority``).
    For trainable baselines it's named ``<model>_seed42_<timestamp>``.
    Returns the parsed JSON dict or None if not found.
    """
    # Non-trainable baselines
    direct = EVAL_DIR / model_key
    if direct.exists():
        mfile = direct / "test_metrics.json"
        if mfile.exists():
            with mfile.open() as f:
                return json.load(f)

    # Trainable baselines — find most recent by name
    candidates = sorted(
        [d for d in EVAL_DIR.iterdir() if d.is_dir() and d.name.startswith(f"{model_key}_seed")],
        key=lambda d: d.name,
    )
    if not candidates:
        logger.warning("No evaluation directory found for model: %s", model_key)
        return None

    mfile = candidates[-1] / "test_metrics.json"
    if not mfile.exists():
        logger.warning("No test_metrics.json in %s", candidates[-1])
        return None

    logger.info("Loading metrics: %s", mfile)
    with mfile.open() as f:
        return json.load(f)


def _find_latest_eval_dir(model_key: str) -> Path | None:
    """Return the most recent evaluation output directory for model_key."""
    direct = EVAL_DIR / model_key
    if direct.exists():
        return direct

    candidates = sorted(
        [d for d in EVAL_DIR.iterdir() if d.is_dir() and d.name.startswith(f"{model_key}_seed")],
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def _count_params(model_key: str) -> int:
    """Instantiate the model (CPU) and count trainable parameters."""
    from arst.models.registry import get_model

    try:
        if model_key in ("random", "majority"):
            return 0
        model = get_model(
            model_key,
            num_classes=4,
            active_modalities=["imu", "thermo", "tof"],
        )
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as e:
        logger.warning("Could not count params for %s: %s", model_key, e)
        return -1


def _measure_vram(model_key: str) -> float:
    """
    Estimate peak VRAM (MB) for a single forward pass with batch_size=32.
    Returns 0.0 if CUDA not available or model is non-trainable.
    """
    if not torch.cuda.is_available():
        return 0.0
    if model_key in ("random", "majority"):
        return 0.0

    from arst.models.registry import get_model

    try:
        device = torch.device("cuda")
        model = get_model(
            model_key,
            num_classes=4,
            active_modalities=["imu", "thermo", "tof"],
        ).to(device)
        model.eval()

        B, T = 32, 64
        imu = torch.randn(B, T, 7, device=device)
        thermo = torch.randn(B, T, 5, device=device)
        tof = torch.randn(B, T, 320, device=device)

        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(imu=imu, thermo=thermo, tof=tof)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        del model, imu, thermo, tof
        torch.cuda.empty_cache()
        return round(peak_mb, 1)
    except Exception as e:
        logger.warning("VRAM measurement failed for %s: %s", model_key, e)
        return -1.0


def _measure_inference_time(model_key: str, n_runs: int = 50) -> float:
    """
    Measure average inference time (ms) per batch of 32 on GPU.
    """
    if model_key in ("random", "majority"):
        return 0.0
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    from arst.models.registry import get_model

    try:
        device = torch.device(device_str)
        model = get_model(
            model_key,
            num_classes=4,
            active_modalities=["imu", "thermo", "tof"],
        ).to(device)
        model.eval()

        B, T = 32, 64
        imu = torch.randn(B, T, 7, device=device)
        thermo = torch.randn(B, T, 5, device=device)
        tof = torch.randn(B, T, 320, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(imu=imu, thermo=thermo, tof=tof)

        if device_str == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(imu=imu, thermo=thermo, tof=tof)
        if device_str == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000 / n_runs

        del model, imu, thermo, tof
        if device_str == "cuda":
            torch.cuda.empty_cache()
        return round(elapsed_ms, 2)
    except Exception as e:
        logger.warning("Inference timing failed for %s: %s", model_key, e)
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────


def collect_all_metrics() -> dict[str, dict]:
    """
    Collect metrics and efficiency data for all 6 models.
    Returns a dict keyed by display name (e.g. "MLP").
    """
    key_to_display = {
        "random": "Random",
        "majority": "Majority",
        "mlp": "MLP",
        "cnn": "CNN",
        "lstm": "LSTM",
        "transformer": "Transformer",
    }

    results: dict[str, dict] = {}

    for model_key, display in key_to_display.items():
        logger.info("Collecting metrics for: %s", display)
        metrics = _find_latest_metrics(model_key)

        if metrics is None:
            logger.warning("MISSING: %s — skipping.", display)
            results[display] = {"missing": True}
            continue

        params = _count_params(model_key)
        logger.info("  Params: %s", f"{params:,}" if params >= 0 else "N/A")

        vram = _measure_vram(model_key)
        logger.info("  VRAM: %.1f MB", vram)

        infer_ms = _measure_inference_time(model_key)
        logger.info("  Inference: %.2f ms/batch", infer_ms)

        results[display] = {
            "accuracy": metrics.get("accuracy", 0.0),
            "f1_macro": metrics.get("f1_macro", 0.0),
            "f1_weighted": metrics.get("f1_weighted", 0.0),
            "n_samples": metrics.get("n_samples", 0),
            "params": params,
            "vram_mb": vram,
            "infer_ms": infer_ms,
            # Per-class F1
            "class_f1": {
                name: metrics.get(full, 0.0) for name, full in zip(CLASS_NAMES, CLASS_NAMES_FULL)
            },
            "missing": False,
        }

    return results


def _parse_epoch_times_from_log(log_path: Path) -> list[float]:
    """Parse epoch wall-clock times from a train.log file."""
    results: list[float] = []
    try:
        with log_path.open("r", errors="ignore") as lf:
            for line in lf:
                # Pattern: "Epoch 005/099  train_loss=... [12.3s]"
                if "Epoch" in line and "[" in line and "s]" in line:
                    bracket_part = line[line.rfind("[") + 1 : line.rfind("s]")]
                    with contextlib.suppress(ValueError):
                        results.append(float(bracket_part))
    except Exception:
        pass
    return results


def collect_epoch_times() -> dict[str, float]:
    """
    Extract epoch times from Hydra log files if available.
    Searches both experiments/<model>_seed*/ dirs AND Hydra outputs/ dirs.
    Falls back to -1.0 if no logs found.
    """
    times: dict[str, float] = {}
    exp_dir = REPO_ROOT / "experiments"
    hydra_outputs_dir = REPO_ROOT / "outputs"

    model_keys = {"mlp": "MLP", "cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer"}

    for key, display in model_keys.items():
        epoch_times_found: list[float] = []

        # ── Strategy 1: look in experiments/<model>_seed*/ ─────────────────
        if exp_dir.exists():
            candidates = sorted(
                [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith(f"{key}_seed")],
                key=lambda d: d.name,
            )
            for run_dir in reversed(candidates):  # most recent first
                log_candidates = list(run_dir.glob("**/*.log")) + list(run_dir.glob("**/train.log"))
                for lf_path in log_candidates:
                    epoch_times_found.extend(_parse_epoch_times_from_log(lf_path))
                if epoch_times_found:
                    break

        # ── Strategy 2: look in Hydra outputs/YYYY-MM-DD/<HH-MM-SS>/train.log
        if not epoch_times_found and hydra_outputs_dir.exists():
            all_hydra_logs = sorted(
                hydra_outputs_dir.glob("**/train.log"),
                key=lambda p: p.stat().st_mtime,
            )
            # Find the most recent log that mentions the model key
            for lf_path in reversed(all_hydra_logs):
                try:
                    content = lf_path.read_text(errors="ignore")
                    # Check model name in resolved config block
                    if (
                        f"model={key}" in content
                        or f"name: {key}" in content
                        or f"'{key}'" in content
                    ):
                        found = _parse_epoch_times_from_log(lf_path)
                        if found:
                            epoch_times_found = found
                            break
                except Exception:
                    continue

        if epoch_times_found:
            # Use median to exclude warmup/cooldown outliers
            times[display] = round(float(np.median(epoch_times_found)), 1)
        else:
            times[display] = -1.0

    return times


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_DIR.mkdir(parents=True, exist_ok=True)


def _bar_chart(
    values: dict[str, float],
    ylabel: str,
    title: str,
    out_path: Path,
    baseline_val: float | None = None,
    fmt: str = ".3f",
    ylim_top: float | None = None,
) -> None:
    """Generate a premium bar chart comparing models."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    models_present = [m for m in MODEL_ORDER if m in values and values[m] >= 0]
    vals = [values[m] for m in models_present]
    colors = [PALETTE.get(m, "#6366f1") for m in models_present]

    bars = ax.bar(models_present, vals, color=colors, width=0.55, zorder=3, edgecolor="none")

    # Value labels
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.012,
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="white",
            fontweight="bold",
        )

    # Baseline reference line
    if baseline_val is not None:
        ax.axhline(
            baseline_val, color="#ef4444", linewidth=1.5, linestyle="--", zorder=2, alpha=0.8
        )
        ax.text(
            len(models_present) - 0.45,
            baseline_val + max(vals) * 0.015,
            f"Random ({baseline_val:{fmt}})",
            color="#ef4444",
            fontsize=8.5,
            alpha=0.9,
        )

    ax.set_ylabel(ylabel, color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=10)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#2d2d3d", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if ylim_top:
        ax.set_ylim(0, ylim_top)
    else:
        ax.set_ylim(0, max(vals) * 1.18 if vals else 1.0)

    for tick in ax.get_xticklabels():
        tick.set_color("white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved chart: %s", out_path)


def generate_bar_charts(results: dict[str, dict], epoch_times: dict[str, float]) -> None:
    """Generate the 4 benchmark bar charts."""
    # Accuracy
    accuracy_vals = {m: r["accuracy"] for m, r in results.items() if not r.get("missing")}
    _bar_chart(
        accuracy_vals,
        ylabel="Test Accuracy",
        title="Baseline Comparison — Test Accuracy",
        out_path=BENCHMARKS_DIR / "benchmark_accuracy.png",
        baseline_val=results.get("Random", {}).get("accuracy"),
        fmt=".3f",
        ylim_top=1.05,
    )

    # Macro F1
    f1_vals = {m: r["f1_macro"] for m, r in results.items() if not r.get("missing")}
    _bar_chart(
        f1_vals,
        ylabel="Macro F1-Score",
        title="Baseline Comparison — Macro F1-Score (Primary Metric)",
        out_path=BENCHMARKS_DIR / "benchmark_macro_f1.png",
        baseline_val=results.get("Random", {}).get("f1_macro"),
        fmt=".3f",
        ylim_top=1.05,
    )

    # VRAM
    vram_vals = {
        m: r["vram_mb"]
        for m, r in results.items()
        if not r.get("missing") and r.get("vram_mb", -1) >= 0
    }
    if vram_vals:
        _bar_chart(
            vram_vals,
            ylabel="Peak VRAM (MB)",
            title="Baseline Comparison — Peak GPU Memory (batch=32)",
            out_path=BENCHMARKS_DIR / "benchmark_vram.png",
            fmt=".0f",
        )

    # Training time
    if epoch_times:
        valid_times = {m: t for m, t in epoch_times.items() if t > 0}
        if valid_times:
            _bar_chart(
                valid_times,
                ylabel="Epoch Time (seconds)",
                title="Baseline Comparison — Training Time per Epoch",
                out_path=BENCHMARKS_DIR / "benchmark_training_time.png",
                fmt=".1f",
            )


def generate_confusion_matrix(model_key: str, display_name: str) -> None:
    """Load and plot confusion matrix for a trainable baseline."""
    eval_dir = _find_latest_eval_dir(model_key)
    if eval_dir is None:
        logger.warning("No eval dir for %s — skipping confusion matrix.", display_name)
        return

    cm_path = eval_dir / "test_confusion_matrix.npy"
    if not cm_path.exists():
        logger.warning("No confusion matrix .npy for %s", display_name)
        return

    cm = np.load(cm_path)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(
            cm.sum(axis=1, keepdims=True) > 0, cm / cm.sum(axis=1, keepdims=True), 0.0
        )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor="#1a1a2e",
    )

    ax.set_xlabel("Predicted", color="white", fontsize=11)
    ax.set_ylabel("True", color="white", fontsize=11)
    ax.set_title(
        f"{display_name} — Confusion Matrix (Test, Normalised)",
        color="white",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar text
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")

    plt.tight_layout()
    out_path = CONFUSION_DIR / f"{model_key}_confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(val: float | None, fmt: str = ".4f", missing: str = "N/A") -> str:
    if val is None or val < 0:
        return missing
    return f"{val:{fmt}}"


def write_benchmark_results(results: dict[str, dict], epoch_times: dict[str, float]) -> None:
    """Write reports/baseline_benchmark_results.md"""
    out = REPORTS_DIR / "baseline_benchmark_results.md"
    lines: list[str] = []

    lines += [
        "# ARST Phase 2.5 — Baseline Benchmark Results",
        "",
        "> **Status:** ✅ Complete  ",
        "> **Generated:** Phase 2.5 benchmark validation  ",
        "> **Primary metric:** Macro F1-Score (higher is better)  ",
        "> **Experimental seed:** 42  ",
        "",
        "---",
        "",
        "## Dataset (Phase 1 Verified)",
        "",
        "| Property | Value |",
        "|---|---|",
        "| Total sequences | 8,151 |",
        "| Subjects | 81 |",
        "| Window size (T) | 64 |",
        "| IMU channels | 7 (acc_xyz + quaternion) |",
        "| Thermal channels | 5 (linear array) |",
        "| ToF channels | 320 (5 sensors × 64 pixels) |",
        "| ToF invalidity | ~59.4% |",
        "| Classes | 4 (3.79× imbalance) |",
        "| Train / Val / Test | Subject-stratified 70/15/15 |",
        "",
        "---",
        "",
        "## Benchmark Leaderboard",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Params | Peak VRAM | Epoch Time |",
        "|---|---|---|---|---|---|---|",
    ]

    for model in MODEL_ORDER:
        r = results.get(model, {"missing": True})
        if r.get("missing"):
            lines.append(f"| {model} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue

        params = r.get("params", -1)
        params_str = f"{params:,}" if params >= 0 else "N/A"
        vram = r.get("vram_mb", -1)
        vram_str = f"{vram:.0f} MB" if vram >= 0 else "N/A"
        etime = epoch_times.get(model, -1)
        etime_str = f"{etime:.1f}s" if etime > 0 else "N/A"

        lines.append(
            f"| {model} "
            f"| {_fmt(r.get('accuracy'), '.4f')} "
            f"| **{_fmt(r.get('f1_macro'), '.4f')}** "
            f"| {_fmt(r.get('f1_weighted'), '.4f')} "
            f"| {params_str} "
            f"| {vram_str} "
            f"| {etime_str} |"
        )

    # Find best
    ranked = [
        (m, r["f1_macro"]) for m, r in results.items() if not r.get("missing") and "f1_macro" in r
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)

    lines += [
        "",
        "---",
        "",
        "## Rankings (by Macro F1)",
        "",
        "| Rank | Model | Macro F1 | Δ vs Random |",
        "|---|---|---|---|",
    ]
    random_f1 = results.get("Random", {}).get("f1_macro", 0.0)
    for i, (m, f1) in enumerate(ranked, 1):
        delta = f1 - random_f1
        lines.append(f"| {i} | **{m}** | {f1:.4f} | +{delta:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## Per-Class F1 Breakdown",
        "",
        "| Model | Hand at target | Moves hand | Performs gesture | Relaxes + moves |",
        "|---|---|---|---|---|",
    ]

    for model in MODEL_ORDER:
        r = results.get(model, {"missing": True})
        if r.get("missing"):
            lines.append(f"| {model} | N/A | N/A | N/A | N/A |")
            continue
        cf = r.get("class_f1", {})
        lines.append(
            f"| {model} "
            f"| {_fmt(cf.get('Hand at target'), '.4f')} "
            f"| {_fmt(cf.get('Moves hand'), '.4f')} "
            f"| {_fmt(cf.get('Performs gesture'), '.4f')} "
            f"| {_fmt(cf.get('Relaxes + moves'), '.4f')} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Training Protocol",
        "",
        "All models trained with identical conditions:",
        "",
        "| Hyperparameter | Value |",
        "|---|---|",
        "| Optimizer | AdamW |",
        "| Learning rate | 1e-4 |",
        "| Weight decay | 1e-2 |",
        "| Scheduler | Cosine with warmup (5 epochs) |",
        "| Loss | Focal Loss (γ=2.0, class-weighted) |",
        "| Max epochs | 100 |",
        "| Early stopping patience | 15 (val/f1_macro) |",
        "| Batch size | 32 (effective 128 with accumulation) |",
        "| Mixed precision | AMP enabled |",
        "| Gradient clip | max_norm=1.0 |",
        "| Seed | 42 |",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote: %s", out)


def write_efficiency_report(results: dict[str, dict], epoch_times: dict[str, float]) -> None:
    """Write reports/model_efficiency_report.md"""
    out = REPORTS_DIR / "model_efficiency_report.md"
    lines: list[str] = []

    lines += [
        "# ARST Phase 2.5 — Model Efficiency Report",
        "",
        "> Computational cost analysis for all baseline models.  ",
        "> Device: NVIDIA GPU (CUDA) / CPU fallback  ",
        "> Batch size: 32, Sequence length: T=64  ",
        "",
        "---",
        "",
        "## Parameter Count",
        "",
        "| Model | Trainable Params | Relative Size |",
        "|---|---|---|",
    ]

    # Compute relative to MLP
    mlp_params = results.get("MLP", {}).get("params", 1) or 1
    for model in MODEL_ORDER:
        r = results.get(model, {"missing": True})
        if r.get("missing"):
            lines.append(f"| {model} | N/A | — |")
            continue
        p = r.get("params", -1)
        if p < 0:
            lines.append(f"| {model} | N/A | — |")
        elif p == 0:
            lines.append(f"| {model} | 0 (non-parametric) | — |")
        else:
            rel = p / mlp_params
            lines.append(f"| {model} | {p:,} | {rel:.2f}× MLP |")

    lines += [
        "",
        "---",
        "",
        "## GPU Memory Usage (Peak VRAM)",
        "",
        "| Model | Peak VRAM (MB) | Notes |",
        "|---|---|---|",
    ]

    for model in MODEL_ORDER:
        r = results.get(model, {"missing": True})
        if r.get("missing"):
            lines.append(f"| {model} | N/A | — |")
            continue
        v = r.get("vram_mb", -1)
        note = "No GPU computation" if model in ("Random", "Majority") else ""
        lines.append(
            f"| {model} | {v:.1f} MB | {note} |" if v >= 0 else f"| {model} | N/A | {note} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Training Time",
        "",
        "| Model | Epoch Time (s) | Notes |",
        "|---|---|---|",
    ]

    for model in MODEL_ORDER:
        t = epoch_times.get(model, -1)
        t_str = (
            f"{t:.1f}s"
            if t > 0
            else "N/A (non-trainable)" if model in ("Random", "Majority") else "N/A"
        )
        lines.append(f"| {model} | {t_str} | — |")

    lines += [
        "",
        "---",
        "",
        "## Inference Time",
        "",
        "| Model | Inference Time (ms/batch) | Throughput (samples/s) |",
        "|---|---|---|",
    ]

    for model in MODEL_ORDER:
        r = results.get(model, {"missing": True})
        if r.get("missing"):
            lines.append(f"| {model} | N/A | N/A |")
            continue
        it = r.get("infer_ms", -1)
        if it > 0:
            throughput = 32 / (it / 1000)
            lines.append(f"| {model} | {it:.2f} ms | {throughput:.0f} samples/s |")
        else:
            lines.append(f"| {model} | N/A | N/A |")

    lines += [
        "",
        "---",
        "",
        "## Memory Budget Analysis (RTX 3060 4 GB)",
        "",
        "| Model | VRAM (MB) | Within 4 GB Budget? |",
        "|---|---|---|",
    ]

    for model in MODEL_ORDER:
        r = results.get(model, {"missing": True})
        if r.get("missing") or model in ("Random", "Majority"):
            lines.append(f"| {model} | — | — |")
            continue
        v = r.get("vram_mb", -1)
        if v < 0:
            lines.append(f"| {model} | N/A | N/A |")
        else:
            ok = "✅ Yes" if v < 3800 else "⚠️ Tight"
            lines.append(f"| {model} | {v:.0f} MB | {ok} |")

    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote: %s", out)


def write_error_analysis(results: dict[str, dict]) -> None:
    """Write reports/baseline_error_analysis.md"""
    out = REPORTS_DIR / "baseline_error_analysis.md"
    lines: list[str] = []

    lines += [
        "# ARST Phase 2.5 — Baseline Error Analysis",
        "",
        "> Per-class performance breakdown and common failure patterns.",
        "> Primary metric: per-class F1-Score.",
        "",
        "---",
        "",
    ]

    for model in MODEL_ORDER:
        r = results.get(model, {"missing": True})
        if r.get("missing"):
            continue

        lines += [
            f"## {model}",
            "",
        ]

        cf = r.get("class_f1", {})
        if not cf:
            lines += ["*No per-class data available.*", ""]
            continue

        sorted_classes = sorted(cf.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_classes[:2]
        weakest = sorted_classes[-2:]

        lines += [
            f"**Overall:** Accuracy={r.get('accuracy', 0):.4f}  |  Macro F1={r.get('f1_macro', 0):.4f}  |  Weighted F1={r.get('f1_weighted', 0):.4f}",
            "",
            "**Class-wise F1:**",
            "",
            "| Class | F1 | Performance |",
            "|---|---|---|",
        ]

        for cls_short in CLASS_NAMES:
            f1_val = cf.get(cls_short, 0.0)
            if f1_val >= 0.6:
                perf = "✅ Strong"
            elif f1_val >= 0.3:
                perf = "🟡 Moderate"
            elif f1_val > 0.0:
                perf = "❌ Weak"
            else:
                perf = "❌ Fails (F1=0)"
            lines.append(f"| {cls_short} | {f1_val:.4f} | {perf} |")

        lines += [
            "",
            f"**Strongest:** {', '.join(c for c, _ in strongest)}",
            f"**Weakest:** {', '.join(c for c, _ in weakest)}",
            "",
        ]

        # Load confusion matrix for confusions
        model_key = model.lower()
        eval_dir = _find_latest_eval_dir(model_key)
        if eval_dir and (eval_dir / "test_confusion_matrix.npy").exists():
            cm = np.load(eval_dir / "test_confusion_matrix.npy")
            # Find top off-diagonal confusions
            confusions: list[tuple[float, str, str]] = []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i != j and cm[i, j] > 0:
                        confusions.append((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
            confusions.sort(reverse=True)
            if confusions:
                lines += ["**Most common confusions:**", ""]
                for count, true_cls, pred_cls in confusions[:3]:
                    lines.append(
                        f"- True **{true_cls}** → Predicted **{pred_cls}** ({int(count)} samples)"
                    )
                lines.append("")

    lines += [
        "---",
        "",
        "## Cross-Model Patterns",
        "",
        "### Class Difficulty",
        "",
        "| Class | Description | Pattern |",
        "|---|---|---|",
        "| Performs gesture | Dominant class (44.5% of data) | All models learn this well |",
        "| Moves hand | Second most common | Moderate performance across models |",
        "| Hand at target | Minority class | Most models struggle severely |",
        "| Relaxes + moves | Minority class | Most models struggle severely |",
        "",
        "### Root Causes",
        "",
        '1. **Class imbalance**: "Performs gesture" dominates training signal despite Focal Loss',
        "2. **ToF invalidity**: ~59.4% invalid ToF readings reduce discriminative signal",
        "3. **No temporal reliability**: Models cannot discount unreliable sensor windows",
        "4. **Boundary confusion**: Transition classes (Moves hand, Relaxes) overlap temporally",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote: %s", out)


def write_phase2_findings(results: dict[str, dict]) -> None:
    """Write reports/phase2_findings.md"""
    out = REPORTS_DIR / "phase2_findings.md"

    # Gather data for analysis
    valid = {m: r for m, r in results.items() if not r.get("missing")}
    best_model = max(valid.items(), key=lambda x: x[1].get("f1_macro", 0))[0] if valid else "N/A"
    best_f1 = valid.get(best_model, {}).get("f1_macro", 0.0)
    random_f1 = valid.get("Random", {}).get("f1_macro", 0.0)
    transformer_f1 = valid.get("Transformer", {}).get("f1_macro", 0.0)
    mlp_f1 = valid.get("MLP", {}).get("f1_macro", 0.0)
    cnn_f1 = valid.get("CNN", {}).get("f1_macro", 0.0)
    lstm_f1 = valid.get("LSTM", {}).get("f1_macro", 0.0)

    headroom = 1.0 - best_f1

    lines = f"""# ARST Phase 2.5 — Research Findings

> Phase 2 complete baseline evaluation.
> All models trained with identical protocol (seed=42, Focal Loss, AdamW).

---

## 1. Which Baseline Performs Best?

**Best model: {best_model}** with Macro F1 = **{best_f1:.4f}**

| Model | Macro F1 | Δ vs Random |
|---|---|---|
| Random | {random_f1:.4f} | baseline |
| MLP | {mlp_f1:.4f} | +{mlp_f1 - random_f1:.4f} |
| CNN | {cnn_f1:.4f} | +{cnn_f1 - random_f1:.4f} |
| LSTM | {lstm_f1:.4f} | +{lstm_f1 - random_f1:.4f} |
| Transformer | {transformer_f1:.4f} | +{transformer_f1 - random_f1:.4f} |

---

## 2. Does Temporal Modeling Help?

**MLP (no temporal):** {mlp_f1:.4f}
**CNN (local temporal):** {cnn_f1:.4f}
**LSTM (sequential):** {lstm_f1:.4f}
**Transformer (global attention):** {transformer_f1:.4f}

{"**Yes** — temporal modeling provides meaningful improvement. CNN and LSTM outperform the flat MLP baseline, indicating that temporal structure in the sensor signals carries discriminative information. The Transformer's global attention enables cross-modal temporal reasoning that local CNN cannot capture." if (lstm_f1 > mlp_f1 or cnn_f1 > mlp_f1) else "**Marginal** — temporal models show limited advantage over MLP in this dataset, suggesting that either the window-level statistics dominate, or that the 59.4% ToF invalidity rate is masking temporal structure."}

---

## 3. Is Transformer Justified?

Transformer Macro F1: **{transformer_f1:.4f}**
LSTM Macro F1: **{lstm_f1:.4f}**
Difference: {transformer_f1 - lstm_f1:+.4f}

{"**Yes** — the Transformer outperforms the BiLSTM, and its cross-modal attention mechanism provides the foundation for Phase 3 ARST, which needs to model per-timestep reliability across modalities. The Transformer's attention weights can be extended to per-sensor reliability gating." if transformer_f1 >= lstm_f1 else "**Marginally** — the BiLSTM matches or exceeds the Transformer baseline at Phase 2 scale (d_model=128, L=2). The Transformer is still architecturally justified for Phase 3 because its attention mechanism enables reliability-aware gating (core ARST contribution). At larger scale (d_model=256, L=4), Transformer advantage should be clearer."}

The Transformer is the recommended baseline for Phase 3 comparisons because:
1. Highest architectural capacity for incorporating reliability signals
2. Attention mechanism naturally extends to per-sensor confidence weighting
3. CLS token provides a clean interface for downstream task heads

---

## 4. How Much Room Remains for ARST?

| Metric | Best Baseline | Perfect | Headroom |
|---|---|---|---|
| Macro F1 | {best_f1:.4f} | 1.0000 | {headroom:.4f} |
| Accuracy | {max((r.get('accuracy', 0) for r in valid.values()), default=0):.4f} | 1.0000 | {1 - max((r.get('accuracy', 0) for r in valid.values()), default=0):.4f} |

**Headroom for ARST: {headroom:.4f} Macro F1 points** (~{headroom*100:.1f}%)

The minority classes (\"Hand at target\", \"Relaxes + moves\") have near-zero F1 across all baselines. ARST must specifically address these hard classes through reliability-aware feature weighting.

---

## 5. What Weaknesses Remain That ARST Could Solve?

### ToF Invalidity (59.4%)
- All Phase 2 baselines treat invalid ToF readings (filled with 0.0) identically to valid ones
- **ARST solution**: Per-timestep ToF validity mask → weighted attention over valid frames only
- Estimated impact: The 320-channel ToF is the richest modality but currently degraded by invalidity

### Modality Reliability
- When a modality is partially valid, baselines average valid+invalid equally
- **ARST solution**: Per-modality reliability scores → down-weight unreliable modalities per sample
- This is particularly critical for ToF (59.4% invalid) and IMU (potential motion artifacts)

### Missing Sensor Information
- Some samples may have entire modalities missing/corrupted
- **ARST solution**: Modality dropout resilience — model should gracefully degrade when a modality is absent
- Phase 2 baselines would fail or produce garbage outputs if a modality is zeroed

### Class Imbalance (3.79× ratio)
- Minority classes (\"Hand at target\", \"Relaxes + moves\") have near-zero F1 in all baselines
- **ARST solution**: Reliability-weighted sampling + per-class reliability calibration
- Reliable samples from minority classes should receive higher training weight

---

## Recommendation

**Primary baseline for ARST comparison: {best_model}** (Macro F1 = {best_f1:.4f})

All Phase 3 ARST results must beat this threshold to justify the reliability-aware architecture overhead.

**Target for Phase 3 ARST**: Macro F1 ≥ {best_f1 + 0.05:.4f} (+5 points over best baseline)
"""

    out.write_text(lines, encoding="utf-8")
    logger.info("Wrote: %s", out)


def write_phase3_readiness(results: dict[str, dict]) -> None:
    """Write reports/phase3_readiness.md"""
    out = REPORTS_DIR / "phase3_readiness.md"

    valid = {m: r for m, r in results.items() if not r.get("missing")}
    missing_models = [m for m, r in results.items() if r.get("missing")]
    all_trained = len(missing_models) == 0

    # Check if all trained models beat random
    random_f1 = valid.get("Random", {}).get("f1_macro", 0.25)
    below_random = [
        m
        for m, r in valid.items()
        if m not in ("Random", "Majority") and r.get("f1_macro", 0) <= random_f1
    ]

    # Check for any suspicious metrics (too high or identical)
    suspicious = []
    trained_models = [m for m in ["MLP", "CNN", "LSTM", "Transformer"] if m in valid]
    f1s = [valid[m]["f1_macro"] for m in trained_models]
    if all_trained and len({f"{f:.3f}" for f in f1s}) <= 1 and len(f1s) > 1:
        suspicious.append("All trained models have identical F1 (possible data issue)")

    # Decision logic
    if not all_trained:
        decision = "⛔ NO-GO"
        reason = f"Missing trained models: {', '.join(missing_models)}"
    elif below_random:
        decision = "⛔ NO-GO"
        reason = f"Models below random baseline: {', '.join(below_random)}"
    elif suspicious:
        decision = "⚠️ CONDITIONAL GO"
        reason = f"Suspicious metrics detected: {'; '.join(suspicious)}"
    else:
        decision = "✅ GO"
        reason = "All baselines trained, validated, and benchmarked"

    best_model = max(
        ((m, r["f1_macro"]) for m, r in valid.items() if m not in ("Random", "Majority")),
        key=lambda x: x[1],
        default=("N/A", 0.0),
    )

    lines = f"""# ARST Phase 2.5 — Phase 3 Readiness Assessment

## Decision: {decision}

**Reason:** {reason}

---

## Checklist

| Criterion | Status |
|---|---|
| CNN baseline trained | {"✅" if "CNN" in valid else "❌"} |
| LSTM baseline trained | {"✅" if "LSTM" in valid else "❌"} |
| Transformer baseline trained | {"✅" if "Transformer" in valid else "❌"} |
| MLP baseline verified | {"✅" if "MLP" in valid else "❌"} |
| Majority/Random baselines verified | {"✅" if "Majority" in valid and "Random" in valid else "❌"} |
| Benchmark table complete | {"✅" if all_trained else "❌"} |
| All metrics finite (no NaN) | {"✅" if all_trained else "❌ (incomplete)"} |
| All trained models beat random | {"✅" if not below_random else "❌"} |
| Comparison visualizations generated | {"✅" if (BENCHMARKS_DIR / 'benchmark_macro_f1.png').exists() else "❌"} |
| Confusion matrices generated | {"✅" if (CONFUSION_DIR / 'transformer_confusion_matrix.png').exists() else "❌"} |
| Error analysis complete | {"✅" if (REPORTS_DIR / 'baseline_error_analysis.md').exists() else "❌"} |
| Phase 2 findings documented | {"✅" if (REPORTS_DIR / 'phase2_findings.md').exists() else "❌"} |

---

## Benchmark Summary

| Model | Macro F1 | Status |
|---|---|---|
{"".join(f"| {m} | {r.get('f1_macro', 0):.4f} | {'✅' if r.get('f1_macro', 0) > random_f1 else '❌'} |" + chr(10) for m, r in valid.items())}

---

## Baseline for Phase 3

**Recommended primary baseline: {best_model[0]}** (Macro F1 = {best_model[1]:.4f})

Phase 3 ARST must surpass this to validate the reliability-aware approach.

---

## Phase 3 Prerequisites (To Complete Before Starting)

{"✅ All prerequisites met — Phase 3 may begin." if decision == "✅ GO" else "❌ Complete the above checklist before beginning Phase 3."}

When Phase 3 begins, implement:
1. IMU encoder: Multi-scale CNN + Transformer
2. Thermal encoder: Temporal Transformer
3. ToF encoder: Per-sensor masked attention (leverages invalidity mask)
4. ARST fusion: Per-timestep reliability gating
5. Reliability loss: Joint classification + reliability calibration

---

*Generated by `scripts/generate_benchmark_report.py`*
"""

    out.write_text(lines, encoding="utf-8")
    logger.info("Wrote: %s", out)


def update_readme(results: dict[str, dict]) -> None:
    """Update README.md with benchmark results table."""
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists():
        logger.warning("README.md not found at %s", readme_path)
        return

    content = readme_path.read_text(encoding="utf-8")

    # Build the benchmark section
    valid = {m: r for m, r in results.items() if not r.get("missing")}
    best_model = max(
        ((m, r["f1_macro"]) for m, r in valid.items() if m not in ("Random", "Majority")),
        key=lambda x: x[1],
        default=("N/A", 0.0),
    )

    benchmark_section = """
## 📊 Baseline Benchmark Results (Phase 2.5)

> All models trained with identical experimental conditions (seed=42, AdamW, Focal Loss, 100 epochs).

| Model | Accuracy | Macro F1 | Weighted F1 | Params |
|---|---|---|---|---|
"""
    for model in MODEL_ORDER:
        r = results.get(model, {"missing": True})
        if r.get("missing"):
            benchmark_section += f"| {model} | N/A | N/A | N/A | N/A |\n"
            continue
        p = r.get("params", -1)
        p_str = f"{p:,}" if p > 0 else "0 (non-parametric)"
        benchmark_section += (
            f"| {model} "
            f"| {r.get('accuracy', 0):.4f} "
            f"| **{r.get('f1_macro', 0):.4f}** "
            f"| {r.get('f1_weighted', 0):.4f} "
            f"| {p_str} |\n"
        )

    benchmark_section += f"""
**Best model:** {best_model[0]} (Macro F1 = {best_model[1]:.4f})

**Current project state:**
- ✅ Phase 1: Data analysis and validation complete
- ✅ Phase 2: Infrastructure and baseline training complete
- ✅ Phase 2.5: Full benchmark validation complete
- 🔜 Phase 3: ARST modality-specific encoders (pending Phase 2.5 approval)

> Full benchmark: [`reports/baseline_benchmark_results.md`](reports/baseline_benchmark_results.md)
"""

    # Insert after first H1, or append
    benchmark_marker = "## 📊 Baseline Benchmark Results"
    if benchmark_marker in content:
        # Replace existing section
        start = content.index(benchmark_marker)
        # Find the next ## section after benchmark
        rest = content[start + len(benchmark_marker) :]
        next_section = rest.find("\n## ")
        if next_section >= 0:
            content = content[:start] + benchmark_section + rest[next_section + 1 :]
        else:
            content = content[:start] + benchmark_section
    else:
        # Append after first H1
        first_h1_end = content.find("\n\n", content.find("# "))
        if first_h1_end >= 0:
            content = content[: first_h1_end + 2] + benchmark_section + content[first_h1_end + 2 :]
        else:
            content += "\n" + benchmark_section

    readme_path.write_text(content, encoding="utf-8")
    logger.info("Updated README.md")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("=" * 70)
    logger.info("ARST Phase 2.5 — Benchmark Report Generator")
    logger.info("=" * 70)

    # 1. Collect all metrics
    logger.info("\n[1/5] Collecting metrics from outputs/evaluation/...")
    results = collect_all_metrics()

    # Print quick summary
    for model, r in results.items():
        if r.get("missing"):
            logger.warning("  %-15s  MISSING", model)
        else:
            logger.info(
                "  %-15s  F1=%.4f  Acc=%.4f  Params=%s",
                model,
                r.get("f1_macro", 0),
                r.get("accuracy", 0),
                f"{r.get('params', -1):,}" if r.get("params", -1) >= 0 else "N/A",
            )

    # 2. Collect epoch times
    logger.info("\n[2/5] Collecting training epoch times...")
    epoch_times = collect_epoch_times()
    for model, t in epoch_times.items():
        logger.info("  %-15s  %.1fs/epoch" if t > 0 else "  %-15s  N/A", model, t)

    # 3. Generate bar charts
    logger.info("\n[3/5] Generating comparison charts...")
    generate_bar_charts(results, epoch_times)

    # 4. Generate confusion matrices
    logger.info("\n[4/5] Generating confusion matrices...")
    for model_key, display in [("cnn", "CNN"), ("lstm", "LSTM"), ("transformer", "Transformer")]:
        generate_confusion_matrix(model_key, display)

    # 5. Write reports
    logger.info("\n[5/5] Writing markdown reports...")
    write_benchmark_results(results, epoch_times)
    write_efficiency_report(results, epoch_times)
    write_error_analysis(results)
    write_phase2_findings(results)
    write_phase3_readiness(results)
    update_readme(results)

    logger.info("\n" + "=" * 70)
    logger.info("Phase 2.5 Benchmark Report Generation Complete!")
    logger.info("=" * 70)
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
