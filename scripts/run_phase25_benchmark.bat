@echo off
REM ============================================================
REM ARST Phase 2.5 - Full Baseline Benchmark Pipeline
REM ============================================================
REM Run from repo root: scripts\run_phase25_benchmark.bat
REM
REM This script:
REM   1. Trains CNN, LSTM, Transformer baselines
REM   2. Generates all reports and visualizations
REM   3. Commits all results
REM ============================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0.."

echo ============================================================
echo ARST Phase 2.5 - Baseline Benchmark Pipeline
echo ============================================================
echo.

REM ── TASK 1: Train CNN Baseline ────────────────────────────────
echo [1/4] Training CNN Baseline...
echo Command: python train.py model=cnn
python train.py model=cnn
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CNN training failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo CNN training complete!
echo.

REM ── TASK 2: Train LSTM Baseline ───────────────────────────────
echo [2/4] Training LSTM Baseline...
echo Command: python train.py model=lstm
python train.py model=lstm
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: LSTM training failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo LSTM training complete!
echo.

REM ── TASK 3: Train Transformer Baseline ────────────────────────
echo [3/4] Training Transformer Baseline...
echo Command: python train.py model=transformer
python train.py model=transformer
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Transformer training failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo Transformer training complete!
echo.

REM ── TASK 4-10: Generate Reports and Visualizations ────────────
echo [4/4] Generating benchmark reports and visualizations...
echo Command: python scripts/generate_benchmark_report.py
python scripts/generate_benchmark_report.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Report generation failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo Report generation complete!
echo.

REM ── Git Commit ────────────────────────────────────────────────
echo Committing all changes...
git add -A
git commit -m "feat: complete phase 2 benchmark validation and analysis"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Git commit failed - please commit manually
) else (
    echo Git commit successful!
)

echo.
echo ============================================================
echo Phase 2.5 Complete!
echo ============================================================
echo.
echo Outputs generated:
echo   outputs/benchmarks/benchmark_accuracy.png
echo   outputs/benchmarks/benchmark_macro_f1.png
echo   outputs/benchmarks/benchmark_vram.png
echo   outputs/benchmarks/benchmark_training_time.png
echo   outputs/benchmarks/confusion_matrices/cnn_confusion_matrix.png
echo   outputs/benchmarks/confusion_matrices/lstm_confusion_matrix.png
echo   outputs/benchmarks/confusion_matrices/transformer_confusion_matrix.png
echo   reports/baseline_benchmark_results.md
echo   reports/model_efficiency_report.md
echo   reports/baseline_error_analysis.md
echo   reports/phase2_findings.md
echo   reports/phase3_readiness.md
echo   README.md (updated)
echo.

pause
