<#
.SYNOPSIS
    ARST Phase 2.5 — Full Baseline Benchmark Pipeline
.DESCRIPTION
    Trains CNN, BiLSTM, and Transformer baselines sequentially with identical
    experimental conditions as the verified MLP baseline, then generates all
    benchmark reports, visualizations, and commits results.

    Usage (from repo root in PowerShell):
        .\scripts\run_phase25.ps1

    Options:
        -SkipTraining   : Skip training, only generate reports
        -SkipCommit     : Skip git commit at the end
        -ModelsToTrain  : Comma-separated list of models (default: cnn,lstm,transformer)

    Example:
        .\scripts\run_phase25.ps1 -ModelsToTrain cnn,lstm
        .\scripts\run_phase25.ps1 -SkipTraining
#>

param(
    [string[]]$ModelsToTrain = @("cnn", "lstm", "transformer"),
    [switch]$SkipTraining = $false,
    [switch]$SkipCommit = $false
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Set-Location $RepoRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "ARST Phase 2.5 - Baseline Benchmark Pipeline" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Repo root   : $RepoRoot"
Write-Host "Models      : $($ModelsToTrain -join ', ')"
Write-Host "Skip train  : $SkipTraining"
Write-Host ""

$StartTime = Get-Date

# ── Helper function ─────────────────────────────────────────────────────────
function Run-Step {
    param([string]$Label, [string[]]$Cmd)
    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "STEP: $Label" -ForegroundColor Yellow
    Write-Host "CMD : $($Cmd -join ' ')" -ForegroundColor DarkYellow
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    $t0 = Get-Date
    & $Cmd[0] $Cmd[1..($Cmd.Length-1)]
    $exit = $LASTEXITCODE
    $elapsed = (Get-Date) - $t0
    if ($exit -ne 0) {
        Write-Host ""
        Write-Host "ERROR: '$Label' failed with exit code $exit" -ForegroundColor Red
        Write-Host "Elapsed: $($elapsed.TotalSeconds.ToString('0.1'))s" -ForegroundColor Red
        exit $exit
    }
    Write-Host ""
    Write-Host "DONE: $Label completed in $($elapsed.TotalSeconds.ToString('0.1'))s ($($elapsed.TotalMinutes.ToString('0.1')) min)" -ForegroundColor Green
}

# ── Task 1-3: Training ───────────────────────────────────────────────────────
if (-not $SkipTraining) {
    $ModelLabels = @{
        "cnn"         = "CNN Baseline (1D-CNN per-modality branches)"
        "lstm"        = "BiLSTM Baseline (attention-pooled per-modality)"
        "transformer" = "Transformer Baseline (cross-modal attention)"
    }

    $i = 1
    foreach ($model in $ModelsToTrain) {
        $label = $ModelLabels[$model]
        if (-not $label) { $label = $model }
        Write-Host ""
        Write-Host "[$i/$($ModelsToTrain.Count)] Training $label..." -ForegroundColor Cyan
        Run-Step -Label "$label training" -Cmd @("python", "train.py", "model=$model")
        $i++
    }

    Write-Host ""
    Write-Host "✅ All baselines trained!" -ForegroundColor Green
}

# ── Task 4-10: Reports + Visualizations ──────────────────────────────────────
Write-Host ""
Write-Host "[Final] Generating benchmark reports and visualizations..." -ForegroundColor Cyan
Run-Step -Label "Benchmark report generation" -Cmd @("python", "scripts/generate_benchmark_report.py")
Write-Host ""
Write-Host "✅ All reports generated!" -ForegroundColor Green

# ── Git Commit ───────────────────────────────────────────────────────────────
if (-not $SkipCommit) {
    Write-Host ""
    Write-Host "Committing all results..." -ForegroundColor Cyan
    try {
        git add -A
        git commit -m "feat: complete phase 2 benchmark validation and analysis"
        Write-Host "✅ Git commit successful!" -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Git commit failed - please commit manually" -ForegroundColor Yellow
        Write-Host "  git add -A"
        Write-Host "  git commit -m `"feat: complete phase 2 benchmark validation and analysis`""
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
$TotalTime = (Get-Date) - $StartTime
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Phase 2.5 Complete!" -ForegroundColor Cyan
Write-Host "Total time: $($TotalTime.TotalMinutes.ToString('0.1')) minutes" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Generated outputs:" -ForegroundColor White
Write-Host "  outputs/benchmarks/benchmark_accuracy.png"
Write-Host "  outputs/benchmarks/benchmark_macro_f1.png"
Write-Host "  outputs/benchmarks/benchmark_vram.png"
Write-Host "  outputs/benchmarks/benchmark_training_time.png"
Write-Host "  outputs/benchmarks/confusion_matrices/cnn_confusion_matrix.png"
Write-Host "  outputs/benchmarks/confusion_matrices/lstm_confusion_matrix.png"
Write-Host "  outputs/benchmarks/confusion_matrices/transformer_confusion_matrix.png"
Write-Host "  reports/baseline_benchmark_results.md"
Write-Host "  reports/model_efficiency_report.md"
Write-Host "  reports/baseline_error_analysis.md"
Write-Host "  reports/phase2_findings.md"
Write-Host "  reports/phase3_readiness.md"
Write-Host "  README.md (updated)"
Write-Host ""
