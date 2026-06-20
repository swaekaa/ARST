@echo off
REM ============================================================
REM Phase 2.6 — Architecture Validation and Repair
REM Complete runner script
REM ============================================================

echo ============================================================
echo Phase 2.6 — Architecture Validation and Repair
echo ============================================================
echo.

REM Step 0: Run existing tests to ensure nothing is broken
echo [Step 0] Running unit tests...
python -m pytest tests/ -v --tb=short 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some tests failed. Check output above.
)
echo.

REM Step 1: Commit architecture fixes
echo [Step 1] Committing architecture fixes...
git add src/arst/models/baselines/cnn.py configs/model/cnn.yaml
git commit -m "fix: correct CNN - add LayerNorm pre-head, reduce dropout 0.3->0.1"
git add src/arst/models/baselines/transformer.py
git commit -m "fix: correct Transformer - skip input proj in _init_weights, add sqrt(d_model) scaling, input norm"
git add reports/cnn_debug_report.md reports/transformer_debug_report.md
git commit -m "docs: add CNN and Transformer debug reports for Phase 2.6"
echo.

REM Step 2: Tiny overfit sanity check
echo [Step 2] Running tiny dataset overfit test...
python scripts/tiny_dataset_overfit.py 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Tiny overfit test failed!
    pause
    exit /b 1
)
git add scripts/tiny_dataset_overfit.py
git commit -m "test: add tiny dataset overfit sanity check (Phase 2.6)"
echo.

REM Step 3: Learning rate study
echo [Step 3] Running learning rate study...
python scripts/lr_study.py 2>&1
git add scripts/lr_study.py reports/lr_study.md reports/lr_study_results.json
git commit -m "experiment: LR study for CNN and Transformer (Phase 2.6)"
echo.

REM Step 4: Ablation study
echo [Step 4] Running ablation study...
python scripts/ablation_study.py 2>&1
git add scripts/ablation_study.py reports/baseline_architecture_ablations.md reports/ablation_results.json
git commit -m "experiment: architecture ablations - Transformer pooling, CNN kernels (Phase 2.6)"
echo.

REM Step 5: Full benchmark retrain - CNN
echo [Step 5a] Retraining CNN (full benchmark)...
python train.py model=cnn training.epochs=100 wandb.enabled=false 2>&1
git add -A
git commit -m "benchmark: retrain CNN with fixed architecture (Phase 2.6)"
echo.

REM Step 5b: Full benchmark retrain - Transformer
echo [Step 5b] Retraining Transformer (full benchmark)...
python train.py model=transformer training.epochs=100 wandb.enabled=false 2>&1
git add -A
git commit -m "benchmark: retrain Transformer with fixed architecture (Phase 2.6)"
echo.

echo ============================================================
echo Phase 2.6 Complete! Check reports/ for results.
echo ============================================================
echo.
echo Next steps:
echo 1. Check reports/baseline_benchmark_results.md for updated results
echo 2. Verify CNN F1 ^> 0.32 (MLP baseline)
echo 3. Verify Transformer F1 ^>= 0.39 (LSTM baseline)
echo 4. If conditions met, Phase 3 can begin
echo.
pause
