@echo off
REM =============================================================================
REM Test Pipeline for RangeViT Training
REM =============================================================================
REM This script runs a quick test of the training pipeline to verify everything
REM works before running a full training session.
REM
REM What it tests:
REM - Model initialization
REM - Data loading (train + val)
REM - Forward/backward pass
REM - Training loop
REM - Validation loop
REM - Checkpoint saving
REM - Test set validation (optional)
REM
REM Configuration:
REM - 2 epochs with only 10 iterations each (very fast!)
REM - Small batch size (2)
REM - Frequent logging (every 10 steps)
REM - All augmentations disabled
REM =============================================================================

echo ========================================
echo Starting RangeViT Test Pipeline
echo ========================================
echo.

REM Clean up previous test output
if exist "test_output" (
    echo Cleaning up previous test output...
    rmdir /s /q "test_output"
    echo.
)


REM Check if data directory exists
if not exist "..\dataset\SemanticKitti\data_odometry_velodyne\dataset\sequences" (
    echo ERROR: Data directory not found!
    echo Expected: ..\dataset\SemanticKitti\data_odometry_velodyne\dataset\sequences
    echo Please update the path in this script if your data is located elsewhere.
    pause
    exit /b 1
)

REM Check if pretrained model exists
if not exist "..\pretrained_model\vit_tiny_p16_384.pth" (
    echo WARNING: Pretrained model not found!
    echo Expected: ..\pretrained_model\vit_tiny_p16_384.pth
    echo Training will continue but may fail during model initialization.
    echo.
    pause
)

echo ========================================
echo STEP 1: Training Test (2 epochs)
echo ========================================
echo Config: configs/config_kitti_test.yaml
echo Duration: 2 epochs x 10 iterations = 20 training steps total
echo Batch size: 2
echo.

python main.py configs/config_kitti_test.yaml ^
    --data_root "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences" ^
    --save_path "test_output" ^
    --num_workers 2 ^
    --log_frequency 10

set TRAIN_EXIT_CODE=%ERRORLEVEL%

if %TRAIN_EXIT_CODE% NEQ 0 (
    echo.
    echo ========================================
    echo Training Test Failed!
    echo ========================================
    echo.
    echo Please check the error messages above to diagnose the issue.
    echo Common issues:
    echo - Missing data files
    echo - Missing pretrained model
    echo - CUDA out of memory ^(reduce batch_size in configs/config_kitti_test.yaml^)
    echo - Dependencies not installed
    echo.
    pause
    exit /b %TRAIN_EXIT_CODE%
)

echo.
echo ========================================
echo STEP 2: Validation on Val Set
echo ========================================
echo Testing inference pipeline with checkpoint...
echo.

REM Wait a moment and verify checkpoint exists
timeout /t 2 /nobreak >nul 2>&1
if not exist "test_output\log_test_run\checkpoint\checkpoint.pth" (
    echo ERROR: Checkpoint file not found!
    echo Expected: test_output\log_test_run\checkpoint\checkpoint.pth
    echo.
    pause
    exit /b 1
)
echo Found checkpoint: test_output\log_test_run\checkpoint\checkpoint.pth
echo.

python main.py configs/config_kitti_test.yaml ^
    --data_root "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences" ^
    --save_path "test_output" ^
    --checkpoint "test_output\log_test_run\checkpoint\checkpoint.pth" ^
    --num_workers 2 ^
    --val_only

set VAL_EXIT_CODE=%ERRORLEVEL%

if %VAL_EXIT_CODE% NEQ 0 (
    echo.
    echo ========================================
    echo Validation Test Failed!
    echo ========================================
    echo.
    pause
    exit /b %VAL_EXIT_CODE%
)

echo.
echo ========================================
echo STEP 3: Validation on Test Set
echo ========================================
echo Testing inference on test split with result saving...
echo.

python main.py configs/config_kitti_test.yaml ^
    --data_root "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences" ^
    --save_path "test_output" ^
    --checkpoint "test_output\log_test_run\checkpoint\checkpoint.pth" ^
    --num_workers 2 ^
    --val_only ^
    --test_split ^
    --save_eval_results

set TEST_EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================
if %TEST_EXIT_CODE% EQU 0 (
    echo All Tests Completed Successfully!
    echo ========================================
    echo.
    echo Summary:
    echo [PASS] Training test ^(2 epochs^)
    echo [PASS] Validation on val set
    echo [PASS] Validation on test set with result saving
    echo.
    echo Next steps:
    echo 1. Check test_output/log_test_run/ for logs and checkpoints
    echo 2. Check test_output/Eval_test_run/preds/ for prediction results
    echo 3. If everything looks good, run full training with configs/config_kitti.yaml
    echo 4. Use the following command for full training:
    echo    python main.py configs/config_kitti.yaml --data_root "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences" --save_path "save_path"
) else (
    echo Test Set Validation Failed!
    echo ========================================
)
echo.

pause
exit /b %EXIT_CODE%
