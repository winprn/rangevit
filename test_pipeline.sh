#!/bin/bash
# =============================================================================
# Test Pipeline for RangeViT Training
# =============================================================================
# This script runs a quick test of the training pipeline to verify everything
# works before running a full training session.
#
# What it tests:
# - Model initialization
# - Data loading (train + val)
# - Forward/backward pass
# - Training loop
# - Validation loop
# - Checkpoint saving
# - Test set validation (optional)
#
# Configuration:
# - 2 epochs with only 10 iterations each (very fast!)
# - Small batch size (2)
# - Frequent logging (every 10 steps)
# - All augmentations disabled
# =============================================================================

echo "========================================"
echo "Starting RangeViT Test Pipeline"
echo "========================================"
echo ""

# Clean up previous test output
if [ -d "test_output" ]; then
    echo "Cleaning up previous test output..."
    rm -rf "test_output"
    echo ""
fi

# Check if data directory exists
if [ ! -d "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences" ]; then
    echo "ERROR: Data directory not found!"
    echo "Expected: ../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences"
    echo "Please update the path in this script if your data is located elsewhere."
    exit 1
fi

# Check if pretrained model exists
if [ ! -f "../pretrained_model/vit_tiny_p16_384.pth" ]; then
    echo "WARNING: Pretrained model not found!"
    echo "Expected: ../pretrained_model/vit_tiny_p16_384.pth"
    echo "Training will continue but may fail during model initialization."
    echo ""
    read -p "Press enter to continue..."
fi

echo "========================================"
echo "STEP 1: Training Test (2 epochs)"
echo "========================================"
echo "Config: configs/config_kitti_test.yaml"
echo "Duration: 2 epochs x 10 iterations = 20 training steps total"
echo "Batch size: 2"
echo ""

python main.py configs/config_kitti_test.yaml \
    --data_root "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences" \
    --save_path "test_output" \
    --num_workers 2 \
    --log_frequency 10

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "Training Test Failed!"
    echo "========================================"
    echo ""
    echo "Please check the error messages above to diagnose the issue."
    echo "Common issues:"
    echo "- Missing data files"
    echo "- Missing pretrained model"
    echo "- CUDA out of memory (reduce batch_size in configs/config_kitti_test.yaml)"
    echo "- Dependencies not installed"
    echo ""
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "========================================"
echo "STEP 2: Validation on Val Set"
echo "========================================"
echo "Testing inference pipeline with checkpoint..."
echo ""

# Wait a moment and verify checkpoint exists
sleep 2
if [ ! -f "test_output/log_test_run/checkpoint/checkpoint.pth" ]; then
    echo "ERROR: Checkpoint file not found!"
    echo "Expected: test_output/log_test_run/checkpoint/checkpoint.pth"
    echo ""
    exit 1
fi
echo "Found checkpoint: test_output/log_test_run/checkpoint/checkpoint.pth"
echo ""

python main.py configs/config_kitti_test.yaml \
    --data_root "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences" \
    --save_path "test_output" \
    --checkpoint "test_output/log_test_run/checkpoint/checkpoint.pth" \
    --num_workers 2 \
    --val_only

VAL_EXIT_CODE=$?

if [ $VAL_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "Validation Test Failed!"
    echo "========================================"
    echo ""
    exit $VAL_EXIT_CODE
fi

echo ""
echo "========================================"
echo "STEP 3: Validation on Test Set"
echo "========================================"
echo "Testing inference on test split with result saving..."
echo ""

python main.py configs/config_kitti_test.yaml \
    --data_root "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences" \
    --save_path "test_output" \
    --checkpoint "test_output/log_test_run/checkpoint/checkpoint.pth" \
    --num_workers 2 \
    --val_only \
    --test_split \
    --save_eval_results

TEST_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "All Tests Completed Successfully!"
    echo "========================================"
    echo ""
    echo "Summary:"
    echo "[PASS] Training test (2 epochs)"
    echo "[PASS] Validation on val set"
    echo "[PASS] Validation on test set with result saving"
    echo ""
    echo "Next steps:"
    echo "1. Check test_output/log_test_run/ for logs and checkpoints"
    echo "2. Check test_output/Eval_test_run/preds/ for prediction results"
    echo "3. If everything looks good, run full training with configs/config_kitti.yaml"
    echo "4. Use the following command for full training:"
    echo "   python main.py configs/config_kitti.yaml --data_root \"../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences\" --save_path \"save_path\""
else
    echo "Test Set Validation Failed!"
    echo "========================================"
fi
echo ""

exit $TEST_EXIT_CODE
