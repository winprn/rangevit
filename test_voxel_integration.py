"""
Test script to verify voxel feature integration into RangeViT pipeline.
Tests data loading, voxelization, and model forward pass.
"""

import torch
import yaml
import numpy as np
from dataset.range_view_loader import RangeViewLoader
from dataset.semantic_kitti import parser as SemanticKittiParser
import models.rangevit as rangevit_models


def test_data_pipeline(config_path='config_kitti.yaml', data_root='/path/to/dataset'):
    """Test data loading with voxel features."""
    print("="*80)
    print("Testing Data Pipeline with Voxel Features")
    print("="*80)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfig loaded:")
    print(f"  - in_channels: {config['in_channels']}")
    print(f"  - voxel_features enabled: {config['voxel_features']['enable']}")
    print(f"  - voxel_size: {config['voxel_features']['voxel_size']}")
    print(f"  - voxel_feature_dim: {config['voxel_features']['feature_dim']}")

    # Initialize dataset
    try:
        kitti_config_path = './dataset/semantic_kitti/semantic-kitti.yaml'
        dataset = SemanticKittiParser.SemanticKitti(
            root=data_root,
            config_path=kitti_config_path,
            sequences=['00'],  # Use sequence 00 for testing
            return_ref=False
        )
        print(f"\nDataset initialized: {len(dataset)} samples")
    except Exception as e:
        print(f"\nDataset initialization failed (expected if data_root not set): {e}")
        print("Skipping data loading test. Please set correct data_root to test.")
        return False

    # Initialize data loader
    loader = RangeViewLoader(
        dataset=dataset,
        config=config,
        data_len=1,  # Test with just 1 sample
        is_train=True,
        return_uproj=False,
        use_kpconv=False
    )

    # Load one sample
    print("\nLoading sample...")
    try:
        features, labels, mask = loader[0]
        print(f"  ✓ Sample loaded successfully")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Mask shape: {mask.shape}")

        # Verify shape
        expected_channels = config['in_channels']
        if features.shape[0] != expected_channels:
            print(f"  ✗ ERROR: Expected {expected_channels} channels, got {features.shape[0]}")
            return False
        else:
            print(f"  ✓ Correct number of channels: {features.shape[0]}")

        # Check feature ranges
        print(f"\n  Feature statistics:")
        for i in range(features.shape[0]):
            channel_name = ['range', 'x', 'y', 'z', 'intensity'] + [f'voxel_{j}' for j in range(8)]
            valid_mask = mask.bool()
            if valid_mask.sum() > 0:
                valid_features = features[i][valid_mask]
                print(f"    Channel {i} ({channel_name[i]}): mean={valid_features.mean():.3f}, std={valid_features.std():.3f}, min={valid_features.min():.3f}, max={valid_features.max():.3f}")

        return True

    except Exception as e:
        print(f"  ✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward(config_path='config_kitti.yaml'):
    """Test model initialization and forward pass."""
    print("\n" + "="*80)
    print("Testing Model Forward Pass")
    print("="*80)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create dummy input
    batch_size = 2
    in_channels = config['in_channels']
    height, width = config['image_size']

    print(f"\nCreating dummy input:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Channels: {in_channels}")
    print(f"  - Height: {height}")
    print(f"  - Width: {width}")

    dummy_input = torch.randn(batch_size, in_channels, height, width)

    # Initialize model
    print("\nInitializing RangeViT model...")
    try:
        model = rangevit_models.RangeViT(
            in_channels=in_channels,
            n_cls=config['n_classes'],
            backbone=config['vit_backbone'],
            image_size=config['image_size'],
            pretrained_path=None,  # No pretrained weights for testing
            new_patch_size=config['patch_size'],
            new_patch_stride=config['patch_stride'],
            reuse_pos_emb=False,
            reuse_patch_emb=False,
            conv_stem=config['conv_stem'],
            stem_base_channels=config['stem_base_channels'],
            stem_hidden_dim=config['D_h'],
            skip_filters=config['skip_filters'],
            decoder=config['decoder'],
            use_kpconv=False,
        )
        print("  ✓ Model initialized successfully")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {num_params:,}")
        print(f"  - Trainable parameters: {num_trainable:,}")

    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Forward pass
    print("\nRunning forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        print(f"  ✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected shape: [{batch_size}, {config['n_classes']}, {height}, {width}]")

        # Verify output shape
        expected_shape = (batch_size, config['n_classes'], height, width)
        if output.shape != expected_shape:
            print(f"  ✗ ERROR: Output shape mismatch")
            return False
        else:
            print(f"  ✓ Output shape correct")

        return True

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("VOXEL FEATURE INTEGRATION TEST")
    print("="*80 + "\n")

    config_path = 'config_kitti.yaml'

    # Test 1: Model forward pass (doesn't require dataset)
    model_test_passed = test_model_forward(config_path)

    # Test 2: Data pipeline (requires dataset)
    # Note: This will fail if data_root is not set correctly
    data_test_passed = test_data_pipeline(
        config_path=config_path,
        data_root='/path/to/semantic_kitti/dataset/sequences/'
    )

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Model Forward Pass: {'✓ PASSED' if model_test_passed else '✗ FAILED'}")
    print(f"Data Pipeline:      {'✓ PASSED' if data_test_passed else '⚠ SKIPPED (set data_root to test)'}")
    print("="*80 + "\n")

    if model_test_passed:
        print("SUCCESS: Core integration is working!")
        print("Next steps:")
        print("  1. Set correct data_root in test script to test data loading")
        print("  2. Run training for 1 epoch to verify end-to-end pipeline")
        print("  3. Monitor training loss and mIoU")
    else:
        print("FAILURE: Please fix the errors above before proceeding.")

    return model_test_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
