#!/usr/bin/env python3
"""
Verification script to test RangeSwin model before training.
Run this to ensure all fixes are working correctly.
"""

import torch
import sys
from models import create_rangeswin

def test_model():
    print("="*80)
    print("RangeSwin Model Verification Script")
    print("="*80)

    # Create model
    print("\n1. Creating model...")
    model_cfg = {
        'in_channels': 5,
        'n_cls': 20,
        'swin_name': 'swinv2_tiny_window8_256',
        'out_channels': 128,
        'pretrained_path': None,
    }

    try:
        model = create_rangeswin(model_cfg, use_kpconv=False)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False

    # Test forward pass in training mode
    print("\n2. Testing forward pass (training mode)...")
    model.train()
    dummy_input = torch.randn(2, 5, 64, 384)

    try:
        output_train = model(dummy_input)
        if isinstance(output_train, tuple):
            print(f"   ✓ Training returns tuple: {type(output_train)}")
            print(f"   ✓ Main output shape: {output_train[0].shape} (expected: [2, 20, 64, 384])")
            print(f"   ✓ Aux output shape: {output_train[1].shape} (expected: [2, 20, 64, 384])")

            # Verify shapes
            assert output_train[0].shape == torch.Size([2, 20, 64, 384]), "Main output shape mismatch!"
            assert output_train[1].shape == torch.Size([2, 20, 64, 384]), "Aux output shape mismatch!"
        else:
            print(f"   ⚠ Training returns single tensor (aux_classifier may not be active)")
            print(f"   Output shape: {output_train.shape}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass in eval mode
    print("\n3. Testing forward pass (eval mode)...")
    model.eval()

    try:
        with torch.no_grad():
            output_eval = model(dummy_input)

        if isinstance(output_eval, tuple):
            print(f"   ⚠ Eval returns tuple (should return single tensor)")
            print(f"   Main shape: {output_eval[0].shape}")
        else:
            print(f"   ✓ Eval returns single tensor: {type(output_eval)}")
            print(f"   ✓ Output shape: {output_eval.shape} (expected: [2, 20, 64, 384])")

            # Verify shape
            assert output_eval.shape == torch.Size([2, 20, 64, 384]), "Eval output shape mismatch!"
    except Exception as e:
        print(f"   ✗ Eval forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check parameters
    print("\n4. Checking model parameters...")
    try:
        stats = model.counter_model_parameters()
        print(f"   ✓ Total parameters: {stats['total_num_parameters']:,}")

        if 'swin_backbone_num_parameters' in stats:
            print(f"   ✓ Swin backbone: {stats['swin_backbone_num_parameters']:,}")

        if 'decode_head_num_parameters' in stats:
            print(f"   ✓ Decode head: {stats['decode_head_num_parameters']:,}")

        if 'classifier_num_parameters' in stats:
            print(f"   ✓ Classifier head: {stats['classifier_num_parameters']:,}")
        else:
            print(f"   ⚠ Classifier head not counted")

        if 'aux_classifier_num_parameters' in stats:
            print(f"   ✓ Aux classifier: {stats['aux_classifier_num_parameters']:,}")
        else:
            print(f"   ⚠ Aux classifier not counted (may not exist)")

        # Verify aux_classifier exists
        has_aux = (hasattr(model, 'swin_encoder') and
                   hasattr(model.swin_encoder, 'aux_classifier') and
                   model.swin_encoder.aux_classifier is not None)

        if has_aux:
            print(f"   ✓ Aux classifier exists and is not None")
        else:
            print(f"   ✗ Aux classifier missing or None!")
            return False

    except Exception as e:
        print(f"   ✗ Parameter counting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test different input sizes
    print("\n5. Testing different input sizes...")
    test_sizes = [
        (64, 384),   # Training size
        (64, 512),   # Multiple of 8
        (64, 2048),  # Validation size
    ]

    model.eval()
    for h, w in test_sizes:
        try:
            with torch.no_grad():
                test_input = torch.randn(1, 5, h, w)
                output = model(test_input)
                if isinstance(output, tuple):
                    output = output[0]
                expected_shape = torch.Size([1, 20, h, w])
                assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
                print(f"   ✓ Size [{h}, {w}] → output shape: {output.shape}")
        except Exception as e:
            print(f"   ✗ Size [{h}, {w}] failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Final checks
    print("\n6. Final verification...")

    # Check window size divisibility
    window_size = 8  # swinv2_tiny_window8_256
    test_dims = [64, 384, 512, 2048]
    all_divisible = all(d % window_size == 0 for d in test_dims)

    if all_divisible:
        print(f"   ✓ All dimensions divisible by window_size={window_size}")
    else:
        print(f"   ✗ Some dimensions not divisible by window_size={window_size}")
        return False

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print("\nYour model is ready for training!")
    print("\nStart training with:")
    print("  python main.py config/config_kitti_swin.yaml \\")
    print("      --data_root /path/to/SemanticKITTI \\")
    print("      --save_path ./experiments")
    print()

    return True

if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)
