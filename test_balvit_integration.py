#!/usr/bin/env python3
"""
Test script to verify BALViT integration and backward compatibility.
This tests that the model works correctly with and without BEV inputs.
"""

import torch
import sys
import yaml
from models.rangevit import RangeViT

def test_backward_compatibility():
    """Test that the model works without BEV (backward compatible with original RangeViT)"""
    print("=" * 80)
    print("Test 1: Backward Compatibility (without BEV)")
    print("=" * 80)

    # Create model without BEV
    model = RangeViT(
        in_channels=5,
        n_cls=20,
        backbone='vit_tiny_patch16_384',
        image_size=(64, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
        conv_stem='ConvStem',
        stem_base_channels=32,
        stem_hidden_dim=128,
        decoder='up_conv',
        up_conv_d_decoder=128,
        up_conv_scale_factor=(2, 8),
        use_kpconv=False,
        # No BEV parameters
        bev_channels=None,
    ).eval()

    # Test forward pass without BEV
    dummy_input = torch.randn(2, 5, 64, 384)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected output shape: [2, 20, 64, 384]")
    assert output.shape == (2, 20, 64, 384), f"Expected shape (2, 20, 64, 384), got {output.shape}"
    print("✓ Test PASSED: Model works without BEV\n")
    return True


def test_with_bev_disabled():
    """Test that BEV components are not created when bev_channels=None"""
    print("=" * 80)
    print("Test 2: BEV Components Disabled (bev_channels=None)")
    print("=" * 80)

    model = RangeViT(
        in_channels=5,
        n_cls=20,
        backbone='vit_tiny_patch16_384',
        image_size=(64, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
        conv_stem='ConvStem',
        decoder='up_conv',
        use_kpconv=False,
        bev_channels=None,  # Explicitly disable
    ).eval()

    # Check that BEV components are not created
    assert model.rangevit.encoder.bev_encoder is None, "BEV encoder should be None"
    assert len(model.rangevit.encoder.cross_modal_adapters) == 0, "No adapters should be created"
    print("✓ BEV encoder is None")
    print("✓ No cross-modal adapters created")
    print("✓ Test PASSED: BEV components properly disabled\n")
    return True


def test_with_bev_enabled():
    """Test that the model works WITH BEV inputs"""
    print("=" * 80)
    print("Test 3: BALViT Mode (with BEV)")
    print("=" * 80)

    # Create model WITH BEV
    model = RangeViT(
        in_channels=5,
        n_cls=20,
        backbone='vit_tiny_patch16_384',
        image_size=(64, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
        conv_stem='ConvStem',
        stem_base_channels=32,
        stem_hidden_dim=128,
        decoder='up_conv',
        up_conv_d_decoder=128,
        up_conv_scale_factor=(2, 8),
        use_kpconv=False,
        # Enable BEV
        bev_channels=8,
        bev_base_channels=64,
        bev_num_layers=3,
        adapter_indices=[3, 7, 11],  # Insert adapters after blocks 3, 7, 11
        adapter_mlp_ratio=4.0,
        use_bev_decoder=True,
        bev_decoder_hidden=128,
        use_bev_fusion=True,
    ).eval()

    # Check that BEV components are created
    assert model.rangevit.encoder.bev_encoder is not None, "BEV encoder should exist"
    assert len(model.rangevit.encoder.cross_modal_adapters) == 3, f"Should have 3 adapters, got {len(model.rangevit.encoder.cross_modal_adapters)}"
    assert model.rangevit.bev_decoder is not None, "BEV decoder should exist"
    print(f"✓ BEV encoder created")
    print(f"✓ {len(model.rangevit.encoder.cross_modal_adapters)} cross-modal adapters created at indices {model.rangevit.encoder.adapter_indices}")
    print(f"✓ BEV decoder created")

    # Test forward pass with BEV
    dummy_rv = torch.randn(2, 5, 64, 384)
    dummy_bev = torch.randn(2, 8, 256, 256)  # BEV with 8 channels

    with torch.no_grad():
        output = model(dummy_rv, bev_image=dummy_bev)

    print(f"✓ RV input shape: {dummy_rv.shape}")
    print(f"✓ BEV input shape: {dummy_bev.shape}")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (2, 20, 64, 384), f"Expected shape (2, 20, 64, 384), got {output.shape}"
    print("✓ Test PASSED: Model works with BEV inputs\n")
    return True


def test_bev_optional_at_runtime():
    """Test that BEV is optional at runtime even when model supports it"""
    print("=" * 80)
    print("Test 4: BEV Optional at Runtime")
    print("=" * 80)

    # Create model that SUPPORTS BEV
    model = RangeViT(
        in_channels=5,
        n_cls=20,
        backbone='vit_tiny_patch16_384',
        image_size=(64, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
        conv_stem='ConvStem',
        decoder='up_conv',
        use_kpconv=False,
        bev_channels=8,
        adapter_indices=[3, 7, 11],
    ).eval()

    dummy_rv = torch.randn(2, 5, 64, 384)

    # Test 1: Pass BEV=None
    with torch.no_grad():
        output1 = model(dummy_rv, bev_image=None)
    print(f"✓ Forward with bev_image=None: output shape {output1.shape}")

    # Test 2: Don't pass BEV argument at all
    with torch.no_grad():
        output2 = model(dummy_rv)
    print(f"✓ Forward without bev_image arg: output shape {output2.shape}")

    # Both should produce same output
    assert torch.allclose(output1, output2), "Outputs should be identical"
    print("✓ Both methods produce identical outputs")
    print("✓ Test PASSED: BEV is optional at runtime\n")
    return True


def test_adapter_insertion():
    """Test that adapters are inserted at correct positions"""
    print("=" * 80)
    print("Test 5: Adapter Insertion at Correct Indices")
    print("=" * 80)

    adapter_indices = [2, 5, 8, 11]
    model = RangeViT(
        in_channels=5,
        n_cls=20,
        backbone='vit_tiny_patch16_384',
        image_size=(64, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
        conv_stem='ConvStem',
        decoder='up_conv',
        use_kpconv=False,
        bev_channels=8,
        adapter_indices=adapter_indices,
    ).eval()

    encoder = model.rangevit.encoder
    print(f"✓ Number of ViT blocks: {len(encoder.blocks)}")
    print(f"✓ Adapter indices: {adapter_indices}")
    print(f"✓ Number of adapters: {len(encoder.cross_modal_adapters)}")

    # Check that adapters exist at the right indices
    for idx in adapter_indices:
        assert str(idx) in encoder.cross_modal_adapters, f"Adapter at index {idx} should exist"
    print(f"✓ All {len(adapter_indices)} adapters created at correct indices")
    print("✓ Test PASSED: Adapters inserted correctly\n")
    return True


def test_freeze_vit():
    """Test that ViT freezing works correctly"""
    print("=" * 80)
    print("Test 6: ViT Backbone Freezing")
    print("=" * 80)

    model = RangeViT(
        in_channels=5,
        n_cls=20,
        backbone='vit_tiny_patch16_384',
        image_size=(64, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
        conv_stem='ConvStem',
        decoder='up_conv',
        use_kpconv=False,
        bev_channels=8,
        adapter_indices=[3, 7, 11],
        freeze_vit=True,  # Freeze the ViT backbone
    )

    # Count trainable vs frozen parameters
    frozen_params = 0
    trainable_params = 0
    bev_params = 0
    adapter_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            if 'bev_encoder' in name:
                bev_params += param.numel()
            elif 'cross_modal_adapters' in name:
                adapter_params += param.numel()
        else:
            frozen_params += param.numel()

    print(f"✓ Frozen parameters: {frozen_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ BEV encoder trainable params: {bev_params:,}")
    print(f"✓ Adapter trainable params: {adapter_params:,}")

    assert trainable_params > 0, "Should have trainable parameters"
    assert frozen_params > 0, "Should have frozen parameters"
    assert bev_params > 0, "BEV encoder should be trainable"
    assert adapter_params > 0, "Adapters should be trainable"

    print("✓ Test PASSED: ViT freezing works correctly\n")
    return True


def main():
    print("\n" + "=" * 80)
    print("BALViT Integration Test Suite")
    print("=" * 80 + "\n")

    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("BEV Components Disabled", test_with_bev_disabled),
        ("BALViT Mode (with BEV)", test_with_bev_enabled),
        ("BEV Optional at Runtime", test_bev_optional_at_runtime),
        ("Adapter Insertion", test_adapter_insertion),
        ("ViT Freezing", test_freeze_vit),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"✗ Test FAILED with exception: {e}\n")
            results.append((test_name, f"FAILED: {e}"))
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, result in results:
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")

    all_passed = all(result == "PASSED" for _, result in results)
    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
