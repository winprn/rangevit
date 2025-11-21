#!/usr/bin/env python3
"""
Simplified test to verify BALViT core components without full dependencies.
This directly tests the model components in isolation.
"""

import torch
import torch.nn as nn
import sys

# Import only the core components we need
sys.path.insert(0, '.')
from models.rangevit import BEVEncoder, CrossModalAdapter, BEVDecoder, VisionTransformer

def test_bev_encoder():
    """Test BEV encoder module"""
    print("=" * 80)
    print("Test 1: BEV Encoder")
    print("=" * 80)

    bev_encoder = BEVEncoder(
        in_channels=8,
        embed_dim=192,  # vit_tiny dimension
        base_channels=64,
        num_layers=3,
    )

    dummy_bev = torch.randn(2, 8, 256, 256)
    output = bev_encoder(dummy_bev)

    print(f"✓ Input shape: {dummy_bev.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: [2, 192, 256, 256]")
    assert output.shape == (2, 192, 256, 256), f"Expected (2, 192, 256, 256), got {output.shape}"
    print("✓ Test PASSED\n")
    return True


def test_cross_modal_adapter():
    """Test cross-modal adapter module"""
    print("=" * 80)
    print("Test 2: Cross-Modal Adapter")
    print("=" * 80)

    adapter = CrossModalAdapter(
        dim=192,
        num_heads=6,
        mlp_ratio=4.0,
    )

    rv_tokens = torch.randn(2, 100, 192)  # Range-view tokens
    bev_tokens = torch.randn(2, 256, 192)  # BEV tokens

    output = adapter(rv_tokens, bev_tokens)

    print(f"✓ RV tokens shape: {rv_tokens.shape}")
    print(f"✓ BEV tokens shape: {bev_tokens.shape}")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == rv_tokens.shape, f"Expected {rv_tokens.shape}, got {output.shape}"
    print("✓ Test PASSED\n")
    return True


def test_bev_decoder():
    """Test BEV decoder module"""
    print("=" * 80)
    print("Test 3: BEV Decoder")
    print("=" * 80)

    bev_decoder = BEVDecoder(
        in_channels=192,
        n_cls=20,
        hidden_channels=128,
    )

    bev_feat = torch.randn(2, 192, 64, 64)
    output = bev_decoder(bev_feat)

    print(f"✓ Input shape: {bev_feat.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: [2, 20, 64, 64]")
    assert output.shape == (2, 20, 64, 64), f"Expected (2, 20, 64, 64), got {output.shape}"
    print("✓ Test PASSED\n")
    return True


def test_vit_without_bev():
    """Test VisionTransformer without BEV (backward compatibility)"""
    print("=" * 80)
    print("Test 4: ViT Encoder Without BEV (Backward Compatible)")
    print("=" * 80)

    vit = VisionTransformer(
        image_size=(64, 384),
        patch_size=(2, 8),
        n_layers=12,
        d_model=192,
        d_ff=768,
        n_heads=6,
        n_cls=20,
        channels=5,
        patch_stride=(2, 8),
        conv_stem='ConvStem',
        stem_base_channels=32,
        # No BEV parameters
        bev_channels=None,
    ).eval()

    assert vit.bev_encoder is None, "BEV encoder should be None"
    assert len(vit.cross_modal_adapters) == 0, "No adapters should exist"
    print("✓ BEV encoder is None")
    print("✓ No adapters created")

    dummy_input = torch.randn(2, 5, 64, 384)
    with torch.no_grad():
        x, skip = vit(dummy_input)

    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {x.shape}")
    print("✓ Test PASSED\n")
    return True


def test_vit_with_bev():
    """Test VisionTransformer WITH BEV"""
    print("=" * 80)
    print("Test 5: ViT Encoder With BEV")
    print("=" * 80)

    vit = VisionTransformer(
        image_size=(64, 384),
        patch_size=(2, 8),
        n_layers=12,
        d_model=192,
        d_ff=768,
        n_heads=6,
        n_cls=20,
        channels=5,
        patch_stride=(2, 8),
        conv_stem='ConvStem',
        stem_base_channels=32,
        # Enable BEV
        bev_channels=8,
        bev_base_channels=64,
        bev_num_layers=3,
        adapter_indices=[3, 7, 11],
    ).eval()

    assert vit.bev_encoder is not None, "BEV encoder should exist"
    assert len(vit.cross_modal_adapters) == 3, f"Should have 3 adapters, got {len(vit.cross_modal_adapters)}"
    print(f"✓ BEV encoder created")
    print(f"✓ {len(vit.cross_modal_adapters)} adapters created")

    dummy_rv = torch.randn(2, 5, 64, 384)
    dummy_bev = torch.randn(2, 8, 256, 256)

    with torch.no_grad():
        x, skip, bev_ctx = vit(dummy_rv, bev_image=dummy_bev)

    print(f"✓ RV input shape: {dummy_rv.shape}")
    print(f"✓ BEV input shape: {dummy_bev.shape}")
    print(f"✓ Output shape: {x.shape}")
    print(f"✓ BEV context keys: {list(bev_ctx.keys())}")
    assert 'bev_feat' in bev_ctx, "BEV context should contain bev_feat"
    assert 'bev_tokens' in bev_ctx, "BEV context should contain bev_tokens"
    print("✓ Test PASSED\n")
    return True


def test_bev_optional_at_runtime():
    """Test that BEV can be omitted at runtime"""
    print("=" * 80)
    print("Test 6: BEV Optional at Runtime")
    print("=" * 80)

    vit = VisionTransformer(
        image_size=(64, 384),
        patch_size=(2, 8),
        n_layers=12,
        d_model=192,
        d_ff=768,
        n_heads=6,
        n_cls=20,
        channels=5,
        patch_stride=(2, 8),
        conv_stem='ConvStem',
        stem_base_channels=32,
        bev_channels=8,  # BEV enabled
        adapter_indices=[3, 7, 11],
    ).eval()

    dummy_rv = torch.randn(2, 5, 64, 384)

    # Test with bev_image=None
    with torch.no_grad():
        x1, skip1 = vit(dummy_rv, bev_image=None)
        print(f"✓ Forward with bev_image=None: output shape {x1.shape}")

    # Test without bev_image argument
    with torch.no_grad():
        x2, skip2 = vit(dummy_rv)
        print(f"✓ Forward without bev_image arg: output shape {x2.shape}")

    assert torch.allclose(x1, x2), "Outputs should be identical"
    print("✓ Both outputs are identical")
    print("✓ Test PASSED\n")
    return True


def test_adapter_indices():
    """Test that adapters are created at correct indices"""
    print("=" * 80)
    print("Test 7: Adapter Indices")
    print("=" * 80)

    adapter_indices = [1, 4, 7, 10]
    vit = VisionTransformer(
        image_size=(64, 384),
        patch_size=(2, 8),
        n_layers=12,
        d_model=192,
        d_ff=768,
        n_heads=6,
        n_cls=20,
        channels=5,
        patch_stride=(2, 8),
        conv_stem='ConvStem',
        bev_channels=8,
        adapter_indices=adapter_indices,
    )

    print(f"✓ Requested adapter indices: {adapter_indices}")
    print(f"✓ Number of adapters created: {len(vit.cross_modal_adapters)}")

    for idx in adapter_indices:
        assert str(idx) in vit.cross_modal_adapters, f"Adapter {idx} should exist"
        print(f"  ✓ Adapter at index {idx} exists")

    print("✓ Test PASSED\n")
    return True


def main():
    print("\n" + "=" * 80)
    print("BALViT Component Test Suite (Simplified)")
    print("=" * 80 + "\n")

    tests = [
        ("BEV Encoder", test_bev_encoder),
        ("Cross-Modal Adapter", test_cross_modal_adapter),
        ("BEV Decoder", test_bev_decoder),
        ("ViT Without BEV", test_vit_without_bev),
        ("ViT With BEV", test_vit_with_bev),
        ("BEV Optional at Runtime", test_bev_optional_at_runtime),
        ("Adapter Indices", test_adapter_indices),
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
        print("\nThe BALViT integration is complete and working correctly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
