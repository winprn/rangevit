# Test Script for RangeFormer Assertions
# Validates that all assertions work correctly and catch errors

import torch
import numpy as np
import sys


def test_rem_assertions():
    """Test REM (Range Embedding Module) assertions."""
    print("\n" + "="*60)
    print("Testing REM Assertions")
    print("="*60)

    from models.rangeformer.backbone import REM

    rem = REM()

    # Test 1: Valid input
    print("\n1. Testing valid input (B=2, C=6, H=64, W=1024)...")
    try:
        x = torch.randn(2, 6, 64, 1024)
        out = rem(x)
        assert out.shape == (2, 128, 64, 1024)
        print("   ✓ Passed: Output shape correct")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: Wrong number of channels
    print("\n2. Testing wrong number of channels (5 instead of 6)...")
    try:
        x_wrong = torch.randn(2, 5, 64, 1024)
        rem(x_wrong)
        print("   ✗ Failed: Should have caught wrong channel count")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error - {str(e)[:80]}...")

    # Test 3: Wrong dimensions
    print("\n3. Testing wrong dimensions (3D instead of 4D)...")
    try:
        x_wrong = torch.randn(2, 6, 64)
        rem(x_wrong)
        print("   ✗ Failed: Should have caught wrong dimensions")
        return False
    except (AssertionError, RuntimeError) as e:
        print(f"   ✓ Passed: Correctly caught error")

    return True


def test_patch_embed_assertions():
    """Test PatchEmbedOverlap assertions."""
    print("\n" + "="*60)
    print("Testing PatchEmbedOverlap Assertions")
    print("="*60)

    from models.rangeformer.backbone import PatchEmbedOverlap

    # Test 1: Valid stride=1
    print("\n1. Testing valid input with stride=1...")
    try:
        patch_embed = PatchEmbedOverlap(128, 128, stride=1)
        x = torch.randn(2, 128, 64, 1024)
        out = patch_embed(x)
        assert out.shape == (2, 128, 64, 1024)
        print("   ✓ Passed: Output shape correct for stride=1")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: Valid stride=2
    print("\n2. Testing valid input with stride=2...")
    try:
        patch_embed = PatchEmbedOverlap(128, 256, stride=2)
        x = torch.randn(2, 128, 64, 1024)
        out = patch_embed(x)
        assert out.shape == (2, 256, 32, 512)
        print("   ✓ Passed: Output shape correct for stride=2")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 3: Wrong input channels
    print("\n3. Testing wrong input channels...")
    try:
        patch_embed = PatchEmbedOverlap(128, 256, stride=2)
        x_wrong = torch.randn(2, 64, 64, 1024)  # Wrong channels
        patch_embed(x_wrong)
        print("   ✗ Failed: Should have caught wrong channel count")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    return True


def test_transformer_block_assertions():
    """Test TransformerBlock2D assertions."""
    print("\n" + "="*60)
    print("Testing TransformerBlock2D Assertions")
    print("="*60)

    from models.rangeformer.backbone import TransformerBlock2D

    # Test 1: Valid configuration
    print("\n1. Testing valid input (dim=128, heads=4)...")
    try:
        block = TransformerBlock2D(dim=128, num_heads=4)
        x = torch.randn(2, 128, 32, 512)
        out = block(x)
        assert out.shape == x.shape
        print("   ✓ Passed: Output shape matches input")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: Invalid heads (not divisible)
    print("\n2. Testing invalid num_heads (dim=128, heads=7)...")
    try:
        block = TransformerBlock2D(dim=128, num_heads=7)
        x = torch.randn(2, 128, 32, 512)
        block(x)
        print("   ✗ Failed: Should have caught invalid head count")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    # Test 3: Wrong channel dimension
    print("\n3. Testing wrong channel dimension...")
    try:
        block = TransformerBlock2D(dim=128, num_heads=4)
        x_wrong = torch.randn(2, 256, 32, 512)  # Wrong dim
        block(x_wrong)
        print("   ✗ Failed: Should have caught wrong dimension")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    return True


def test_backbone_assertions():
    """Test RangeFormerBackbone assertions."""
    print("\n" + "="*60)
    print("Testing RangeFormerBackbone Assertions")
    print("="*60)

    from models.rangeformer.backbone import RangeFormerBackbone

    # Test 1: Valid configuration
    print("\n1. Testing valid configuration...")
    try:
        backbone = RangeFormerBackbone(
            H=64, W=1024, num_classes=19,
            depths=[2, 2, 6, 2],
            stage_channels=[128, 128, 320, 512],
            heads=[3, 4, 6, 3]
        )
        x = torch.randn(2, 6, 64, 1024)
        features = backbone(x)
        assert len(features) == 4
        assert features[0].shape == (2, 128, 64, 1024)  # Stage 1
        assert features[1].shape == (2, 128, 32, 512)   # Stage 2
        assert features[2].shape == (2, 320, 16, 256)   # Stage 3
        assert features[3].shape == (2, 512, 8, 128)    # Stage 4
        print("   ✓ Passed: All stage outputs correct")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: Invalid depths configuration
    print("\n2. Testing invalid depths (only 3 values)...")
    try:
        backbone = RangeFormerBackbone(
            H=64, W=1024, num_classes=19,
            depths=[2, 2, 6],  # Only 3!
            stage_channels=[128, 128, 320, 512],
            heads=[3, 4, 6, 3]
        )
        print("   ✗ Failed: Should have caught invalid depths")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    # Test 3: H not divisible by 8
    print("\n3. Testing H not divisible by 8 (H=63)...")
    try:
        backbone = RangeFormerBackbone(
            H=63, W=1024, num_classes=19  # 63 not divisible by 8
        )
        x = torch.randn(2, 6, 63, 1024)
        backbone(x)
        print("   ✗ Failed: Should have caught invalid H")
        return False
    except (AssertionError, RuntimeError) as e:
        print(f"   ✓ Passed: Correctly caught error")

    # Test 4: Channels not divisible by heads
    print("\n4. Testing channels not divisible by heads...")
    try:
        backbone = RangeFormerBackbone(
            H=64, W=1024, num_classes=19,
            stage_channels=[128, 128, 320, 512],
            heads=[3, 7, 6, 3]  # 128 not divisible by 7!
        )
        print("   ✗ Failed: Should have caught invalid channel/head config")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    return True


def test_decoder_assertions():
    """Test SegmentationHead assertions."""
    print("\n" + "="*60)
    print("Testing SegmentationHead Assertions")
    print("="*60)

    from models.rangeformer.decoder import SegmentationHead

    # Test 1: Valid input
    print("\n1. Testing valid input...")
    try:
        head = SegmentationHead(
            stage_channels=[128, 128, 320, 512],
            out_ch_unify=256,
            num_classes=19,
            H=64, W=1024
        )

        # Create fake features
        features = [
            torch.randn(2, 128, 64, 1024),
            torch.randn(2, 128, 32, 512),
            torch.randn(2, 320, 16, 256),
            torch.randn(2, 512, 8, 128)
        ]

        logits, auxs = head(features)
        assert logits.shape == (2, 19, 64, 1024)
        assert len(auxs) == 4
        print("   ✓ Passed: Decoder output correct")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: Wrong number of features
    print("\n2. Testing wrong number of features (3 instead of 4)...")
    try:
        features_wrong = [
            torch.randn(2, 128, 64, 1024),
            torch.randn(2, 128, 32, 512),
            torch.randn(2, 320, 16, 256)
        ]
        head(features_wrong)
        print("   ✗ Failed: Should have caught wrong number of features")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    # Test 3: Batch size mismatch
    print("\n3. Testing batch size mismatch...")
    try:
        features_wrong = [
            torch.randn(2, 128, 64, 1024),
            torch.randn(4, 128, 32, 512),  # Different batch size!
            torch.randn(2, 320, 16, 256),
            torch.randn(2, 512, 8, 128)
        ]
        head(features_wrong)
        print("   ✗ Failed: Should have caught batch size mismatch")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    return True


def test_rangeformer_model_assertions():
    """Test complete RangeFormer model assertions."""
    print("\n" + "="*60)
    print("Testing RangeFormer Model Assertions")
    print("="*60)

    from models.rangeformer import RangeFormer

    # Test 1: Valid configuration
    print("\n1. Testing valid model...")
    try:
        model = RangeFormer(H=64, W=1024, num_classes=19)
        x = torch.randn(2, 6, 64, 1024)
        logits, auxs = model(x)
        assert logits.shape == (2, 19, 64, 1024)
        assert len(auxs) == 4
        print("   ✓ Passed: Model output correct")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: H not divisible by 8
    print("\n2. Testing H not divisible by 8...")
    try:
        model = RangeFormer(H=63, W=1024, num_classes=19)
        print("   ✗ Failed: Should have caught invalid H")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    # Test 3: W not divisible by 8
    print("\n3. Testing W not divisible by 8...")
    try:
        model = RangeFormer(H=64, W=1000, num_classes=19)
        print("   ✗ Failed: Should have caught invalid W")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    # Test 4: Wrong input channels
    print("\n4. Testing wrong input channels (5 instead of 6)...")
    try:
        model = RangeFormer(H=64, W=1024, num_classes=19)
        x_wrong = torch.randn(2, 5, 64, 1024)
        model(x_wrong)
        print("   ✗ Failed: Should have caught wrong channel count")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    # Test 5: Invalid depths
    print("\n5. Testing invalid depths configuration...")
    try:
        model = RangeFormer(
            H=64, W=1024, num_classes=19,
            depths=[2, 2, 6]  # Only 3 values!
        )
        print("   ✗ Failed: Should have caught invalid depths")
        return False
    except AssertionError as e:
        print(f"   ✓ Passed: Correctly caught error")

    return True


def main():
    """Run all assertion tests."""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "RangeFormer Assertion Tests" + " "*16 + "║")
    print("╚" + "="*58 + "╝")

    tests = [
        ("REM Module", test_rem_assertions),
        ("PatchEmbedOverlap Module", test_patch_embed_assertions),
        ("TransformerBlock2D Module", test_transformer_block_assertions),
        ("RangeFormerBackbone Module", test_backbone_assertions),
        ("SegmentationHead Module", test_decoder_assertions),
        ("Complete RangeFormer Model", test_rangeformer_model_assertions),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:12} {name}")

    print("="*60)
    print(f"Results: {passed}/{total} test suites passed")
    print("="*60)

    if passed == total:
        print("\n🎉 All assertion tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
