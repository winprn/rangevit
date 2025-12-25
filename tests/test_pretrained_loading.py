"""
Test suite for pretrained weight loading with channel adaptation.

Tests the adapt_input_conv function with both 'repeat' and 'grayscale' methods
for adapting RGB (3-channel) pretrained weights to LiDAR (5-channel) input.
"""

import torch
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_utils import adapt_input_conv


def test_adapt_repeat():
    """Test the 'repeat' method for channel adaptation."""
    print("\n=== Testing 'repeat' method ===")

    # Create dummy RGB weight tensor
    weight = torch.randn(64, 3, 3, 3)  # [out_channels, in_channels, kernel_h, kernel_w]
    print(f"Original weight shape: {weight.shape}")

    # Adapt to 5 channels using repeat method
    adapted = adapt_input_conv(5, weight, method='repeat')
    print(f"Adapted weight shape: {adapted.shape}")

    # Verify shape
    assert adapted.shape == (64, 5, 3, 3), f"Expected shape (64, 5, 3, 3), got {adapted.shape}"

    # Verify scaling factor (should be multiplied by 3/5)
    expected_scale = 3 / 5
    # First 3 channels should match original * (3/5)
    assert torch.allclose(adapted[:, :3, :, :], weight * expected_scale, rtol=1e-5), \
        "First 3 channels don't match expected scaling"

    # Channels 3 and 4 should match channels 0 and 1 with scaling
    assert torch.allclose(adapted[:, 3, :, :], weight[:, 0, :, :] * expected_scale, rtol=1e-5), \
        "Channel 3 doesn't match channel 0 with scaling"
    assert torch.allclose(adapted[:, 4, :, :], weight[:, 1, :, :] * expected_scale, rtol=1e-5), \
        "Channel 4 doesn't match channel 1 with scaling"

    print("✓ Repeat method: shape correct")
    print("✓ Repeat method: scaling factor correct")
    print("✓ Repeat method: channel replication correct")


def test_adapt_grayscale():
    """Test the 'grayscale' method for channel adaptation."""
    print("\n=== Testing 'grayscale' method ===")

    # Create dummy RGB weight tensor
    weight = torch.randn(64, 3, 3, 3)
    print(f"Original weight shape: {weight.shape}")

    # Adapt to 5 channels using grayscale method
    adapted = adapt_input_conv(5, weight, method='grayscale')
    print(f"Adapted weight shape: {adapted.shape}")

    # Verify shape
    assert adapted.shape == (64, 5, 3, 3), f"Expected shape (64, 5, 3, 3), got {adapted.shape}"

    # Verify all 5 channels are identical (grayscale duplicated)
    for i in range(1, 5):
        assert torch.allclose(adapted[:, i, :, :], adapted[:, 0, :, :], rtol=1e-5), \
            f"Channel {i} is not identical to channel 0"

    # Verify grayscale computation using ITU-R BT.601 weights
    gray_weights = torch.tensor([0.299, 0.587, 0.114])
    expected_gray = (weight * gray_weights[None, :, None, None]).sum(dim=1)

    for i in range(5):
        assert torch.allclose(adapted[:, i, :, :], expected_gray, rtol=1e-5), \
            f"Channel {i} doesn't match expected grayscale conversion"

    print("✓ Grayscale method: shape correct")
    print("✓ Grayscale method: all channels identical")
    print("✓ Grayscale method: ITU-R BT.601 weights correct")


def test_adapt_different_target_channels():
    """Test adaptation to different target channel counts."""
    print("\n=== Testing different target channel counts ===")

    weight = torch.randn(32, 3, 5, 5)

    # Test 1 channel (should average)
    adapted_1 = adapt_input_conv(1, weight)
    assert adapted_1.shape == (32, 1, 5, 5)
    print("✓ Adaptation to 1 channel works")

    # Test 3 channels (should return unchanged)
    adapted_3 = adapt_input_conv(3, weight, method='repeat')
    assert adapted_3.shape == (32, 3, 5, 5)
    assert torch.allclose(adapted_3, weight)
    print("✓ Adaptation to 3 channels returns unchanged")

    # Test 7 channels with repeat
    adapted_7 = adapt_input_conv(7, weight, method='repeat')
    assert adapted_7.shape == (32, 7, 5, 5)
    print("✓ Adaptation to 7 channels works")

    # Test 10 channels with grayscale
    adapted_10 = adapt_input_conv(10, weight, method='grayscale')
    assert adapted_10.shape == (32, 10, 5, 5)
    print("✓ Adaptation to 10 channels works")


def test_dtype_preservation():
    """Test that the dtype is preserved during adaptation."""
    print("\n=== Testing dtype preservation ===")

    # Test with float32
    weight_f32 = torch.randn(16, 3, 3, 3, dtype=torch.float32)
    adapted_f32 = adapt_input_conv(5, weight_f32, method='repeat')
    assert adapted_f32.dtype == torch.float32
    print("✓ Float32 dtype preserved")

    # Test with float16
    weight_f16 = torch.randn(16, 3, 3, 3, dtype=torch.float16)
    adapted_f16 = adapt_input_conv(5, weight_f16, method='grayscale')
    assert adapted_f16.dtype == torch.float16
    print("✓ Float16 dtype preserved")


def test_invalid_method():
    """Test that invalid methods raise ValueError."""
    print("\n=== Testing invalid method handling ===")

    weight = torch.randn(16, 3, 3, 3)

    try:
        adapted = adapt_input_conv(5, weight, method='invalid_method')
        assert False, "Should have raised ValueError for invalid method"
    except ValueError as e:
        assert "Unknown adaptation method" in str(e)
        print("✓ ValueError raised for invalid method")


def test_magnitude_comparison():
    """Compare magnitude of weights between methods."""
    print("\n=== Comparing magnitude between methods ===")

    weight = torch.randn(64, 3, 3, 3)

    adapted_repeat = adapt_input_conv(5, weight, method='repeat')
    adapted_gray = adapt_input_conv(5, weight, method='grayscale')

    mean_repeat = adapted_repeat.abs().mean()
    mean_gray = adapted_gray.abs().mean()

    print(f"Mean magnitude (repeat): {mean_repeat:.6f}")
    print(f"Mean magnitude (grayscale): {mean_gray:.6f}")

    # Both should have reasonable magnitudes (not zero or extreme)
    assert mean_repeat > 0.01, "Repeat method has too small magnitude"
    assert mean_gray > 0.01, "Grayscale method has too small magnitude"
    print("✓ Both methods produce reasonable magnitudes")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*60)
    print("Running Pretrained Weight Loading Tests")
    print("="*60)

    try:
        test_adapt_repeat()
        test_adapt_grayscale()
        test_adapt_different_target_channels()
        test_dtype_preservation()
        test_invalid_method()
        test_magnitude_comparison()

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        return True

    except AssertionError as e:
        print("\n" + "="*60)
        print(f"❌ Test failed: {e}")
        print("="*60)
        return False
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ Unexpected error: {e}")
        print("="*60)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
