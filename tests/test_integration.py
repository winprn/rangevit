"""
Integration tests for RangeViT-Fusion.

This module contains end-to-end integration tests that verify the full
training and inference pipelines work correctly together.
"""

import pytest
import torch
import torch.nn as nn

# Device constant for cuda/cpu detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_full_training_step():
    """Test a complete forward-backward pass."""
    from models.rangevit_fusion import RangeViTFusion

    model = RangeViTFusion(
        in_channels=5,
        n_cls=17,
        backbone="vit_small_patch16_384",
        image_size=(32, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate training step
    model.train()
    batch_size = 2
    n_points = 500

    images = torch.randn(batch_size, 5, 32, 384, device=DEVICE)
    point_attrs = torch.randn(batch_size * n_points, 5, device=DEVICE)
    coords = torch.zeros(batch_size * n_points, 3, dtype=torch.long, device=DEVICE)
    coords[:, 0] = torch.arange(batch_size, device=DEVICE).repeat_interleave(n_points)
    coords[:, 1] = torch.randint(0, 32, (batch_size * n_points,), device=DEVICE)
    coords[:, 2] = torch.randint(0, 384, (batch_size * n_points,), device=DEVICE)
    labels = torch.randint(1, 17, (batch_size * n_points,), device=DEVICE)

    # Forward
    outputs = model(images, point_attrs, coords, labels)
    loss = outputs["losses"]["loss"]

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check gradients flowed to all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_inference_speed():
    """Basic inference speed sanity check."""
    import time

    from models.rangevit_fusion import RangeViTFusion

    model = RangeViTFusion(
        in_channels=5,
        n_cls=17,
        backbone="vit_small_patch16_384",
        image_size=(32, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
    ).to(DEVICE)
    model.eval()

    images = torch.randn(1, 5, 32, 384, device=DEVICE)
    point_attrs = torch.randn(10000, 5, device=DEVICE)
    coords = torch.zeros(10000, 3, dtype=torch.long, device=DEVICE)
    coords[:, 1] = torch.randint(0, 32, (10000,), device=DEVICE)
    coords[:, 2] = torch.randint(0, 384, (10000,), device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(images, point_attrs, coords)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Time inference
    start = time.time()
    n_runs = 10
    with torch.no_grad():
        for _ in range(n_runs):
            model(images, point_attrs, coords)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.time() - start) / n_runs
    print(f"Inference time: {elapsed*1000:.2f}ms")

    # Should be reasonably fast
    if DEVICE.type == "cuda":
        assert elapsed < 1.0, f"Inference too slow: {elapsed}s"


def test_model_save_load():
    """Test model can be saved and loaded."""
    import tempfile

    from models.rangevit_fusion import RangeViTFusion

    model = RangeViTFusion(
        in_channels=5,
        n_cls=17,
        backbone="vit_small_patch16_384",
        image_size=(32, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
    ).to(DEVICE)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(model.state_dict(), f.name)

        # Load
        model2 = RangeViTFusion(
            in_channels=5,
            n_cls=17,
            backbone="vit_small_patch16_384",
            image_size=(32, 384),
            new_patch_size=(2, 8),
            new_patch_stride=(2, 8),
        ).to(DEVICE)
        model2.load_state_dict(torch.load(f.name))

    # Verify outputs match
    model.eval()
    model2.eval()

    images = torch.randn(1, 5, 32, 384, device=DEVICE)
    point_attrs = torch.randn(100, 5, device=DEVICE)
    coords = torch.zeros(100, 3, dtype=torch.long, device=DEVICE)
    coords[:, 1] = torch.randint(0, 32, (100,), device=DEVICE)
    coords[:, 2] = torch.randint(0, 384, (100,), device=DEVICE)

    with torch.no_grad():
        out1 = model(images, point_attrs, coords)
        out2 = model2(images, point_attrs, coords)

    assert torch.allclose(out1["point_logits"], out2["point_logits"])


def test_parameter_count():
    """Verify model parameter count."""
    from models.rangevit_fusion import RangeViTFusion

    model = RangeViTFusion(
        in_channels=5,
        n_cls=17,
        backbone="vit_small_patch16_384",
        image_size=(32, 384),
        new_patch_size=(2, 8),
        new_patch_stride=(2, 8),
    )

    stats = model.count_parameters()

    # ViT-Small has ~22M params, fusion adds ~2-3M
    assert stats["total"] > 20_000_000
    assert stats["total"] < 30_000_000
