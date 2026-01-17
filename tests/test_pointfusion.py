# tests/test_pointfusion.py
"""Integration tests for PointFusion model."""

import torch
import pytest


def test_point_mlp_encoder():
    """Test PointMLPEncoder forward pass."""
    from models.fusion.point_encoder import PointMLPEncoder

    encoder = PointMLPEncoder(in_channels=7, hidden_dim=256, if_dist=False)
    x = torch.randn(1000, 7)  # 1000 points, 7 features
    out = encoder(x)

    assert out.shape == (1000, 256), f"Expected (1000, 256), got {out.shape}"
    print("PointMLPEncoder test passed!")


def test_cross_attention():
    """Test PointToRangeCrossAttention forward pass."""
    from models.fusion.fusion_modules import PointToRangeCrossAttention

    attn = PointToRangeCrossAttention(dim=256, window_size=3, num_heads=4, if_dist=False)

    point_feats = torch.randn(100, 256)  # 100 points
    vit_feats = torch.randn(64, 768, 256)  # Range image features
    proj_y = torch.randint(0, 64, (100,))
    proj_x = torch.randint(0, 768, (100,))

    out = attn(point_feats, vit_feats, proj_y, proj_x)

    assert out.shape == (100, 256), f"Expected (100, 256), got {out.shape}"
    print("CrossAttention test passed!")


def test_pointfusion_model():
    """Test PointFusionRangeViT forward pass."""
    from models.fusion.pointfusion_rangevit import PointFusionRangeViT

    model = PointFusionRangeViT(
        range_in_channels=5,
        n_cls=20,
        vit_backbone='vit_small_patch16_384',
        image_size=(64, 768),
        patch_size=(2, 8),
        patch_stride=(2, 8),
        point_in_channels=7,
        point_hidden_dim=256,
        cross_attn_window=3,
        cross_attn_heads=4,
        if_dist=False,
    )

    # Simulate batch
    B = 2
    H, W = 64, 768
    N_per_batch = 5000

    range_image = torch.randn(B, 5, H, W)
    point_features = torch.randn(B * N_per_batch, 4)
    cluster_offset = torch.randn(B * N_per_batch, 3)
    batch_indices = torch.cat([
        torch.zeros(N_per_batch, dtype=torch.long),
        torch.ones(N_per_batch, dtype=torch.long),
    ])
    range_pxpy = torch.rand(B * N_per_batch, 2) * 2 - 1  # [-1, 1]

    with torch.no_grad():
        out = model(range_image, point_features, cluster_offset, batch_indices, range_pxpy)

    assert out.shape == (B * N_per_batch, 20), f"Expected ({B * N_per_batch}, 20), got {out.shape}"
    print("PointFusionRangeViT test passed!")

    # Print parameter counts
    params = model.count_parameters()
    print("\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")


if __name__ == '__main__':
    test_point_mlp_encoder()
    test_cross_attention()
    test_pointfusion_model()
    print("\nAll tests passed!")
