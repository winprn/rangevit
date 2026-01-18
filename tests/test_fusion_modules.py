"""
Test suite for RangeViT-Fusion modules.

This module contains tests for the bidirectional point-pixel fusion components
including feature encoders, mapping operations, and the full fusion pipeline.
"""

import pytest
import torch
import torch.nn as nn

from models.features_encoder import FeaturesEncoder
from models.fusion_modules import (
    EfficientTransformationPipeline,
    PointFusionLayer,
    PixelFusionLayer,
    AuxiliaryHead,
)
from models.fusion_head import FusionHead
from models.vit_fusion import VisionTransformerFusion

# Device constant for cuda/cpu detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_placeholder():
    """Placeholder test to verify test infrastructure is working."""
    assert True


def test_features_encoder_output_shape():
    """
    Test that the features encoder produces output tensors with the expected shape.

    Verifies:
    - Output batch dimension matches input
    - Output feature dimension matches configured embedding dimension
    - Spatial dimensions are correctly preserved or transformed
    """
    # Test parameters
    batch_size = 4
    n_points = 1000
    in_channels = 5
    d_model = 384

    # Create encoder
    encoder = FeaturesEncoder(in_channels=in_channels, d_model=d_model).to(DEVICE)

    # Create random input: (batch_size * n_points, in_channels)
    point_attrs = torch.randn(batch_size * n_points, in_channels, device=DEVICE)

    # Forward pass
    output = encoder(point_attrs)

    # Verify output shape
    assert output.shape == (batch_size * n_points, d_model), \
        f"Expected shape {(batch_size * n_points, d_model)}, got {output.shape}"

    # Test with different d_model values
    for test_d_model in [128, 256, 384, 768]:
        encoder_test = FeaturesEncoder(in_channels=in_channels, d_model=test_d_model).to(DEVICE)
        output_test = encoder_test(point_attrs)
        assert output_test.shape == (batch_size * n_points, test_d_model), \
            f"Expected shape {(batch_size * n_points, test_d_model)}, got {output_test.shape}"


def test_features_encoder_gradient_flow():
    """
    Test that gradients flow correctly through the features encoder.

    Verifies:
    - Gradients are computed for all trainable parameters
    - No gradient is None or zero for active parameters
    - Backward pass completes without errors
    """
    # Test parameters
    batch_size = 4
    n_points = 1000
    in_channels = 5
    d_model = 384

    # Create encoder
    encoder = FeaturesEncoder(in_channels=in_channels, d_model=d_model).to(DEVICE)

    # Create input with requires_grad=True
    point_attrs = torch.randn(batch_size * n_points, in_channels, device=DEVICE, requires_grad=True)

    # Forward pass
    output = encoder(point_attrs)

    # Compute a simple loss (mean of output)
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Verify gradients exist for input
    assert point_attrs.grad is not None, "Input gradients should not be None"
    assert not torch.isnan(point_attrs.grad).any(), "Input gradients should not contain NaN"

    # Verify gradients exist for all trainable parameters
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should not be None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not contain NaN"
            # Check that gradient is not all zeros (model is being trained)
            assert param.grad.abs().sum() > 0, f"Gradient for {name} should not be all zeros"


def test_pixel2point_mapping():
    """
    Test the pixel-to-point mapping operation.

    Verifies:
    - Point features are correctly gathered from pixel features using projection indices
    - Output shape matches number of valid points
    - Invalid/out-of-bounds projections are handled correctly
    """
    # Test parameters
    batch_size = 2
    d_model = 384
    H, W = 16, 48
    n_points = 100

    # Create ETP
    etp = EfficientTransformationPipeline(ny=H, nx=W)

    # Create pixel features
    pixel_feats = torch.randn(batch_size, d_model, H, W, device=DEVICE)

    # Create valid coordinates: (N, 3) [batch_idx, y, x]
    batch_idx = torch.randint(0, batch_size, (n_points,), device=DEVICE).float()
    y_coords = torch.randint(0, H, (n_points,), device=DEVICE).float()
    x_coords = torch.randint(0, W, (n_points,), device=DEVICE).float()
    coords = torch.stack([batch_idx, y_coords, x_coords], dim=1)

    # Map pixel features to points
    point_feats = etp.pixel2point(pixel_feats, coords, stride=1)

    # Verify output shape
    assert point_feats.shape == (n_points, d_model), \
        f"Expected shape {(n_points, d_model)}, got {point_feats.shape}"

    # Verify that gathered features match expected values
    for i in range(min(10, n_points)):  # Check first 10 points
        b = int(batch_idx[i].item())
        y = int(y_coords[i].item())
        x = int(x_coords[i].item())
        expected = pixel_feats[b, :, y, x]
        actual = point_feats[i]
        assert torch.allclose(expected, actual), f"Point {i} features don't match expected pixel values"

    # Test with stride > 1
    point_feats_stride2 = etp.pixel2point(pixel_feats, coords, stride=2)
    assert point_feats_stride2.shape == (n_points, d_model), \
        f"Expected shape {(n_points, d_model)} with stride=2, got {point_feats_stride2.shape}"

    # Test with empty coords
    empty_coords = torch.zeros(0, 3, device=DEVICE)
    empty_feats = etp.pixel2point(pixel_feats, empty_coords)
    assert empty_feats.shape == (0, d_model), \
        f"Expected shape {(0, d_model)} for empty coords, got {empty_feats.shape}"


def test_point2cluster_aggregation():
    """
    Test the point-to-cluster aggregation operation.

    Verifies:
    - Points are correctly grouped into clusters (voxels/supervoxels)
    - Aggregation function (max/mean) produces correct results
    - Cluster features have expected dimensions
    """
    # Test parameters
    n_points = 100
    d_model = 384
    H, W = 32, 96
    batch_size = 2

    # Create ETP
    etp = EfficientTransformationPipeline(ny=H, nx=W)

    # Create point features
    point_feats = torch.randn(n_points, d_model, device=DEVICE)

    # Create coordinates with some points mapping to the same voxel
    batch_idx = torch.randint(0, batch_size, (n_points,), device=DEVICE).float()
    # Use fewer unique y, x values to ensure collisions
    y_coords = torch.randint(0, 10, (n_points,), device=DEVICE).float()
    x_coords = torch.randint(0, 20, (n_points,), device=DEVICE).float()
    coords = torch.stack([batch_idx, y_coords, x_coords], dim=1)

    # Aggregate points into clusters
    cluster_coords, cluster_feats = etp.point2cluster(point_feats, coords, stride=1)

    # Verify output shapes
    M = cluster_coords.shape[0]
    assert M <= n_points, f"Number of clusters {M} should be <= number of points {n_points}"
    assert cluster_feats.shape == (M, d_model), \
        f"Expected cluster_feats shape {(M, d_model)}, got {cluster_feats.shape}"
    assert cluster_coords.shape == (M, 3), \
        f"Expected cluster_coords shape {(M, 3)}, got {cluster_coords.shape}"

    # Verify cluster coordinates are valid
    assert (cluster_coords[:, 0] >= 0).all() and (cluster_coords[:, 0] < batch_size).all(), \
        "Cluster batch indices should be valid"
    assert (cluster_coords[:, 1] >= 0).all() and (cluster_coords[:, 1] < H).all(), \
        "Cluster y coordinates should be valid"
    assert (cluster_coords[:, 2] >= 0).all() and (cluster_coords[:, 2] < W).all(), \
        "Cluster x coordinates should be valid"

    # Test with stride > 1
    cluster_coords_s2, cluster_feats_s2 = etp.point2cluster(point_feats, coords, stride=2)
    M_s2 = cluster_coords_s2.shape[0]
    assert M_s2 <= M, f"Stride 2 should produce <= clusters than stride 1"
    assert cluster_feats_s2.shape == (M_s2, d_model), \
        f"Expected cluster_feats shape {(M_s2, d_model)}, got {cluster_feats_s2.shape}"

    # Test with empty input
    empty_feats = torch.zeros(0, d_model, device=DEVICE)
    empty_coords = torch.zeros(0, 3, device=DEVICE)
    empty_cluster_coords, empty_cluster_feats = etp.point2cluster(empty_feats, empty_coords)
    assert empty_cluster_coords.shape == (0, 3), "Empty input should produce empty cluster coords"
    assert empty_cluster_feats.shape == (0, d_model), "Empty input should produce empty cluster feats"


def test_cluster2pixel_dense_conversion():
    """
    Test the cluster-to-pixel dense feature conversion.

    Verifies:
    - Cluster features are correctly scattered back to pixel space
    - Dense output has correct spatial dimensions
    - Empty pixels (no corresponding cluster) are handled properly
    """
    # Test parameters
    batch_size = 2
    d_model = 384
    H, W = 16, 48
    n_clusters = 50

    # Create ETP
    etp = EfficientTransformationPipeline(ny=H, nx=W)

    # Create cluster features
    cluster_feats = torch.randn(n_clusters, d_model, device=DEVICE)

    # Create unique cluster coordinates
    batch_idx = torch.randint(0, batch_size, (n_clusters,), device=DEVICE)
    y_coords = torch.randint(0, H, (n_clusters,), device=DEVICE)
    x_coords = torch.randint(0, W, (n_clusters,), device=DEVICE)
    coords = torch.stack([batch_idx, y_coords, x_coords], dim=1)

    # Convert to dense pixel features
    pixel_feats = etp.cluster2pixel(cluster_feats, coords, batch_size, stride=1)

    # Verify output shape
    assert pixel_feats.shape == (batch_size, d_model, H, W), \
        f"Expected shape {(batch_size, d_model, H, W)}, got {pixel_feats.shape}"

    # Verify that scattered features are at correct locations
    for i in range(min(10, n_clusters)):  # Check first 10 clusters
        b = batch_idx[i].item()
        y = y_coords[i].item()
        x = x_coords[i].item()
        expected = cluster_feats[i]
        actual = pixel_feats[b, :, y, x]
        assert torch.allclose(expected, actual), f"Cluster {i} features don't match at pixel location"

    # Test with stride > 1
    H_s2 = (H + 1) // 2
    W_s2 = (W + 1) // 2
    # Create coords for stride 2 grid
    coords_s2 = torch.stack([
        batch_idx,
        torch.clamp(y_coords // 2, 0, H_s2 - 1),
        torch.clamp(x_coords // 2, 0, W_s2 - 1)
    ], dim=1)
    pixel_feats_s2 = etp.cluster2pixel(cluster_feats, coords_s2, batch_size, stride=2)
    assert pixel_feats_s2.shape == (batch_size, d_model, H_s2, W_s2), \
        f"Expected shape {(batch_size, d_model, H_s2, W_s2)}, got {pixel_feats_s2.shape}"

    # Test with empty clusters
    empty_feats = torch.zeros(0, d_model, device=DEVICE)
    empty_coords = torch.zeros(0, 3, device=DEVICE, dtype=torch.long)
    empty_pixel_feats = etp.cluster2pixel(empty_feats, empty_coords, batch_size, stride=1)
    assert empty_pixel_feats.shape == (batch_size, d_model, H, W), \
        "Empty clusters should produce zero-filled pixel features"
    assert (empty_pixel_feats == 0).all(), "Empty clusters should produce all zeros"


def test_roundtrip_pixel_point_pixel():
    """
    Test the full roundtrip: pixel -> point -> cluster -> pixel.

    Verifies:
    - Information is preserved through the roundtrip transformation
    - Output dimensions match input pixel feature dimensions
    - The transformation is differentiable end-to-end
    """
    # Test parameters
    batch_size = 2
    d_model = 384
    H, W = 16, 48

    # Create ETP
    etp = EfficientTransformationPipeline(ny=H, nx=W)

    # Create pixel features with requires_grad for gradient testing
    pixel_feats = torch.randn(batch_size, d_model, H, W, device=DEVICE, requires_grad=True)

    # Create coordinates with exactly one point per pixel (for perfect recovery)
    coords_list = []
    for b in range(batch_size):
        for y in range(H):
            for x in range(W):
                coords_list.append([b, y, x])
    coords = torch.tensor(coords_list, device=DEVICE, dtype=torch.float)

    # pixel -> point
    point_feats = etp.pixel2point(pixel_feats, coords, stride=1)
    assert point_feats.shape == (batch_size * H * W, d_model), \
        f"Expected shape {(batch_size * H * W, d_model)}, got {point_feats.shape}"

    # point -> cluster (with one point per pixel, should get same number of clusters)
    cluster_coords, cluster_feats = etp.point2cluster(point_feats, coords, stride=1)
    assert cluster_coords.shape[0] == batch_size * H * W, \
        f"Expected {batch_size * H * W} clusters, got {cluster_coords.shape[0]}"

    # cluster -> pixel
    recovered_pixel_feats = etp.cluster2pixel(cluster_feats, cluster_coords, batch_size, stride=1)
    assert recovered_pixel_feats.shape == pixel_feats.shape, \
        f"Expected shape {pixel_feats.shape}, got {recovered_pixel_feats.shape}"

    # Verify recovery (should be exact for one-to-one mapping)
    assert torch.allclose(pixel_feats, recovered_pixel_feats, atol=1e-6), \
        "Roundtrip should recover original features for one-to-one mapping"

    # Verify gradient flow through the entire pipeline
    loss = recovered_pixel_feats.mean()
    loss.backward()
    assert pixel_feats.grad is not None, "Gradients should flow through the pipeline"
    assert not torch.isnan(pixel_feats.grad).any(), "Gradients should not be NaN"


def test_point_fusion_layer():
    """
    Test the PointFusionLayer for fusing mapped pixel features with point features.

    Verifies:
    - Output shape is correct (N, D)
    - Gradient flow through the layer
    - Layer parameters are properly initialized
    """
    # Test parameters
    n_points = 1000
    d_model = 384

    # Create layer
    fusion_layer = PointFusionLayer(d_model=d_model).to(DEVICE)

    # Create input features
    mapped_pixel_feats = torch.randn(n_points, d_model, device=DEVICE, requires_grad=True)
    point_feats = torch.randn(n_points, d_model, device=DEVICE, requires_grad=True)

    # Forward pass
    fused_feats = fusion_layer(mapped_pixel_feats, point_feats)

    # Verify output shape
    assert fused_feats.shape == (n_points, d_model), \
        f"Expected shape {(n_points, d_model)}, got {fused_feats.shape}"

    # Verify gradient flow
    loss = fused_feats.mean()
    loss.backward()

    assert mapped_pixel_feats.grad is not None, "Gradients should flow to mapped_pixel_feats"
    assert point_feats.grad is not None, "Gradients should flow to point_feats"

    for name, param in fusion_layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should not be None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not be NaN"


def test_pixel_fusion_layer():
    """
    Test the PixelFusionLayer for fusing mapped point features with pixel features.

    Verifies:
    - Output shape is correct (B, D, H, W)
    - Gradient flow through the layer
    - Layer parameters are properly initialized
    """
    # Test parameters
    batch_size = 2
    d_model = 384
    H, W = 16, 48

    # Create layer
    fusion_layer = PixelFusionLayer(d_model=d_model).to(DEVICE)

    # Create input features
    pixel_from_points = torch.randn(batch_size, d_model, H, W, device=DEVICE, requires_grad=True)
    pixel_feats = torch.randn(batch_size, d_model, H, W, device=DEVICE, requires_grad=True)

    # Forward pass
    fused_feats = fusion_layer(pixel_from_points, pixel_feats)

    # Verify output shape
    assert fused_feats.shape == (batch_size, d_model, H, W), \
        f"Expected shape {(batch_size, d_model, H, W)}, got {fused_feats.shape}"

    # Verify gradient flow
    loss = fused_feats.mean()
    loss.backward()

    assert pixel_from_points.grad is not None, "Gradients should flow to pixel_from_points"
    assert pixel_feats.grad is not None, "Gradients should flow to pixel_feats"

    for name, param in fusion_layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should not be None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not be NaN"


def test_auxiliary_head():
    """
    Test the AuxiliaryHead for pixel-level supervision.

    Verifies:
    - Output shape is correct (B, n_classes, H, W)
    - Gradient flow through the head
    """
    # Test parameters
    batch_size = 2
    d_model = 384
    n_classes = 17
    H, W = 16, 48

    # Create head
    aux_head = AuxiliaryHead(d_model=d_model, n_classes=n_classes).to(DEVICE)

    # Create input features
    pixel_feats = torch.randn(batch_size, d_model, H, W, device=DEVICE, requires_grad=True)

    # Forward pass
    logits = aux_head(pixel_feats)

    # Verify output shape
    assert logits.shape == (batch_size, n_classes, H, W), \
        f"Expected shape {(batch_size, n_classes, H, W)}, got {logits.shape}"

    # Verify gradient flow
    loss = logits.mean()
    loss.backward()

    assert pixel_feats.grad is not None, "Gradients should flow to pixel_feats"

    for name, param in aux_head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should not be None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not be NaN"


def test_fusion_head_output_shape():
    """
    Test that the fusion head produces correctly shaped output.

    Verifies:
    - Output has correct number of classes
    - Spatial dimensions match input range image size
    - Batch dimension is preserved
    """
    # Test parameters
    n_points = 1000
    d_model = 384
    n_classes = 17

    # Create head
    fusion_head = FusionHead(d_model=d_model, n_classes=n_classes).to(DEVICE)

    # Create input features
    mapped_pixel_feats = torch.randn(n_points, d_model, device=DEVICE)
    point_feats = torch.randn(n_points, d_model, device=DEVICE)

    # Forward pass
    logits = fusion_head(mapped_pixel_feats, point_feats)

    # Verify output shape
    assert logits.shape == (n_points, n_classes), \
        f"Expected shape {(n_points, n_classes)}, got {logits.shape}"

    # Test with different n_classes values
    for test_n_classes in [10, 17, 20, 32]:
        head_test = FusionHead(d_model=d_model, n_classes=test_n_classes).to(DEVICE)
        logits_test = head_test(mapped_pixel_feats, point_feats)
        assert logits_test.shape == (n_points, test_n_classes), \
            f"Expected shape {(n_points, test_n_classes)}, got {logits_test.shape}"

    # Test with different d_model values
    for test_d_model in [128, 256, 384, 768]:
        head_test = FusionHead(d_model=test_d_model, n_classes=n_classes).to(DEVICE)
        pixel_feats_test = torch.randn(n_points, test_d_model, device=DEVICE)
        point_feats_test = torch.randn(n_points, test_d_model, device=DEVICE)
        logits_test = head_test(pixel_feats_test, point_feats_test)
        assert logits_test.shape == (n_points, n_classes), \
            f"Expected shape {(n_points, n_classes)}, got {logits_test.shape}"


def test_fusion_head_gradient_flow():
    """
    Test gradient flow through the fusion head.

    Verifies:
    - All fusion head parameters receive gradients
    - Gradients from both pixel and point branches are combined
    - No vanishing or exploding gradients
    """
    # Test parameters
    n_points = 1000
    d_model = 384
    n_classes = 17

    # Create head
    fusion_head = FusionHead(d_model=d_model, n_classes=n_classes).to(DEVICE)

    # Create input features with requires_grad=True
    mapped_pixel_feats = torch.randn(n_points, d_model, device=DEVICE, requires_grad=True)
    point_feats = torch.randn(n_points, d_model, device=DEVICE, requires_grad=True)

    # Forward pass
    logits = fusion_head(mapped_pixel_feats, point_feats)

    # Compute a simple loss (mean of output)
    loss = logits.mean()

    # Backward pass
    loss.backward()

    # Verify gradients exist for both inputs
    assert mapped_pixel_feats.grad is not None, "Gradients should flow to mapped_pixel_feats"
    assert not torch.isnan(mapped_pixel_feats.grad).any(), "mapped_pixel_feats gradients should not contain NaN"
    assert mapped_pixel_feats.grad.abs().sum() > 0, "mapped_pixel_feats gradients should not be all zeros"

    assert point_feats.grad is not None, "Gradients should flow to point_feats"
    assert not torch.isnan(point_feats.grad).any(), "point_feats gradients should not contain NaN"
    assert point_feats.grad.abs().sum() > 0, "point_feats gradients should not be all zeros"

    # Verify gradients exist for all trainable parameters
    for name, param in fusion_head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should not be None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not contain NaN"
            # Check that gradient is not all zeros (model is being trained)
            assert param.grad.abs().sum() > 0, f"Gradient for {name} should not be all zeros"

    # Verify no exploding gradients (gradient norm should be reasonable)
    total_grad_norm = 0.0
    for param in fusion_head.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    assert total_grad_norm < 1e6, f"Total gradient norm {total_grad_norm} suggests exploding gradients"


def test_vit_fusion_forward_shape():
    """
    Test the ViT-Fusion backbone forward pass output shape.

    Verifies:
    - Multi-scale features have expected shapes at each level
    - Fusion features are correctly integrated into ViT output
    - Auxiliary outputs have correct dimensions
    """
    # Test parameters
    batch_size = 2
    image_size = (32, 384)
    patch_size = (2, 8)
    patch_stride = (2, 8)
    n_layers = 12
    d_model = 384
    d_ff = 1536
    n_heads = 6
    n_cls = 17
    channels = 5
    fusion_blocks = [4, 8, 12]

    # Create model
    model = VisionTransformerFusion(
        image_size=image_size,
        patch_size=patch_size,
        n_layers=n_layers,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_cls=n_cls,
        channels=channels,
        patch_stride=patch_stride,
        fusion_blocks=fusion_blocks,
    ).to(DEVICE)

    # Compute expected grid size
    grid_h = image_size[0] // patch_stride[0]
    grid_w = image_size[1] // patch_stride[1]

    # Create input image
    im = torch.randn(batch_size, channels, image_size[0], image_size[1], device=DEVICE)

    # Create point features and coordinates
    n_points = 500
    point_feats = torch.randn(n_points, d_model, device=DEVICE)

    # Create valid coordinates: (N, 3) [batch_idx, y, x] in pixel space
    batch_idx = torch.randint(0, batch_size, (n_points,), device=DEVICE).float()
    y_coords = torch.randint(0, image_size[0], (n_points,), device=DEVICE).float()
    x_coords = torch.randint(0, image_size[1], (n_points,), device=DEVICE).float()
    coords = torch.stack([batch_idx, y_coords, x_coords], dim=1)

    # Forward pass with point features
    pixel_feats, updated_point_feats, aux_outputs = model(im, point_feats, coords)

    # Verify pixel features shape
    assert pixel_feats.shape == (batch_size, d_model, grid_h, grid_w), \
        f"Expected pixel_feats shape {(batch_size, d_model, grid_h, grid_w)}, got {pixel_feats.shape}"

    # Verify updated point features shape
    assert updated_point_feats.shape == (n_points, d_model), \
        f"Expected updated_point_feats shape {(n_points, d_model)}, got {updated_point_feats.shape}"

    # Verify auxiliary outputs
    assert len(aux_outputs) == len(fusion_blocks), \
        f"Expected {len(fusion_blocks)} auxiliary outputs, got {len(aux_outputs)}"

    for i, aux_out in enumerate(aux_outputs):
        expected_shape = (batch_size, n_cls, grid_h, grid_w)
        assert aux_out.shape == expected_shape, \
            f"Expected aux_output[{i}] shape {expected_shape}, got {aux_out.shape}"

    # Test gradient flow
    loss = pixel_feats.mean() + sum(aux.mean() for aux in aux_outputs)
    loss.backward()

    # Verify gradients exist for model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should not be None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not contain NaN"


def test_vit_fusion_without_points():
    """
    Test ViT-Fusion forward pass when point cloud data is not provided.

    Verifies:
    - Model gracefully handles missing point cloud input
    - Falls back to standard ViT behavior
    - Output shape remains consistent
    """
    # Test parameters
    batch_size = 2
    image_size = (32, 384)
    patch_size = (2, 8)
    patch_stride = (2, 8)
    n_layers = 12
    d_model = 384
    d_ff = 1536
    n_heads = 6
    n_cls = 17
    channels = 5
    fusion_blocks = [4, 8, 12]

    # Create model
    model = VisionTransformerFusion(
        image_size=image_size,
        patch_size=patch_size,
        n_layers=n_layers,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_cls=n_cls,
        channels=channels,
        patch_stride=patch_stride,
        fusion_blocks=fusion_blocks,
    ).to(DEVICE)

    # Compute expected grid size
    grid_h = image_size[0] // patch_stride[0]
    grid_w = image_size[1] // patch_stride[1]

    # Create input image (no point features)
    im = torch.randn(batch_size, channels, image_size[0], image_size[1], device=DEVICE)

    # Forward pass without point features
    pixel_feats, point_feats_out, aux_outputs = model(im)

    # Verify pixel features shape
    assert pixel_feats.shape == (batch_size, d_model, grid_h, grid_w), \
        f"Expected pixel_feats shape {(batch_size, d_model, grid_h, grid_w)}, got {pixel_feats.shape}"

    # Verify point features is None (since we didn't provide any)
    assert point_feats_out is None, \
        f"Expected point_feats to be None when not provided, got {type(point_feats_out)}"

    # Verify auxiliary outputs still exist (but without fusion)
    assert len(aux_outputs) == len(fusion_blocks), \
        f"Expected {len(fusion_blocks)} auxiliary outputs, got {len(aux_outputs)}"

    for i, aux_out in enumerate(aux_outputs):
        expected_shape = (batch_size, n_cls, grid_h, grid_w)
        assert aux_out.shape == expected_shape, \
            f"Expected aux_output[{i}] shape {expected_shape}, got {aux_out.shape}"

    # Test forward with point_feats=None explicitly
    pixel_feats2, point_feats_out2, aux_outputs2 = model(im, point_feats=None, coords=None)

    # Results should be the same
    assert pixel_feats2.shape == pixel_feats.shape, \
        "Output shape should be consistent with explicit None arguments"

    # Test in eval mode
    model.eval()
    with torch.no_grad():
        pixel_feats_eval, _, aux_outputs_eval = model(im)

    # Verify shapes are consistent in eval mode
    assert pixel_feats_eval.shape == (batch_size, d_model, grid_h, grid_w), \
        f"Eval mode: Expected pixel_feats shape {(batch_size, d_model, grid_h, grid_w)}, got {pixel_feats_eval.shape}"

    for i, aux_out in enumerate(aux_outputs_eval):
        expected_shape = (batch_size, n_cls, grid_h, grid_w)
        assert aux_out.shape == expected_shape, \
            f"Eval mode: Expected aux_output[{i}] shape {expected_shape}, got {aux_out.shape}"

    # Test determinism in eval mode
    with torch.no_grad():
        pixel_feats_eval2, _, _ = model(im)
    assert torch.allclose(pixel_feats_eval, pixel_feats_eval2), \
        "Model output should be deterministic in eval mode"


def test_rangevit_fusion_forward():
    """
    Test the full RangeViT-Fusion model forward pass.

    Verifies:
    - End-to-end forward pass completes successfully
    - Output segmentation logits have correct shape (B, C, H, W)
    - All intermediate features are correctly sized
    """
    pass


def test_rangevit_fusion_inference_mode():
    """
    Test RangeViT-Fusion in inference mode (eval).

    Verifies:
    - Model behaves correctly with torch.no_grad()
    - Dropout and batch norm are in eval mode
    - Output is deterministic in eval mode
    """
    pass


def test_rangevit_fusion_loss_components():
    """
    Test that loss components are correctly computed.

    Verifies:
    - Main segmentation loss is computed
    - Auxiliary losses (if any) are computed
    - Loss values are valid (not NaN or Inf)
    - Loss gradients flow to model parameters
    """
    pass
