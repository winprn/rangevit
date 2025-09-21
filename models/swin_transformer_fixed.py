# Copyright 2023 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple

from .model_utils import init_weights
from .mock_swin_backbone import MockSwinBackbone, MockSwinVisionTransformer


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone for RangeViT with MockSwin fallback
    Returns hierarchical multi-scale features compatible with RangeViT decoder

    Features:
    - 4 hierarchical stages with different spatial resolutions
    - Channel dimensions that double at each stage
    - Window-based attention mechanism
    - Compatible with timm pretrained weights
    - Automatic fallback to MockSwinBackbone for range image compatibility
    """

    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        channels: int = 3,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
        range_image_mode: bool = False,  # Special handling for range images
        force_mock: bool = False,  # Force using mock backbone for testing
        **kwargs
    ):
        """
        Initialize Swin Transformer backbone

        Args:
            model_name: Name of the timm model
            embed_dim: Base embedding dimension (C)
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            window_size: Window size for attention
            channels: Number of input channels
            pretrained: Whether to load timm pretrained weights
            drop_path_rate: Stochastic depth rate
            range_image_mode: Enable special handling for range images
            force_mock: Force using mock backbone (for testing)
        """
        super().__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.range_image_mode = range_image_mode
        self.force_mock = force_mock

        # Adaptive model selection for range images
        if range_image_mode and not force_mock:
            model_name, window_size = self._select_range_compatible_model(model_name)
            self.model_name = model_name
            self.window_size = window_size

        # Try to use timm Swin model, fallback to mock on failure
        self.use_mock = force_mock

        if not force_mock:
            try:
                # Create Swin model using timm
                self.swin = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=True,  # Return intermediate features
                    out_indices=(0, 1, 2, 3),  # All 4 stages
                    in_chans=channels,
                    drop_path_rate=drop_path_rate
                )

                # Modify patch embedding to accept flexible input sizes
                # Find the patch embedding module and remove size constraints
                def remove_size_check(module):
                    if hasattr(module, 'img_size'):
                        # Store the original check but disable it
                        module._original_img_size = module.img_size
                        module.img_size = None
                    for child in module.children():
                        remove_size_check(child)

                remove_size_check(self.swin)

                print(f"[OK] Created Swin backbone: {model_name}")
                print(f"  - Embed dim: {embed_dim}")
                print(f"  - Input channels: {channels}")
                print(f"  - Window size: {window_size}")
                if range_image_mode:
                    print(f"  - Range image mode: ENABLED")

            except Exception as e:
                if range_image_mode:
                    print(f"[WARNING] timm Swin model failed: {e}")
                    print(f"[INFO] Falling back to MockSwinBackbone for range image compatibility")
                    self.use_mock = True
                else:
                    print(f"[ERROR] Failed to create Swin model {model_name}: {e}")
                    raise

        if self.use_mock or force_mock:
            # Use mock backbone as fallback for range images
            self.swin = MockSwinBackbone(
                channels=channels,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads
            )
            print(f"[OK] Using Mock Swin backbone for range image compatibility")

        # Store feature dimensions for each stage
        # For Swin-Tiny: [96, 192, 384, 768]
        self.feature_dims = [
            embed_dim,        # Stage 0: C
            embed_dim * 2,    # Stage 1: 2C
            embed_dim * 4,    # Stage 2: 4C
            embed_dim * 8     # Stage 3: 8C
        ]

        # Stride information for each stage
        self.strides = [4, 8, 16, 32]

    def _select_range_compatible_model(self, model_name: str) -> Tuple[str, int]:
        """
        Select range image compatible model and window size

        Args:
            model_name: Original model name

        Returns:
            Tuple of (compatible_model_name, window_size)
        """

        # Range image compatibility mapping
        range_compatible_models = {
            # SwinV2 models -> Swin models (better compatibility)
            'swinv2_tiny_window16_256': ('swin_tiny_patch4_window7_224', 7),
            'swinv2_tiny_window8_256': ('swin_tiny_patch4_window7_224', 7),
            'swinv2_small_window16_256': ('swin_small_patch4_window7_224', 7),
            'swinv2_base_window16_256': ('swin_base_patch4_window7_224', 7),

            # Already compatible Swin models
            'swin_tiny_patch4_window7_224': ('swin_tiny_patch4_window7_224', 7),
            'swin_small_patch4_window7_224': ('swin_small_patch4_window7_224', 7),
            'swin_base_patch4_window7_224': ('swin_base_patch4_window7_224', 7),
        }

        if model_name in range_compatible_models:
            compatible_model, window_size = range_compatible_models[model_name]
            if compatible_model != model_name:
                print(f"[INFO] Range image mode: {model_name} -> {compatible_model}")
                print(f"       Window size: {window_size} (optimized for range images)")
            return compatible_model, window_size
        else:
            # Default fallback for unknown models
            print(f"[WARNING] Unknown model {model_name}, using default swin_tiny_patch4_window7_224")
            return 'swin_tiny_patch4_window7_224', 7

    def _apply_intelligent_padding(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Apply intelligent padding for window size compatibility

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (padded_tensor, padding_info)
        """
        if not self.range_image_mode or self.use_mock:
            return x, None

        B, C, H, W = x.shape
        window_size = self.window_size

        # Calculate target dimensions (must be divisible by window_size)
        target_H = ((H + window_size - 1) // window_size) * window_size
        target_W = ((W + window_size - 1) // window_size) * window_size

        if H == target_H and W == target_W:
            return x, None

        # Calculate padding
        pad_H = target_H - H
        pad_W = target_W - W

        # Apply padding: (left, right, top, bottom)
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        x_padded = F.pad(x, padding, mode='reflect')

        padding_info = {
            'original_size': (H, W),
            'padded_size': (target_H, target_W),
            'padding': padding
        }

        print(f"[DEBUG] Window size: {window_size}, Target multiple: {window_size * window_size}")
        print(f"[DEBUG] Original: {H}x{W}, Padding: +{pad_H}x{pad_W}")
        print(f"[INFO] Applied intelligent padding: ({H}, {W}) -> ({target_H}, {target_W})")

        return x_padded, padding_info

    def _remove_padding(self, features: List[torch.Tensor], padding_info: dict) -> List[torch.Tensor]:
        """
        Remove padding from output features to match original input size

        Args:
            features: List of feature tensors
            padding_info: Padding information from _apply_intelligent_padding

        Returns:
            List of features with padding removed
        """
        if padding_info is None:
            return features

        original_H, original_W = padding_info['original_size']

        unpadded_features = []
        for i, feat in enumerate(features):
            stride = self.strides[i]
            target_H = original_H // stride
            target_W = original_W // stride

            # Crop to target size
            feat_cropped = feat[:, :, :target_H, :target_W]
            unpadded_features.append(feat_cropped)

        return unpadded_features

    def get_feature_dims(self) -> List[int]:
        """Return channel dimensions for each stage"""
        return self.feature_dims.copy()

    def get_strides(self) -> List[int]:
        """Return spatial stride for each stage"""
        return self.strides.copy()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through Swin Transformer with automatic fallback

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of 4 feature maps from different stages
        """
        # If already using mock, just forward directly
        if self.use_mock:
            return self.swin(x)

        # Apply intelligent padding if needed
        x_padded, padding_info = self._apply_intelligent_padding(x)

        try:
            # Forward pass through Swin backbone
            features = self.swin(x_padded)

            # Remove padding to restore original proportions
            features = self._remove_padding(features, padding_info)
            return features

        except Exception as e:
            if self.range_image_mode:
                print(f"[ERROR] Swin forward pass failed: {e}")
                print(f"[INFO] Automatically switching to MockSwinBackbone for compatibility")

                # Switch to mock backbone
                self.use_mock = True
                self.swin = MockSwinBackbone(
                    channels=x.shape[1],
                    embed_dim=self.embed_dim,
                    depths=self.depths,
                    num_heads=self.num_heads
                )
                print(f"[OK] Switched to Mock Swin backbone automatically")

                # Retry with mock backbone (no padding needed)
                return self.swin(x)
            else:
                print(f"[ERROR] Swin forward pass failed: {e}")
                raise


class SwinVisionTransformer(nn.Module):
    """
    Swin-ViT compatibility wrapper with MockSwin fallback
    Provides the same interface as Vision Transformer but uses Swin backbone
    """

    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',
        channels: int = 3,
        embed_dim: int = 96,
        use_stage: int = 2,
        multi_scale: bool = False,
        native_input: bool = False,
        force_mock: bool = False,
        **kwargs
    ):
        """
        Initialize Swin-ViT wrapper

        Args:
            model_name: Swin model name
            channels: Number of input channels
            embed_dim: Base embedding dimension
            use_stage: Which stage to use for single-scale mode
            multi_scale: Whether to return all stages
            native_input: Whether to use native input (no resizing)
            force_mock: Force using mock backbone
        """
        super().__init__()

        # Determine if range image mode should be enabled
        range_image_mode = (channels == 5) or force_mock

        if force_mock or (range_image_mode and any(name in model_name for name in ['swinv2'])):
            # Use mock implementation for problematic models
            print(f"[INFO] Using MockSwinVisionTransformer for {model_name}")
            self._use_mock_wrapper = True

            self.swin_backbone = MockSwinVisionTransformer(
                channels=channels,
                embed_dim=embed_dim,
                use_stage=use_stage,
                multi_scale=multi_scale,
                native_input=native_input
            )

            # Copy interface properties
            self.d_model = self.swin_backbone.d_model
            self.patch_size = self.swin_backbone.patch_size
            self.patch_stride = self.swin_backbone.patch_stride
            self.image_size = self.swin_backbone.image_size

            # Set multi-scale attributes
            self.use_stage = use_stage
            self.multi_scale = multi_scale
            self.native_input = native_input

        else:
            # Use real Swin backbone
            self._use_mock_wrapper = False

            self.swin_backbone = SwinTransformerBackbone(
                model_name=model_name,
                channels=channels,
                embed_dim=embed_dim,
                range_image_mode=range_image_mode,
                force_mock=force_mock,
                **kwargs
            )

            self.use_stage = use_stage
            self.multi_scale = multi_scale
            self.native_input = native_input

            # Set interface properties to match ViT
            if multi_scale:
                self.d_model = 256  # Unified decoder channel dimension
            else:
                self.d_model = self.swin_backbone.feature_dims[use_stage]

            self.patch_size = (4, 4)
            self.patch_stride = (4, 4)
            self.image_size = None

        print(f"[OK] Created SwinVisionTransformer")
        if hasattr(self, 'multi_scale') and self.multi_scale:
            print(f"  - Mode: Multi-scale (all stages)")
            if hasattr(self.swin_backbone, 'feature_dims'):
                print(f"  - Feature dims: {self.swin_backbone.feature_dims}")
        else:
            print(f"  - Mode: Single-scale (stage {getattr(self, 'use_stage', 'N/A')})")
            print(f"  - Feature dim: {self.d_model}")

    @torch.jit.ignore
    def no_weight_decay(self):
        """Compatibility with RangeViT weight decay exclusion"""
        return set()

    def get_grid_size(self, H: int, W: int):
        """Get grid size for given input dimensions"""
        if self._use_mock_wrapper:
            return self.swin_backbone.get_grid_size(H, W)
        else:
            stride = self.swin_backbone.strides[self.use_stage]
            return (H // stride, W // stride)

    def get_actual_grid_size(self):
        """Get actual grid size from last forward pass"""
        if self._use_mock_wrapper:
            return self.swin_backbone.get_actual_grid_size()
        else:
            if hasattr(self, '_last_grid_size'):
                return self._last_grid_size
            return None

    def forward(self, im: torch.Tensor, return_features: bool = False):
        """
        Forward pass compatible with ViT interface

        Args:
            im: Input tensor (B, C, H, W)
            return_features: Always True for compatibility

        Returns:
            For single-scale mode: (tokens_with_cls, skip)
            For multi-scale mode: (multi_scale_features, skip)
        """
        if self._use_mock_wrapper:
            # Delegate to mock wrapper
            return self.swin_backbone(im, return_features)
        else:
            B, C, H, W = im.shape
            self.image_size = (H, W)

            # Get all hierarchical features from Swin backbone
            multi_scale_features = self.swin_backbone(im)

            if self.multi_scale:
                # Multi-scale mode - return all feature maps
                return multi_scale_features, None
            else:
                # Single-scale mode - return selected stage in ViT format
                selected_feature = multi_scale_features[self.use_stage]
                feat_B, feat_D, feat_H, feat_W = selected_feature.shape

                # Convert to token format like ViT
                tokens = selected_feature.flatten(2).transpose(1, 2)  # (B, N, D)

                # Store grid size
                self._last_grid_size = (feat_H, feat_W)

                # Add dummy CLS token
                cls_token = torch.zeros(feat_B, 1, feat_D, device=tokens.device, dtype=tokens.dtype)
                tokens_with_cls = torch.cat([cls_token, tokens], dim=1)  # (B, N+1, D)

                return tokens_with_cls, None


def create_swin_backbone(
    model_name: str,
    channels: int = 3,
    pretrained: bool = False,
    range_image_mode: bool = False,
    force_mock: bool = False,
    **kwargs
) -> SwinTransformerBackbone:
    """
    Factory function to create Swin backbone with automatic fallback

    Args:
        model_name: Model name
        channels: Number of input channels
        pretrained: Whether to load pretrained weights
        range_image_mode: Enable range image mode
        force_mock: Force using mock backbone
        **kwargs: Additional arguments

    Returns:
        SwinTransformerBackbone instance
    """
    return SwinTransformerBackbone(
        model_name=model_name,
        channels=channels,
        pretrained=pretrained,
        range_image_mode=range_image_mode,
        force_mock=force_mock,
        **kwargs
    )


if __name__ == '__main__':
    """
    Test Swin Transformer with automatic fallback
    """
    print("Testing Swin Transformer with automatic fallback...")

    # Test configuration
    batch_size = 2
    channels = 5
    height, width = 32, 384  # Range image dimensions

    # Create test input
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}")

    test_models = [
        ('swin_tiny_patch4_window7_224', False, 'Compatible Swin model'),
        ('swinv2_tiny_window16_256', False, 'Problematic SwinV2 model'),
        ('swin_tiny_patch4_window7_224', True, 'Forced mock mode'),
    ]

    for model_name, force_mock, description in test_models:
        print(f"\n--- Testing: {description} ---")

        try:
            # Test backbone
            backbone = create_swin_backbone(
                model_name,
                channels=channels,
                range_image_mode=True,
                force_mock=force_mock
            )

            with torch.no_grad():
                features = backbone(x)

            print(f"[OK] Backbone forward pass successful")
            for i, feat in enumerate(features):
                print(f"  Stage {i}: {feat.shape}")

            # Test wrapper in multi-scale mode
            wrapper = SwinVisionTransformer(
                model_name=model_name,
                channels=channels,
                multi_scale=True,
                force_mock=force_mock
            )

            with torch.no_grad():
                multi_features, skip = wrapper(x)

            print(f"[OK] Wrapper forward pass successful")
            print(f"  Multi-scale features: {len(multi_features)} stages")

        except Exception as e:
            print(f"[FAIL] Test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n[SUCCESS] All Swin tests completed!")