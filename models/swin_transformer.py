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


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone for RangeViT
    Returns hierarchical multi-scale features compatible with RangeViT decoder

    Features:
    - 4 hierarchical stages with different spatial resolutions
    - Channel dimensions that double at each stage
    - Window-based attention mechanism
    - Compatible with timm pretrained weights
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
        """
        super().__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.range_image_mode = range_image_mode

        # Adaptive model selection for range images
        if range_image_mode:
            model_name, window_size = self._select_range_compatible_model(model_name)
            self.model_name = model_name
            self.window_size = window_size

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
            print(f"[ERROR] Failed to create Swin model {model_name}: {e}")
            raise

        # Store feature dimensions for each stage
        # For Swin-Tiny: [96, 192, 384, 768]
        self.feature_dims = [
            embed_dim,        # Stage 0: C
            embed_dim * 2,    # Stage 1: 2C
            embed_dim * 4,    # Stage 2: 4C
            embed_dim * 8     # Stage 3: 8C
        ]

        # Store stride information for each stage
        self.strides = [4, 8, 16, 32]

        # Initialize any additional layers
        self.apply(init_weights)

    def _select_range_compatible_model(self, original_model: str) -> Tuple[str, int]:
        """
        Select Swin model variant that's compatible with range images

        Args:
            original_model: Original model name requested

        Returns:
            Tuple of (compatible_model_name, compatible_window_size)
        """
        # Range image compatibility matrix
        # For 32×384 and 64×384 range images, we need window sizes that divide evenly
        range_compatible_models = {
            'swinv2_tiny_window16_256': ('swin_tiny_patch4_window7_224', 7),  # window7 works better
            'swinv2_tiny_patch4_window16_256': ('swin_tiny_patch4_window7_224', 7),
            'swin_tiny_patch4_window7_224': ('swin_tiny_patch4_window7_224', 7),  # Already compatible
            'swin_small_patch4_window7_224': ('swin_small_patch4_window7_224', 7),
            'swin_base_patch4_window7_224': ('swin_base_patch4_window7_224', 7),
        }

        if original_model in range_compatible_models:
            compatible_model, window_size = range_compatible_models[original_model]
            if compatible_model != original_model:
                print(f"[INFO] Range image mode: {original_model} -> {compatible_model}")
                print(f"       Window size: {window_size} (optimized for range images)")
            return compatible_model, window_size
        else:
            # Default fallback - use Swin-Tiny with window7
            print(f"[WARN] Unknown model {original_model}, using swin_tiny_patch4_window7_224")
            return 'swin_tiny_patch4_window7_224', 7

    def get_feature_dims(self) -> List[int]:
        """Return channel dimensions for each stage"""
        return self.feature_dims.copy()

    def get_strides(self) -> List[int]:
        """Return spatial stride for each stage"""
        return self.strides.copy()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through Swin Transformer with intelligent padding for range images

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of 4 feature maps from different stages:
            - F0: (B, C,   H/4,  W/4)   - Stage 0, stride 4
            - F1: (B, 2C,  H/8,  W/8)   - Stage 1, stride 8
            - F2: (B, 4C,  H/16, W/16)  - Stage 2, stride 16
            - F3: (B, 8C,  H/32, W/32)  - Stage 3, stride 32

            For Swin-Tiny (embed_dim=96):
            - F0: (B, 96,  H/4,  W/4)
            - F1: (B, 192, H/8,  W/8)
            - F2: (B, 384, H/16, W/16)
            - F3: (B, 768, H/32, W/32)
        """
        try:
            B, C, H_orig, W_orig = x.shape

            # Apply intelligent padding if in range image mode
            if self.range_image_mode:
                x_padded, pad_info = self._apply_intelligent_padding(x)
                # Update dimensions
                _, _, H_padded, W_padded = x_padded.shape
            else:
                x_padded = x
                pad_info = None
                H_padded, W_padded = H_orig, W_orig

            # Pass through Swin backbone
            features = self.swin(x_padded)

            # Convert from timm format (B, H, W, C) to standard format (B, C, H, W)
            features = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]

            # Remove padding from features if applied
            if self.range_image_mode and pad_info is not None:
                features = self._remove_padding_from_features(features, pad_info, H_orig, W_orig)

            # Validate output shapes
            expected_shapes = [
                (B, self.feature_dims[0], H_orig//4, W_orig//4),
                (B, self.feature_dims[1], H_orig//8, W_orig//8),
                (B, self.feature_dims[2], H_orig//16, W_orig//16),
                (B, self.feature_dims[3], H_orig//32, W_orig//32),
            ]

            for i, (feat, expected) in enumerate(zip(features, expected_shapes)):
                if feat.shape != expected:
                    print(f"Warning: Stage {i} shape mismatch. Got {feat.shape}, expected {expected}")

            return features

        except Exception as e:
            print(f"[ERROR] Swin forward pass failed: {e}")
            print(f"  Input shape: {x.shape}")
            print(f"  Window size: {self.window_size}")
            print(f"  Range image mode: {self.range_image_mode}")
            raise

    def _apply_intelligent_padding(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Apply intelligent padding to make input compatible with Swin window requirements

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (padded_tensor, padding_info)
        """
        B, C, H, W = x.shape

        # For Swin Transformer with window_size=7, use empirically determined working sizes
        # Based on testing, these dimensions work reliably with timm Swin models
        window_size = self.window_size

        if window_size == 7:
            # For window_size=7, pad to multiples of 7×7=49 for both dimensions
            # This ensures proper window partitioning and attention computation
            target_multiple = 7 * 7  # 49
            pad_h = (target_multiple - H % target_multiple) % target_multiple
            pad_w = (target_multiple - W % target_multiple) % target_multiple
        else:
            # For other window sizes, use window_size as the base multiple
            # This is a fallback that should work for most cases
            target_multiple = window_size * window_size
            pad_h = (target_multiple - H % target_multiple) % target_multiple
            pad_w = (target_multiple - W % target_multiple) % target_multiple

        print(f"[DEBUG] Window size: {window_size}, Target multiple: {target_multiple}")
        print(f"[DEBUG] Original: {H}x{W}, Padding: +{pad_h}x{pad_w}")

        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            # Use reflection padding to minimize information distortion
            # Padding format: (left, right, top, bottom)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

            pad_info = {
                'original_size': (H, W),
                'padded_size': (H + pad_h, W + pad_w),
                'padding': (pad_left, pad_right, pad_top, pad_bottom),
                'pad_h': pad_h,
                'pad_w': pad_w
            }

            print(f"[INFO] Applied intelligent padding: {(H, W)} -> {(H + pad_h, W + pad_w)}")
        else:
            x_padded = x
            pad_info = None

        return x_padded, pad_info

    def _remove_padding_from_features(self, features: List[torch.Tensor], pad_info: dict, H_orig: int, W_orig: int) -> List[torch.Tensor]:
        """
        Remove padding from feature maps to restore original proportions

        Args:
            features: List of feature tensors from Swin backbone
            pad_info: Padding information from _apply_intelligent_padding
            H_orig, W_orig: Original input dimensions

        Returns:
            List of feature tensors with padding removed
        """
        if pad_info is None:
            return features

        cleaned_features = []

        for i, feat in enumerate(features):
            stride = self.strides[i]

            # Calculate expected output size at this stride
            expected_H = H_orig // stride
            expected_W = W_orig // stride

            # Current feature dimensions
            B, C, feat_H, feat_W = feat.shape

            # Calculate how much padding to remove at this scale
            # The padding scales down by the stride factor
            pad_h_scaled = pad_info['pad_h'] // stride
            pad_w_scaled = pad_info['pad_w'] // stride

            if pad_h_scaled > 0 or pad_w_scaled > 0:
                # Calculate crop region
                pad_top_scaled = pad_info['padding'][2] // stride  # top padding
                pad_left_scaled = pad_info['padding'][0] // stride  # left padding

                # Crop to remove padding
                feat_cropped = feat[
                    :, :,
                    pad_top_scaled:pad_top_scaled + expected_H,
                    pad_left_scaled:pad_left_scaled + expected_W
                ]

                cleaned_features.append(feat_cropped)
            else:
                cleaned_features.append(feat)

        return cleaned_features


class SwinVisionTransformer(nn.Module):
    """
    Wrapper to make Swin Transformer compatible with RangeViT's ViT interface

    This class provides the same interface as the original VisionTransformer class
    but uses Swin Transformer backbone internally. For Phase 1, it uses only the
    F2 stage (stride 16) to match ViT behavior.
    """

    def __init__(
        self,
        swin_backbone: SwinTransformerBackbone,
        use_stage: int = 2,  # Use F2 by default (stride 16, matches ViT)
        multi_scale: bool = False,  # Phase 2A: Enable multi-scale output
        native_input: bool = False,  # Phase 2C: Disable input resizing
    ):
        """
        Initialize Swin-ViT compatibility wrapper

        Args:
            swin_backbone: SwinTransformerBackbone instance
            use_stage: Which Swin stage to use (0-3). Default 2 (F2, stride 16)
            multi_scale: If True, return all stages [F0,F1,F2,F3]. If False, return single stage
            native_input: If True, use native input size without resizing
        """
        super().__init__()

        self.swin = swin_backbone
        self.use_stage = use_stage
        self.multi_scale = multi_scale
        self.native_input = native_input

        # Set interface properties to match VisionTransformer
        if multi_scale:
            # For multi-scale, d_model represents the unified decoder channel dimension
            self.d_model = 256  # Will be set by decoder configuration
        else:
            # For single-scale, use the selected stage dimension
            self.d_model = self.swin.feature_dims[use_stage]  # Feature dimension

        self.patch_size = (4, 4)  # Swin patch size
        self.patch_stride = (4, 4)  # Same as patch size for Swin

        # Image size will be set when needed
        self.image_size = None

        print(f"[OK] Created SwinVisionTransformer wrapper")
        if multi_scale:
            print(f"  - Mode: Multi-scale (all stages)")
            print(f"  - Feature dims: {self.swin.feature_dims}")
        else:
            print(f"  - Mode: Single-scale (stage {use_stage})")
            print(f"  - Feature dim: {self.d_model}")
        print(f"  - Native input: {native_input}")

    @torch.jit.ignore
    def no_weight_decay(self):
        """Compatibility with RangeViT weight decay exclusion"""
        # Swin doesn't use pos_embed or cls_token, so return empty set
        return set()

    def get_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """
        Get grid size for given input dimensions
        Compatible with RangeViT's get_grid_size interface
        """
        stride = self.swin.strides[self.use_stage]
        return (H // stride, W // stride)

    def get_actual_grid_size(self) -> Tuple[int, int]:
        """
        Get the actual grid size from the last forward pass
        This is needed because Swin may have different actual dimensions
        than what the decoder expects
        """
        if hasattr(self, '_last_grid_size'):
            return self._last_grid_size
        return None

    def forward(self, im: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass compatible with VisionTransformer interface

        Args:
            im: Input tensor (B, C, H, W)
            return_features: Always True for compatibility

        Returns:
            For single-scale mode (multi_scale=False):
                Tuple of (tokens, skip):
                - tokens: Feature tokens in ViT format (B, N+1, D) where N+1 includes CLS token
                - skip: Skip connection (None for Swin)

            For multi-scale mode (multi_scale=True):
                Tuple of (multi_scale_features, skip):
                - multi_scale_features: List of [F0, F1, F2, F3] feature maps
                - skip: Skip connection (None for Swin)
        """
        B, C, H, W = im.shape
        self.image_size = (H, W)  # Store for compatibility

        # Input processing: resize or use native input
        if self.native_input:
            # Phase 2C: Use native input size with intelligent padding if needed
            im_processed = self._apply_intelligent_padding(im)
        else:
            # Phase 1/2A: Resize to square for pretrained compatibility
            target_size = 256  # Standard size for swinv2_tiny_window16_256
            if H != target_size or W != target_size:
                im_processed = torch.nn.functional.interpolate(
                    im, size=(target_size, target_size), mode='bilinear', align_corners=False
                )
            else:
                im_processed = im

        # Get all hierarchical features from Swin
        multi_scale_features = self.swin(im_processed)  # [F0, F1, F2, F3]

        # Adjust feature map sizes back to original proportions if input was resized
        if not self.native_input and (H != 256 or W != 256):
            multi_scale_features = self._adjust_feature_sizes(multi_scale_features, H, W)

        # Return based on mode
        if self.multi_scale:
            # Phase 2A: Multi-scale mode - return all feature maps
            return multi_scale_features, None
        else:
            # Phase 1: Single-scale mode - return selected stage in ViT format
            return self._format_single_stage_output(multi_scale_features[self.use_stage])

    def _apply_intelligent_padding(self, im: torch.Tensor) -> torch.Tensor:
        """
        Apply intelligent padding to make input compatible with window size

        Args:
            im: Input tensor (B, C, H, W)

        Returns:
            Padded tensor compatible with Swin window size
        """
        B, C, H, W = im.shape
        window_size = 16  # Default window size for SwinV2

        # Calculate padding needed
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            # Apply reflection padding to minimize information loss
            im = F.pad(im, (0, pad_w, 0, pad_h), mode='reflect')

        return im

    def _adjust_feature_sizes(self, features: List[torch.Tensor], orig_H: int, orig_W: int) -> List[torch.Tensor]:
        """
        Adjust feature map sizes back to expected proportions based on original input

        Args:
            features: List of [F0, F1, F2, F3] feature maps
            orig_H, orig_W: Original input dimensions

        Returns:
            Adjusted feature maps
        """
        adjusted_features = []
        strides = self.swin.strides

        for i, feat in enumerate(features):
            expected_H = orig_H // strides[i]
            expected_W = orig_W // strides[i]

            # Only resize if dimensions don't match
            if feat.shape[2] != expected_H or feat.shape[3] != expected_W:
                feat_adjusted = F.interpolate(
                    feat,
                    size=(expected_H, expected_W),
                    mode='bilinear',
                    align_corners=False
                )
                adjusted_features.append(feat_adjusted)
            else:
                adjusted_features.append(feat)

        return adjusted_features

    def _format_single_stage_output(self, selected_feature: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Format single stage output to match ViT interface

        Args:
            selected_feature: Selected feature map (B, D, H, W)

        Returns:
            Tuple of (tokens_with_cls, skip)
        """
        feat_B, feat_D, feat_H, feat_W = selected_feature.shape

        # Convert to token format like ViT
        tokens = selected_feature.flatten(2).transpose(1, 2)  # (B, N, D) where N = feat_H * feat_W

        # Store the actual grid size for decoder compatibility
        self._last_grid_size = (feat_H, feat_W)

        # Add dummy CLS token to match ViT interface (will be removed by caller)
        cls_token = torch.zeros(feat_B, 1, feat_D, device=tokens.device, dtype=tokens.dtype)
        tokens_with_cls = torch.cat([cls_token, tokens], dim=1)  # (B, N+1, D)

        # No skip connection for now
        skip = None

        return tokens_with_cls, skip


def create_swin_backbone(
    model_name: str,
    channels: int = 3,
    pretrained: bool = False,
    range_image_mode: bool = True,  # Enable range image mode by default
    **kwargs
) -> SwinTransformerBackbone:
    """
    Factory function to create Swin backbone with predefined configurations

    Args:
        model_name: Name of the Swin model
        channels: Number of input channels
        pretrained: Whether to use pretrained weights
        range_image_mode: Enable range image compatibility mode
        **kwargs: Additional arguments

    Returns:
        SwinTransformerBackbone instance
    """

    # Predefined configurations for common Swin models
    swin_configs = {
        'swin_tiny_patch4_window7_224': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
        },
        'swin_small_patch4_window7_224': {
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
        },
        'swin_base_patch4_window7_224': {
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': 7,
        },
        'swinv2_tiny_window16_256': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 16,
        },
        'swinv2_tiny_patch4_window16_256': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 16,
        }
    }

    if model_name not in swin_configs:
        raise ValueError(f"Unknown Swin model: {model_name}. Available: {list(swin_configs.keys())}")

    config = swin_configs[model_name]
    config.update(kwargs)  # Allow override of default config

    return SwinTransformerBackbone(
        model_name=model_name,
        channels=channels,
        pretrained=pretrained,
        range_image_mode=range_image_mode,
        **config
    )


if __name__ == '__main__':
    """
    Test script for Swin Transformer backbone
    """
    print("Testing Swin Transformer Backbone...")

    # Test configuration - use dimensions compatible with window size
    batch_size = 2
    channels = 5  # Range image channels
    height, width = 224, 224  # Test with standard size first

    # Create test input
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}")

    try:
        # Test backbone creation
        backbone = create_swin_backbone(
            'swin_tiny_patch4_window7_224',
            channels=channels,
            pretrained=False
        )
        print(f"[OK] Backbone created successfully")
        print(f"Feature dims: {backbone.get_feature_dims()}")
        print(f"Strides: {backbone.get_strides()}")

        # Test forward pass
        with torch.no_grad():
            features = backbone(x)

        print(f"[OK] Forward pass successful")
        for i, feat in enumerate(features):
            print(f"  Stage {i}: {feat.shape}")

        # Test wrapper
        wrapper = SwinVisionTransformer(backbone, use_stage=2)
        print(f"[OK] Wrapper created successfully")
        print(f"d_model: {wrapper.d_model}")

        # Test wrapper forward pass
        with torch.no_grad():
            tokens, skip = wrapper(x, return_features=True)

        print(f"[OK] Wrapper forward pass successful")
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Skip: {skip}")

        print("[SUCCESS] All tests passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()