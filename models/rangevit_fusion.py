# Copyright 2026 - RangeViT-Fusion
# Main RangeViT-Fusion model combining all components

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Tuple, List

from .features_encoder import FeaturesEncoder
from .vit_fusion import VisionTransformerFusion
from .fusion_head import FusionHead
from .fusion_modules import EfficientTransformationPipeline
from .model_utils import resize_pos_embed, adapt_input_conv
from utils.optim.focal_softmax import FocalSoftmaxLoss
from utils.optim.lovasz_softmax import Lovasz_softmax


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RangeViTFusion(nn.Module):
    """
    RangeViT-Fusion: Vision Transformer with bidirectional point-pixel fusion
    for 3D semantic segmentation.

    This model combines:
    1. FeaturesEncoder: Encodes raw point attributes to feature vectors
    2. VisionTransformerFusion: ViT backbone with bidirectional fusion at specified blocks
    3. FusionHead: Combines pixel and point features for per-point classification

    Architecture Flow:
        1. point_attrs -> FeaturesEncoder -> point_feats
        2. images + point_feats + coords -> VisionTransformerFusion -> pixel_feats, point_feats, aux_outputs
        3. pixel_feats -> pixel2point -> mapped_pixel_feats
        4. mapped_pixel_feats + point_feats -> FusionHead -> point_logits
        5. Compute losses (point-level + auxiliary pixel-level)

    Args:
        in_channels: Number of input channels for range image (default: 5)
        point_channels: Number of input channels per point (default: 5 for xyz + intensity + range)
        n_cls: Number of semantic classes
        backbone: ViT backbone variant ('vit_small_patch16_384', 'vit_base_patch16_384', etc.)
        image_size: Input range image size (H, W)
        pretrained_path: Path to pretrained weights or 'timmImageNet21k' for timm weights
        new_patch_size: New patch size for ViT encoder (optional)
        new_patch_stride: New patch stride for ViT encoder (optional)
        reuse_pos_emb: Whether to reuse pretrained positional embeddings
        conv_stem: Type of convolutional stem ('none' or 'ConvStem')
        stem_base_channels: Base channels for ConvStem
        stem_hidden_dim: Hidden dimension for ConvStem
        fusion_blocks: List of block indices (1-indexed) for fusion operations
        aux_loss_weight: Weight for auxiliary pixel-level losses
        ignore_index: Label index to ignore in loss computation
    """

    def __init__(
        self,
        in_channels: int = 5,
        point_channels: int = 5,
        n_cls: int = 17,
        backbone: str = 'vit_small_patch16_384',
        image_size: Tuple[int, int] = (32, 384),
        pretrained_path: Optional[str] = None,
        new_patch_size: Optional[Tuple[int, int]] = None,
        new_patch_stride: Optional[Tuple[int, int]] = None,
        reuse_pos_emb: bool = False,
        conv_stem: str = 'none',
        stem_base_channels: int = 32,
        stem_hidden_dim: Optional[int] = None,
        fusion_blocks: List[int] = [4, 8, 12],
        aux_loss_weight: float = 0.4,
        ignore_index: int = 0,
    ):
        super().__init__()

        self.n_cls = n_cls
        self.aux_loss_weight = aux_loss_weight
        self.ignore_index = ignore_index
        self.image_size = image_size

        # Get backbone configuration
        backbone_config = self._get_backbone_config(backbone)
        n_heads = backbone_config['n_heads']
        n_layers = backbone_config['n_layers']
        patch_size = backbone_config['patch_size']
        dropout = backbone_config['dropout']
        drop_path_rate = backbone_config['drop_path_rate']
        d_model = backbone_config['d_model']

        self.d_model = d_model

        # Handle patch size and stride
        if new_patch_size is not None:
            patch_size = new_patch_size
        if new_patch_stride is None:
            patch_stride = patch_size
        else:
            patch_stride = new_patch_stride

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Compute grid size
        if isinstance(patch_stride, (list, tuple)):
            self.grid_h = image_size[0] // patch_stride[0]
            self.grid_w = image_size[1] // patch_stride[1]
        else:
            self.grid_h = image_size[0] // patch_stride
            self.grid_w = image_size[1] // patch_stride

        # 1. Features Encoder: encodes raw point attributes to d_model features
        self.features_encoder = FeaturesEncoder(
            in_channels=point_channels,
            d_model=d_model,
        )

        # 2. Vision Transformer with Fusion
        self.vit_fusion = VisionTransformerFusion(
            image_size=image_size,
            patch_size=patch_size,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=4 * d_model,  # Standard MLP expansion ratio
            n_heads=n_heads,
            n_cls=n_cls,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            channels=in_channels,
            patch_stride=patch_stride,
            fusion_blocks=fusion_blocks,
            conv_stem=conv_stem,
            stem_base_channels=stem_base_channels,
            stem_hidden_dim=stem_hidden_dim,
        )

        # 3. Efficient Transformation Pipeline for final pixel2point mapping
        self.etp = EfficientTransformationPipeline(self.grid_h, self.grid_w)

        # 4. Fusion Head: combines mapped pixel features with point features
        self.fusion_head = FusionHead(
            d_model=d_model,
            n_classes=n_cls,
        )

        # 5. Loss functions
        self.focal_loss = FocalSoftmaxLoss(n_classes=n_cls, gamma=2, alpha=0.25)
        self.lovasz_loss = Lovasz_softmax(ignore=ignore_index)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained(
                pretrained_path=pretrained_path,
                backbone=backbone,
                in_channels=in_channels,
                reuse_pos_emb=reuse_pos_emb,
                new_patch_size=new_patch_size,
                new_patch_stride=new_patch_stride,
            )

    def _get_backbone_config(self, backbone: str) -> Dict:
        """Get configuration for a specific backbone variant."""
        configs = {
            'vit_small_patch16_384': {
                'n_heads': 6,
                'n_layers': 12,
                'patch_size': (16, 16),
                'dropout': 0.0,
                'drop_path_rate': 0.1,
                'd_model': 384,
            },
            'vit_base_patch16_384': {
                'n_heads': 12,
                'n_layers': 12,
                'patch_size': (16, 16),
                'dropout': 0.0,
                'drop_path_rate': 0.1,
                'd_model': 768,
            },
            'vit_large_patch16_384': {
                'n_heads': 16,
                'n_layers': 24,
                'patch_size': (16, 16),
                'dropout': 0.0,
                'drop_path_rate': 0.1,
                'd_model': 1024,
            },
        }

        if backbone not in configs:
            raise ValueError(f"Unknown backbone: {backbone}. "
                           f"Available: {list(configs.keys())}")

        return configs[backbone]

    def _load_pretrained(
        self,
        pretrained_path: str,
        backbone: str,
        in_channels: int,
        reuse_pos_emb: bool,
        new_patch_size: Optional[Tuple[int, int]],
        new_patch_stride: Optional[Tuple[int, int]],
    ):
        """Load pretrained ViT weights into the model."""
        print(f'Loading pretrained parameters from {pretrained_path}')

        if pretrained_path == 'timmImageNet21k':
            # Load from timm pretrained models
            vit_imagenet = timm.create_model(backbone, pretrained=True)
            pretrained_state_dict = vit_imagenet.state_dict()

            # Remap keys to match vit_fusion structure
            all_keys = list(pretrained_state_dict.keys())
            for key in all_keys:
                pretrained_state_dict['vit_fusion.' + key] = pretrained_state_dict.pop(key)
        else:
            # Load from checkpoint file
            pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['state_dict']
            elif 'model' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['model']

            # Handle different checkpoint formats
            all_keys = list(pretrained_state_dict.keys())
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', 'vit_fusion.')
                    pretrained_state_dict[new_key] = pretrained_state_dict.pop(key)
                elif key.startswith('encoder.'):
                    new_key = key.replace('encoder.', 'vit_fusion.')
                    pretrained_state_dict[new_key] = pretrained_state_dict.pop(key)
                elif not key.startswith('vit_fusion.'):
                    pretrained_state_dict['vit_fusion.' + key] = pretrained_state_dict.pop(key)

        # Reuse pre-trained positional embeddings
        if reuse_pos_emb and new_patch_size is not None and new_patch_stride is not None:
            print('Reusing positional embeddings.')
            gs_new_h = int((self.image_size[0] - new_patch_size[0]) // new_patch_stride[0] + 1)
            gs_new_w = int((self.image_size[1] - new_patch_size[1]) // new_patch_stride[1] + 1)
            num_extra_tokens = 1

            pos_embed_key = 'vit_fusion.pos_embed'
            if pos_embed_key in pretrained_state_dict:
                resized_pos_emb = resize_pos_embed(
                    pretrained_state_dict[pos_embed_key],
                    grid_old_shape=None,
                    grid_new_shape=(gs_new_h, gs_new_w),
                    num_extra_tokens=num_extra_tokens
                )
                pretrained_state_dict[pos_embed_key] = resized_pos_emb

        # Filter out keys that don't match our model
        model_state_dict = self.state_dict()
        filtered_state_dict = {}

        for key, value in pretrained_state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f'Skipping {key}: shape mismatch '
                          f'({value.shape} vs {model_state_dict[key].shape})')
            else:
                # Try without vit_fusion prefix for backwards compatibility
                pass  # Skip keys that don't exist in our model

        msg = self.load_state_dict(filtered_state_dict, strict=False)
        print(f'Loaded pretrained weights: {msg}')

    def _convert_coords_to_patch_space(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert pixel-space coordinates to patch-space coordinates."""
        patch_coords = coords.clone()
        if isinstance(self.patch_stride, (list, tuple)):
            patch_coords[:, 1] = coords[:, 1] // self.patch_stride[0]
            patch_coords[:, 2] = coords[:, 2] // self.patch_stride[1]
        else:
            patch_coords[:, 1] = coords[:, 1] // self.patch_stride
            patch_coords[:, 2] = coords[:, 2] // self.patch_stride
        return patch_coords

    def _create_pseudo_labels(
        self,
        labels: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        Create pixel pseudo-labels from point labels for auxiliary supervision.

        Uses majority voting within each pixel to determine the pixel label.

        Args:
            labels: (N,) point labels
            coords: (N, 3) coordinates [batch_idx, y, x] in patch space
            batch_size: Number of samples in batch
            grid_h: Actual grid height
            grid_w: Actual grid width

        Returns:
            pixel_labels: (B, grid_H, grid_W) pixel pseudo-labels
        """
        device = labels.device
        H, W = grid_h, grid_w

        # Initialize with ignore_index
        pixel_labels = torch.full(
            (batch_size, H, W),
            self.ignore_index,
            dtype=torch.long,
            device=device
        )

        if labels.numel() == 0:
            return pixel_labels

        # Extract coordinates
        batch_idx = coords[:, 0].long()
        y = coords[:, 1].long().clamp(0, H - 1)
        x = coords[:, 2].long().clamp(0, W - 1)

        # For simplicity, use the label of the last point at each pixel
        # A more sophisticated approach would use majority voting
        valid_mask = labels != self.ignore_index
        valid_batch = batch_idx[valid_mask]
        valid_y = y[valid_mask]
        valid_x = x[valid_mask]
        valid_labels = labels[valid_mask]

        if valid_labels.numel() > 0:
            pixel_labels[valid_batch, valid_y, valid_x] = valid_labels

        return pixel_labels

    def _compute_loss(
        self,
        point_logits: torch.Tensor,
        labels: torch.Tensor,
        aux_outputs: List[torch.Tensor],
        pixel_pseudo_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute point-level and auxiliary pixel-level losses.

        Uses Focal loss + Lovasz loss for point-level supervision, which are
        better suited for semantic segmentation with class imbalance.

        Args:
            point_logits: (N, n_cls) point classification logits
            labels: (N,) ground truth point labels
            aux_outputs: List of (B, n_cls, H, W) auxiliary pixel logits
            pixel_pseudo_labels: (B, H, W) pseudo-labels for pixel supervision

        Returns:
            Dictionary containing loss components:
            - 'loss': total combined loss
            - 'focal_loss': focal loss component
            - 'lovasz_loss': lovasz loss component
            - 'point_loss': combined point loss (focal + lovasz)
            - 'aux_loss': auxiliary pixel-level loss
        """
        # Focal loss for point-level supervision
        focal = self.focal_loss(point_logits, labels)

        # Lovasz loss for point-level supervision
        # Lovasz expects probabilities, not logits
        point_probs = F.softmax(point_logits, dim=1)
        # Reshape for lovasz: (N, C) -> (1, C, N, 1) to match expected (B, C, H, W) format
        point_probs_reshaped = point_probs.unsqueeze(0).unsqueeze(-1).permute(0, 2, 1, 3)
        labels_reshaped = labels.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        lovasz = self.lovasz_loss(point_probs_reshaped, labels_reshaped)

        # Point loss = focal + lovasz
        point_loss = focal + lovasz

        # Auxiliary pixel-level losses (use cross-entropy for simplicity)
        aux_loss = torch.tensor(0.0, device=point_logits.device)
        if self.training and len(aux_outputs) > 0:
            for aux_logits in aux_outputs:
                aux_loss = aux_loss + F.cross_entropy(
                    aux_logits,
                    pixel_pseudo_labels,
                    ignore_index=self.ignore_index,
                )
            aux_loss = aux_loss / len(aux_outputs) * self.aux_loss_weight

        # Total loss
        total_loss = point_loss + aux_loss

        return {
            'loss': total_loss,
            'focal_loss': focal,
            'lovasz_loss': lovasz,
            'point_loss': point_loss,
            'aux_loss': aux_loss,
        }

    def forward(
        self,
        images: torch.Tensor,
        point_attrs: torch.Tensor,
        coords: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of RangeViT-Fusion.

        Args:
            images: (B, C, H, W) input range images
            point_attrs: (N, point_channels) raw point attributes (xyz, intensity, range)
            coords: (N, 3) coordinates [batch_idx, y, x] in pixel space
            labels: (N,) ground truth labels (optional, for training)

        Returns:
            Dictionary containing:
                - 'point_logits': (N, n_cls) per-point classification logits
                - 'pixel_feats': (B, D, grid_H, grid_W) final pixel features
                - 'point_feats': (N, D) final point features
                - 'aux_outputs': List of auxiliary pixel logits
                - 'losses': Dictionary of losses (if labels provided)
        """
        B, _, H, W = images.shape

        # Compute actual grid size from input dimensions (handles variable-size inputs)
        if isinstance(self.patch_stride, (list, tuple)):
            actual_grid_h = H // self.patch_stride[0]
            actual_grid_w = W // self.patch_stride[1]
        else:
            actual_grid_h = H // self.patch_stride
            actual_grid_w = W // self.patch_stride

        # 1. Encode point attributes to feature vectors
        point_feats = self.features_encoder(point_attrs)  # (N, d_model)

        # 2. Process through ViT with bidirectional fusion
        pixel_feats, point_feats, aux_outputs = self.vit_fusion(
            images, point_feats, coords
        )  # pixel_feats: (B, d_model, grid_H, grid_W), point_feats: (N, d_model)

        # 3. Map final pixel features to point locations
        # Create dynamic ETP for actual grid size
        dynamic_etp = EfficientTransformationPipeline(actual_grid_h, actual_grid_w)
        patch_coords = self._convert_coords_to_patch_space(coords)
        mapped_pixel_feats = dynamic_etp.pixel2point(pixel_feats, patch_coords)  # (N, d_model)

        # 4. Predict point-level logits using fusion head
        point_logits = self.fusion_head(mapped_pixel_feats, point_feats)  # (N, n_cls)

        # Build output dictionary
        outputs = {
            'point_logits': point_logits,
            'pixel_feats': pixel_feats,
            'point_feats': point_feats,
            'aux_outputs': aux_outputs,
        }

        # 5. Compute losses if labels are provided
        if labels is not None:
            # Create pseudo-labels for auxiliary pixel supervision
            pixel_pseudo_labels = self._create_pseudo_labels(
                labels, patch_coords, B, actual_grid_h, actual_grid_w
            )

            losses = self._compute_loss(
                point_logits=point_logits,
                labels=labels,
                aux_outputs=aux_outputs,
                pixel_pseudo_labels=pixel_pseudo_labels,
            )
            outputs['losses'] = losses

        return outputs

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts for different components."""
        stats = {
            'total': count_parameters(self),
            'features_encoder': count_parameters(self.features_encoder),
            'vit_fusion': count_parameters(self.vit_fusion),
            'fusion_head': count_parameters(self.fusion_head),
        }

        # Breakdown of vit_fusion
        stats['vit_fusion_patch_embed'] = count_parameters(self.vit_fusion.patch_embed)
        stats['vit_fusion_blocks'] = count_parameters(self.vit_fusion.blocks)
        stats['vit_fusion_fusion_layers'] = (
            count_parameters(self.vit_fusion.point_fusion_layers) +
            count_parameters(self.vit_fusion.pixel_fusion_layers)
        )
        stats['vit_fusion_aux_heads'] = count_parameters(self.vit_fusion.aux_heads)

        return stats

    def counter_model_parameters(self) -> Dict[str, int]:
        """Return parameter counts in format compatible with main.py."""
        stats = {}
        stats['total_num_parameters'] = count_parameters(self)
        stats['encoder_num_parameters'] = count_parameters(self.vit_fusion)
        stats['stem_num_parameters'] = count_parameters(self.vit_fusion.patch_embed)
        stats['decoder_num_parameters'] = count_parameters(self.fusion_head)
        stats['features_encoder_num_parameters'] = count_parameters(self.features_encoder)
        stats['fusion_layers_num_parameters'] = (
            count_parameters(self.vit_fusion.point_fusion_layers) +
            count_parameters(self.vit_fusion.pixel_fusion_layers)
        )
        return stats

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return parameter names that should not have weight decay."""
        nwd = {'vit_fusion.pos_embed', 'vit_fusion.cls_token'}
        return nwd
