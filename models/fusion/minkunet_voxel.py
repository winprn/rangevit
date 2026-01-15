# Copyright 2024 - Fusion Extension for RangeViT
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

"""
MinkUNet Voxel Encoder for multi-view fusion.

Sparse 3D U-Net architecture based on:
- NVIDIA MinkowskiEngine
- MIT Han Lab SPVNAS
- OpenPCSeg MinkUNet

Adapted for fusion with RangeViT, exposing features at 3 fusion points.
"""

import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply

__all__ = ['MinkUNetVoxelEncoder']


class SyncBatchNorm(nn.SyncBatchNorm):
    """Sparse SyncBatchNorm wrapper for SparseTensor."""
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BatchNorm(nn.BatchNorm1d):
    """Sparse BatchNorm wrapper for SparseTensor."""
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BasicConvolutionBlock(nn.Module):
    """Basic 3D sparse convolution block with BN and ReLU."""

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(False),
        )

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    """Basic 3D sparse transposed convolution block for upsampling."""

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                stride=stride,
                transposed=True,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(False),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block for sparse 3D convolution."""
    expansion = 1

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(False),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(False)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for sparse 3D convolution (expansion=4)."""
    expansion = 4

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                stride=stride,
                bias=False,
                dilation=dilation,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc * self.expansion,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(False)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkUNetVoxelEncoder(nn.Module):
    """
    Sparse 3D U-Net voxel encoder for multi-view fusion (3-stage variant).

    Exposes features at 3 fusion points matching RangeViT's architecture:
      - After stem: cs[0] channels
      - After stage3/bottleneck: cs[3] * expansion channels
      - After up3/final: cs[6] * expansion channels

    Architecture (3-stage to avoid Z-dimension collapse):
        stem → stage1 → stage2 → stage3 (bottleneck)
               ↓skip    ↓skip    ↓skip
               up3   ←  up2   ←  up1

    Args:
        in_feature_dim: Input feature dimension (default: 4 for x, y, z, intensity)
        num_layer: Number of blocks per stage [stage1, stage2, stage3, up1, up2, up3]
        block_type: 'ResBlock' or 'Bottleneck'
        cr: Channel ratio multiplier
        planes: Base channel dimensions for each stage [stem, s1, s2, s3, up1, up2, up3]
        pres: Point resolution for voxelization
        vres: Voxel resolution
        if_dist: Whether to use SyncBatchNorm for distributed training
        dropout_p: Dropout probability
    """

    def __init__(
        self,
        in_feature_dim: int = 4,
        num_layer: list = None,
        block_type: str = 'Bottleneck',
        cr: float = 1.0,
        planes: list = None,
        pres: float = 0.05,
        vres: float = 0.05,
        if_dist: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        if num_layer is None:
            num_layer = [2, 3, 4, 2, 2, 2]  # 3 encoder + 3 decoder stages
        if planes is None:
            planes = [32, 32, 64, 128, 128, 96, 96]  # 7 stages total

        self.in_feature_dim = in_feature_dim
        self.num_layer = num_layer
        self.block = {'ResBlock': ResidualBlock, 'Bottleneck': Bottleneck}[block_type]
        self.block_type = block_type
        self.expansion = self.block.expansion

        # Scale channels by cr
        cs = [int(cr * x) for x in planes]
        self.cs = cs

        self.pres = pres
        self.vres = vres

        # Stem: two 3x3 convolutions
        self.stem = nn.Sequential(
            spnn.Conv3d(in_feature_dim, cs[0], kernel_size=3, stride=1),
            SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
            spnn.ReLU(False),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
            spnn.ReLU(False),
        )

        # Encoder stages (3 stages to avoid Z-dimension collapse)
        self.in_channels = cs[0]

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, if_dist=if_dist),
            *self._make_layer(self.block, cs[1], num_layer[0], if_dist=if_dist),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, if_dist=if_dist),
            *self._make_layer(self.block, cs[2], num_layer[1], if_dist=if_dist),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, if_dist=if_dist),
            *self._make_layer(self.block, cs[3], num_layer[2], if_dist=if_dist),
        )
        # stage3 is now the bottleneck (removed stage4)

        # Decoder up blocks with skip connections (3 up blocks)
        # up1: deconv + skip from stage2 + residual blocks
        self.up1_deconv = BasicDeconvolutionBlock(self.in_channels, cs[4], ks=2, stride=2, if_dist=if_dist)
        self.in_channels = cs[4] + cs[2] * self.expansion
        self.up1_blocks = nn.Sequential(*self._make_layer(self.block, cs[4], num_layer[3], if_dist=if_dist))

        # up2: deconv + skip from stage1 + residual blocks
        self.up2_deconv = BasicDeconvolutionBlock(self.in_channels, cs[5], ks=2, stride=2, if_dist=if_dist)
        self.in_channels = cs[5] + cs[1] * self.expansion
        self.up2_blocks = nn.Sequential(*self._make_layer(self.block, cs[5], num_layer[4], if_dist=if_dist))

        # up3: deconv + skip from stem + residual blocks
        self.up3_deconv = BasicDeconvolutionBlock(self.in_channels, cs[6], ks=2, stride=2, if_dist=if_dist)
        self.in_channels = cs[6] + cs[0]
        self.up3_blocks = nn.Sequential(*self._make_layer(self.block, cs[6], num_layer[5], if_dist=if_dist))

        self.dropout = nn.Dropout(dropout_p, inplace=False)  # Must be False to avoid modifying bottleneck_out.F

        # Initialize weights
        self._weight_initialization()

        # Store output dimensions for fusion (3-stage architecture)
        self.stem_out_channels = cs[0]
        self.bottleneck_out_channels = cs[3] * self.expansion  # stage3 is bottleneck
        self.final_out_channels = cs[6] * self.expansion  # up3 is final

    def _make_layer(self, block, out_channels, num_block, stride=1, if_dist=False):
        """Create a sequence of residual blocks."""
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, if_dist=if_dist))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels, if_dist=if_dist))
        return layers

    def _weight_initialization(self):
        """Initialize batch norm weights."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x0: SparseTensor) -> dict:
        """
        Forward pass through the voxel encoder (3-stage architecture).

        Args:
            x0: SparseTensor with voxelized point features

        Returns:
            dict with:
                'stem': SparseTensor after stem (for fusion point 1)
                'bottleneck': SparseTensor after stage3 (for fusion point 2)
                'final': SparseTensor after up3 (for fusion point 3)
                'x0': stem output for skip connection
                'x1': stage1 output for skip connection
                'x2': stage2 output for skip connection
        """
        # Stem
        x0 = self.stem(x0)
        stem_out = x0

        # Encoder (3 stages)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        bottleneck_out = x3  # stage3 is now bottleneck

        # Decoder with skip connections (3 up blocks)
        x3 = SparseTensor(self.dropout(x3.F), x3.C, x3.s)
        x3._caches = bottleneck_out._caches

        y1 = self.up1_deconv(x3)
        y1 = torchsparse.cat([y1, x2])  # skip from stage2
        y1 = self.up1_blocks(y1)

        y2 = self.up2_deconv(y1)
        y2 = torchsparse.cat([y2, x1])  # skip from stage1
        y2 = self.up2_blocks(y2)

        # Note: Removed mid-decoder dropout to avoid inplace operation conflicts
        # Dropout is already applied at bottleneck (x3) which is standard practice

        y3 = self.up3_deconv(y2)
        y3 = torchsparse.cat([y3, x0])  # skip from stem
        y3 = self.up3_blocks(y3)
        final_out = y3

        return {
            'stem': stem_out,
            'bottleneck': bottleneck_out,
            'final': final_out,
            'x0': x0,
            'x1': x1,
            'x2': x2,
        }

    def forward_with_intermediates(self, x0: SparseTensor):
        """
        Forward pass returning all intermediate features for multi-stage fusion.

        This method allows external fusion modules to inject fused features
        and continue processing.

        Args:
            x0: SparseTensor with voxelized point features

        Returns:
            dict with all intermediate SparseTensors
        """
        return self.forward(x0)

    def forward_encoder_only(self, x0: SparseTensor) -> SparseTensor:
        """
        Forward pass through encoder only (stem + stages 1-3).

        Args:
            x0: SparseTensor with voxelized point features

        Returns:
            SparseTensor at bottleneck (after stage3)
        """
        x0 = self.stem(x0)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x3

    def forward_decoder_only(
        self,
        x3: SparseTensor,
        x0: SparseTensor,
        x1: SparseTensor,
        x2: SparseTensor,
    ) -> SparseTensor:
        """
        Forward pass through decoder only (up blocks 1-3).

        Args:
            x3: SparseTensor at bottleneck (stage3 output)
            x0, x1, x2: Skip connection features from encoder

        Returns:
            SparseTensor at final resolution
        """
        x3_dropped = SparseTensor(self.dropout(x3.F), x3.C, x3.s)
        x3_dropped._caches = x3._caches  # Preserve caches for transposed convolutions

        y1 = self.up1_deconv(x3_dropped)
        y1 = torchsparse.cat([y1, x2])  # skip from stage2
        y1 = self.up1_blocks(y1)

        y2 = self.up2_deconv(y1)
        y2 = torchsparse.cat([y2, x1])  # skip from stage1
        y2 = self.up2_blocks(y2)

        # Note: Removed mid-decoder dropout to avoid inplace operation conflicts

        y3 = self.up3_deconv(y2)
        y3 = torchsparse.cat([y3, x0])  # skip from stem
        y3 = self.up3_blocks(y3)

        return y3
