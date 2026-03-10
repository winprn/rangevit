import torch
import torch.nn as nn

from .metaformer import metaformer_baseline as metaformer


class MetaFormerAdapter(nn.Module):
    def __init__(
        self,
        backbone_name='convformer_s18',
        image_size=(64, 2048),
        patch_size=None,
        patch_stride=None,
        channels=3,
        in_channels=None,
        pretrained_path=None,
        load_pretrained_stem=False,
        d_model=None,
        use_fpn_decoder=False,
        **kwargs,
    ):
        if in_channels is None:
            in_channels = channels
        super().__init__()

        if not hasattr(metaformer, backbone_name):
            raise ValueError(f"Unknown MetaFormer backbone: {backbone_name}")

        # Build model from the local registry. We won't use the classifier head.
        ctor = getattr(metaformer, backbone_name)
        self.model = ctor(pretrained=False, in_chans=in_channels, num_classes=0)

        # Collect per-stage dims from downsample layers.
        self.embed_dims = [layer.conv.out_channels for layer in self.model.downsample_layers]
        self.d_model = self.embed_dims[-1]

        # Compute effective stride from downsample layers (e.g. 4 * 2 * 2 * 2 = 32).
        total_stride_h = 1
        total_stride_w = 1
        for layer in self.model.downsample_layers:
            sh, sw = layer.conv.stride
            total_stride_h *= sh
            total_stride_w *= sw
        self.patch_size = (total_stride_h, total_stride_w)
        self.patch_stride = self.patch_size

        self.use_fpn_decoder = use_fpn_decoder
        self.in_channels = in_channels

    def forward(self, x, return_features=True):
        # x: [B, C, H, W]
        stage_features = []
        for i in range(self.model.num_stage):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            stage_features.append(x)

        if return_features:
            if self.use_fpn_decoder:
                # FPN expects [B, C, H, W] per stage.
                stage_features = [f.permute(0, 3, 1, 2).contiguous() for f in stage_features]
                return stage_features, None

            # Return tokens for linear/up_conv decoders.
            last = stage_features[-1]
            B, H, W, C = last.shape
            tokens = last.reshape(B, H * W, C)
            return tokens, None

        return stage_features[-1]
