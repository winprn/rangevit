import copy
import torch.nn as nn

from .segman_encoder import SegMANEncoder
from .segman_decoder import SegMANDecoder


def _normalize_variant(name):
    if name is None:
        return None
    name = str(name).lower()
    if name.startswith("segman_"):
        return name.split("_", 1)[1]
    return name


class SegMANRangeSeg(nn.Module):
    def __init__(self, in_channels, n_cls, image_size, segman_cfg):
        super().__init__()
        segman_cfg = copy.deepcopy(segman_cfg or {})

        variant = _normalize_variant(segman_cfg.pop("variant", None))
        embed_dims = segman_cfg.pop("embed_dims", None)
        depths = segman_cfg.pop("depths", None)
        num_heads = segman_cfg.pop("num_heads", None)
        window_size = segman_cfg.pop("window_size", None)
        window_dilation = segman_cfg.pop("window_dilation", None)
        sr_ratio = segman_cfg.pop("sr_ratio", None)
        mlp_ratios = segman_cfg.pop("mlp_ratios", None)
        drop_path_rate = segman_cfg.pop("drop_path_rate", 0.0)
        use_rpb = segman_cfg.pop("use_rpb", True)
        layerscales = segman_cfg.pop("layerscales", None)
        layer_init_values = segman_cfg.pop("layer_init_values", 1e-6)
        ssm_ratio = segman_cfg.pop("ssm_ratio", 1.0)
        ssm_split = segman_cfg.pop("ssm_split", False)
        fused_na = segman_cfg.pop("fused_na", False)
        pretrained = segman_cfg.pop("pretrained", None)
        feat_proj_dim = segman_cfg.pop("feat_proj_dim", 256)
        channel_split = segman_cfg.pop("channel_split", False)
        short_cut = segman_cfg.pop("short_cut", False)
        interpolate_mode = segman_cfg.pop("interpolate_mode", "bilinear")

        if embed_dims is None:
            if variant == "t":
                embed_dims = [32, 64, 144, 192]
                depths = depths or [2, 2, 4, 2]
                num_heads = num_heads or [1, 2, 4, 8]
                window_size = window_size or [11, 9, 9, 7]
                mlp_ratios = mlp_ratios or [4, 4, 3, 3]
                layerscales = layerscales or [False, False, False, False]
            elif variant == "b":
                embed_dims = [96, 160, 364, 560]
                depths = depths or [4, 4, 18, 4]
                num_heads = num_heads or [4, 8, 13, 20]
                window_size = window_size or [11, 9, 7, 7]
                mlp_ratios = mlp_ratios or [4, 4, 3, 3]
                layerscales = layerscales or [True, True, True, True]
            elif variant == "l":
                embed_dims = [96, 192, 432, 640]
                depths = depths or [4, 4, 28, 4]
                num_heads = num_heads or [4, 8, 12, 20]
                window_size = window_size or [11, 9, 7, 7]
                mlp_ratios = mlp_ratios or [4, 4, 3, 3]
                layerscales = layerscales or [True, True, True, True]
            else:
                embed_dims = [64, 144, 288, 512]
                depths = depths or [2, 2, 10, 4]
                num_heads = num_heads or [2, 4, 8, 16]
                window_size = window_size or [11, 9, 7, 7]
                mlp_ratios = mlp_ratios or [4, 4, 3.4, 3.4]
                layerscales = layerscales or [False, False, False, False]

        window_dilation = window_dilation or [1, 1, 1, 1]
        sr_ratio = sr_ratio or [8, 4, 2, 1]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        layerscales = layerscales or [False, False, False, False]

        self.encoder = SegMANEncoder(
            image_size=image_size,
            in_chans=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            window_dilation=window_dilation,
            use_rpb=use_rpb,
            sr_ratio=sr_ratio,
            mlp_ratios=mlp_ratios,
            drop_path_rate=drop_path_rate,
            layerscales=layerscales,
            layer_init_values=layer_init_values,
            ssm_split=ssm_split,
            fused_na=fused_na,
            ssm_ratio=ssm_ratio,
            pretrained=pretrained,
        )

        self.decoder = SegMANDecoder(
            in_channels=embed_dims,
            channels=embed_dims[0],
            num_classes=n_cls,
            feat_proj_dim=feat_proj_dim,
            image_size=image_size,
            channel_split=channel_split,
            short_cut=short_cut,
            interpolate_mode=interpolate_mode,
            use_rpb=use_rpb,
        )

        self.n_cls = n_cls

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(feats)

    @property
    def rangevit(self):
        return self

    def counter_model_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return {
            "total_num_parameters": total,
            "encoder_num_parameters": encoder,
            "decoder_num_parameters": decoder,
        }
