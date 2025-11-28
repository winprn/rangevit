import torch
import torch.nn as nn

from .encoder import RangeFormerBackbone
from .decoder import SegmentationHead
from ..rangevit_kpconv import KPClassifier


class RangeFormerKPConv(nn.Module):
    """
    End-to-end RangeFormer with optional KPConv refinement for 3D supervision.
    - forward_2d_features: returns 2D logits for sliding-window inference
    - forward: samples logits to points and refines with KPConv
    """

    def __init__(self, backbone: RangeFormerBackbone, head: SegmentationHead, kpclassifier: KPClassifier, n_cls: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.kpclassifier = kpclassifier
        self.n_cls = n_cls

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def _forward_logits(self, im: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = self.backbone(im)
        logits, _ = self.head([f1, f2, f3, f4])
        return logits

    def forward_2d_features(self, im: torch.Tensor) -> torch.Tensor:
        return self._forward_logits(im)

    def forward(self, im: torch.Tensor, px, py, pxyz, pknn, num_points):
        logits_2d = self.forward_2d_features(im)
        masks3d = self.kpclassifier(logits_2d, px, py, pxyz, pknn, num_points)
        return masks3d


class RangeFormerModel(nn.Module):
    """
    Thin wrapper to keep the same interface used by the training pipeline.
    Exposes the RangeFormer under the `.rangevit` attribute for compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        n_cls: int,
        backbone_depths=(3, 4, 6, 3), # layers per stage
        backbone_embed_dims=(128, 128, 320, 512),
        backbone_heads=(1, 2, 5, 8),
        backbone_sr_ratios=(8, 4, 2, 1),
        drop_path_rate: float = 0.1,
        use_kpconv: bool = False,
    ):
        super().__init__()
        backbone = RangeFormerBackbone(
            depths=backbone_depths,
            embed_dims=backbone_embed_dims,
            num_heads=backbone_heads,
            sr_ratios=backbone_sr_ratios,
            drop_path_rate=drop_path_rate,
            in_channels=in_channels,
        )
        head = SegmentationHead(stage_channels=list(backbone_embed_dims), out_ch_unify=256, num_classes=n_cls)

        if use_kpconv:
            kpclassifier = KPClassifier(in_channels=n_cls, out_channels=n_cls, num_classes=n_cls)
            self.rangevit = RangeFormerKPConv(backbone, head, kpclassifier, n_cls=n_cls)
        else:
            # Fallback: just return logits in forward
            class _RangeFormerNoKP(nn.Module):
                def __init__(self, backbone, head, n_cls):
                    super().__init__()
                    self.backbone = backbone
                    self.head = head
                    self.n_cls = n_cls

                def forward_2d_features(self, im):
                    f1, f2, f3, f4 = self.backbone(im)
                    logits, _ = self.head([f1, f2, f3, f4])
                    return logits

                def forward(self, im, *args, **kwargs):
                    return self.forward_2d_features(im)

            self.rangevit = _RangeFormerNoKP(backbone, head, n_cls=n_cls)

    def forward(self, *args, **kwargs):
        return self.rangevit(*args, **kwargs)

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = count_parameters(self.rangevit)
        return stats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # params count test
    model = RangeFormerModel(in_channels=5, n_cls=20, use_kpconv=False)
    stats = model.counter_model_parameters()
    print(stats)
