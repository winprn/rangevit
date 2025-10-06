# rangeformer_reference.py
# Reference PyTorch implementation of the RangeFormer flow (paper: Kong et al. 2023).
# - Rasterize 3D point cloud -> range image
# - RangeAug (RangeMix, RangeUnion, RangePaste, RangeShift) per paper pseudo-code
# - REM (6 -> 64 -> 128 -> 128) using 1x1 convs
# - Overlap patch embed (3x3) + hierarchical transformer stages
# - Decoder: channel unification to 256, bilinear upsample to HxW, concat -> MLP to classes
# - RangePost & STR helpers (per paper pseudocode)
#
# NOTE: The paper gives stage channel dims and heads per stage; it doesn't list exact
#       transformer block depths in one place. I choose reasonable defaults which you can
#       change in `depths`. See comments where assumptions were made.
#
# Required: torch >= 1.8
import math
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 3D -> 2D rasterization utils
# ---------------------------
class RangeRasterizer:
    """
    Rasterize a point cloud into a range image with channels:
       [x, y, z, depth, intensity, existence]
    - points: (N, >=4) columns = (x,y,z,intensity[, ...])
    - labels (optional): (N,) ints
    Returns:
      rv: numpy array shape (C=6, H, W)
      index_map: numpy array shape (H, W) -> index of chosen point in original array or -1
      label_map: numpy array shape (H, W) for labels (if provided), else -1
    Implementation choices:
      - If multiple points fall into the same pixel, keep the one with **smallest depth** (closest to sensor).
      - Projection formula follows standard spherical projection used in range-image LiDAR works:
          col ~ azimuth, row ~ elevation.
      - Default sensor vertical FOV settable (defaults chosen for HDL-64E like sensors).
    """
    def __init__(self, H: int = 64, W: int = 2048, fov_up: float = 3.0, fov_down: float = -25.0):
        """
        H: vertical pixels (beam number)
        W: horizontal resolution (#azimuth bins)
        fov_up/down: degrees (fov_up positive, fov_down negative usually)
        """
        self.H = H
        self.W = W
        self.fov_up = fov_up
        self.fov_down = fov_down
        # convert to radians for calculation
        self.fov_up_rad = math.radians(fov_up)
        self.fov_down_rad = math.radians(fov_down)
        self.fov_rad = abs(self.fov_up_rad) + abs(self.fov_down_rad)

    def rasterize(self, points: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        points: numpy array (N, >=4) (x, y, z, intensity, ...)
        labels: optional (N,) int labels
        returns: rv (6, H, W), index_map (H, W), label_map (H, W)
        """
        N = points.shape[0]
        x = points[:, 0].astype(np.float32)
        y = points[:, 1].astype(np.float32)
        z = points[:, 2].astype(np.float32)
        intensity = points[:, 3].astype(np.float32) if points.shape[1] > 3 else np.zeros(N, dtype=np.float32)

        # Compute depth (range)
        depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # (N,)

        # Spherical coords
        # elevation: angle above horizon (arctan(z / sqrt(x^2 + y^2)))
        elev = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))  # [-pi/2, pi/2]
        azim = np.arctan2(y, x)  # [-pi, pi]

        # Project to image coords
        # column (u) from azimuth. Map [-pi, pi] -> [0, W)
        u = (0.5 * (azim / math.pi + 1.0)) * (self.W - 1)
        # row (v) from elevation. Map [fov_down, fov_up] -> [H-1..0] (top row is high elevation)
        # convert elevation to fraction of total fov
        # note: ensure elevation outside FOV are ignored
        v = (1.0 - (elev - self.fov_down_rad) / self.fov_rad) * (self.H - 1)

        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)

        # bounds
        mask = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H) \
               & (elev <= self.fov_up_rad + 1e-6) & (elev >= self.fov_down_rad - 1e-6)
        valid_idx = np.nonzero(mask)[0]

        # initialize maps
        rv = np.zeros((6, self.H, self.W), dtype=np.float32)  # x,y,z,depth,intensity,existence
        index_map = -np.ones((self.H, self.W), dtype=np.int32)
        depth_map = np.full((self.H, self.W), fill_value=np.inf, dtype=np.float32)
        label_map = -np.ones((self.H, self.W), dtype=np.int32) if labels is not None else None

        # iterate valid points: choose point with smallest depth for each pixel
        for i in valid_idx:
            ri = v[i]
            cj = u[i]
            d = depth[i]
            if d < depth_map[ri, cj]:
                depth_map[ri, cj] = d
                index_map[ri, cj] = i
                rv[0, ri, cj] = x[i]
                rv[1, ri, cj] = y[i]
                rv[2, ri, cj] = z[i]
                rv[3, ri, cj] = d
                rv[4, ri, cj] = intensity[i]
                rv[5, ri, cj] = 1.0
                if labels is not None:
                    label_map[ri, cj] = int(labels[i])

        return rv, index_map, label_map


# ---------------------------
# RangeAug (RangeMix, RangeUnion, RangePaste, RangeShift)
# ---------------------------
def range_mix(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray, mix_strategy: Tuple[int,int]):
    """
    RangeMix per paper pseudo-code (grid mixing).
    xa, xb: rv arrays (C, H, W)
    ya, yb: label maps (H, W)
    mix_strategy: tuple (phi, theta) dividing H and W into blocks
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    phi, theta = mix_strategy
    mix_h = int(xa.shape[1] / phi)
    mix_w = int(xa.shape[2] / theta)
    for i in range(1, mix_h + 1):
        for j in range(1, mix_w + 1):
            r0 = (i - 1) * phi
            c0 = (j - 1) * theta
            r1 = min(i * phi, xa.shape[1])
            c1 = min(j * theta, xa.shape[2])
            xa_[:, r0:r1, c0:c1] = xb[:, r0:r1, c0:c1]
            ya_[r0:r1, c0:c1] = yb[r0:r1, c0:c1]
    return xa_, ya_


def range_union(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray, kunion: float = 0.5):
    """
    RangeUnion: fill void pixels (existence channel = 0) in A with B
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    mask = xa_[-1, :, :]  # existence
    void = mask == 0
    # choose random subset of voids (kunion fraction)
    void_coords = np.stack(np.where(void), axis=1)
    if void_coords.shape[0] == 0:
        return xa_, ya_
    K = int(void_coords.shape[0] * kunion)
    pick_idx = np.random.choice(void_coords.shape[0], size=K, replace=False)
    chosen = void_coords[pick_idx]
    for (r, c) in chosen:
        xa_[:, r, c] = xb[:, r, c]
        ya_[r, c] = yb[r, c]
    return xa_, ya_


def range_paste(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray, sem_classes: List[int]):
    """
    RangePaste: paste pixels from xb that belong to rare semantic classes into xa
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    for sem_class in sem_classes:
        pix = (yb == sem_class)
        if pix.sum() == 0:
            continue
        xa_[:, pix] = xb[:, pix]
        ya_[pix] = yb[pix]
    return xa_, ya_


def range_shift(xa: np.ndarray, ya: np.ndarray):
    """
    RangeShift: shift image along width (azimuth) by random kshift in [W/4, 3W/4]
    """
    xa_ = xa.copy()
    ya_ = ya.copy()
    h, w = xa.shape[1], xa.shape[2]
    p = random.randint(int(0.25 * w), int(0.75 * w))
    xa_ = np.concatenate([xa[:, :, p:], xa[:, :, :p]], axis=2)
    ya_ = np.concatenate([ya[:, p:], ya[:, :p]], axis=1)
    return xa_, ya_


# ---------------------------
# REM, PatchEmbed, TransformerBlock, Backbone, Decoder
# ---------------------------
class REM(nn.Module):
    """Range Embedding Module: 6 -> 64 -> 128 -> 128 (1x1 convs) with BN+GELU"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

    def forward(self, x):
        # x: (B, 6, H, W)
        return self.net(x)  # (B, 128, H, W)


class PatchEmbedOverlap(nn.Module):
    """3x3 overlapping patch embedding (paper: patch size 3x3, stride=1 for stage1, stride=2 for later stages)"""
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TransformerBlock2D(nn.Module):
    """
    Transformer block for 2D range image. Implements:
      - LayerNorm -> MultiHeadAttention (on flattened patches)
      - FFN (Linear-GELU-Linear) + a 3x3 conv branch (as in the paper ff detail)
      - Residual connections
    Notes:
      - Input: (B, C, H, W)
      - Flatten to (B, N, C) for attention where N = H*W
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        # PyTorch's MultiheadAttention expects (B, N, C) with batch_first=True (newer versions)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp_fc1 = nn.Linear(dim, hidden_dim)
        self.mlp_fc2 = nn.Linear(hidden_dim, dim)
        # convolution branch in FFN to inject local spatial info
        self.conv_branch = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        # flatten spatial dims and transpose to (B, N, C)
        x_flat = x.view(B, C, N).permute(0, 2, 1).contiguous()  # (B, N, C)

        # attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)  # (B, N, C)
        x2 = x_flat + attn_out

        # FFN
        x2_norm = self.norm2(x2)
        ffn = self.mlp_fc2(self.act(self.mlp_fc1(x2_norm)))  # (B, N, C)

        # conv branch: operate on original spatial map
        x_spatial = x  # (B, C, H, W)
        conv_out = self.conv_branch(x_spatial)  # (B, C, H, W)
        conv_out_flat = conv_out.view(B, C, N).permute(0, 2, 1).contiguous()

        x_out = x2 + ffn + conv_out_flat  # residual add
        # reshape back to (B, C, H, W)
        x_out = x_out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return x_out


class RangeFormerBackbone(nn.Module):
    """
    Backbone implementing REM -> 4 stages of patch embedding + stacks of TransformerBlock2D
    Follows paper: stage channels [128, 128, 320, 512], heads [3,4,6,3].
    Depths per stage are not specified explicitly in the paper; here we set defaults that can be altered.
    """
    def __init__(self, H: int, W: int, num_classes: int,
                 depths: List[int] = [2, 2, 6, 2],   # assumption: you can change as needed
                 stage_channels: List[int] = [128, 128, 320, 512],
                 heads: List[int] = [3, 4, 6, 3]):
        super().__init__()
        assert len(depths) == 4 and len(stage_channels) == 4 and len(heads) == 4
        self.H = H
        self.W = W
        self.rem = REM()  # outputs (B, 128, H, W)
        # stage 1: patch embed stride 1
        self.patch1 = PatchEmbedOverlap(128, stage_channels[0], stride=1)
        # stage2..4: patch embed stride 2 (downsampling)
        self.patch2 = PatchEmbedOverlap(stage_channels[0], stage_channels[1], stride=2)
        self.patch3 = PatchEmbedOverlap(stage_channels[1], stage_channels[2], stride=2)
        self.patch4 = PatchEmbedOverlap(stage_channels[2], stage_channels[3], stride=2)

        # transformer stacks
        self.stage1_blocks = nn.ModuleList([TransformerBlock2D(stage_channels[0], num_heads=heads[0]) for _ in range(depths[0])])
        self.stage2_blocks = nn.ModuleList([TransformerBlock2D(stage_channels[1], num_heads=heads[1]) for _ in range(depths[1])])
        self.stage3_blocks = nn.ModuleList([TransformerBlock2D(stage_channels[2], num_heads=heads[2]) for _ in range(depths[2])])
        self.stage4_blocks = nn.ModuleList([TransformerBlock2D(stage_channels[3], num_heads=heads[3]) for _ in range(depths[3])])

    def forward(self, x):
        """
        x: (B, 6, H, W) range image
        returns list of stage features: [F1, F2, F3, F4] with shapes:
          F1: (B, C1, H, W)
          F2: (B, C2, H/2, W/2)
          F3: (B, C3, H/4, W/4)
          F4: (B, C4, H/8, W/8)
        """
        B = x.shape[0]
        x = self.rem(x)  # (B, 128, H, W)
        # stage 1
        x1 = self.patch1(x)
        for blk in self.stage1_blocks:
            x1 = blk(x1)
        # stage 2
        x2 = self.patch2(x1)
        for blk in self.stage2_blocks:
            x2 = blk(x2)
        # stage 3
        x3 = self.patch3(x2)
        for blk in self.stage3_blocks:
            x3 = blk(x3)
        # stage 4
        x4 = self.patch4(x3)
        for blk in self.stage4_blocks:
            x4 = blk(x4)
        return [x1, x2, x3, x4]


class SegmentationHead(nn.Module):
    """
    Decoder head:
      - Channel unification: map each Fi (with dim di) -> 256 via 1x1 conv
      - Spatial unify: upsample Fi (for i>1) to HxW using bilinear interp
      - Concatenate four 256-feature maps -> MLP (conv1x1 + GELU + conv1x1) to classes
      - Auxiliary heads: 1x1 conv per Fi to classes (for auxiliary losses)
    """
    def __init__(self, stage_channels: List[int], out_ch_unify: int = 256, num_classes: int = 19, H: int = 64, W: int = 2048):
        super().__init__()
        self.unify_layers = nn.ModuleList([nn.Conv2d(c, out_ch_unify, kernel_size=1) for c in stage_channels])
        self.aux_heads = nn.ModuleList([nn.Sequential(nn.Conv2d(out_ch_unify, out_ch_unify//2, 1), nn.GELU(), nn.Conv2d(out_ch_unify//2, num_classes, 1)) for _ in stage_channels])
        self.main_mlp = nn.Sequential(
            nn.Conv2d(out_ch_unify * 4, out_ch_unify, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch_unify),
            nn.GELU(),
            nn.Conv2d(out_ch_unify, num_classes, kernel_size=1)
        )
        self.H = H
        self.W = W

    def forward(self, features: List[torch.Tensor]):
        """
        features: [F1, F2, F3, F4] as output by backbone
        return:
          logits_main: (B, num_classes, H, W)
          aux_logits: list of (B, num_classes, H, W) from each stage (after upsample/interp)
        """
        ups = []
        auxs = []
        for i, f in enumerate(features):
            f_unify = self.unify_layers[i](f)  # (B, 256, Hi, Wi)
            # upsample to (H, W)
            f_up = F.interpolate(f_unify, size=(self.H, self.W), mode='bilinear', align_corners=False)
            ups.append(f_up)
            # aux
            aux = self.aux_heads[i](f_unify)
            aux_up = F.interpolate(aux, size=(self.H, self.W), mode='bilinear', align_corners=False)
            auxs.append(aux_up)
        cat = torch.cat(ups, dim=1)  # (B, 256*4, H, W)
        logits = self.main_mlp(cat)  # (B, num_classes, H, W)
        return logits, auxs


# ---------------------------
# Full Model wrapper
# ---------------------------
class RangeFormer(nn.Module):
    def __init__(self, H: int, W: int, num_classes: int, depths: List[int] = [2, 2, 6, 2]):
        super().__init__()
        self.backbone = RangeFormerBackbone(H=H, W=W, num_classes=num_classes, depths=depths)
        stage_channels = [128, 128, 320, 512]
        self.head = SegmentationHead(stage_channels=stage_channels, out_ch_unify=256, num_classes=num_classes, H=H, W=W)

    def forward(self, rv: torch.Tensor):
        """
        rv: (B, 6, H, W) range image tensor (float)
        returns: logits_main (B, C, H, W), aux_logits list
        """
        features = self.backbone(rv)
        logits, auxs = self.head(features)
        return logits, auxs


# ---------------------------
# Helper: map 2D predictions back to original points
# ---------------------------
def rv_preds_to_point_labels(pred_map: np.ndarray, index_map: np.ndarray, num_points: int, default_label: int = 0):
    """
    pred_map: (H, W) predicted class ids (np int)
    index_map: (H, W) mapping pixel -> original point index or -1
    returns: per-point labels array shape (num_points,)
    Points not selected in rasterization will remain default_label (usually 0)
    Note: For many-to-one conflicts some points are never assigned a pixel; RangePost helps recover them.
    """
    point_labels = np.ones((num_points,), dtype=np.int32) * default_label
    H, W = index_map.shape
    for r in range(H):
        for c in range(W):
            idx = index_map[r, c]
            if idx >= 0:
                point_labels[idx] = int(pred_map[r, c])
    return point_labels


# ---------------------------
# STR helpers (split pointcloud into Z azimuthal "views")
# ---------------------------
def split_into_views(points: np.ndarray, Z: int):
    """
    Split scan into Z views based on azimuth angle theta = arctan2(y, x).
    Returns a list of points arrays, each representing a view.
    Paper's STR: during training sample only one view randomly; inference rasterize all.
    Implementation: returns a list where view_i contains points whose azimuth falls into i-th bin.
    """
    azim = np.arctan2(points[:, 1], points[:, 0])  # [-pi, pi]
    # Map to [0, 2pi) for easier binning
    azim_pos = azim.copy()
    azim_pos[azim_pos < 0] += 2 * math.pi
    bins = np.linspace(0, 2 * math.pi, num=Z + 1)
    views = []
    for i in range(Z):
        mask = (azim_pos >= bins[i]) & (azim_pos < bins[i + 1])
        views.append(points[mask])
    return views


# ---------------------------
# RangePost pseudocode implementation (per paper)
# ---------------------------
def range_post_inference(model: nn.Module, rasterize_fn, scan: np.ndarray, num_sub: int, knn_postproc_fn=None):
    """
    Implements Algorithm 3 RangePost in the paper.
    - scan: (N, c)
    - rasterize_fn: callable scan->(rv, index_map, label_map) (we reuse RangeRasterizer.rasterize)
    - num_sub: number of sub-clouds (paper splits whole scan into equal-interval subclouds)
    - knn_postproc_fn: optional k-NN post-processing per-subcloud preds
    returns: pred (N,) int array of labels
    """
    # Step 1: Split
    subclouds = []
    indices = []
    for i in range(num_sub):
        sub = scan[i::num_sub, :]
        subclouds.append(sub)
        # indices in original scan
        idxs = np.arange(i, scan.shape[0], num_sub)
        indices.append(idxs)

    # Step 2: stack rasterized subclouds
    rvs = []
    index_maps = []
    for sub in subclouds:
        rv, index_map, _ = rasterize_fn(sub, None)
        rvs.append(rv)
        index_maps.append(index_map)

    # Convert to torch batch and forward
    batch_rv = torch.from_numpy(np.stack(rvs, axis=0)).float()  # (num_sub, C, H, W)
    with torch.no_grad():
        logits_batch, _ = model(batch_rv)  # (num_sub, num_classes, H, W)
        preds_batch = logits_batch.argmax(dim=1).cpu().numpy()  # (num_sub, H, W)

    # Step 4: unstack / map predictions back
    final_pred = np.zeros(scan.shape[0], dtype=np.int32)
    for j in range(len(preds_batch)):
        pred_j = preds_batch[j]  # (H, W)
        idxs = indices[j]
        # Map back using index_map
        ind_map = index_maps[j]
        for r in range(ind_map.shape[0]):
            for c in range(ind_map.shape[1]):
                idx = ind_map[r, c]
                if idx >= 0:
                    # idx is index within subcloud; convert to global index
                    global_idx = idxs[idx]
                    final_pred[global_idx] = int(pred_j[r, c])

    # Optionally apply knn_postproc on each sub result (if provided)
    if knn_postproc_fn is not None:
        final_pred = knn_postproc_fn(final_pred, scan)

    return final_pred


# ---------------------------
# Example usage snippet
# ---------------------------
if __name__ == "__main__":
    # small sanity run with fake data
    H, W = 64, 1024  # horizontal resolution can be e.g. 512,1024,2048 (paper experiments)
    num_classes = 19

    # generate fake point cloud with 10000 points: (x,y,z,intensity)
    N = 120000  # typical SemanticKITTI ~120k
    # Here we generate a simplified spherical point distribution for demo only
    rng = np.random.RandomState(42)
    thetas = rng.rand(N) * 2 * math.pi - math.pi
    phis = (rng.rand(N) - 0.5) * math.radians(28)  # small vertical angle range
    ranges = rng.rand(N) * 80 + 1.0
    x = ranges * np.cos(phis) * np.cos(thetas)
    y = ranges * np.cos(phis) * np.sin(thetas)
    z = ranges * np.sin(phis)
    intensity = rng.rand(N).astype(np.float32)
    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)

    # create rasterizer and rasterize
    raster = RangeRasterizer(H=H, W=W, fov_up=3.0, fov_down=-25.0)
    rv_np, idx_map, label_map = raster.rasterize(points, labels=None)  # rv_np shape (6,H,W)

    # convert to torch and run model
    model = RangeFormer(H=H, W=W, num_classes=num_classes)
    model.eval()
    rv_t = torch.from_numpy(rv_np[None, ...]).float()  # (1, 6, H, W)
    with torch.no_grad():
        logits, auxs = model(rv_t)  # (1, num_classes, H, W)
        preds2d = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

    # map back to points
    per_point_labels = rv_preds_to_point_labels(preds2d, idx_map, num_points=N, default_label=0)
    print("Per-point labels shape:", per_point_labels.shape)
    print("Unique labels in demo:", np.unique(per_point_labels))
