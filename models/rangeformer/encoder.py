import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RangeEmbeddingModule(nn.Module):
    """
    REM: maps range image R(u,v) of shape (B, C_in, H, W)
    to an embedding F0 of shape (B, 128, H, W).
    """

    def __init__(self, in_channels=5, embed_dim=128):
        super().__init__()
        assert embed_dim == 128, "Paper uses 128-dim embedding after REM."

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        return: (B, 128, H, W)
        """
        return self.layers(x)

class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding implemented as a Conv2d + LayerNorm.

    Input:  (B, C_in, H, W)
    Output: x: (B, N, C_out) with N = H_out * W_out, and H_out, W_out
    """

    def __init__(self,
                 in_channels,
                 embed_dim,
                 patch_size=3,
                 stride=1):
        super().__init__()
        padding = patch_size // 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        Returns:
            x_flat: (B, N, C_out)
            H_out, W_out: spatial size after conv
        """
        B, C, H, W = x.shape
        x = self.proj(x)           # (B, C_out, H_out, W_out)
        B, C_out, H_out, W_out = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C_out)
        x = self.norm(x)
        return x, H_out, W_out

class SpatialReductionAttention(nn.Module):
    """
    Multi-head self-attention with optional spatial reduction (sr_ratio).

    Input:  x: (B, N, C)
            H, W: spatial resolution so that N = H * W
    Output: (B, N, C)
    """

    def __init__(self,
                 dim,
                 num_heads,
                 sr_ratio=1,
                 qkv_bias=True,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio,
                                stride=sr_ratio, padding=0)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None

    def forward(self, x, H, W):
        B, N, C = x.shape

        # Q from full sequence
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

        # K, V possibly from spatially-reduced sequence
        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)   # (B, C, H, W)
            x_ = self.sr(x_)                             # (B, C, H', W')
            H_sr, W_sr = x_.shape[2], x_.shape[3]
            x_ = x_.reshape(B, C, -1).transpose(1, 2)    # (B, N_sr, C)
            x_ = self.norm(x_)
            kv = self.kv(x_)                             # (B, N_sr, 2C)
        else:
            kv = self.kv(x)                              # (B, N, 2C)

        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)                  # (2, B, heads, N_kv, head_dim)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)      # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MlpWithConv(nn.Module):
    """
    FFN:
        Linear -> (optional 3x3 depthwise conv in spatial domain) -> GELU -> Drop -> Linear
    The 3x3 conv injects positional information directly into the embeddings.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 act_layer=nn.GELU,
                 drop=0.0,
                 use_3x3_conv=True):
        super().__init__()
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.use_3x3_conv = use_3x3_conv

        if use_3x3_conv:
            self.dwconv = nn.Conv2d(hidden_features, hidden_features,
                                    kernel_size=3, stride=1, padding=1,
                                    groups=hidden_features)
        else:
            self.dwconv = None

        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x, H, W):
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        x = self.fc1(x)  # (B, N, hidden)
        if self.use_3x3_conv:
            # go to spatial domain
            x_spatial = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, hidden, H, W)
            x_spatial = self.dwconv(x_spatial)
            x = x_spatial.flatten(2).transpose(1, 2)            # (B, N, hidden)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    Single transformer block used in each stage.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 sr_ratio=1,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpWithConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            use_3x3_conv=True,
        )

        # Simple DropPath implementation (stochastic depth)
        self.drop_path_rate = drop_path

    def drop_path(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    def forward(self, x, H, W):
        # x: (B, N, C)
        residual = x
        x = self.norm1(x)
        x = self.attn(x, H, W)
        x = residual + self.drop_path(x)

        residual = x
        x = self.norm2(x)
        x = self.mlp(x, H, W)
        x = residual + self.drop_path(x)
        return x

class RangeFormerBackbone(nn.Module):
    """
    RangeFormer-style 4-stage transformer backbone.

    Input:
        range_img: (B, 6, H, W)

    Output:
        F1: (B, 128, H,   W)
        F2: (B, 128, H/2, W/2)
        F3: (B, 320, H/4, W/4)
        F4: (B, 512, H/8, W/8)
    """

    def __init__(self,
                 depths=(2, 2, 6, 2), # number of transformer blocks in each stage
                 embed_dims=(128, 128, 320, 512),
                 num_heads=(2, 2, 5, 8),
                 sr_ratios=(8, 4, 2, 1),
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 device=None,
                 use_amp=False,
                 in_channels=5):
        super().__init__()

        # Optional device pinning so the backbone can live on GPU even if the caller forgets.
        self.device = torch.device(device) if device is not None else None
        self.use_amp = use_amp
        self._runtime_device = self.device  # updated lazily in _ensure_device

        self.rem = RangeEmbeddingModule(in_channels=in_channels, embed_dim=128)

        d1, d2, d3, d4 = depths
        c1, c2, c3, c4 = embed_dims
        h1, h2, h3, h4 = num_heads
        sr1, sr2, sr3, sr4 = sr_ratios

        # Stochastic depth decay across all blocks
        total_blocks = d1 + d2 + d3 + d4
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        idx = 0

        # Stage 1 (no downsampling)
        self.patch_embed1 = OverlapPatchEmbed(
            in_channels=128,
            embed_dim=c1,
            patch_size=3,
            stride=1,  # keep H, W
        )
        self.block1 = nn.ModuleList([
            TransformerBlock(
                dim=c1,
                num_heads=h1,
                mlp_ratio=4.0,
                sr_ratio=sr1,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[idx + i],
            )
            for i in range(d1)
        ])
        idx += d1

        # Stage 2 (downsample by 2)
        self.patch_embed2 = OverlapPatchEmbed(
            in_channels=c1,
            embed_dim=c2,
            patch_size=3,
            stride=2,
        )
        self.block2 = nn.ModuleList([
            TransformerBlock(
                dim=c2,
                num_heads=h2,
                mlp_ratio=4.0,
                sr_ratio=sr2,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[idx + i],
            )
            for i in range(d2)
        ])
        idx += d2

        # Stage 3 (downsample by 2)
        self.patch_embed3 = OverlapPatchEmbed(
            in_channels=c2,
            embed_dim=c3,
            patch_size=3,
            stride=2,
        )
        self.block3 = nn.ModuleList([
            TransformerBlock(
                dim=c3,
                num_heads=h3,
                mlp_ratio=4.0,
                sr_ratio=sr3,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[idx + i],
            )
            for i in range(d3)
        ])
        idx += d3

        # Stage 4 (downsample by 2)
        self.patch_embed4 = OverlapPatchEmbed(
            in_channels=c3,
            embed_dim=c4,
            patch_size=3,
            stride=2,
        )
        self.block4 = nn.ModuleList([
            TransformerBlock(
                dim=c4,
                num_heads=h4,
                mlp_ratio=4.0,
                sr_ratio=sr4,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[idx + i],
            )
            for i in range(d4)
        ])

        if self.device is not None:
            self.to(self.device)

    def _ensure_device(self, range_img):
        """
        Keep the module and the input on the same device.
        If a target device was provided at init, move both there once.
        """
        target_device = self.device or range_img.device
        # Lazily move the whole module once; avoids silent CPU execution.
        if self._runtime_device != target_device:
            self.to(target_device)
            self._runtime_device = target_device

        if range_img.device != target_device:
            range_img = range_img.to(target_device, non_blocking=True)
        return range_img, target_device

    def _run_stage(self, x_in, patch_embed, blocks):
        """
        Helper: run patch embedding + all transformer blocks,
        and return both sequence and feature map.
        """
        B, C, H, W = x_in.shape
        x, H_out, W_out = patch_embed(x_in)  # (B, N, C_out)

        for blk in blocks:
            x = blk(x, H_out, W_out)

        # reshape back to (B, C_out, H_out, W_out)
        C_out = x.shape[-1]
        feat = x.transpose(1, 2).reshape(B, C_out, H_out, W_out)
        return x, feat, H_out, W_out

    def forward(self, range_img):
        """
        range_img: (B, 6, H, W)
        Returns:
            F1, F2, F3, F4 feature maps.
        """
        range_img, target_device = self._ensure_device(range_img)
        autocast_ctx = torch.amp.autocast(
            "cuda",
            enabled=self.use_amp and target_device.type == "cuda",
        )

        # REM
        with autocast_ctx:
            x = self.rem(range_img)  # (B, 128, H, W)

            # Stage 1
            _, F1, H1, W1 = self._run_stage(x, self.patch_embed1, self.block1)

            # Stage 2
            _, F2, H2, W2 = self._run_stage(F1, self.patch_embed2, self.block2)

            # Stage 3
            _, F3, H3, W3 = self._run_stage(F2, self.patch_embed3, self.block3)

            # Stage 4
            _, F4, H4, W4 = self._run_stage(F3, self.patch_embed4, self.block4)

        return F1, F2, F3, F4

if __name__ == "__main__":
    print("Testing RangeFormerBackbone...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RangeFormerBackbone(device=device)
    x = torch.randn(2, 6, 64, 2048, device=device)  # e.g., SemanticKITTI config
    F1, F2, F3, F4 = model(x)
    print(F1.shape)  # (2, 128, 64, 2048)
    print(F2.shape)  # (2, 128, 32, 1024)
    print(F3.shape)  # (2, 320, 16, 512)
    print(F4.shape)  # (2, 512, 8, 256)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
