import torch
from typing import Tuple, Optional

Tensor = torch.Tensor

def _ensure_hw_first(x: Tensor, channels_last: bool) -> Tuple[Tensor, bool]:
    """Return tensor as (H, W, C)."""
    if channels_last:
        assert x.ndim == 3, "Expected (H, W, C)"
        return x, True
    else:
        assert x.ndim == 3, "Expected (C, H, W)"
        return x.permute(1, 2, 0), False

def _restore_layout(x_hw_c: Tensor, channels_last: bool) -> Tensor:
    return x_hw_c if channels_last else x_hw_c.permute(2, 0, 1)

class ScanAugmentor:
    @staticmethod
    @torch.no_grad()
    def range_mix(
        xa: Tensor, ya: Tensor,
        xb: Tensor, yb: Tensor,
        *,
        kmix: Optional[int] = None,
        bin_select: str = "random",    # {"random","alternate"}
        channels_last: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        RangeMix: split the vertical (inclination) axis into k equal bins and
        replace the corresponding rows in (xa, ya) with those from (xb, yb).
        - kmix: if None, sample from {2,3,4,5,6} as in the paper.
        - bin_select:
            "random"   -> independently choose each bin with p=0.5 to replace
            "alternate"-> replace every other bin starting from a random offset
        Shapes:
        xa, xb : (H, W, C) if channels_last else (C, H, W)
        ya, yb : (H, W)   integer labels
        """
        # Put features to (H, W, C)
        xa_hw, _cl = _ensure_hw_first(xa, channels_last)
        xb_hw, _   = _ensure_hw_first(xb, channels_last)
        H, W, C = xa_hw.shape
        assert xb_hw.shape == (H, W, C)
        assert ya.shape == (H, W) and yb.shape == (H, W)

        # Sample kmix per paper
        if kmix is None:
            kmix = int(torch.tensor([2,3,4,5,6])[torch.randint(0, 5, (1,))])

        # Compute vertical bin edges (equal-span inclination ranges)
        edges = torch.linspace(0, H, kmix + 1, dtype=torch.int64)
        xa_out = xa_hw.clone()
        ya_out = ya.clone()

        if bin_select == "random":
            take = torch.rand(kmix) < 0.5
        elif bin_select == "alternate":
            start = int(torch.randint(0, 2, (1,)))
            take = torch.zeros(kmix, dtype=torch.bool)
            take[start::2] = True
        else:
            raise ValueError("bin_select must be 'random' or 'alternate'.")

        for i in range(kmix):
            if take[i]:
                r0, r1 = int(edges[i]), int(edges[i+1])
                xa_out[r0:r1, :, :] = xb_hw[r0:r1, :, :]
                ya_out[r0:r1, :]    = yb[r0:r1, :]

        return _restore_layout(xa_out, channels_last), ya_out
