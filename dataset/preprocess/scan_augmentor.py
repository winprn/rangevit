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
        xa: torch.Tensor, ya: torch.Tensor,
        xb: torch.Tensor, yb: torch.Tensor,
        *,
        kmix: Optional[int] = None,
        bin_select: str = "random",      # {"random","alternate"}
        channels_last: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RangeMix: split the vertical (inclination) axis into k equal bins and
        replace the corresponding rows in (xa, ya) with those from (xb, yb).

        - kmix: if None, sample from {2,3,4,5,6}
        - bin_select:
            "random"    -> independently choose each bin with p=0.5 to replace
            "alternate" -> replace every other bin starting from a random offset

        Shapes:
        xa, xb : (H, W, C) if channels_last else (C, H, W)
        ya, yb : (H, W)    (any dtype); will be row-swapped to stay consistent.

        Notes:
        * Operates IN-PLACE on xa/ya, so passing views like proj_tensor[1:4] is fine.
        * Returns (xa, ya) for convenience, but caller may ignore.
        """
        # Determine layout and sizes
        if channels_last:
            H, W, C = xa.shape
            assert xb.shape == (H, W, C), f"xb shape {xb.shape} != {(H, W, C)}"
            vdim = 0  # vertical dimension (rows)
        else:
            C, H, W = xa.shape
            assert xb.shape == (C, H, W), f"xb shape {xb.shape} != {(C, H, W)}"
            vdim = 1

        assert ya.shape == (H, W), f"ya shape {ya.shape} != {(H, W)}"
        assert yb.shape == (H, W), f"yb shape {yb.shape} != {(H, W)}"

        dev = xa.device

        # Sample kmix per paper
        if kmix is None:
            kmix = [2, 3, 4, 5, 6][int(torch.randint(0, 5, (1,), device=dev))]

        # Compute vertical bin edges
        # (Use CPU int64 for indexing, values are small; that’s OK even if tensors are on GPU)
        edges = torch.linspace(0, H, kmix + 1, dtype=torch.int64)

        # Choose which bins to take from xb/yb
        if bin_select == "random":
            take = (torch.rand(kmix, device=dev) < 0.5)
        elif bin_select == "alternate":
            start = int(torch.randint(0, 2, (1,), device=dev))
            take = torch.zeros(kmix, dtype=torch.bool, device=dev)
            take[start::2] = True
        else:
            raise ValueError("bin_select must be 'random' or 'alternate'.")

        # Ensure at least one bin is swapped
        if not bool(take.any()):
            take[int(torch.randint(0, kmix, (1,), device=dev))] = True

        # Do the row-wise swaps IN-PLACE
        for i in range(kmix):
            if take[i]:
                r0 = int(edges[i].item())
                r1 = int(edges[i + 1].item())

                if channels_last:
                    xa[r0:r1, :, :] = xb[r0:r1, :, :]
                else:
                    xa[:, r0:r1, :] = xb[:, r0:r1, :]

                ya[r0:r1, :] = yb[r0:r1, :]

        return xa, ya
