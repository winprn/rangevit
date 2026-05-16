import torch


SUPPORTED_ROBUST_EVAL_TYPES = {"none", "range_noise", "point_dropout", "beam_dropout"}


def apply_robust_eval(input_feature, input_mask, corruption_type, severity, seed):
    """Apply deterministic validation-time range-image corruption.

    This is a controlled sensitivity test on the normalized model input. It does
    not alter labels, projected depth, unprojection indices, or KNN geometry.
    """
    corruption_type = str(corruption_type).lower()
    if corruption_type == "none":
        return input_feature
    if corruption_type not in SUPPORTED_ROBUST_EVAL_TYPES:
        raise ValueError(
            f"Unsupported robust_eval type '{corruption_type}'. "
            f"Expected one of {sorted(SUPPORTED_ROBUST_EVAL_TYPES)}."
        )
    if input_feature.ndim != 4 or input_feature.shape[1] < 5:
        raise ValueError("input_feature must have shape [B, C>=5, H, W].")

    severity = float(severity)
    if severity < 0.0 or severity > 1.0:
        raise ValueError("robust_eval severity must be in [0, 1].")

    valid_mask = input_mask.bool()
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    elif valid_mask.ndim != 4:
        raise ValueError("input_mask must have shape [B, H, W] or [B, 1, H, W].")

    generator = torch.Generator(device=input_feature.device)
    generator.manual_seed(int(seed))
    corrupted = input_feature.clone()

    if corruption_type == "range_noise":
        noise = torch.randn(
            corrupted[:, :4].shape,
            device=corrupted.device,
            dtype=corrupted.dtype,
            generator=generator,
        ) * severity
        corrupted[:, :4] = torch.where(valid_mask, corrupted[:, :4] + noise, corrupted[:, :4])
        return corrupted

    if corruption_type == "point_dropout":
        drop_mask = (
            torch.rand(
                (corrupted.shape[0], 1, corrupted.shape[2], corrupted.shape[3]),
                device=corrupted.device,
                generator=generator,
            ) < severity
        ) & valid_mask
        corrupted = corrupted.masked_fill(drop_mask.expand_as(corrupted), 0.0)
        return corrupted

    if corruption_type == "beam_dropout":
        drop_rows = torch.rand(
            (corrupted.shape[0], 1, corrupted.shape[2], 1),
            device=corrupted.device,
            generator=generator,
        ) < severity
        drop_mask = drop_rows.expand(-1, 1, -1, corrupted.shape[3]) & valid_mask
        corrupted = corrupted.masked_fill(drop_mask.expand_as(corrupted), 0.0)
        return corrupted

    return input_feature
