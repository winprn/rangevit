class EncoderInterface:
    """
    Minimal contract for RangeViT/RangeSeg encoders.

    Expected attributes:
      - d_model
      - patch_size
      - patch_stride

    Expected forward:
      forward(im, return_features=True) -> (tokens, skip_or_features)
    """

    pass
