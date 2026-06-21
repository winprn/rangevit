# TinyViM Decoder Family: FPN, FPN-Gated, and FPN-Gated-Detail

This document summarizes the three active TinyViM decoder variants used for comparison:

- `fpn`: original baseline decoder
- `fpn_gated`: V1 capacity-matched gated FPN decoder
- `fpn_gated_detail`: V2 gated FPN decoder with shallow detail reinjection

These variants are intended to answer two staged questions:

1. Does replacing hard final summation with learned gated fusion improve performance when decoder capacity remains similar?
2. After improving fusion, does adding a lightweight shallow-detail branch further improve thin structures and boundaries?

## Shared Input Features

All three decoders consume the same 4 TinyViM stage features from the current adapter path:

- `f0`: `[B, 48, 64, 2048]`
- `f1`: `[B, 96, 64, 1024]`
- `f2`: `[B, 192, 64, 512]`
- `f3`: `[B, 384, 64, 256]`

## Baseline: `fpn`

Implemented in [fpn_decoder.py](/E:/KLTN/RangeTinyVim/models/tinyvim/fpn_decoder.py) as `TinyViMFPNDecoder`.

Pipeline:

1. Lateral `1x1` projection of each stage to `256` channels
2. Top-down FPN addition
3. Per-level FPN convs
4. Per-level head convs to `128` channels
5. Upsample all heads to full resolution
6. Hard summation of all head features
7. Final `3x3` fuse conv
8. Dropout
9. `1x1` classifier

Final fusion:

```text
fused = h0 + h1 + h2 + h3
```

Parameter count:

- `3,089,556`

## V1: `fpn_gated`

Implemented in [fpn_decoder.py](/E:/KLTN/RangeTinyVim/models/tinyvim/fpn_decoder.py) as `TinyViMFPNGatedDecoder`.

V1 keeps the baseline FPN structure and width almost unchanged:

- same lateral convs
- same top-down pathway
- same FPN convs
- same head convs
- same full-resolution head upsampling
- same final `fuse_conv`
- same classifier

Only the final fusion rule changes.

Instead of hard summation, V1 does:

1. Concatenate full-resolution head features
2. Predict 4 spatial gate maps
3. Apply `softmax` across the 4 scales
4. Compute weighted sum of the head features

Fusion logic:

```text
cat = concat(h0, h1, h2, h3)
gates = softmax(gate(cat), dim=scale)
fused = g0*h0 + g1*h1 + g2*h2 + g3*h3
```

Parameter count:

- `3,155,864`

## V2: `fpn_gated_detail`

Implemented in [fpn_decoder.py](/E:/KLTN/RangeTinyVim/models/tinyvim/fpn_decoder.py) as `TinyViMFPNGatedDetailDecoder`.

V2 keeps the full V1 main branch unchanged and adds a lightweight shallow-detail branch.

### Main branch

Exactly the same as `fpn_gated`:

- lateral projections
- top-down FPN
- per-level FPN convs
- head convs
- head upsampling
- gated fusion

This produces:

- `fused_main`: `[B, 128, 64, 2048]`

### Detail branch

Uses shallow encoder features:

- `f0`: `[B, 48, 64, 2048]`
- `f1`: `[B, 96, 64, 1024]`

Processing:

1. `detail_proj0`: `48 -> 64` via `1x1`
2. `detail_proj1`: `96 -> 64` via `1x1`
3. Upsample projected `f1` to `f0` size
4. Fuse as `detail = d0 + up(d1)`
5. Apply residual detail refinement
6. Project detail from `64 -> 128`
7. Add it to the main fused feature

Final merge:

```text
x = fused_main + detail_to_main(detail)
x = fuse_conv(x)
x = dropout(x)
logits = classifier(x)
```

Parameter count:

- `3,182,680`

## Output Contract

All three decoders preserve the same interface:

- input: list of 4 TinyViM stage features
- output logits: `[B, n_classes, 64, 2048]`
- `return_features=True` returns decoder features before classification

That keeps the training loop, inference path, and KPConv compatibility unchanged.

## Ablation Order

Recommended comparison:

1. `fpn`
2. `fpn_gated`
3. `fpn_gated_detail`

Keep unchanged across all runs:

- backbone
- losses
- KNN
- augmentation
- training schedule

That isolates:

- V1 = better fusion
- V2 = better fusion + shallow detail reinjection

## Config Files

- Baseline: [config/kitti/main/config_tinyvim.yaml](/E:/KLTN/RangeTinyVim/config/kitti/main/config_tinyvim.yaml)
- V1: [config/kitti/ablation/decoder/config_fpn_gated.yaml](/E:/KLTN/RangeTinyVim/config/kitti/ablation/decoder/config_fpn_gated.yaml)
- V2: [config/kitti/ablation/decoder/config_fpn_gated_detail.yaml](/E:/KLTN/RangeTinyVim/config/kitti/ablation/decoder/config_fpn_gated_detail.yaml)
