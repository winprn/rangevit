The **first improvement** you should do is:

## Replace the plain FPN fusion with a **better decoder fusion block**

More specifically:

**keep TinyViM-base unchanged**,
**keep KNN unchanged for now**,
and **upgrade only the decoder** first.

That should be your first step.

### Why this should be first

Because right now the clearest weakness in your pipeline is not the encoder. Your encoder already gives you:

* multi-scale features
* local modeling from `LocalBlock`
* long-range modeling from `TViMBlock / SS2D`
* anisotropic width-preserving semantics that fit range images reasonably well

But the decoder is still relatively simple: project, upsample, sum, classify. That means the rich encoder features are probably **not being fused as intelligently as they could be**. Your current FPN is likely the biggest bottleneck between “good encoder features” and “good final segmentation.”

So the best first move is the one that gives the highest chance of gain while keeping the experiment clean.

---

## The exact first improvement I recommend

Do this first:

### **FPN v1: replace plain summation with learned fusion**

Do **not** redesign everything yet.
Do **not** add boundary head yet.
Do **not** add point refinement yet.

Just change:

```text
upsample all scales -> sum
```

into:

```text
upsample all scales -> concatenate or gated-weight fusion -> fuse conv
```

### Minimal version

For each scale:

* project to same channels, e.g. 128
* upsample to `64 x 2048`

Then instead of:

```text
fused = u0 + u1 + u2 + u3
```

use:

```text
fused = Conv3x3(Concat[u0, u1, u2, u3])
```

or slightly better:

```text
gates = sigmoid(Conv1x1(Concat[u0, u1, u2, u3]))
fused = g0*u0 + g1*u1 + g2*u2 + g3*u3
```

Then:

```text
fused -> conv -> dropout -> classifier
```

---

## Why not other improvements first?

### Not backbone first

Because if you change TinyViM now, you will not know whether gains came from:

* stronger encoder
* or fixing poor fusion

Also the encoder already seems decent.

### Not KNN first

Because if the 2D logits are still suboptimal, improving KNN first is like polishing the output of a weak decoder. Better to improve the 2D representation first.

### Not boundary loss first

Boundary supervision can help, but if your multi-scale fusion is weak, the boundary signal still sits on top of a mediocre fused feature.

### Not detail branch first

A detail branch is a very good second step, but it is best added **after** you improve the main fusion. Otherwise you do not know whether the issue is poor global fusion or missing shallow detail.

---

## So the first experiment should be:

### **Experiment 1**

**Baseline:** TinyViM-base + current FPN + KNN

### **Experiment 2**

**TinyViM-base + improved fusion decoder + KNN**

Where “improved fusion decoder” means:

* same 4 encoder inputs
* same per-scale projection
* same upsampling to full resolution
* but replace **sum fusion** with **learned fusion**

That is the cleanest and strongest first step.

---

## My recommendation for the simplest good version

Start with this, because it is easy and strong enough to test the idea:

### Decoder change

1. project each scale to 128 channels
2. upsample all to `64 x 2048`
3. concatenate all 4 maps
4. apply:

   * `1x1 conv` to reduce channels
   * `3x3 conv`
   * classifier

So:

```text
u0, u1, u2, u3  # each [B,128,64,2048]

cat = Concat([u0,u1,u2,u3])   # [B,512,64,2048]
fused = Conv1x1(cat)          # [B,128,64,2048]
fused = Conv3x3(fused)        # [B,128,64,2048]
logits = Conv1x1(fused)       # [B,20,64,2048]
```

This is already better than naive summation because the network can learn **how to mix scales** instead of assuming all scales should be equally added.

---

## If you want the slightly better first version

Use **gated fusion**:

```text
cat = Concat([u0,u1,u2,u3])
gates = Conv1x1(cat) -> [B,4,64,2048]
gates = softmax over scale dimension

fused = g0*u0 + g1*u1 + g2*u2 + g3*u3
fused = Conv3x3(fused)
logits = Conv1x1(fused)
```

This is probably the best “first upgrade” if you are comfortable implementing it.

---

## In one sentence

Your first improvement should be:

**Keep the current TinyViM encoder and KNN unchanged, and first replace the plain FPN sum fusion with a learned multi-scale fusion decoder.**

After that, the next step should probably be a **detail branch**.

If you want, I can now write the **exact decoder architecture you should implement first** in PyTorch-style module form.

Here is the **exact first decoder upgrade** I recommend you implement first.

## Goal

Replace the current plain FPN final fusion:

```text
upsample all levels -> sum -> fuse conv -> classifier
```

with:

```text
upsample all levels -> learned fusion -> refine -> classifier
```

This keeps your experiment clean:

* **encoder unchanged**
* **training setup unchanged**
* **KNN unchanged**
* only **decoder fusion** changes

That makes it the best first ablation.

---

# 1. Proposed first decoder: `TinyViMFusionDecoderV1`

## Input

Same 4 encoder outputs as now: 

* `f0: [B, 48, 64, 2048]`
* `f1: [B, 96, 64, 1024]`
* `f2: [B, 192, 64, 512]`
* `f3: [B, 384, 64, 256]`

## Output

* fused feature: `[B, 128, 64, 2048]`
* logits: `[B, num_classes, 64, 2048]`

---

# 2. What changes from your current FPN

Your current decoder already does:

* per-level lateral projection
* top-down fusion
* upsample all to full size
* sum them
* classifier 

The part I want you to change first is only the **final multi-scale fusion rule**.

Instead of:

```python
fused = h0 + h1 + h2 + h3
```

use:

```python
cat = torch.cat([h0, h1, h2, h3], dim=1)
fused = learned_mix(cat)
```

That alone is already a meaningful upgrade.

---

# 3. Recommended architecture

## Version A: simplest strong baseline

This is the one I recommend you build first.

### Step-by-step

For each scale:

1. `1x1 conv` project to 128 channels
2. upsample to `(64, 2048)`

Then:
3. concatenate all 4 features
4. `1x1 conv` reduce channels
5. `3x3 conv` refine
6. dropout
7. `1x1 conv` classifier

---

## Structure diagram

```text
f0 [B,48,64,2048]  -> proj0 -> u0 [B,128,64,2048]
f1 [B,96,64,1024]  -> proj1 -> up -> u1 [B,128,64,2048]
f2 [B,192,64,512]  -> proj2 -> up -> u2 [B,128,64,2048]
f3 [B,384,64,256]  -> proj3 -> up -> u3 [B,128,64,2048]

cat = concat(u0,u1,u2,u3)     # [B,512,64,2048]

fused = Conv1x1(512 -> 128)
fused = BN + GELU
fused = Conv3x3(128 -> 128, padding=1)
fused = BN + GELU
fused = Dropout2d
logits = Conv1x1(128 -> num_classes)
```

This is the best “first implementation” because it is:

* simple
* clean
* easy to debug
* clearly stronger than plain sum fusion

---

# 4. PyTorch-style implementation skeleton

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TinyViMFusionDecoderV1(nn.Module):
    """
    First recommended upgrade:
    - keep encoder unchanged
    - keep KNN unchanged
    - replace plain sum fusion with learned concat fusion
    """

    def __init__(
        self,
        in_channels=(48, 96, 192, 384),
        inner_channels=128,
        num_classes=20,
        dropout=0.1,
        align_corners=False,
    ):
        super().__init__()
        self.align_corners = align_corners

        # Per-scale projections
        self.proj0 = ConvBNAct(in_channels[0], inner_channels, kernel_size=1, padding=0)
        self.proj1 = ConvBNAct(in_channels[1], inner_channels, kernel_size=1, padding=0)
        self.proj2 = ConvBNAct(in_channels[2], inner_channels, kernel_size=1, padding=0)
        self.proj3 = ConvBNAct(in_channels[3], inner_channels, kernel_size=1, padding=0)

        # Learned fusion after concatenation
        self.fuse = nn.Sequential(
            ConvBNAct(inner_channels * 4, inner_channels, kernel_size=1, padding=0),
            ConvBNAct(inner_channels, inner_channels, kernel_size=3, padding=1),
        )

        self.dropout = nn.Dropout2d(dropout)
        self.cls = nn.Conv2d(inner_channels, num_classes, kernel_size=1)

    def forward(self, features):
        """
        features:
            f0: [B,48,64,2048]
            f1: [B,96,64,1024]
            f2: [B,192,64,512]
            f3: [B,384,64,256]
        """
        f0, f1, f2, f3 = features

        target_size = f0.shape[-2:]  # (64, 2048)

        u0 = self.proj0(f0)
        u1 = F.interpolate(self.proj1(f1), size=target_size, mode="bilinear", align_corners=self.align_corners)
        u2 = F.interpolate(self.proj2(f2), size=target_size, mode="bilinear", align_corners=self.align_corners)
        u3 = F.interpolate(self.proj3(f3), size=target_size, mode="bilinear", align_corners=self.align_corners)

        x = torch.cat([u0, u1, u2, u3], dim=1)   # [B, 512, 64, 2048]
        x = self.fuse(x)                         # [B, 128, 64, 2048]
        x = self.dropout(x)
        logits = self.cls(x)                    # [B, num_classes, 64, 2048]

        return logits
```

---

# 5. Why this is a good first step

This decoder is better than simple summation for one main reason:

## Plain sum assumes all scales contribute equally

But in your model:

* shallow levels carry more detail
* deep levels carry more context
* different image regions need different scales
* TinyViM features are not uniform CNN features; they already contain different local/global characteristics across stages 

So letting the network **learn scale mixing after concatenation** is immediately more expressive than hard-coded addition.

---

# 6. If you want a slightly better first version

After Version A works, the next small upgrade is:

## Version B: gated fusion

Instead of:

```python
x = torch.cat([u0, u1, u2, u3], dim=1)
x = self.fuse(x)
```

predict per-scale weights.

### Idea

```text
cat -> Conv1x1 -> 4-channel gate map -> softmax over 4 scales
fused = g0*u0 + g1*u1 + g2*u2 + g3*u3
```

That lets the decoder choose:

* shallow features for thin objects
* deeper features for road/building context
* mixed features for ambiguous regions

---

## PyTorch skeleton for gated version

```python
class TinyViMFusionDecoderV2(nn.Module):
    def __init__(
        self,
        in_channels=(48, 96, 192, 384),
        inner_channels=128,
        num_classes=20,
        dropout=0.1,
        align_corners=False,
    ):
        super().__init__()
        self.align_corners = align_corners

        self.proj0 = ConvBNAct(in_channels[0], inner_channels, kernel_size=1, padding=0)
        self.proj1 = ConvBNAct(in_channels[1], inner_channels, kernel_size=1, padding=0)
        self.proj2 = ConvBNAct(in_channels[2], inner_channels, kernel_size=1, padding=0)
        self.proj3 = ConvBNAct(in_channels[3], inner_channels, kernel_size=1, padding=0)

        self.gate = nn.Sequential(
            ConvBNAct(inner_channels * 4, inner_channels, kernel_size=1, padding=0),
            nn.Conv2d(inner_channels, 4, kernel_size=1, bias=True),
        )

        self.refine = nn.Sequential(
            ConvBNAct(inner_channels, inner_channels, kernel_size=3, padding=1),
            ConvBNAct(inner_channels, inner_channels, kernel_size=3, padding=1),
        )

        self.dropout = nn.Dropout2d(dropout)
        self.cls = nn.Conv2d(inner_channels, num_classes, kernel_size=1)

    def forward(self, features):
        f0, f1, f2, f3 = features
        target_size = f0.shape[-2:]

        u0 = self.proj0(f0)
        u1 = F.interpolate(self.proj1(f1), size=target_size, mode="bilinear", align_corners=self.align_corners)
        u2 = F.interpolate(self.proj2(f2), size=target_size, mode="bilinear", align_corners=self.align_corners)
        u3 = F.interpolate(self.proj3(f3), size=target_size, mode="bilinear", align_corners=self.align_corners)

        cat = torch.cat([u0, u1, u2, u3], dim=1)
        gates = self.gate(cat)                  # [B,4,H,W]
        gates = torch.softmax(gates, dim=1)

        x = (
            gates[:, 0:1] * u0 +
            gates[:, 1:2] * u1 +
            gates[:, 2:3] * u2 +
            gates[:, 3:4] * u3
        )

        x = self.refine(x)
        x = self.dropout(x)
        logits = self.cls(x)
        return logits
```

---

# 7. Which version should you implement first?

My recommendation:

## First implement **V1**, not V2

Because V1 is:

* easier
* less likely to introduce bugs
* already a meaningful architectural improvement
* easier to compare against your current FPN

Then if V1 helps, move to V2.

So the correct ablation order is:

1. current FPN baseline
2. **FusionDecoderV1**
3. FusionDecoderV2 gated
4. add detail branch
5. add boundary branch
6. improve KNN or add point refinement

---

# 8. Important implementation advice for your setup

## Keep these unchanged in the first experiment

Do not change:

* encoder
* losses
* KNN settings
* augmentation
* training schedule

Otherwise you will not know whether the gain came from decoder fusion or something else.

---

## Keep channel width moderate

Use:

* `inner_channels = 128`

Do **not** jump to 256 first.
Because your full-resolution map is `64 x 2048`, so memory grows quickly.

---

## Keep bilinear upsampling

Use bilinear upsampling first. It is simple and stable.

---

## Keep one final prediction head only

Do not add aux heads in this first experiment.

---

# 9. What result would count as success?

For this first decoder improvement, success means any of these:

* better overall mIoU
* better small-object classes
* sharper boundaries
* less over-smoothing than plain FPN
* same or slightly better performance at modest extra cost

Even a moderate gain is meaningful here, because this is still only the first clean decoder upgrade.

---

# 10. My direct recommendation

If you want the exact first thing to code:

## Implement `TinyViMFusionDecoderV1`

with:

* 4 per-scale `1x1` projections to 128 channels
* upsample all to full resolution
* concatenate
* `1x1 -> 3x3 -> dropout -> classifier`

That is the best first improvement.