# RangeViT-Fusion: Mid-term Progress Report

## Presentation Materials

**Duration:** 20-30 minutes
**Audience:** Thesis advisor (knows CV deeply, knows RangeViT, not familiar with HARP-NeXt)
**Purpose:** Mid-term thesis progress report

---

## Slide 1: Title

**Title:** RangeViT-Fusion: Integrating Bidirectional Point-Pixel Fusion into Vision Transformers for LiDAR Semantic Segmentation

**Subtitle:** Mid-term Progress Report

**Your name, Advisor name, Institution, Date**

---

## Slide 2: Motivation & Problem

**Slide Title:** The Gap in Range Image Methods

**Key Points:**
- Range image projection enables efficient 2D processing of LiDAR point clouds
- Current ViT-based methods (RangeViT) process only in 2D pixel space
- **Problem:** 3D geometric information is lost during projection
- Points that are spatially distant in 3D can become neighbors in 2D
- Post-hoc 3D refiners (KPConv) are expensive and disconnected from encoder

**Diagram - The Projection Problem:**

```mermaid
flowchart LR
    subgraph 3D["3D Point Cloud"]
        A["Point A<br/>(x=10, y=0, z=1)"]
        B["Point B<br/>(x=10, y=5, z=1)"]
        distance["5m apart in 3D"]
    end

    subgraph Proj["Spherical Projection"]
        arrow["→ Project to 2D →"]
    end

    subgraph 2D["Range Image"]
        C["Pixel (row=5, col=100)"]
        D["Pixel (row=5, col=101)"]
        adjacent["Adjacent pixels!"]
    end

    A --> arrow
    B --> arrow
    arrow --> C
    arrow --> D

    style distance fill:#ffcccc,stroke:#cc0000
    style adjacent fill:#ffcccc,stroke:#cc0000
```

**Diagram - Current Pipeline Limitation:**

```mermaid
flowchart LR
    PC[("3D Point<br/>Cloud")] --> Proj["Spherical<br/>Projection"]
    Proj --> RI["Range<br/>Image"]
    RI --> ViT["ViT Encoder<br/>(2D only)"]
    ViT --> Dec["Decoder"]
    Dec --> KP["KPConv<br/>3D Refiner"]
    KP --> Out[("Predictions")]

    style ViT fill:#ffeeee,stroke:#cc0000
    style KP fill:#fff3cd,stroke:#cc9900

    ViT -.- Note1["❌ No 3D awareness"]
    KP -.- Note2["⚠️ Expensive post-hoc fix"]
```

**Talking Points:**
> "While RangeViT achieves strong results by applying Vision Transformers to range images, there's a fundamental limitation: the 2D encoder operates entirely in pixel space and has no awareness of the underlying 3D geometry. Two points that are 10 meters apart in 3D could end up as neighboring pixels due to the spherical projection. Current solutions apply 3D refinement after the encoder, but this is computationally expensive and the encoder never benefits from 3D information during feature learning."

---

## Slide 3: RangeViT Recap

**Slide Title:** RangeViT: ViT for Range Image Segmentation (CVPR 2023)

**Key Points:**
- Adapts Vision Transformer for range image semantic segmentation
- ConvStem for patch embedding (better than linear projection)
- Pretrained on ImageNet, fine-tuned for LiDAR
- Optional KPConv post-processor for 3D refinement
- State-of-the-art on nuScenes and SemanticKITTI

**Diagram - RangeViT Architecture:**

```mermaid
flowchart LR
    subgraph Input
        PC[("Point<br/>Cloud")]
        RI["Range Image<br/>(H×W×5)"]
    end

    subgraph Encoder["ViT Encoder"]
        CS["ConvStem"]
        TB["Transformer<br/>Blocks ×12"]
    end

    subgraph Decoder
        DEC["Linear/Conv<br/>Decoder"]
    end

    subgraph Optional["Optional 3D Refiner"]
        KP["KPConv"]
    end

    PC --> RI
    RI --> CS
    CS --> TB
    TB --> DEC
    DEC --> KP
    KP --> Out[("Per-point<br/>Predictions")]

    style Optional fill:#f5f5f5,stroke:#999,stroke-dasharray: 5 5
```

**Talking Points:**
> "As you know, RangeViT projects LiDAR points to a range image and processes it with a Vision Transformer. The ConvStem handles patch embedding, 12 transformer blocks extract features, and a decoder produces pixel-wise predictions. These are then mapped back to points. Optionally, KPConv refines predictions in 3D. The key limitation I'm addressing: the ViT encoder has no knowledge of 3D structure during processing."

---

## Slide 4: HARP-NeXt Key Insight

**Slide Title:** HARP-NeXt: Bidirectional Point-Pixel Fusion

**Key Points:**
- Maintains **parallel branches**: pixel features AND point features
- **Continuous fusion** throughout the network, not just at the end
- At each fusion point:
  - Pixel → Point: map 2D features to 3D locations
  - Point → Pixel: aggregate 3D features back to 2D grid
- Result: Both branches benefit from each other's information
- No need for expensive post-hoc 3D refinement

**Diagram - Bidirectional Fusion Concept:**

```mermaid
flowchart TB
    subgraph Pixel["Pixel Branch (2D)"]
        P1["Pixel<br/>Features"] --> P2["Updated<br/>Pixel"] --> P3["Updated<br/>Pixel"] --> PF["Final<br/>Pixel"]
    end

    subgraph Point["Point Branch (3D)"]
        Q1["Point<br/>Features"] --> Q2["Updated<br/>Point"] --> Q3["Updated<br/>Point"] --> QF["Final<br/>Point"]
    end

    P1 -.->|"Px→Pt"| Q2
    Q1 -.->|"Pt→Px"| P2

    P2 -.->|"Px→Pt"| Q3
    Q2 -.->|"Pt→Px"| P3

    P3 -.->|"Px→Pt"| QF
    Q3 -.->|"Pt→Px"| PF

    style Pixel fill:#e3f2fd,stroke:#1976d2
    style Point fill:#fff3e0,stroke:#f57c00
```

**Diagram - Single Fusion Operation:**

```mermaid
flowchart LR
    subgraph Px2Pt["Pixel → Point"]
        PF1["Pixel Features<br/>(B,D,H,W)"]
        Map1["Gather at<br/>(y,x) coords"]
        PT1["Mapped Features<br/>(N,D)"]
        PF1 --> Map1 --> PT1
    end

    subgraph Fuse1["Point Fusion"]
        PT1 --> Cat1["Concat"]
        PointF["Point Features<br/>(N,D)"] --> Cat1
        Cat1 --> MLP1["MLP"]
        MLP1 --> NewPt["Updated Point<br/>(N,D)"]
    end

    subgraph Pt2Px["Point → Pixel"]
        NewPt --> Agg["Max-pool<br/>per pixel"]
        Agg --> Grid["Sparse Grid<br/>(B,D,H,W)"]
    end

    subgraph Fuse2["Pixel Fusion"]
        Grid --> Cat2["Concat"]
        PF1 --> Cat2
        Cat2 --> Conv["Conv 1×1"]
        Conv --> NewPx["Updated Pixel<br/>(B,D,H,W)"]
    end
```

**Talking Points:**
> "HARP-NeXt's key insight is maintaining two parallel branches throughout the network - one for 2D pixel features and one for 3D point features - with continuous bidirectional fusion. At each fusion point, pixel features are gathered at point locations and fused with point features. Then point features are aggregated back into a 2D grid via max-pooling and fused with pixel features. This creates a feedback loop where both branches continuously inform each other. The 2D branch gains 3D awareness, and the 3D branch benefits from the 2D encoder's powerful representations."

---

## Slide 5: Your Contribution

**Slide Title:** RangeViT-Fusion: Our Novel Integration

**Key Points:**
- **Core idea:** Replace HARP-NeXt's CNN pixel branch with Vision Transformer
- Inject bidirectional fusion directly INTO the ViT encoder
- Fusion at transformer blocks 4, 8, and 12 (early, mid, late)
- Remove KPConv entirely - fusion handles 3D awareness
- Keep ViT's pretrained weights, add lightweight fusion layers

**Diagram - What's New:**

```mermaid
flowchart LR
    subgraph Before["RangeViT (Before)"]
        B1["ViT Encoder<br/>(2D only)"] --> B2["Decoder"] --> B3["KPConv<br/>(3D)"]
    end

    subgraph After["RangeViT-Fusion (Ours)"]
        A1["ViT Encoder<br/>+ Fusion"] --> A2["Fusion<br/>Head"]
    end

    style B1 fill:#ffeeee,stroke:#cc0000
    style B3 fill:#fff3cd,stroke:#cc9900
    style A1 fill:#e8f5e9,stroke:#4caf50
    style A2 fill:#e8f5e9,stroke:#4caf50
```

**Table - Contribution Summary:**

| Aspect | RangeViT | HARP-NeXt | **Ours** |
|--------|----------|-----------|----------|
| Pixel Branch | ViT | ConvNeXt | **ViT** |
| Point Branch | None | MLP layers | **MLP layers** |
| Fusion | None (post-hoc) | Continuous | **Continuous** |
| 3D Refiner | KPConv | None | **None** |
| Pretrained | ImageNet ViT | ImageNet CNN | **ImageNet ViT** |

**Talking Points:**
> "Our contribution is combining the best of both worlds. We take RangeViT's powerful ViT encoder with ImageNet pretraining, and inject HARP-NeXt's bidirectional fusion mechanism directly into it. Instead of processing purely in 2D and fixing it later with KPConv, our encoder maintains 3D awareness throughout. We add fusion operations at blocks 4, 8, and 12 - giving the model opportunities to exchange information at early, middle, and late stages. This lets us remove KPConv entirely while potentially achieving better results."

---

## Slide 6: Architecture Overview

**Slide Title:** RangeViT-Fusion Architecture

**Key Points:**
- **Input:** Range image (B, 5, H, W) + Point attributes (N, 5)
- **FeaturesEncoder:** MLP encodes point attributes to D-dimensional features
- **VisionTransformerFusion:** ViT with fusion at blocks 4, 8, 12
- **FusionHead:** Combines final pixel & point features for prediction
- **Losses:** Focal + Lovász on points, auxiliary CE on pixels

**Diagram - Full Architecture:**

```mermaid
flowchart TB
    subgraph Input["Input"]
        RI["Range Image<br/>(B, 5, H, W)"]
        PA["Point Attributes<br/>(N, 5)<br/>xyz, intensity, range"]
        CO["Point Coords<br/>(N, 3)<br/>batch, y, x"]
    end

    subgraph FE["Features Encoder"]
        PA --> FE1["Linear(5→64)"]
        FE1 --> FE2["BN + ReLU"]
        FE2 --> FE3["Linear(64→128)"]
        FE3 --> FE4["BN + ReLU"]
        FE4 --> FE5["Linear(128→D)"]
        FE5 --> PtFeat["Point Features<br/>(N, D)"]
    end

    subgraph ViT["Vision Transformer with Fusion"]
        RI --> CS["ConvStem"]
        CS --> PE["+ Pos Embed"]
        PE --> B1["Blocks 1-3"]
        B1 --> B4["Block 4"]
        B4 --> F1["⟷ Fusion 1"]
        F1 --> B5["Blocks 5-7"]
        B5 --> B8["Block 8"]
        B8 --> F2["⟷ Fusion 2"]
        F2 --> B9["Blocks 9-11"]
        B9 --> B12["Block 12"]
        B12 --> F3["⟷ Fusion 3"]
        F3 --> PxFeat["Pixel Features<br/>(B, D, H', W')"]
    end

    PtFeat -.->|"fusion"| F1
    PtFeat -.->|"fusion"| F2
    PtFeat -.->|"fusion"| F3

    subgraph Head["Fusion Head"]
        PxFeat --> P2P["pixel2point"]
        P2P --> MPF["Mapped Pixel<br/>(N, D)"]
        MPF --> CAT["Concat"]
        PtFeat2["Final Point<br/>(N, D)"] --> CAT
        CAT --> MLP["MLP<br/>(2D→D→N_cls)"]
        MLP --> OUT["Predictions<br/>(N, N_cls)"]
    end

    F3 --> PtFeat2

    style F1 fill:#fff3e0,stroke:#f57c00
    style F2 fill:#fff3e0,stroke:#f57c00
    style F3 fill:#fff3e0,stroke:#f57c00
```

**Talking Points:**
> "Here's the complete architecture. Raw point attributes - xyz coordinates, intensity, and range - go through a FeaturesEncoder MLP to produce D-dimensional point features. The range image goes through our modified ViT. At blocks 4, 8, and 12, we pause the transformer and perform bidirectional fusion with the point branch. After the final fusion, we map the pixel features to point locations, concatenate with the final point features, and pass through the FusionHead MLP to get per-point predictions. We supervise with Focal and Lovász losses on points, plus auxiliary cross-entropy on pixels at each fusion point."

---

## Slide 7: Fusion Mechanism Detail

**Slide Title:** Bidirectional Fusion at Each Block

**Key Points:**
- **EfficientTransformationPipeline:** Handles all coordinate mappings
- **pixel2point:** Gather pixel features at point coordinates
- **point2cluster:** Max-pool points into voxel grid
- **cluster2pixel:** Scatter voxel features to dense pixel grid
- **Fusion layers:** Lightweight concat + linear/conv

**Diagram - Detailed Fusion Operation:**

```mermaid
flowchart TB
    subgraph Step1["Step 1: Pixel → Point"]
        PX["Pixel Features<br/>(B, D, H, W)"]
        COORD["Point Coords<br/>(N, 3)"]
        PX --> |"gather at (b,y,x)"| MPX["Mapped Pixel<br/>(N, D)"]
        COORD --> MPX
    end

    subgraph Step2["Step 2: Point Fusion"]
        MPX --> CONCAT1["Concat"]
        PT["Point Features<br/>(N, D)"] --> CONCAT1
        CONCAT1 --> |"(N, 2D)"| LIN1["Linear(2D→D)<br/>+ BN + ReLU"]
        LIN1 --> PT_NEW["Updated Point<br/>(N, D)"]
    end

    subgraph Step3["Step 3: Point → Cluster"]
        PT_NEW --> VOXEL["Group by<br/>(b, y//s, x//s)"]
        VOXEL --> MAXPOOL["Max Pool<br/>per voxel"]
        MAXPOOL --> CLUSTER["Cluster Features<br/>(M, D)"]
    end

    subgraph Step4["Step 4: Cluster → Pixel"]
        CLUSTER --> SCATTER["Scatter to<br/>dense grid"]
        SCATTER --> PX_FROM_PT["Pixel from Points<br/>(B, D, H, W)"]
    end

    subgraph Step5["Step 5: Pixel Fusion"]
        PX_FROM_PT --> CONCAT2["Concat"]
        PX --> CONCAT2
        CONCAT2 --> |"(B, 2D, H, W)"| CONV["Conv1×1(2D→D)<br/>+ BN + Hardswish"]
        CONV --> PX_NEW["Updated Pixel<br/>(B, D, H, W)"]
    end

    subgraph Aux["Auxiliary Output"]
        PX_NEW --> AUXHEAD["Conv1×1(D→C)"]
        AUXHEAD --> AUXOUT["Aux Logits<br/>(B, C, H, W)"]
    end

    style Step1 fill:#e3f2fd,stroke:#1976d2
    style Step2 fill:#fff3e0,stroke:#f57c00
    style Step3 fill:#fff3e0,stroke:#f57c00
    style Step4 fill:#e3f2fd,stroke:#1976d2
    style Step5 fill:#e3f2fd,stroke:#1976d2
```

**Talking Points:**
> "Let me walk through one fusion operation in detail. First, we gather pixel features at each point's projected coordinates - this is a simple indexing operation. Then we concatenate these with point features and pass through an MLP - this is the point fusion. Next, we need to go back to pixel space. We group points by their pixel locations and max-pool within each group to get cluster features. These sparse clusters are scattered to a dense grid. Finally, we concatenate this with the original pixel features and apply a 1x1 conv - the pixel fusion. We also output auxiliary logits for intermediate supervision. This entire operation adds minimal compute but creates rich bidirectional information flow."

---

## Slide 8: Implementation Details

**Slide Title:** Technical Choices & Configuration

**Key Points:**
- **Backbone:** ViT-Small (22M params) or ViT-Base (86M params)
- **Fusion blocks:** [4, 8, 12] - early, mid, late fusion
- **Point encoding:** 5 → 64 → 128 → D (same as HARP-NeXt)
- **Loss weights:** Focal(γ=2) + Lovász + 0.4 × Auxiliary CE
- **Dynamic grid:** Handles variable image sizes (train windows vs full validation)

**Table - Model Configurations:**

| Config | Backbone | d_model | Params | Dataset |
|--------|----------|---------|--------|---------|
| Small | ViT-S/16 | 384 | ~25M | nuScenes |
| Base | ViT-B/16 | 768 | ~100M | SemanticKITTI |

**Table - Key Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Fusion blocks | [4, 8, 12] | Early/mid/late feature exchange |
| Aux loss weight | 0.4 | Balance point vs pixel supervision |
| Focal gamma | 2.0 | Handle class imbalance |
| Patch stride | [2, 8] | Match RangeViT settings |
| Learning rate | 4e-4 | Standard for ViT fine-tuning |

**Diagram - Loss Computation:**

```mermaid
flowchart LR
    subgraph PointLoss["Point-Level Loss"]
        PL["Point Logits<br/>(N, C)"]
        GT["Ground Truth<br/>(N,)"]
        PL --> FOCAL["Focal Loss<br/>γ=2, α=0.25"]
        PL --> LOVASZ["Lovász Loss"]
        GT --> FOCAL
        GT --> LOVASZ
        FOCAL --> PSUM["Point Loss"]
        LOVASZ --> PSUM
    end

    subgraph AuxLoss["Auxiliary Pixel Loss"]
        AUX1["Aux Logits 1"]
        AUX2["Aux Logits 2"]
        AUX3["Aux Logits 3"]
        PSEUDO["Pseudo Labels<br/>(from points)"]
        AUX1 --> CE1["CE"]
        AUX2 --> CE2["CE"]
        AUX3 --> CE3["CE"]
        PSEUDO --> CE1
        PSEUDO --> CE2
        PSEUDO --> CE3
        CE1 --> ASUM["× 0.4"]
        CE2 --> ASUM
        CE3 --> ASUM
    end

    PSUM --> TOTAL["Total Loss"]
    ASUM --> TOTAL
```

**Talking Points:**
> "For implementation, we support both ViT-Small for faster experiments and ViT-Base for maximum performance. Fusion happens at blocks 4, 8, and 12 - we found this spacing gives good coverage of early, mid, and late features. For losses, we use Focal loss to handle the severe class imbalance in LiDAR data, plus Lovász loss which directly optimizes IoU. Auxiliary losses at each fusion point provide intermediate supervision. One technical detail: we compute grid sizes dynamically from input dimensions, so the same model handles both training windows and full validation images without issues."

---

## Slide 9: Current Progress

**Slide Title:** Implementation Status

**Key Points:**
- ✅ Full architecture implemented and tested
- ✅ FeaturesEncoder, VisionTransformerFusion, FusionHead
- ✅ EfficientTransformationPipeline for coordinate mappings
- ✅ Focal + Lovász + Auxiliary losses
- ✅ Config files for nuScenes and SemanticKITTI
- ✅ Integration with existing training pipeline
- 🔄 Currently training on SemanticKITTI

**Diagram - Implementation Components:**

```mermaid
flowchart TB
    subgraph Complete["✅ Completed"]
        FE["features_encoder.py<br/>Point attribute MLP"]
        VF["vit_fusion.py<br/>ViT with fusion hooks"]
        FM["fusion_modules.py<br/>ETP, fusion layers"]
        FH["fusion_head.py<br/>Final prediction head"]
        RF["rangevit_fusion.py<br/>Main model class"]
        CFG["config_fusion_*.yaml<br/>Configurations"]
        TR["train.py updates<br/>Fusion training loop"]
        DL["dataloader updates<br/>Point data loading"]
    end

    subgraph InProgress["🔄 In Progress"]
        TRAIN["Model Training<br/>SemanticKITTI"]
    end

    subgraph Todo["📋 To Do"]
        EVAL["Full Evaluation"]
        NUSC["nuScenes Training"]
        ABL["Ablation Studies"]
    end

    Complete --> InProgress --> Todo
```

**Table - Files Created/Modified:**

| File | Status | Description |
|------|--------|-------------|
| `models/features_encoder.py` | ✅ New | Point attribute encoder |
| `models/vit_fusion.py` | ✅ New | ViT with fusion |
| `models/fusion_modules.py` | ✅ New | ETP, fusion layers |
| `models/fusion_head.py` | ✅ New | Prediction head |
| `models/rangevit_fusion.py` | ✅ New | Main model |
| `config_fusion_nusc.yaml` | ✅ New | nuScenes config |
| `config_fusion_kitti.yaml` | ✅ New | KITTI config |
| `train.py` | ✅ Modified | Fusion training loop |
| `dataset/range_view_loader.py` | ✅ Modified | Point data loading |

**Talking Points:**
> "On the implementation side, we've completed all core components. The FeaturesEncoder handles point attributes, VisionTransformerFusion implements the modified ViT with fusion hooks, and the FusionHead produces final predictions. We've created configs for both datasets and integrated everything with the existing training pipeline. The model is currently training on SemanticKITTI with ViT-Base backbone."

---

## Slide 10: Preliminary Results

**Slide Title:** Training Progress & Early Observations

**Key Points:**
- Training currently running on SemanticKITTI
- Monitoring: loss curves, mIoU, per-class IoU
- Parameter count verified: ~25M (Small) / ~100M (Base)
- No convergence issues observed
- [Add actual numbers when available]

**Diagram - Expected Metrics to Report:**

```mermaid
flowchart LR
    subgraph Metrics["Metrics Being Tracked"]
        LOSS["Training Loss<br/>↓ Decreasing"]
        MIOU["Validation mIoU<br/>↑ Increasing"]
        PCLS["Per-class IoU<br/>Focus on rare classes"]
    end

    subgraph Compare["Comparison Points"]
        RV["RangeViT<br/>(baseline)"]
        OURS["Ours<br/>(RangeViT-Fusion)"]
        HN["HARP-NeXt<br/>(reference)"]
    end

    Metrics --> Compare
```

**Placeholder for Actual Results:**

| Model | mIoU (val) | car | bicycle | person | road | building |
|-------|------------|-----|---------|--------|------|----------|
| RangeViT (paper) | 64.0 | - | - | - | - | - |
| HARP-NeXt (paper) | 67.5 | - | - | - | - | - |
| **Ours (current)** | TBD | TBD | TBD | TBD | TBD | TBD |

**Talking Points:**
> "The model is currently training, so I'll show the latest metrics. [Update with actual numbers]. We're tracking overall mIoU as well as per-class performance, particularly for challenging classes like bicycles and pedestrians where 3D context should help most. Our hypothesis is that continuous fusion will particularly improve these small, sparse object classes that suffer most from the 2D projection ambiguity."

---

## Slide 11: Next Steps

**Slide Title:** Remaining Work

**Key Points:**
- Complete SemanticKITTI training and evaluation
- Run nuScenes experiments
- Ablation studies:
  - Number of fusion points (1 vs 2 vs 3)
  - Fusion block positions
  - With/without auxiliary losses
- Compare against RangeViT and HARP-NeXt baselines
- Analyze per-class improvements
- Write thesis chapter on methodology and results

**Diagram - Remaining Experiments:**

```mermaid
flowchart TB
    subgraph Main["Main Experiments"]
        KITTI["SemanticKITTI<br/>Full training"]
        NUSC["nuScenes<br/>Full training"]
    end

    subgraph Ablation["Ablation Studies"]
        ABL1["Fusion points:<br/>[12] vs [4,12] vs [4,8,12]"]
        ABL2["Block positions:<br/>[4,8,12] vs [3,6,9,12]"]
        ABL3["Aux loss:<br/>with vs without"]
        ABL4["Backbone:<br/>ViT-S vs ViT-B"]
    end

    subgraph Analysis["Analysis"]
        CLS["Per-class breakdown"]
        VIS["Qualitative visualization"]
        SPEED["Inference speed"]
    end

    Main --> Ablation --> Analysis
```

**Table - Proposed Ablation Studies:**

| Experiment | Variants | Purpose |
|------------|----------|---------|
| Fusion frequency | 1, 2, 3 fusion points | Find optimal fusion density |
| Fusion positions | Early vs late vs distributed | Where does fusion help most? |
| Auxiliary loss | 0, 0.2, 0.4, 0.6 weight | Importance of pixel supervision |
| Backbone size | ViT-S, ViT-B | Accuracy vs efficiency tradeoff |

**Talking Points:**
> "Moving forward, I'll complete the SemanticKITTI experiments and run the same setup on nuScenes. Then I'll conduct ablation studies to understand which design choices matter most - how many fusion points we need, where they should be placed, and how important the auxiliary pixel losses are. I'll also compare inference speed against KPConv-based approaches to quantify the efficiency gains. Finally, I'll do per-class analysis to verify our hypothesis that fusion particularly helps small object classes."

---

## Summary: Key Takeaways

1. **Problem:** RangeViT's 2D-only processing loses 3D geometric context
2. **Solution:** Integrate HARP-NeXt's bidirectional fusion INTO the ViT encoder
3. **Innovation:** First work combining ViT backbone with continuous point-pixel fusion
4. **Progress:** Full implementation complete, currently training
5. **Expected outcome:** Better accuracy without expensive KPConv post-processing
