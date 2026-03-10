# CuVision-Engine — Segmentation Module

> **Attention U-Net** implemented natively in CUDA / cuDNN / cuBLAS.  
> Pixel-wise semantic segmentation — no high-level frameworks, pure GPU primitives.

---

## Table of Contents

1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
   - [Encoder — ResNet-style Backbone](#encoder--resnet-style-backbone)
   - [Bottleneck — Dilated ASPP](#bottleneck--dilated-aspp)
   - [Decoder — U-Net with Attention Gates](#decoder--u-net-with-attention-gates)
   - [Output Head](#output-head)
3. [Attention Gate — Additive Attention](#attention-gate--additive-attention)
4. [Loss Functions](#loss-functions)
5. [Data Augmentation](#data-augmentation)
6. [Training Pipeline](#training-pipeline)
7. [File Structure](#file-structure)
8. [Getting Started](#getting-started)
9. [Reference Papers](#reference-papers)

---

## Overview

This module implements **Attention U-Net** — a state-of-the-art semantic segmentation network combining:

| Component | Design Choice | Justification |
|:---|:---|:---|
| **Encoder** | ResNet-like 4-stage backbone | Hierarchical features; residual blocks prevent vanishing gradients |
| **Bottleneck** | Dilated convolutions (ASPP-style) | Expands receptive field without resolution loss |
| **Decoder** | Transposed-conv upsample + skip connections | Recovers spatial detail lost during downsampling |
| **Skip Attention** | Additive Attention Gates on each skip | Suppresses irrelevant encoder activations before merging |
| **Cls Loss** | Pixel cross-entropy + Dice loss | CE handles per-pixel accuracy; Dice handles class imbalance |
| **Augmentation** | Elastic deformation + paired flip/jitter | Critical for generalisation on medical/natural image tasks |
| **Optimizer** | Momentum-SGD + polynomial LR decay | Smooth convergence; checkpoints every 5 epochs |

---

## Network Architecture

### Full Forward Pass — ASCII Diagram

```
Input Image [B, 3, 256, 256]
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║                        ENCODER                                ║
║                                                               ║
║  Stem  Conv3×3 s=2, BN, ReLU → [B,  64, 128, 128]   ← E1   ║
║         │                                                     ║
║         ▼                                                     ║
║  Stage 2: ResBlock×2 (64→128, s=2) → [B, 128, 64, 64] ← E2  ║
║         │                                                     ║
║         ▼                                                     ║
║  Stage 3: ResBlock×2 (128→256, s=2) → [B, 256, 32, 32] ← E3 ║
║         │                                                     ║
║         ▼                                                     ║
║  Stage 4: ResBlock×2 (256→512, s=2) → [B, 512, 16, 16] ← E4 ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║              BOTTLENECK  (dilated, ASPP-style)                ║
║                                                               ║
║   512 ─► Conv3×3 dilation=2 ─► BN ─► ReLU → 1024 ch        ║
║   1024 ─► Conv3×3 dilation=4 ─► BN ─► ReLU → 1024 ch       ║
║                                                               ║
║   Receptive field expanded without reducing spatial size      ║
╚═══════════════════════════════════════════════════════════════╝
        │                  skip connections from encoder ──────┐
        ▼                                                      │
╔═══════════════════════════════════════════════════════════════╗
║                        DECODER                                ║
║                                                               ║
║  D4: 2×Upsample(1024→512)                                    ║
║       ← AttGate(skip=E4[512], gate=D_up[512])                ║
║       [concat → 1024]  ─► Conv3×3×2 ─► [B, 512, 16, 16]     ║
║         │                                                     ║
║  D3: 2×Upsample(512→256)                                     ║
║       ← AttGate(skip=E3[256], gate=D4[256])                  ║
║       [concat → 512]   ─► Conv3×3×2 ─► [B, 256, 32, 32]     ║
║         │                                                     ║
║  D2: 2×Upsample(256→128)                                     ║
║       ← AttGate(skip=E2[128], gate=D3[128])                  ║
║       [concat → 256]   ─► Conv3×3×2 ─► [B, 128, 64, 64]     ║
║         │                                                     ║
║  D1: 2×Upsample(128→64)       [B, 64, 128, 128]              ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
  Conv 1×1 ─► [B, numClasses, 256, 256]
        │
  Argmax per pixel → Segmentation Mask
```

---

### Encoder — ResNet-style Backbone

Each Residual Block:

```
   Input x  [B, C_in, H, W]
       │
       ├────── Shortcut (1×1 proj if C_in ≠ C_out or stride ≠ 1) ──┐
       │                                                             │
       ▼                                                             │
   Conv 3×3, C_out, pad=1, stride=s                                 │
       │                                                             │
   Batch Norm → ReLU                                                 │
       │                                                             │
   Conv 3×3, C_out, pad=1, stride=1                                 │
       │                                                             │
   Batch Norm                                                        │
       │                                                             │
       └──────────────────── (+) ────────────────────────────────────┘
                              │
                            ReLU
                              │
                        [B, C_out, H/s, W/s]
```

**Encoder spatial resolution table (input 256×256):**

```
Layer            Channels     Spatial       Stride
────────────────────────────────────────────────────
Stem Conv3×3        64        128 × 128       ×2
Stage 2 (×2)       128         64 × 64        ×2
Stage 3 (×2)       256         32 × 32        ×2
Stage 4 (×2)       512         16 × 16        ×2
```

> *He et al., 2016 — Deep Residual Learning for Image Recognition* [[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)]

---

### Bottleneck — Dilated ASPP

Standard convolutions have a **fixed receptive field**. Dilated (atrous) convolutions expand it exponentially with the same parameter count:

```
  Standard Conv3×3  (dilation=1):       ■ · ■ · ■
                                        ·   ·   ·
                                        ■ · ■ · ■    RF = 3×3

  Dilated  Conv3×3  (dilation=2):       ■ · · ■ · · ■
                                        ·   ·   ·
                                        ■ · · ■ · · ■   RF = 7×7
                                        ·   ·   ·
                                        ■ · · ■ · · ■

  Dilated  Conv3×3  (dilation=4):                        RF = 17×17
```

The bottleneck stacks dilation=2 then dilation=4, giving two successively wider receptive fields of the same 1024-channel feature map — similar to ASPP (Atrous Spatial Pyramid Pooling).

> *Chen et al., 2018 — DeepLab v3+* [[arXiv:1802.02611](https://arxiv.org/abs/1802.02611)]  
> *Yu & Koltun, 2016 — Multi-Scale Context Aggregation by Dilated Convolutions* [[arXiv:1511.07122](https://arxiv.org/abs/1511.07122)]

---

### Decoder — U-Net with Attention Gates

Each decoder stage performs four operations:

```
  Deep feature map  [B, C_deep, h, w]
         │
         │  1. Bilinear 2× upsample  (CUDA kernel: upsample2xKernel)
         ▼
  [B, C_deep, 2h, 2w]
         │
         │  2. Attention Gate on skip connection  (see below)
         │
  Encoder skip  [B, C_skip, 2h, 2w]  ───► AttGate ───► Gated skip
         │                                               │
         │  3. Channel concatenation  (concatKernel)     │
         └──────────────────────────────────────────────►(concat)
                                                         │
                                               [B, C_deep+C_skip, 2h, 2w]
                                                         │
                                         4. 2 × Conv3×3 → BN → ReLU
                                                         │
                                               [B, C_out, 2h, 2w]
```

> *Ronneberger et al., 2015 — U-Net: Convolutional Networks for Biomedical Image Segmentation* [[arXiv:1505.04597](https://arxiv.org/abs/1505.04597)]

---

### Output Head

```
  Decoder output [B, 64, 256, 256]
         │
     Conv 1×1 (no padding, stride=1)
         │
     [B, numClasses, 256, 256]
         │
    ─────────────────────────────────
    Per pixel:  softmax → argmax class
    ─────────────────────────────────
         │
    Segmentation mask [B, 256, 256]
```

---

## Attention Gate — Additive Attention

Vanilla U-Net skip connections pass **all** encoder features to the decoder — including irrelevant background regions. Attention Gates learn to **re-weight** skip activations:

```
  Gate signal  g  [B, gC, H, W]   ← from deeper decoder layer
  Skip signal  x  [B, xC, H, W]   ← from encoder at same resolution

  ┌──────────────────────────────────────────────┐
  │         Additive Attention Gate              │
  │                                              │
  │   φ_g  =  Wg · g    (1×1 conv, gC→intC)    │
  │   φ_x  =  Wx · x    (1×1 conv, xC→intC)    │
  │                                              │
  │   φ    =  ReLU(φ_g + φ_x)                  │
  │                                              │
  │   ψ    =  σ(Wψ · φ)   ∈ (0,1)              │
  │           (sigmoid scalar attention map)     │
  │                                              │
  │   x̂   =  ψ ⊙ x       (element-wise scale)  │
  └──────────────────────────────────────────────┘
         │
       x̂  [B, xC, H, W]   — attended skip features

  ψ ≈ 0  →  suppress irrelevant region
  ψ ≈ 1  →  pass through relevant region
```

The gate specifically helps when the object of interest occupies a small portion of the image — forcing the decoder to focus on the correct spatial regions rather than uniformly up-weighting all encoder features.

> *Oktay et al., 2018 — Attention U-Net: Learning Where to Look for the Pancreas* [[arXiv:1804.03999](https://arxiv.org/abs/1804.03999)]

---

## Loss Functions

### 1 — Pixel-wise Cross-Entropy

Applied independently at every pixel. For pixel `p` with ground-truth class `c*`:

```
  CE_p = − log( softmax(logit_{c*,p}) )

  softmax(zₖ) = exp(zₖ) / Σⱼ exp(zⱼ)

  Total loss = (1 / N·H·W) · Σₙ Σₚ CE_p
```

Gradient w.r.t. logit `zₖ`:

```
  ∂CE/∂zₖ = (softmax(zₖ) − 1[k = c*]) / (N·H·W)
```

Implemented with a numerically stable in-kernel max-subtract-then-expsum.

---

### 2 — Dice Loss

Cross-entropy can still fail when class frequencies are extremely imbalanced (e.g., tiny lesion vs. large background). Dice loss directly optimises the **overlap** metric:

```
  Dice = 1 − (2 · |P ∩ G|) / (|P| + |G|)

  where:
    P = predicted probability for class c  (soft, not thresholded)
    G = binary ground-truth mask for class c

  Intersection: Σₚ Pₚ · Gₚ
  Sums        : Σₚ Pₚ + Σₚ Gₚ
```

Atomic accumulation across pixels via `diceLossFwdKernel` (CUDA atomicAdd).

The combined loss used in training:

```
  L_total = L_CrossEntropy + L_Dice
```

> *Milletari et al., 2016 — V-Net: Fully Convolutional Neural Networks for Volumetric Image Segmentation* [[arXiv:1606.04797](https://arxiv.org/abs/1606.04797)]

---

### 3 — mIoU Metric (Evaluation)

```
  IoU_c = |Predicted_c ∩ GT_c| / |Predicted_c ∪ GT_c|

  mIoU  = (1/C) · Σ_c IoU_c

  Classes (Oxford Pet):
    0 → background  (large area)
    1 → pet body    (target)
    2 → boundary    (thin, hard)
```

Reported per batch during training by `computeBatchMIoU()` in `main.cu`.

---

## Data Augmentation

All augmentations operate on **paired (image, mask)** tensors simultaneously in CUDA. Masks use **nearest-neighbour** resampling to preserve integer class labels:

```
Input Pair [B, 3, H, W] + [B, H, W]
      │
      ├─ segHFlipKernel      Horizontal flip: image (bilinear swap) + mask (nearest swap)
      │
      ├─ segVFlipKernel      Vertical flip: same paired approach (25% applied)
      │
      ├─ segColorJitterKernel
      │     Image only: brightness ±0.15 | contrast ×[0.8,1.2] | saturation ×[0.7,1.3]
      │
      ├─ segGaussianNoiseKernel
      │     Image only: additive σ ∈ [0, 0.03] via per-thread cuRAND state
      │
      └─ elasticSampleKernel
            Random displacement field (α=8, σ=4) computed on host
            Image: bilinear interpolation at displaced coordinates
            Mask : nearest-neighbour interpolation (class-label safe)
                   ↓
            Temporarily stored in d_imgTmp / d_maskTmp → swapped back
```

### Elastic Deformation Detail

```
  For each pixel (x, y):
    dx = rand() ∈ [−α, +α]    ← random displacement field
    dy = rand() ∈ [−α, +α]

  Sample source:
    src_x = x + dx
    src_y = y + dy

  Image pixel (bilinear):
    I'(x,y) = (1-wy)·[(1-wx)·I(x0,y0) + wx·I(x1,y0)]
            +    wy ·[(1-wx)·I(x0,y1) + wx·I(x1,y1)]

  Mask  pixel (nearest):
    M'(x,y) = M(round(src_x), round(src_y))
```

Elastic deformation is particularly important for **medical image segmentation** where training data is scarce.

> *Simard et al., 2003 — Best Practices for CNNs applied to Visual Document Analysis* (elastic deformation origin)  
> *Ronneberger et al., 2015 — U-Net* [[arXiv:1505.04597](https://arxiv.org/abs/1505.04597)]

---

## Training Pipeline

```
Epoch Loop
    │
    ├─ loadBatch()           read B (image, mask) pairs from seg_pets.bin
    │                        image → float [0,1]  |  mask → int [0, numCls-1]
    │
    ├─ Upload mask to GPU    cudaMemcpy h_masks → d_masks
    │
    ├─ segNet.forward()      Encoder → Bottleneck → Decoder (with aug in train mode)
    │
    ├─ segNet.backward()     pixelCrossEntropyLossKernel  (in-kernel softmax)
    │                        diceLossFwdKernel            (atomic accumulation)
    │                        momentumSGDKernel on all weight tensors
    │
    ├─ computeBatchMIoU()    argmax on h_logits → compare with h_masks
    │
    ├─ LR Decay              polynomial schedule:
    │                        lr = lr_init · (1 − epoch/epochs)^0.9
    │
    └─ Checkpoint            saveWeights() every 5 epochs
                             "seg_pets_ep{N}.bin"
```

**Polynomial LR decay** is the standard schedule for segmentation tasks (used in DeepLab, PSPNet):

```
  lr_t = lr_init · (1 − t / T)^power       power = 0.9
```

> *Chen et al., 2017 — DeepLab: Semantic Image Segmentation with DCNNs, Atrous Convolution, and Fully Connected CRFs* [[arXiv:1606.00915](https://arxiv.org/abs/1606.00915)]

---

## File Structure

```
segmentation/
├── main.cu                   Training entry point + mIoU tracking + checkpointing
├── compile.ps1               NVCC build script (Windows PowerShell)
├── dataset/
│   └── prepare_dataset.py    Downloads Oxford-IIIT Pet, writes seg_pets.bin
│                             (images + trimap → 3-class masks, 256×256)
└── network/
    ├── cudnn_helper.h        CUDA / cuDNN / cuBLAS error-check macros
    ├── utilities.cu          He init · GPU timer · SGD kernel
    │                         Pixel CE loss · Dice loss · mIoU · Weight I/O
    ├── augmentation.cu       H/V flip · Color jitter · Gaussian noise
    │                         Elastic deformation (paired image+mask)
    └── network.cu            SegmentationNet class
                              (ResNet encoder + ASPP bottleneck +
                               Attention U-Net decoder + 1×1 head)
```

---

## Getting Started

### Prerequisites

- NVIDIA GPU (Pascal / sm_60 or newer)
- CUDA Toolkit 11.x+
- cuDNN 8.x
- Python 3.8+ with `pip install requests numpy pillow`

### 1 — Prepare Dataset

> **Dataset notice:** The Oxford-IIIT Pet Dataset is NOT included in this
> repository. Download images.tar.gz and annotations.tar.gz from the official
> source (CC-BY-SA 4.0 license):
> https://www.robots.ox.ac.uk/~vgg/data/pets/
> Extract both so that `images/` and `annotations/` folders exist next to the script.

```powershell
cd segmentation\dataset
python prepare_dataset.py
# Best case:  Found trimap annotations. Done! Saved → seg_pets.bin
# Worst case: [ERROR] Trimap directory not found. [ABORT] Dataset preparation failed.
cd ..
```

### 2 — Compile

```powershell
cd segmentation
.\compile.ps1
# Output: seg_unet.exe
```

### 3 — Train

```powershell
.\seg_unet.exe
# Epoch  1/15  Batch 100/1847  mIoU: 42.3%  ...
# [Checkpoint] Saved → seg_pets_ep5.bin
# ...
# Weights saved → seg_pets_final.bin
```

---

## Reference Papers

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 1 | **U-Net: Convolutional Networks for Biomedical Image Segmentation** | Ronneberger et al. | MICCAI 2015 | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| 2 | **Attention U-Net: Learning Where to Look for the Pancreas** | Oktay et al. | MIDL 2018 | [arXiv:1804.03999](https://arxiv.org/abs/1804.03999) |
| 3 | **Deep Residual Learning for Image Recognition** | He et al. | CVPR 2016 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 4 | **DeepLab: Semantic Segmentation with DCNNs and CRFs** | Chen et al. | TPAMI 2017 | [arXiv:1606.00915](https://arxiv.org/abs/1606.00915) |
| 5 | **DeepLab v3+: Encoder-Decoder with Atrous Separable Convolution** | Chen et al. | ECCV 2018 | [arXiv:1802.02611](https://arxiv.org/abs/1802.02611) |
| 6 | **Multi-Scale Context Aggregation by Dilated Convolutions** | Yu & Koltun | ICLR 2016 | [arXiv:1511.07122](https://arxiv.org/abs/1511.07122) |
| 7 | **V-Net: Fully CNN for Volumetric Segmentation (Dice Loss)** | Milletari et al. | 3DV 2016 | [arXiv:1606.04797](https://arxiv.org/abs/1606.04797) |
| 8 | **Batch Normalization** | Ioffe & Szegedy | ICML 2015 | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) |
| 9 | **He (Kaiming) Initialization** | He et al. | ICCV 2015 | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) |
| 10 | **Momentum SGD** | Sutskever et al. | ICML 2013 | [ICML Proc.](http://proceedings.mlr.press/v28/sutskever13.html) |
| 11 | **Elastic deformation for CNNs** | Simard et al. | ICDAR 2003 | [IEEE](https://doi.org/10.1109/ICDAR.2003.1227801) |
| 12 | **Oxford-IIIT Pet Dataset** | Parkhi et al. | CVPR 2012 | [Project](https://www.robots.ox.ac.uk/~vgg/data/pets/) |

> **Dataset attribution:** The Oxford-IIIT Pet Dataset (cited above) is not
> included in this repository and is NOT covered by the MIT License.
> It is licensed under CC-BY-SA 4.0 and must be obtained independently.

---

## License

The source code in this module is distributed under the **MIT License**.
See [`../LICENSE`](../LICENSE) for the full text, including dataset and
algorithm attribution notices.

> The MIT License covers source code only. The Oxford-IIIT Pet Dataset
> is not included and is licensed under CC-BY-SA 4.0.
