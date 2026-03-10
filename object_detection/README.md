# CuVision-Engine — Object Detection Module

> **RetinaNet-FPN detector** implemented natively in CUDA / cuDNN / cuBLAS.  
> No PyTorch. No TensorFlow. Pure GPU primitives — maximum throughput.

---

## Table of Contents

1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
   - [Backbone — ResNet-style](#backbone--resnet-style)
   - [Neck — Feature Pyramid Network (FPN)](#neck--feature-pyramid-network-fpn)
   - [Detection Head](#detection-head)
3. [Anchor System](#anchor-system)
4. [Loss Functions](#loss-functions)
5. [Data Augmentation](#data-augmentation)
6. [Training Pipeline](#training-pipeline)
7. [File Structure](#file-structure)
8. [Getting Started](#getting-started)
9. [Reference Papers](#reference-papers)

---

## Overview

This module implements a **one-stage anchor-based object detector** combining:

| Component | Design Choice | Justification |
|:---|:---|:---|
| **Backbone** | ResNet-like 4-stage encoder | Strong hierarchical features; residual skip connections prevent vanishing gradients |
| **Neck** | Feature Pyramid Network (FPN) | Multi-scale detection: small objects on high-res maps, large on low-res |
| **Head** | Shared cls + reg tower (4×conv) | Parameter-efficient; same weights applied across all FPN levels |
| **Cls Loss** | Sigmoid Focal Loss (α=0.25, γ=2) | Solves extreme foreground/background imbalance without hard-example mining |
| **Reg Loss** | Smooth-L1 (Huber) Loss | Robust to outlier boxes; matches RCNN-family delta encoding |
| **Optimizer** | Momentum-SGD + weight decay | Custom CUDA kernel; cosine LR schedule |
| **Init** | He (Kaiming) Normal | Correct variance for ReLU activations |

---

## Network Architecture

### Full Forward Pass — ASCII Diagram

```
Input Image [B, 3, 300, 300]
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                        BACKBONE                               │
│                                                               │
│   Stem Conv3×3 s=2, BN, ReLU → [B, 64, 150, 150]            │
│        │                                                      │
│        ▼                                                      │
│   Stage 2: ResBlock×2  (64→128, s=2) → [B, 128, 75, 75]     │  ← C2
│        │                                                      │
│        ▼                                                      │
│   Stage 3: ResBlock×2  (128→256, s=2) → [B, 256, 38, 38]    │  ← C3
│        │                                                      │
│        ▼                                                      │
│   Stage 4: ResBlock×2  (256→512, s=2) → [B, 512, 19, 19]    │  ← C4
└───────────────────────────────────────────────────────────────┘
        │C4              │C3               │C2
        ▼                ▼                 ▼
┌───────────────────────────────────────────────────────────────┐
│                    NECK — FPN (top-down)                      │
│                                                               │
│  Lateral 1×1                                                  │
│  C4 ──────────────────────────────────► P4 [B,256,19,19]     │
│                                          │ 2× upsample        │
│  C3 ──── lateral ──(+)──── 3×3 ────────► P3 [B,256,38,38]   │
│                      ▲      │ 2× upsample                    │
│  C2 ──── lateral ──(+)──── 3×3 ────────► P2 [B,256,75,75]   │
└───────────────────────────────────────────────────────────────┘
        │P4            │P3              │P2
        ▼              ▼                ▼
┌───────────────────────────────────────────────────────────────┐
│               DETECTION HEAD  (shared weights)                │
│                                                               │
│   ┌─────────────────────┐    ┌─────────────────────┐         │
│   │   CLS TOWER         │    │   REG TOWER          │         │
│   │  4 × Conv3×3,256,BN │    │  4 × Conv3×3,256,BN │         │
│   │  4 × ReLU           │    │  4 × ReLU            │         │
│   │  1×1 → A×numCls     │    │  1×1 → A×4           │         │
│   └─────────────────────┘    └─────────────────────┘         │
│         (applied to each of P2, P3, P4)                       │
│                                                               │
│   Total anchors A = 9 per cell  (3 scales × 3 ratios)        │
│   Total predictions = (75²+38²+19²) × 9 = ~60 k anchors      │
└───────────────────────────────────────────────────────────────┘
        │ cls logits            │ reg deltas
        ▼                       ▼
  Focal Loss             Smooth-L1 Loss
  (classification)       (box regression)
```

---

### Backbone — ResNet-style

Each **Residual Block** consists of:

```
   Input x
     │
     ├────────── Shortcut ──────────┐
     │   (1×1 proj if dim changes)  │
     │                              │
     ▼                              │
  Conv3×3 → BN → ReLU              │
     │                              │
     ▼                              │
  Conv3×3 → BN                     │
     │                              │
     └──────────── (+) ─────────────┘
                    │
                   ReLU
                    │
                  Output
```

**Why residual connections?**  
Without them, gradients vanish across deep networks. The identity shortcut provides a direct gradient highway — enabling training of arbitrarily deep networks.  
> *He et al., 2016 — Deep Residual Learning for Image Recognition* [[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)]

**Stage dimensions (input 300×300):**

```
Layer          Channels   Spatial Size   Stride
─────────────────────────────────────────────────
Stem (Conv3×3)     64     150 × 150        2
Stage 2 (×2)      128      75 × 75         2
Stage 3 (×2)      256      38 × 38         2
Stage 4 (×2)      512      19 × 19         2
```

---

### Neck — Feature Pyramid Network (FPN)

The FPN merges backbone features across scales via **top-down lateral connections**:

```
       Low resolution, semantically rich
       C4 [512, 19×19]
            │
            │  1×1 conv → 256 ch
            ▼
           P4 [256, 19×19]
            │
            │  2× upsample (bilinear)
            ▼
       C3 [256, 38×38]
       ──lateral─►  (+)  ──► 3×3 conv ──► P3 [256, 38×38]
                              smooth
            │
            │  2× upsample
            ▼
       C2 [128, 75×75]
       ──lateral─►  (+)  ──► 3×3 conv ──► P2 [256, 75×75]
                              smooth
       High resolution, positionally precise
```

Each pyramid level captures objects of different sizes:

| FPN Level | Stride | Receptive Field | Best for |
|:---:|:---:|:---:|:---:|
| P2 | 4 | Small | Tiny objects, fine detail |
| P3 | 8 | Medium | Mid-size objects |
| P4 | 16 | Large | Large objects, context |

> *Lin et al., 2017 — Feature Pyramid Networks for Object Detection* [[arXiv:1612.03144](https://arxiv.org/abs/1612.03144)]

---

### Detection Head

```
FPN Feature Map  [B, 256, H, W]
        │
        ├──────────────────────────────┬──────────────────────────────┐
        │   CLS TOWER                  │   REG TOWER                  │
        │                              │                              │
        │  Conv3×3 → BN → ReLU ×4     │  Conv3×3 → BN → ReLU ×4    │
        │                              │                              │
        │  Conv1×1                     │  Conv1×1                     │
        │  [B, A×numCls, H, W]         │  [B, A×4, H, W]             │
        │                              │                              │
        │   ↓ sigmoid                  │   ↓  Δcx, Δcy, Δw, Δh      │
        │  Class probabilities          │  Box deltas                 │
        └──────────────────────────────┴──────────────────────────────┘
                │                                    │
         Focal Loss                          Smooth-L1 Loss
```

The head weights are **shared across all FPN levels** — the same conv filters detect objects regardless of scale. This acts as a form of scale invariance.

---

## Anchor System

Anchors are **pre-defined reference boxes** tiled across every cell of every FPN feature map. The network learns to predict *offsets* from anchors, not absolute boxes.

### Anchor Configuration

```
Per FPN level:
  Base scales  : [32, 64, 128] px
  Aspect ratios: [0.5, 1.0, 2.0]
  ─────────────────────────────
  Anchors/cell : 3 × 3 = 9

  Scale 32, ratio 0.5 → w=22.6, h=45.2
  Scale 32, ratio 1.0 → w=32.0, h=32.0
  Scale 32, ratio 2.0 → w=45.2, h=22.6
  ... (repeat for 64, 128)
```

### IoU-Based Anchor Assignment

```
For each anchor A and ground-truth box G:

  IoU(A, G) ≥ 0.5  →  POSITIVE  (anchor assigned to that class)
  IoU(A, G) < 0.4  →  NEGATIVE  (anchor = background)
  0.4 ≤ IoU < 0.5  →  IGNORED   (not used in loss)
```

### Regression Delta Encoding (RCNN-style)

```
  Given:  anchor (acx, acy, aw, ah)
          ground-truth (gcx, gcy, gw, gh)

  Target deltas:
    Δcx = (gcx - acx) / aw
    Δcy = (gcy - acy) / ah
    Δw  = log(gw / aw)
    Δh  = log(gh / ah)

  At inference:
    pcx = Δcx * aw + acx
    pcy = Δcy * ah + acy
    pw  = aw * exp(Δw)
    ph  = ah * exp(Δh)
```

---

## Loss Functions

### 1 — Sigmoid Focal Loss (Classification)

Standard cross-entropy fails on dense detectors because **easy negatives dominate** the loss (there are ~60 k anchors but only a handful are positive objects).

Focal Loss down-weights easy examples automatically:

```
                FL(pₜ) = −α · (1 − pₜ)^γ · log(pₜ)

  where:
    pₜ  = sigmoid probability of the true class
    α   = 0.25  (class-imbalance balancing factor)
    γ   = 2.0   (focusing parameter)

  Effect:
    pₜ = 0.9  (easy) →  (1-0.9)^2 = 0.01  ← near-zero weight
    pₜ = 0.1  (hard) →  (1-0.1)^2 = 0.81  ← full weight
```

> *Lin et al., 2017 — Focal Loss for Dense Object Detection (RetinaNet)* [[arXiv:1708.02002](https://arxiv.org/abs/1708.02002)]

---

### 2 — Smooth-L1 Loss (Bounding Box Regression)

```
              ┌ 0.5 · δ²          if |δ| < 1
  SmoothL1 = ─┤
              └ |δ| − 0.5         otherwise

  where δ = predicted_delta − target_delta
```

Compared to plain L2, Smooth-L1 is **less sensitive to outlier boxes** (very wrong predictions contribute linearly, not quadratically).

> *Girshick, 2015 — Fast R-CNN* [[arXiv:1504.08083](https://arxiv.org/abs/1504.08083)]

---

### 3 — Non-Maximum Suppression (NMS) — Post-processing

```
Input:  N boxes with confidence scores
Output: M kept boxes (M ≤ N)

Algorithm:
  1. Sort boxes by score (descending)
  2. Keep B₁ (highest score)
  3. For every remaining box Bᵢ:
       if IoU(B₁, Bᵢ) > threshold → suppress Bᵢ
  4. Repeat from step 2 on remaining boxes
```

Implemented host-side in `utilities.cu` → `nonMaxSuppression()`.

---

## Data Augmentation

All augmentations are applied on-device (CUDA kernels in `network/augmentation.cu`):

```
Input Batch [B, 3, H, W]
      │
      ├── horizontalFlipKernel    (50% per image — returns flip mask for bbox mirroring)
      │
      ├── colorJitterKernel       (brightness ±0.15, contrast ×[0.8,1.2], saturation ×[0.7,1.3])
      │
      ├── gaussianNoiseKernel     (additive σ ∈ [0, 0.04] via per-thread cuRAND state)
      │
      └── cutoutKernel            (random square zeroed: 15–25% of min(H,W) side)
             ↓
       Augmented Batch  (in-place, no extra copy)
```

**Bbox coordinate transform after flip:**
```
  if flip_flag[i] == 1:
    x1_new = 1.0 - x2_old
    x2_new = 1.0 - x1_old
```

> *DeVries & Taylor, 2017 — Improved Regularization of CNNs with Cutout* [[arXiv:1708.04552](https://arxiv.org/abs/1708.04552)]

---

## Training Pipeline

```
Epoch Loop
    │
    ├─ loadBatch()             read B images + GT boxes from od_voc2007.bin
    │
    ├─ buildTargets()          IoU match → clsTargets[B×A], regTargets[B×A×4]
    │
    ├─ detector.forward()      Backbone → FPN → Head  (with augmentation)
    │
    ├─ detector.backward()     focalLossKernel + smoothL1LossKernel
    │                          momentumSGDKernel on all weight tensors
    │
    └─ LR Decay                cosine schedule:
                               lr = lr_init · 0.5 · (1 + cos(π·epoch/epochs))
```

**Optimizer — Momentum SGD (custom CUDA kernel):**

```
  v[i]  ← momentum · v[i]  +  lr · (∇L[i] + decay · w[i])
  w[i]  ← w[i] − v[i]
  ∇L[i] ← 0   (zero-gradient in-place)
```

> *Sutskever et al., 2013 — On the importance of initialization and momentum in deep learning* [[ICML](http://proceedings.mlr.press/v28/sutskever13.html)]

---

## File Structure

```
object_detection/
├── main.cu                   Training entry point
├── compile.ps1               NVCC build script (Windows PowerShell)
├── dataset/
│   └── prepare_dataset.py    Downloads Pascal VOC 2007, writes od_voc2007.bin
└── network/
    ├── cudnn_helper.h        CUDA / cuDNN / cuBLAS error-check macros
    ├── utilities.cu          He init · GPU timer · SGD kernel · Smooth-L1
    │                         Focal loss · NMS · Anchor generator · Weight I/O
    ├── augmentation.cu       H-flip · Color jitter · Gaussian noise · Cutout
    └── network.cu            ObjectDetector class (backbone + FPN + head)
```

---

## Getting Started

### Prerequisites

- NVIDIA GPU (Pascal / sm_60 or newer)
- CUDA Toolkit 11.x+
- cuDNN 8.x
- Python 3.8+ with `pip install requests numpy pillow`

### 1 — Prepare Dataset

> **Dataset notice:** Pascal VOC 2007 is NOT included in this repository.
> Download VOCtrainval_06-Nov-2007.tar from the official source:
> http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
> Extract it so that a `VOCdevkit/` folder exists next to this script.

```powershell
cd object_detection\dataset
python prepare_dataset.py
# Best case:  Found valid images in trainval split. Done! Saved → od_voc2007.bin
# Worst case: [ERROR] VOCdevkit/ directory not found. [ABORT] Dataset preparation failed.
cd ..
```

### 2 — Compile

```powershell
cd object_detection
.\compile.ps1
# Output: od_detector.exe
```

### 3 — Train

```powershell
.\od_detector.exe
# Epoch 1/10  Batch 100/1250  Images: 400  ...
# Weights saved → od_voc2007_model.bin
```

---

## Reference Papers

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 1 | **Deep Residual Learning for Image Recognition** | He et al. | CVPR 2016 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 2 | **Feature Pyramid Networks for Object Detection** | Lin et al. | CVPR 2017 | [arXiv:1612.03144](https://arxiv.org/abs/1612.03144) |
| 3 | **Focal Loss for Dense Object Detection (RetinaNet)** | Lin et al. | ICCV 2017 | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) |
| 4 | **SSD: Single Shot MultiBox Detector** | Liu et al. | ECCV 2016 | [arXiv:1512.02325](https://arxiv.org/abs/1512.02325) |
| 5 | **Fast R-CNN** (Smooth-L1 loss) | Girshick | ICCV 2015 | [arXiv:1504.08083](https://arxiv.org/abs/1504.08083) |
| 6 | **Batch Normalization** | Ioffe & Szegedy | ICML 2015 | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) |
| 7 | **He (Kaiming) Initialization** | He et al. | ICCV 2015 | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) |
| 8 | **Improved Regularization with Cutout** | DeVries & Taylor | 2017 | [arXiv:1708.04552](https://arxiv.org/abs/1708.04552) |
| 9 | **Momentum SGD** | Sutskever et al. | ICML 2013 | [ICML Proc.](http://proceedings.mlr.press/v28/sutskever13.html) |
| 10 | **Pascal VOC Challenge** | Everingham et al. | IJCV 2010 | [DOI](https://doi.org/10.1007/s11263-009-0275-4) |

> **Dataset attribution:** The Pascal VOC 2007 dataset (cited above) is not
> included in this repository and is NOT covered by the MIT License.
> It must be obtained independently subject to PASCAL VOC's terms of use.

---

## License

The source code in this module is distributed under the **MIT License**.
See [`../LICENSE`](../LICENSE) for the full text, including dataset and
algorithm attribution notices.

> The MIT License covers source code only. The Pascal VOC 2007 dataset
> is not included and is subject to its own separate license.
