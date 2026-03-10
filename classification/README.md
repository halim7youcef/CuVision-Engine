# CuVision-Engine — Classification Module

> **Deep CNN image classifier** implemented natively in CUDA / cuDNN / cuBLAS.  
> No PyTorch. No TensorFlow. Pure GPU primitives — maximum throughput on NVIDIA hardware.

---

## Table of Contents

1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
   - [Layer-by-Layer Diagram](#layer-by-layer-diagram)
   - [Convolutional Stage](#convolutional-stage)
   - [Fully-Connected Head](#fully-connected-head)
3. [Normalization & Regularization](#normalization--regularization)
   - [Batch Normalization](#batch-normalization)
   - [Dropout](#dropout)
4. [Weight Initialization](#weight-initialization)
5. [Optimizer — Momentum SGD](#optimizer--momentum-sgd)
6. [Loss Function — Softmax Cross-Entropy](#loss-function--softmax-cross-entropy)
7. [Learning Rate Schedule](#learning-rate-schedule)
8. [Data Augmentation](#data-augmentation)
9. [Training Pipeline](#training-pipeline)
10. [Binary Dataset Format](#binary-dataset-format)
11. [File Structure](#file-structure)
12. [Getting Started](#getting-started)
13. [Reference Papers](#reference-papers)

---

## Overview

This module implements a **2-stage Deep CNN** for multi-class image classification, following the foundational design patterns of LeNet → AlexNet → VGGNet but implemented entirely with raw cuDNN descriptors for maximum GPU efficiency.

| Component | Design Choice | Justification |
|:---|:---|:---|
| **Architecture** | 2 × (Conv → BN → ReLU → MaxPool) + FC + Softmax | Hierarchical feature extraction with spatial compression |
| **Normalization** | Batch Normalization (Spatial) | Reduces internal covariate shift; allows higher learning rates |
| **Regularization** | Dropout (50%) + L2 Weight Decay | Prevents co-adaptation; penalises large weights |
| **Initialization** | He (Kaiming) Normal | Correct variance preservation for ReLU activations |
| **Optimizer** | Custom Momentum-SGD CUDA kernel | Direct GPU weight updates; no framework overhead |
| **Loss** | Cross-Entropy over Softmax | Standard for multi-class classification |
| **LR Schedule** | Step decay ×0.8 per epoch | Smooth convergence to a tight minimum |
| **Augmentation** | H-flip + Brightness jitter (CUDA) | Improves generalisation without CPU bottleneck |

---

## Network Architecture

### Layer-by-Layer Diagram

```
Input Image  [B, 3, 32, 32]   ← normalised float CHW
        │
        ▼
┌────────────────────────── STAGE 1 ──────────────────────────────┐
│                                                                  │
│   Conv2d  3×3, pad=1, stride=1                                  │
│   Filters:  [32, 3, 3, 3]   →  [B, 32, 32, 32]                 │
│        │                                                         │
│   Batch Norm (Spatial, 32 channels)                              │
│        │                                                         │
│   ReLU  (in-place via cudnnActivationForward)                    │
│        │                                                         │
│   MaxPool  2×2, stride=2                                         │
│        │                                                         │
│   Output:  [B, 32, 16, 16]                                      │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────── STAGE 2 ──────────────────────────────┐
│                                                                  │
│   Conv2d  3×3, pad=1, stride=1                                  │
│   Filters:  [64, 32, 3, 3]  →  [B, 64, 16, 16]                 │
│        │                                                         │
│   Batch Norm (Spatial, 64 channels)                              │
│        │                                                         │
│   ReLU  (in-place)                                               │
│        │                                                         │
│   MaxPool  2×2, stride=2                                         │
│        │                                                         │
│   Output:  [B, 64, 8, 8]                                        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────── DROPOUT + FC ─────────────────────────────┐
│                                                                  │
│   Dropout  (p=0.5, training only)                                │
│        │                                                         │
│   Flatten:  [B, 64×8×8] = [B, 4096]                            │
│        │                                                         │
│   FC  (via cuBLAS Sgemm):  4096 → numClasses                   │
│        │                                                         │
│   Softmax  (CUDNN_SOFTMAX_ACCURATE, INSTANCE mode)              │
│        │                                                         │
│   Output:  [B, numClasses]   ← class probabilities             │
└──────────────────────────────────────────────────────────────────┘
        │
   Cross-Entropy Loss  +  Backward  +  Momentum-SGD update
```

**Parameter count (10 classes, 32×32 input):**

```
Layer            Parameters
─────────────────────────────────────────
Conv1 weights    32 × 3 × 3 × 3   =    864
Conv1 BN         32 × 2            =     64
Conv2 weights    64 × 32 × 3 × 3  = 18,432
Conv2 BN         64 × 2            =    128
FC weights       4096 × 10         = 40,960
─────────────────────────────────────────
Total                               ≈ 60 k   (deliberately lightweight)
```

---

### Convolutional Stage

Each stage applies the classic **Conv → BN → ReLU → Pool** sequence. Detailed data-flow for Stage 1:

```
  d_input  [B, 3, 32, 32]
       │
       │  cudnnConvolutionForward
       │  filter1: [32, 3, 3, 3]  pad=1  stride=1
       ▼
  d_conv1Out  [B, 32, 32, 32]      ← raw convolution output
       │
       │  cudnnBatchNormalizationForward{Training|Inference}
       │  scale γ, bias β  (32 parameters each, learnable)
       ▼
  d_bn1Out  [B, 32, 32, 32]        ← normalised + scaled + shifted
       │
       │  cudnnActivationForward  (RELU)
       ▼
  d_bn1Out  [B, 32, 32, 32]        ← in-place: negatives zeroed
       │
       │  cudnnPoolingForward  (MAX, 2×2, stride=2)
       ▼
  d_pool1Out  [B, 32, 16, 16]      ← spatially halved
```

The filter algorithm is auto-selected by cuDNN at runtime via  
`CUDNN_CONVOLUTION_FWD_PREFER_FASTEST` — chooses between Winograd, FFT, and implicit GEMM based on available VRAM.

---

### Fully-Connected Head

The flattened feature vector is multiplied by the FC weight matrix using **cuBLAS Sgemm** (single-precision GEMM):

```
  Operation:
    d_fcOut = d_fcWeights^T  ×  d_pool2OutDrop

  Dimensions:
    d_fcWeights   : [flatSize, numClasses]   (column-major in cuBLAS)
    d_pool2OutDrop: [flatSize, B]
    d_fcOut       : [numClasses, B]

  cuBLAS call:
    cublasSgemm(handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      numClasses, B, flatSize,
      &alpha, d_fcWeights, flatSize,
              d_pool2OutDrop, flatSize,
      &beta,  d_fcOut, numClasses)
```

---

## Normalization & Regularization

### Batch Normalization

Applied immediately after each convolution, before activation:

```
  For a mini-batch of size B, channel c:

  μ_c  = (1/B·H·W) · Σ x_{n,c,h,w}          ← batch mean
  σ²_c = (1/B·H·W) · Σ (x − μ_c)²           ← batch variance

  x̂_{n,c,h,w} = (x_{n,c,h,w} − μ_c) / √(σ²_c + ε)

  y_{n,c,h,w}  = γ_c · x̂ + β_c
                 ↑              ↑
           learnable scale   learnable bias

  At inference: uses running mean/var accumulated during training
    (momentum = 0.1 in cudnnBatchNormalizationForwardTraining)
```

**Benefits:**
- Enables significantly higher learning rates
- Reduces sensitivity to weight initialization
- Acts as a mild regularizer (similar effect to Dropout)

> *Ioffe & Szegedy, 2015 — Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* [[arXiv:1502.03167](https://arxiv.org/abs/1502.03167)]

---

### Dropout

Applied **after Stage 2 pooling** and **only during training** — inference bypasses it with a direct `cudaMemcpy`:

```
  Training:
    Each activation independently zeroed with probability p=0.5
    Surviving activations scaled by 1/(1−p) = 2.0 (inverted dropout)
    Reserve-space buffer stored for backward pass

  Inference:
    d_pool2OutDrop ← cudaMemcpy(d_pool2Out)   (identity, no dropout)

  Why after Stage 2?
    Dropout on early conv features destroys spatial structure.
    Applied at the flat-vector boundary it regularises the FC layer
    without disrupting learned conv kernels.
```

> *Srivastava et al., 2014 — Dropout: A Simple Way to Prevent Neural Networks from Overfitting* [[JMLR](https://jmlr.org/papers/v15/srivastava14a.html)]

---

## Weight Initialization

All learnable weight tensors use **He (Kaiming) Normal** initialisation, computed on the CPU and uploaded to GPU once at construction:

```
  For a layer with fan_in input connections:

  std = √(2 / fan_in)

  w ~ N(0, std²)    (Box-Muller transform for Gaussian sampling)

  ┌───────────────────────────────────────────────────────────┐
  │  Layer          fan_in              std                    │
  │  ──────────     ──────────          ────────────           │
  │  Conv1          3 × 3 × 3 = 27     √(2/27)  ≈ 0.272      │
  │  Conv2          32 × 3 × 3 = 288   √(2/288) ≈ 0.083      │
  │  FC             4096               √(2/4096) ≈ 0.022      │
  └───────────────────────────────────────────────────────────┘
```

**Why He init for ReLU?**  
Glorot (Xavier) init assumes symmetric activations (tanh, sigmoid). ReLU kills negative half, halving the effective variance. He init compensates with a factor of 2 in the numerator.

> *He et al., 2015 — Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* [[arXiv:1502.01852](https://arxiv.org/abs/1502.01852)]

---

## Optimizer — Momentum SGD

A custom CUDA kernel updates all parameters directly on-device — eliminating any CPU round-trips during the update step:

```
  CUDA Kernel: applyMomentumSGD  (one thread per parameter)

  For each weight wᵢ at iteration t:

    gᵢ  ←  ∇L_wᵢ  +  λ · wᵢ          ← gradient + L2 decay term
    vᵢ  ←  μ · vᵢ  +  lr · gᵢ          ← velocity accumulation
    wᵢ  ←  wᵢ  −  vᵢ                    ← parameter update
    ∇L_wᵢ ← 0                           ← zero-grad in-place

  Hyper-parameters:
    μ (momentum)   = 0.9
    λ (weight decay)= 0.0005
    lr (initial)   = 0.01

  Updated tensors each backward pass:
    d_filter1, d_filter2    (conv weights)
    d_bn1Scale, d_bn1Bias   (BN γ, β — no weight decay)
    d_bn2Scale, d_bn2Bias
    d_fcWeights             (classifier head)
```

Velocity buffers `v_*` are persistent across iterations — this is what enables effective momentum accumulation:

```
  Without momentum (SGD):      noisy, slow convergence
  ─────────────────────────────────────────────────────
   step1 →  step2 ←  step3 →  step4 ←   (oscillating)

  With momentum (μ=0.9):       smooth, directed descent
  ─────────────────────────────────────────────────────
   step1 ──────────────────────────────► (damped, fast)
```

> *Sutskever et al., 2013 — On the importance of initialization and momentum in deep learning* [[ICML Proc.](http://proceedings.mlr.press/v28/sutskever13.html)]

---

## Loss Function — Softmax Cross-Entropy

```
  Forward:
    Softmax:  p_c = exp(z_c) / Σ_k exp(z_k)

    Cross-Entropy:  L = −(1/B) · Σ_n log(p_{n, c*_n} + ε)

  where c*_n is the ground-truth class label for image n.

  Backward (gradient of loss w.r.t. logit z_c):
    ∂L/∂z_c = (p_c − 1[c = c*]) / B
```

cuDNN computes softmax in `CUDNN_SOFTMAX_ACCURATE` mode (numerically stable via max-subtraction before exponentiation). The CE gradient is computed on the CPU from the downloaded softmax output and then uploaded as `d_diffLogits`.

---

## Learning Rate Schedule

A **step decay** schedule reduces the learning rate by 20% at the end of every epoch:

```
  lr_{t+1} = lr_t × 0.8

  Starting lr = 0.01

  Epoch  1: lr = 0.0100
  Epoch  2: lr = 0.0080
  Epoch  3: lr = 0.0064
  Epoch  4: lr = 0.0051
  Epoch  5: lr = 0.0041

                0.010 ┤━━━━━━━━━━━┐
                0.008 ┤           └──────────┐
                0.006 ┤                      └─────────┐
                0.004 ┤                                └────────
                      └───────────────────────────────────────►
                       ep1        ep2        ep3        ep4   ep5
```

---

## Data Augmentation

Two augmentations are applied **on-device** within a single CUDA kernel (`augmentBatchKernel`):

```
  Input Batch  [B, 3, H, W]  (on GPU)
        │
        ▼
  augmentBatchKernel  <<<blocks, threads>>>
  │
  ├── Horizontal Flip  (50% probability per image)
  │     For x in [0, W/2):
  │       swap pixel[x] ↔ pixel[W-1-x]   (all channels)
  │
  └── Brightness Jitter  (additive, ±0.15)
        b ~ Uniform(−0.15, +0.15)  per image
        pixel_new = clamp(pixel + b, 0.0, 1.0)
        │
        ▼
  Augmented Batch  [B, 3, H, W]  (in-place, no extra allocation)
```

Both operations are fused into one kernel launch — threads handle half-width column pairs to enable the in-place swap without race conditions:

```
  Thread assignment:
    threadIdx.x → x ∈ [0, (W+1)/2)
    threadIdx.y → y ∈ [0, H)
    blockIdx.z  → n ∈ [0, B)

  This avoids the need for a temporary buffer.
```

> *Krizhevsky et al., 2012 — ImageNet Classification with Deep CNNs (AlexNet)* [[NIPS](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)] *(introduced GPU augmentation pipeline)*

---

## Training Pipeline

```
  Epoch Loop  (5 epochs default)
       │
       ├─ loadBatch()          read batchSize records from flowers10.bin
       │                       [label (1 byte)] [R,G,B pixels (3×32×32 bytes)]
       │                       normalise: float = uint8 / 255.0
       │
       ├─ dnn.forward()        h_input → GPU → augment → Conv1→BN→ReLU→Pool
       │                       → Conv2→BN→ReLU→Pool → Dropout → FC → Softmax
       │
       ├─ Loss + Accuracy       CE loss on softmax output  (CPU loop)
       │                        argmax for top-1 accuracy
       │
       ├─ dnn.backward()        logit gradients → FC backprop → Dropout backprop
       │                        → Pool2 backprop → BN2 backprop → Conv2 backprop
       │                        → Pool1 backprop → BN1 backprop → Conv1 backprop
       │                        → momentumSGDKernel on all 7 weight tensors
       │
       └─ LR step decay         lr *= 0.8

  After all epochs:
       dnn.saveWeights("flower_model.bin")
```

**Backward pass order (chain rule):**

```
  Softmax CE grad
       │
  FC backprop          (cublasSgemm for dW_fc and d_diffPool2Drop)
       │
  Dropout backprop     (cudnnDropoutBackward — uses reserve space)
       │
  Pool2 backprop       (cudnnPoolingBackward)
  BN2 backprop         (cudnnBatchNormalizationBackward)
  Conv2 backprop       (cudnnConvolutionBackwardFilter + BackwardData)
       │
  Pool1 backprop
  BN1 backprop
  Conv1 backprop (filter gradients only — no data grad for input)
       │
  Weight update        (applyMomentumSGD kernel per tensor)
```

---

## Binary Dataset Format

`flowers10.bin` is written by `dataset/prepare_dataset.py`:

```
  ┌───────────────────────────────────────────────┐
  │  Header:                                       │
  │    total_images  (int32, 4 bytes)              │
  │                                               │
  │  Per image  (repeated total_images times):    │
  │    label     (uint8,  1 byte  — class 0..9)  │
  │    R channel (uint8, 1024 bytes — 32×32)      │
  │    G channel (uint8, 1024 bytes — 32×32)      │
  │    B channel (uint8, 1024 bytes — 32×32)      │
  │                                               │
  │  Total size ≈ 800 images × 3073 B ≈ 2.4 MB   │
  └───────────────────────────────────────────────┘

  Source: Oxford 17-Flowers dataset (first 10 classes, 80 imgs/class)
  Resize: PIL bilinear 32×32
  Layout: CHW  (channel-first, matching cuDNN NCHW tensors)
```

---

## File Structure

```
classification/
├── main.cu                   Training entry point
├── compile.ps1               NVCC build script (Windows PowerShell)
├── dataset/
│   └── prepare_dataset.py    Downloads 17-Flowers, writes flowers10.bin
└── network/
    ├── cudnn_helper.h        CUDA / cuDNN / cuBLAS error-check macros
    ├── utilities.cu          GpuTimer (cudaEvent) · printDeviceInformation
    ├── augmentation.cu       DataAugmenter class: H-flip + brightness jitter
    └── network.cu            ImageClassifier class
                              (2-stage CNN + Dropout + FC + Softmax)
                              forward() · backward() · saveWeights()
```

---

## Getting Started

### Prerequisites

- NVIDIA GPU (Pascal / sm_60 or newer)
- CUDA Toolkit 11.x+
- cuDNN 8.x
- Python 3.8+ with `pip install requests numpy pillow`

### 1 — Prepare Dataset

> **Dataset notice:** The Oxford 17-Category Flower Dataset is NOT included
> in this repository. Download it from:
> https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
> Extract the `jpg/` folder into `classification/dataset/` before running.

```powershell
cd classification\dataset
python prepare_dataset.py
# Best case:  Successfully created flowers10.bin.
# Worst case: Error: Image directory 'jpg' not found.
cd ..
```

### 2 — Compile

```powershell
cd classification
.\compile.ps1
# Compiles with nvcc -O3 -use_fast_math -lcudnn -lcublas
# Output: dnn_classifier.exe
```

### 3 — Train

```powershell
.\dnn_classifier.exe
# Epoch 1/5 - Loss: 2.1543 - Accuracy: 18.75% - LR: 0.008 - Time: 3.2s
# Epoch 2/5 - Loss: 1.7821 - Accuracy: 36.25% - LR: 0.0064 - Time: 3.0s
# ...
# Model + BN states saved to flower_model.bin
```

---

## Reference Papers

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 1 | **Batch Normalization: Accelerating Deep Network Training** | Ioffe & Szegedy | ICML 2015 | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) |
| 2 | **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** | Srivastava et al. | JMLR 2014 | [JMLR](https://jmlr.org/papers/v15/srivastava14a.html) |
| 3 | **Delving Deep into Rectifiers (He Initialization)** | He et al. | ICCV 2015 | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) |
| 4 | **On the importance of initialization and momentum in deep learning** | Sutskever et al. | ICML 2013 | [ICML Proc.](http://proceedings.mlr.press/v28/sutskever13.html) |
| 5 | **ImageNet Classification with Deep CNNs (AlexNet)** | Krizhevsky et al. | NeurIPS 2012 | [NIPS](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 6 | **Very Deep Convolutional Networks (VGGNet)** | Simonyan & Zisserman | ICLR 2015 | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |
| 7 | **Gradient-Based Learning Applied to Document Recognition (LeNet)** | LeCun et al. | IEEE 1998 | [IEEE](https://doi.org/10.1109/5.726791) |
| 8 | **cuDNN: Efficient Primitives for Deep Learning** | Chetlur et al. | 2014 | [arXiv:1410.0759](https://arxiv.org/abs/1410.0759) |
| 9 | **Oxford 17 Category Flower Dataset** | Nilsback & Zisserman | BMVC 2006 | [Project](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) |

> **Dataset attribution:** The Oxford 17-Flowers dataset (cited above) is not
> included in this repository and is NOT covered by the MIT License.
> It must be obtained independently from the official Oxford VGG source.

---

## License

The source code in this module is distributed under the **MIT License**.
See [`../LICENSE`](../LICENSE) for the full text, including dataset and
algorithm attribution notices.

> The MIT License covers source code only. The Oxford 17-Flowers dataset
> is not included and is subject to its own separate license.
