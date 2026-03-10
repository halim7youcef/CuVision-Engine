# CuVision-Engine

```
  ██████╗██╗   ██╗██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗
 ██╔════╝██║   ██║██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║
 ██║     ██║   ██║██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║
 ██║     ██║   ██║╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║
 ╚██████╗╚██████╔╝ ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║
  ╚═════╝ ╚═════╝   ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                    E  N  G  I  N  E
```

**High-Performance Native Computer Vision for the Edge.**

CuVision-Engine is a low-latency Computer Vision framework written in C++ and CUDA. By targeting cuDNN and cuBLAS primitives directly, it achieves peak hardware utilisation on NVIDIA GPUs — no PyTorch, no TensorFlow, no framework overhead.

---

## Project State  *(March 2026)*

```
  Module                  Status          Technique
  ─────────────────────────────────────────────────────────────────────────
  Classification          ██████████ 100%  2-Stage CNN + BN + Dropout
  Object Detection        ████████░░  85%  RetinaNet-FPN (ResNet backbone)
  Segmentation            ████████░░  85%  Attention U-Net + ASPP Bottleneck
  ─────────────────────────────────────────────────────────────────────────
  TensorRT Integration    ░░░░░░░░░░   0%  (planned)
  Instance Segmentation   ░░░░░░░░░░   0%  (planned)
```

---

## 🗺️ Engine Architecture — Top-Level Overview

```
 ┌────────────────────────────── CuVision-Engine ─────────────────────────────┐
 │                                                                             │
 │   ┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
 │   │   CLASSIFICATION    │  │  OBJECT DETECTION    │  │  SEGMENTATION    │  │
 │   │                     │  │                      │  │                  │  │
 │   │  Input [B,C,32,32]  │  │  Input [B,C,300,300] │  │  Input [B,C,256  │  │
 │   │         │           │  │         │            │  │        ×256]     │  │
 │   │      2×ConvBlock    │  │  ResNet Backbone     │  │  ResNet Encoder  │  │
 │   │    (BN+ReLU+Pool)   │  │  (4 stages, stride2) │  │  (4 stages)      │  │
 │   │         │           │  │         │            │  │       │          │  │
 │   │      Dropout        │  │    FPN Neck          │  │  Dilated ASPP    │  │
 │   │         │           │  │  (P2, P3, P4 @ 256)  │  │  Bottleneck      │  │
 │   │      FC Layer       │  │         │            │  │       │          │  │
 │   │         │           │  │  Shared Det. Head    │  │  Attn U-Net      │  │
 │   │      Softmax        │  │  (cls + reg towers)  │  │  Decoder ×3      │  │
 │   │         │           │  │         │            │  │       │          │  │
 │   │  Cross-Entropy     │  │  Focal + SmoothL1    │  │  CE + Dice Loss  │  │
 │   │  Loss              │  │  Loss                │  │                  │  │
 │   └─────────────────────┘  └──────────────────────┘  └──────────────────┘  │
 │                                                                             │
 │   ┌─────────────────────────── SHARED FOUNDATION ─────────────────────────┐ │
 │   │  Momentum-SGD kernel  │  He init  │  BN (train/infer)  │  Dropout    │ │
 │   │  cuRAND augmentation  │  cuBLAS   │  cudnnConvolution  │  GpuTimer   │ │
 │   └────────────────────────────────────────────────────────────────────────┘ │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
CU_NN/
│
├── README.md                         ← You are here
├── LICENSE                           MIT
│
├── classification/                   ✅  COMPLETE
│   ├── main.cu                       Training loop  (CE loss, step-LR)
│   ├── compile.ps1                   NVCC build → dnn_classifier.exe
│   ├── README.md                     Architecture + math + paper refs
│   ├── dataset/
│   │   └── prepare_dataset.py        Oxford 17-Flowers → flowers10.bin  *
│   └── network/
│       ├── cudnn_helper.h            Error macros (CUDA / cuDNN / cuBLAS)
│       ├── utilities.cu              GpuTimer, printDeviceInformation
│       ├── augmentation.cu           H-flip + brightness jitter kernel
│       └── network.cu                ImageClassifier class (fwd + bwd + save)
│
├── object_detection/                 🔶  IN PROGRESS (network complete)
│   ├── main.cu                       Training loop  (IoU match, cosine-LR)
│   ├── compile.ps1                   NVCC build → od_detector.exe
│   ├── README.md                     Architecture + math + paper refs
│   ├── dataset/
│   │   └── prepare_dataset.py        Pascal VOC 2007 → od_voc2007.bin  *
│   └── network/
│       ├── cudnn_helper.h            Error macros
│       ├── utilities.cu              Smooth-L1, Focal loss, NMS, Anchors
│       ├── augmentation.cu           H-flip (bbox-aware), jitter, noise, cutout
│       └── network.cu                ObjectDetector (backbone→FPN→head)
│
└── segmentation/                     🔶  IN PROGRESS (network complete)
    ├── main.cu                       Training loop  (mIoU tracking, poly-LR)
    ├── compile.ps1                   NVCC build → seg_unet.exe
    ├── README.md                     Architecture + math + paper refs
    ├── dataset/
    │   └── prepare_dataset.py        Oxford-IIIT Pet → seg_pets.bin  *
    └── network/
        ├── cudnn_helper.h            Error macros
        ├── utilities.cu              Pixel CE, Dice loss, mIoU, weight I/O
        ├── augmentation.cu           Paired H/V flip, elastic deform, noise
        └── network.cu                SegmentationNet (encoder→ASPP→attn decoder)

* Dataset files are NOT included. See "Dataset Attribution" below.
```

---

## Model Complexity Comparison

```
  ── Parameters (log scale) ─────────────────────────────────────────────────

  Classification    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~60 K
  Object Detection  ████████████████████░░░░░░░░░  ~21 M  (ResNet + FPN + Head)
  Segmentation      ████████████████████████░░░░░  ~31 M  (ResNet + ASPP + UNet)

  ── Input Resolution ────────────────────────────────────────────────────────

  Classification    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░  32×32
  Object Detection  ████████████████████░░░░░░░░░  300×300
  Segmentation      █████████████████████████░░░░  256×256

  ── Anchors / Output Pixels ─────────────────────────────────────────────────

  Classification    ──   (single label per image)
  Object Detection  ~60,000 anchors across P2+P3+P4
  Segmentation      65,536 pixels labelled per image
```

---

## CUDA Kernel Summary

```
  Kernel                       Module(s)         Purpose
  ─────────────────────────────────────────────────────────────────────────────
  applyMomentumSGD             All               v ← μv + lr·g;  w ← w − v
  setConst / fillConst         All               GPU buffer zero / fill
  augmentBatchKernel           Classification    H-flip + brightness  (fused)
  horizontalFlipKernel         Object Det.       Paired pixel swap (NCHW)
  colorJitterKernel            Object Det.       Brightness/contrast/saturation
  gaussianNoiseKernel          OD + Seg          Additive noise via cuRAND
  cutoutKernel                 Object Det.       Square region zeroing
  segHFlipKernel               Segmentation      H-flip: image + int mask
  segVFlipKernel               Segmentation      V-flip: image + int mask
  segColorJitterKernel         Segmentation      Color jitter (image only)
  segGaussianNoiseKernel       Segmentation      Noise (image only)
  elasticSampleKernel          Segmentation      Bilinear warp + NN mask sample
  upsample2xKernel             Segmentation      Bilinear 2× upscale
  concatKernel                 Segmentation      Channel-wise concat
  attentionScaleKernel         Segmentation      ψ ⊙ skip features
  focalLossKernel              Object Det.       −α(1−pₜ)^γ log(pₜ)
  smoothL1LossKernel           Object Det.       Huber regression loss
  pixelCrossEntropyLossKernel  Segmentation      Pixel-wise CE + softmax grad
  diceLossFwdKernel            Segmentation      |P∩G| / (|P|+|G|) via atomics
  ─────────────────────────────────────────────────────────────────────────────
  Total custom kernels: 20
```

---

## Technique Comparison Across Modules

```
                         Classification   Detection    Segmentation
  ──────────────────────────────────────────────────────────────────
  Backbone               2-stage CNN      ResNet-4     ResNet-4
  Skip connections       ✗                ✓ (residual) ✓ (residual)
  Multi-scale output     ✗                ✓ (FPN)      ✗
  Attention mechanism    ✗                ✗            ✓ (additive)
  Dilated convolution    ✗                ✗            ✓ (ASPP ×2)
  Batch Normalisation    ✓                ✓            ✓
  Dropout                ✓ (0.5 FC)       ✗            ✗
  He Initialisation      ✓                ✓            ✓
  Momentum SGD           ✓                ✓            ✓
  Weight Decay (L2)      ✓                ✓            ✓
  LR Schedule            Step ×0.8/ep     Cosine       Polynomial^0.9
  Augmentation           flip+bright      flip+jitter  elastic+flip
                                          +noise+cut   +noise+jitter
  Loss                   Softmax CE       Focal+SmoothL1  CE+Dice
  Metric                 Accuracy         mAP (IoU)    mIoU
  Dataset                Oxford Flowers*  Pascal VOC07* Oxford-IIIT Pet*
  Classes                10               20           3
  ──────────────────────────────────────────────────────────────────
  * Third-party datasets. Not distributed here. See LICENSE.
```

---

## Learning Rate Schedules — Visual

```
  ── Classification: Step Decay (×0.8 / epoch) ──────────────────────

  lr  0.010 ┤━━━━━━━━━━━━┐
      0.008 ┤            └━━━━━━━━━━┐
      0.006 ┤                       └━━━━━━━━┐
      0.004 ┤                                └━━━━━━━
            └──────────────────────────────────────► epoch
             1            2          3         4   5


  ── Object Detection: Cosine Annealing ─────────────────────────────

  lr  0.001 ┤━━━━━━┐
      0.000 ┤      ╲      ╲
            ┤       ╲      ╲__
            ┤        ╲         ╲___
            ┤         ╲             ╲______━━━
            └──────────────────────────────────► epoch
             1    2    3    4    5    6   ...  10


  ── Segmentation: Polynomial Decay (power=0.9) ─────────────────────

  lr  5e-4 ┤━━━━┐
      4e-4 ┤    ╲━━━┐
      3e-4 ┤        ╲━━━━┐
      2e-4 ┤             ╲━━━━━┐
      1e-4 ┤                   ╲━━━━━━━━━┐
           ┤                             ╲━━━━━
           └──────────────────────────────────► epoch
            1    3    5    7    9   11   13   15
```

---

## Augmentation Pipeline — Visual

```
  CLASSIFICATION
  ──────────────────────────────────────────────────────────────
  Raw batch [B, 3, 32, 32]
       │
       └──[augmentBatchKernel]── H-flip (50%) + Brightness ±0.15
                                 (1 kernel, in-place, fused)


  OBJECT DETECTION
  ──────────────────────────────────────────────────────────────
  Raw batch [B, 3, 300, 300]                    Bounding Boxes
       │                                              │
       ├──[horizontalFlipKernel]──────────────── x' = 1 − x
       ├──[colorJitterKernel]  bright/contrast/saturation
       ├──[gaussianNoiseKernel] σ ∈ [0, 0.04]  (cuRAND)
       └──[cutoutKernel]  random 15–25% square zeroed


  SEGMENTATION
  ──────────────────────────────────────────────────────────────
  Image [B, 3, 256, 256]    +    Mask [B, 256, 256] (int)
       │                              │
       ├──[segHFlipKernel]────────── pixel swap  +  label swap
       ├──[segVFlipKernel]────────── pixel swap  +  label swap
       ├──[segColorJitterKernel]──── image only
       ├──[segGaussianNoiseKernel]── image only (cuRAND)
       └──[elasticSampleKernel]───── bilinear image  +  NN mask
                                     (displacement α=8, σ=4)
```

---

## Loss Functions — Visual

```
  Focal Loss  (Object Detection — classification head)
  ───────────────────────────────────────────────────────
  weight
  1.0 ┤                                      *
      ┤                             *
  0.8 ┤                    *
      ┤           *
  0.4 ┤    *
      ┤  *
  0.1 ┤ *  ← easy (pₜ=0.9) down-weighted to ~0.01
      └──────────────────────────────────────────► pₜ
       0.0  0.1  0.3  0.5  0.7  0.9  1.0

  (1-pₜ)^γ with γ=2.0: easy negatives → near-zero gradient


  Smooth-L1  (Object Detection — regression head)
  ───────────────────────────────────────────────────────
  loss
  2.0 ┤              /  ← linear (|δ|−0.5)
      ┤            /
  1.0 ┤          /
      ┤       ╭──╮  ← quadratic (0.5δ²)
  0.0 ┤──────╯    ╰──────
      └──────────────────────────────────────────► δ
       -2    -1    0    1    2


  Dice Loss  (Segmentation)
  ───────────────────────────────────────────────────────
  Dice = 1 − (2|P∩G|) / (|P|+|G|)

  Overlap  |P∩G|/|P∪G|   vs   Dice score
  0.0 → Dice = 1.0   (worst)
  0.5 → Dice = 0.67
  0.8 → Dice = 0.33
  1.0 → Dice = 0.0   (perfect overlap)
```

---

## Development Roadmap

```
  ✅  DONE ──────────────────────────────────────────────────────────────
  [x]  2-Stage CNN Classifier (Conv→BN→ReLU→Pool ×2 + FC)
  [x]  Batch Normalization (training + inference running stats)
  [x]  Dropout Regularization (50%, inverted)
  [x]  He (Kaiming) Normal weight initialisation
  [x]  Custom Momentum-SGD CUDA kernel (per-param velocity)
  [x]  L2 Weight Decay (fused into SGD kernel)
  [x]  GPU Timer (cudaEvent benchmark utility)
  [x]  Oxford 17-Flowers dataset loader (binary format)
  [x]  CUDA augmentation: H-flip + brightness jitter
  [x]  ResNet-style backbone (4 stages, residual blocks)
  [x]  Feature Pyramid Network (FPN) neck
  [x]  RetinaNet detection head (shared cls + reg towers)
  [x]  Sigmoid Focal Loss (α=0.25, γ=2.0)
  [x]  Smooth-L1 (Huber) regression loss
  [x]  Anchor generation (multi-scale, multi-ratio)
  [x]  IoU-based anchor matching + delta encoding
  [x]  Non-Maximum Suppression (host-side)
  [x]  Object detection augmentation (flip/jitter/noise/cutout)
  [x]  Pascal VOC 2007 dataset loader
  [x]  Attention U-Net decoder with additive attention gates
  [x]  Dilated convolution bottleneck (ASPP-style, d=2,4)
  [x]  Pixel-wise cross-entropy loss (in-kernel numerically stable)
  [x]  Dice loss (atomicAdd accumulation)
  [x]  mIoU metric (per-class intersection-over-union)
  [x]  Bilinear 2× upsample CUDA kernel
  [x]  Channel-wise concatenation CUDA kernel
  [x]  Elastic deformation augmentation (paired image+mask)
  [x]  Oxford-IIIT Pet dataset loader (trimap → class mask)
  [x]  Polynomial + cosine LR schedules
  [x]  Checkpoint saving (every N epochs)
  [x]  Full documentation (README per module, ASCII diagrams, paper refs)

  🔶  REMAINING ─────────────────────────────────────────────────────────
  [ ]  FPN lateral add + spatial upsample (GPU elementwise kernel)
  [ ]  Detection head full forward per FPN level
  [ ]  Segmentation decoder full bilinear dispatch per stage
  [ ]  mAP evaluation (mean Average Precision, PASCAL VOC protocol)
  [ ]  Inference-only mode (load weights, no grad buffers)
  [ ]  ONNX weight export for TensorRT ingestion
  [ ]  TensorRT integration (INT8 / FP16 engine for edge deployment)
  [ ]  Instance segmentation (Mask R-CNN style)
  [ ]  Multi-GPU support (NCCL all-reduce)
```

---

## Getting Started

### Hardware Requirements

```
  Minimum:   NVIDIA Pascal GPU (sm_60)  |  8 GB VRAM  |  CUDA 11+
  Recommend: NVIDIA Ampere  (sm_86)     |  16 GB VRAM |  CUDA 12+
  Edge:      NVIDIA Jetson Xavier/Orin  |  8 GB unified
```

### Build (Windows PowerShell)

```powershell
# Classification
cd classification && .\compile.ps1 && .\dnn_classifier.exe

# Object Detection
cd object_detection && .\compile.ps1 && .\od_detector.exe

# Segmentation
cd segmentation && .\compile.ps1 && .\seg_unet.exe
```

### Dataset Preparation

> **Important:** Datasets are **not included** in this repository. You must
> obtain each dataset independently from the official sources listed below,
> subject to each dataset's own license.

| Module          | Dataset            | Official Source                                                      |
| --------------- | ------------------ | -------------------------------------------------------------------- |
| Classification  | Oxford 17-Flowers  | https://www.robots.ox.ac.uk/~vgg/data/flowers/17/                   |
| Object Detection| Pascal VOC 2007    | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/                     |
| Segmentation    | Oxford-IIIT Pet    | https://www.robots.ox.ac.uk/~vgg/data/pets/                         |

Once downloaded and extracted, run the preparation script for each module:

```powershell
# Place dataset files in the dataset/ folder of each module first, then:
python classification/dataset/prepare_dataset.py
python object_detection/dataset/prepare_dataset.py
python segmentation/dataset/prepare_dataset.py
```

---

## Reference Papers — Full Engine

### Algorithm Papers

| Module          | Paper                                 | Authors              | Venue        | Link                                                                                       |
| --------------- | ------------------------------------- | -------------------- | ------------ | ------------------------------------------------------------------------------------------ |
| All             | **Batch Normalization**         | Ioffe & Szegedy      | ICML 2015    | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)                                          |
| All             | **He (Kaiming) Initialization** | He et al.            | ICCV 2015    | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)                                          |
| All             | **Momentum SGD**                | Sutskever et al.     | ICML 2013    | [ICML](http://proceedings.mlr.press/v28/sutskever13.html)                                     |
| All             | **cuDNN: Efficient Primitives** | Chetlur et al.       | 2014         | [arXiv:1410.0759](https://arxiv.org/abs/1410.0759)                                            |
| Classification  | **Dropout**                     | Srivastava et al.    | JMLR 2014    | [JMLR](https://jmlr.org/papers/v15/srivastava14a.html)                                        |
| Classification  | **AlexNet**                     | Krizhevsky et al.    | NeurIPS 2012 | [NIPS](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| Classification  | **VGGNet**                      | Simonyan & Zisserman | ICLR 2015    | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)                                            |
| Detection + Seg | **Deep Residual Learning**      | He et al.            | CVPR 2016    | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)                                          |
| Detection       | **Feature Pyramid Networks**    | Lin et al.           | CVPR 2017    | [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)                                          |
| Detection       | **Focal Loss / RetinaNet**      | Lin et al.           | ICCV 2017    | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)                                          |
| Detection       | **SSD**                         | Liu et al.           | ECCV 2016    | [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)                                          |
| Detection       | **Fast R-CNN (Smooth-L1)**      | Girshick             | ICCV 2015    | [arXiv:1504.08083](https://arxiv.org/abs/1504.08083)                                          |
| Detection       | **Cutout**                      | DeVries & Taylor     | 2017         | [arXiv:1708.04552](https://arxiv.org/abs/1708.04552)                                          |
| Segmentation    | **U-Net**                       | Ronneberger et al.   | MICCAI 2015  | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)                                          |
| Segmentation    | **Attention U-Net**             | Oktay et al.         | MIDL 2018    | [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)                                          |
| Segmentation    | **DeepLab v3+**                 | Chen et al.          | ECCV 2018    | [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)                                          |
| Segmentation    | **Dilated Convolutions**        | Yu & Koltun          | ICLR 2016    | [arXiv:1511.07122](https://arxiv.org/abs/1511.07122)                                          |
| Segmentation    | **V-Net / Dice Loss**           | Milletari et al.     | 3DV 2016     | [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)                                          |
| Segmentation    | **Elastic Deformation**         | Simard et al.        | ICDAR 2003   | [IEEE](https://doi.org/10.1109/ICDAR.2003.1227801)                                            |

### Dataset Attribution

> **Note:** No dataset files are included in this repository. The following
> datasets are cited for attribution only; they must be downloaded
> independently and are subject to their own respective licenses.

| Module          | Dataset                | Authors              | License                   | Source                                                    |
| --------------- | ---------------------- | -------------------- | ------------------------- | --------------------------------------------------------- |
| Classification  | Oxford 17-Flowers      | Nilsback & Zisserman (BMVC 2006) | Research/Non-commercial  | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) |
| Object Detection| Pascal VOC 2007        | Everingham et al. (IJCV 2010)    | Research/Non-commercial  | [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)    |
| Segmentation    | Oxford-IIIT Pet        | Parkhi et al. (CVPR 2012)        | CC-BY-SA 4.0             | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/pets/) |

---

## License

The source code in this repository is distributed under the **MIT License**.
See [`LICENSE`](LICENSE) for the full text, including:

- Third-party dataset attribution and license notices
- Third-party algorithm and model attribution notices

> **Important:** The MIT License covers only the source code. Dataset files
> (none of which are included here) are subject to their own separate licenses.
