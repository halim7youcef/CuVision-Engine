#!/usr/bin/env python3
"""
CuVision-Engine | Segmentation — Dataset Preparation Script
===========================================================

This script prepares a semantic-segmentation binary dataset compatible
with the CuVision-Engine segmentation module.

DATASET NOTICE
--------------
This script supports the Oxford-IIIT Pet Dataset.
You must obtain that dataset yourself, subject to its own license
(Creative Commons Attribution-ShareAlike 4.0):
  https://www.robots.ox.ac.uk/~vgg/data/pets/

  Parkhi, O. M., Vedaldi, A., Zisserman, A. and Jawahar, C. V. (2012).
  "Cats and Dogs."  IEEE Conference on Computer Vision and Pattern
  Recognition (CVPR).

The dataset is NOT redistributed here and is NOT covered by this
project's license. Please comply with the CC-BY-SA 4.0 terms.

USAGE
-----
1. Download images.tar.gz and annotations.tar.gz from the Oxford-IIIT
   Pet website and extract them. You should have:
     images/        ← pet JPEG images
     annotations/   ← trimap PNG masks (trimaps/ sub-directory)
2. Place both folders next to this script.
3. Run:  python prepare_dataset.py

OUTPUT FORMAT  (seg_pets.bin)
-------------------------------
  Header:
    [num_images  : int32]
    [IMG_H       : int32]   (256)
    [IMG_W       : int32]   (256)
    [num_classes : int32]   (3)

  Per image record:
    [C × H × W uint8 — RGB image, CHW layout]
    [H × W     uint8 — class-index mask]

SEGMENTATION CLASSES (3):
  0 → background / border
  1 → pet foreground
  2 → boundary (thin region between pet and background)

Trimap remapping (Oxford convention → our convention):
  Oxford 1 (foreground) → class 1 (pet)
  Oxford 2 (background) → class 0 (background)
  Oxford 3 (boundary)   → class 2 (boundary)

EXAMPLES
--------
Best-case run (images/ and annotations/ present):

  $ python prepare_dataset.py
  Found trimap annotations.
  Writing valid image-mask pairs → seg_pets.bin
  Done!  Saved → seg_pets.bin

Worst-case run (dataset folders missing):

  $ python prepare_dataset.py
  [ERROR] Trimap directory not found: annotations/trimaps
  Please download the Oxford-IIIT Pet dataset from:
    https://www.robots.ox.ac.uk/~vgg/data/pets/
  Extract images.tar.gz  → images/
  Extract annotations.tar.gz → annotations/
  [ABORT] Dataset preparation failed.

DEPENDENCIES
------------
  pip install numpy pillow
"""

import os
import struct

import numpy as np
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_H, IMG_W  = 256, 256
NUM_CLASSES   = 3
OUTPUT_FILE   = "seg_pets.bin"
IMAGES_DIR    = "images"
ANNOTS_DIR    = "annotations"
# ──────────────────────────────────────────────────────────────────────────────

# Oxford Pet trimap pixel → our class index
TRIMAP_REMAP = {1: 1, 2: 0, 3: 2}


def trimap_to_mask(trimap_path: str) -> np.ndarray:
    """Load a .png trimap and return a uint8 class-index mask (H, W)."""
    tri = np.array(Image.open(trimap_path).convert("L"))
    mask = np.zeros_like(tri, dtype=np.uint8)
    for src, dst in TRIMAP_REMAP.items():
        mask[tri == src] = dst
    return mask


def build_binary():
    trimap_dir = os.path.join(ANNOTS_DIR, "trimaps")

    if not os.path.exists(trimap_dir):
        print(f"[ERROR] Trimap directory not found: {trimap_dir}")
        print("Please download the Oxford-IIIT Pet dataset from:")
        print("  https://www.robots.ox.ac.uk/~vgg/data/pets/")
        print("Extract images.tar.gz  → images/")
        print("Extract annotations.tar.gz → annotations/")
        return False

    trimap_paths = sorted([
        os.path.join(trimap_dir, f)
        for f in os.listdir(trimap_dir)
        if f.lower().endswith(".png")
    ])

    if not trimap_paths:
        print("[ERROR] No trimap .png files found in annotations/trimaps/")
        return False

    records = []
    for tri_path in trimap_paths:
        stem = os.path.splitext(os.path.basename(tri_path))[0]
        img_path = os.path.join(IMAGES_DIR, stem + ".jpg")
        if not os.path.exists(img_path):
            img_path_png = os.path.join(IMAGES_DIR, stem + ".png")
            if os.path.exists(img_path_png):
                img_path = img_path_png
            else:
                continue
        try:
            img  = Image.open(img_path).convert("RGB").resize(
                (IMG_W, IMG_H), Image.BILINEAR)
            mask = trimap_to_mask(tri_path)
            mask = np.array(
                Image.fromarray(mask).resize((IMG_W, IMG_H), Image.NEAREST)
            )
            records.append((img, mask))
        except Exception as e:
            print(f"  [WARN] Skipping {stem}: {e}")

    num_images = len(records)
    if num_images == 0:
        print("[ERROR] No valid image-mask pairs found.")
        return False

    print(f"Found trimap annotations.")
    print(f"Writing valid image-mask pairs → {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "wb") as f:
        f.write(struct.pack("iiii", num_images, IMG_H, IMG_W, NUM_CLASSES))
        for idx, (img, mask) in enumerate(records):
            arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
            f.write(arr.tobytes())
            f.write(mask.astype(np.uint8).tobytes())
            if (idx + 1) % 500 == 0 or (idx + 1) == num_images:
                print(f"\r  {idx+1}/{num_images} pairs written...", end="", flush=True)

    print(f"\nDone!  Saved → {OUTPUT_FILE}")
    return True


if __name__ == "__main__":
    if not build_binary():
        print("[ABORT] Dataset preparation failed.")
