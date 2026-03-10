#!/usr/bin/env python3
"""
CuVision-Engine | Classification — Dataset Preparation Script
=============================================================

This script prepares an image-classification binary dataset compatible
with the CuVision-Engine classification module.

DATASET NOTICE
--------------
This script supports the Oxford 17-Category Flower Dataset.
You must obtain that dataset yourself, subject to its own license:
  https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

  Nilsback, M-E. and Zisserman, A. (2006)
  "A Visual Vocabulary for Flower Classification"
  Proc. IEEE Computer Vision and Pattern Recognition (CVPR)

The dataset is NOT redistributed here and is NOT covered by this
project's license. Please comply with Oxford VGG's terms of use.

USAGE
-----
1. Download and extract the Oxford 17-Flowers dataset manually.
2. Place the extracted `jpg/` folder next to this script.
3. Run:  python prepare_dataset.py

OUTPUT FORMAT  (flowers10.bin)
-------------------------------
  [total_images : int32]
  Per image:
    [label   : uint8   — class index 0..9]
    [R plane : uint8   — 32×32 = 1024 bytes]
    [G plane : uint8   — 32×32 = 1024 bytes]
    [B plane : uint8   — 32×32 = 1024 bytes]

EXAMPLES
--------
Best-case run (all images present, correct format):

  $ python prepare_dataset.py
  Processing 10 classes...
  Progress: 800/800 images processed...
  Successfully created flowers10.bin.

Worst-case run (dataset folder missing):

  $ python prepare_dataset.py
  Error: Image directory 'jpg' not found.
  Please download the Oxford 17-Flowers dataset and place the 'jpg/'
  folder in the same directory as this script.
  Dataset preparation aborted.

DEPENDENCIES
------------
  pip install numpy pillow
"""

import os
import glob
import numpy as np
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_CLASSES      = 10
IMAGES_PER_CLASS = 80
IMAGE_SIZE       = 32          # resize target (square)
OUTPUT_FILE      = "flowers10.bin"
IMAGE_DIR        = "jpg"       # expected sub-directory with .jpg files
# ──────────────────────────────────────────────────────────────────────────────


def process_data():
    """Convert raw JPEG images into flowers10.bin for training."""
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory '{IMAGE_DIR}' not found.")
        print("Please download the Oxford 17-Flowers dataset and place the 'jpg/'")
        print("folder in the same directory as this script.")
        return False

    img_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    required = NUM_CLASSES * IMAGES_PER_CLASS

    if len(img_files) < required:
        print(f"Error: Found {len(img_files)} images, need at least {required}.")
        return False

    print(f"Processing {NUM_CLASSES} classes, {IMAGES_PER_CLASS} images each...")

    # Format: [Label (1 byte)][R (1024 bytes)][G (1024 bytes)][B (1024 bytes)]
    with open(OUTPUT_FILE, "wb") as f:
        total_images = NUM_CLASSES * IMAGES_PER_CLASS
        f.write(np.array([total_images], dtype=np.int32).tobytes())

        for class_id in range(NUM_CLASSES):
            start_idx = class_id * IMAGES_PER_CLASS
            for i in range(IMAGES_PER_CLASS):
                img_path = img_files[start_idx + i]
                try:
                    img = Image.open(img_path).convert("RGB").resize(
                        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
                    )
                    # CHW layout (channel-first) to match cuDNN NCHW tensors
                    img_data = np.array(img).transpose(2, 0, 1)
                    f.write(np.array([class_id], dtype=np.uint8).tobytes())
                    f.write(img_data.tobytes())
                except Exception as e:
                    print(f"\nWarning: Skipping {img_path}: {e}")

                processed = class_id * IMAGES_PER_CLASS + i + 1
                if processed % 100 == 0 or processed == total_images:
                    print(f"\rProgress: {processed}/{total_images} images processed...",
                          end="", flush=True)
        print()

    print(f"Successfully created {OUTPUT_FILE}.")
    return True


if __name__ == "__main__":
    if not process_data():
        print("Dataset preparation aborted.")
