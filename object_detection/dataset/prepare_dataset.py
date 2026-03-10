#!/usr/bin/env python3
"""
CuVision-Engine | Object Detection — Dataset Preparation Script
===============================================================

This script prepares an object-detection binary dataset compatible
with the CuVision-Engine object-detection module.

DATASET NOTICE
--------------
This script supports the Pascal VOC 2007 dataset.
You must obtain that dataset yourself, subject to its own license:
  http://host.robots.ox.ac.uk/pascal/VOC/

  Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and
  Zisserman, A. (2010).  "The Pascal Visual Object Classes (VOC)
  Challenge."  IJCV 88(2):303–338.
  https://doi.org/10.1007/s11263-009-0275-4

The dataset is NOT redistributed here and is NOT covered by this
project's license. Please comply with PASCAL VOC's terms of use.

USAGE
-----
1. Download VOCtrainval_06-Nov-2007.tar from the official PASCAL VOC
   website and extract it. You should have a VOCdevkit/ directory.
2. Place VOCdevkit/ next to this script.
3. Run:  python prepare_dataset.py

OUTPUT FORMAT  (od_voc2007.bin)
---------------------------------
  Header:
    [num_images  : int32]
    [IMG_H       : int32]   (300)
    [IMG_W       : int32]   (300)
    [num_classes : int32]   (20)

  Per image record:
    [num_boxes : int32]
    [num_boxes × 5 floats: class_idx  x1  y1  x2  y2  (normalised 0-1)]
    [C × H × W uint8 pixels — RGB, CHW layout]

VOC CLASSES (20):
  aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
  diningtable, dog, horse, motorbike, person, pottedplant, sheep,
  sofa, train, tvmonitor

EXAMPLES
--------
Best-case run (VOCdevkit/ present, correctly structured):

  $ python prepare_dataset.py
  Found valid images in trainval split.
  Writing valid images → od_voc2007.bin
  Done!  Saved → od_voc2007.bin

Worst-case run (dataset directory missing):

  $ python prepare_dataset.py
  [ERROR] VOCdevkit/ directory not found.
  Please download Pascal VOC 2007 from:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
  and extract VOCtrainval_06-Nov-2007.tar next to this script.
  [ABORT] Dataset preparation failed.

DEPENDENCIES
------------
  pip install numpy pillow
"""

import os
import struct
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_H, IMG_W  = 300, 300
MAX_BOXES     = 30
OUTPUT_FILE   = "od_voc2007.bin"
VOC_ROOT      = "VOCdevkit"

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}
NUM_CLASSES  = len(VOC_CLASSES)
# ──────────────────────────────────────────────────────────────────────────────


def parse_annotation(xml_path, img_w, img_h):
    """Return list of [class_idx, x1_norm, y1_norm, x2_norm, y2_norm]."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip().lower()
        if cls_name not in CLASS_TO_IDX:
            continue
        diff = obj.find("difficult")
        if diff is not None and int(diff.text) == 1:
            continue
        bnd = obj.find("bndbox")
        x1 = max(0.0, float(bnd.find("xmin").text)) / img_w
        y1 = max(0.0, float(bnd.find("ymin").text)) / img_h
        x2 = min(1.0, float(bnd.find("xmax").text)) / img_w
        y2 = min(1.0, float(bnd.find("ymax").text)) / img_h
        if x2 > x1 and y2 > y1:
            boxes.append([CLASS_TO_IDX[cls_name], x1, y1, x2, y2])
    return boxes[:MAX_BOXES]


def build_binary():
    voc_path  = os.path.join(VOC_ROOT, "VOC2007")
    img_dir   = os.path.join(voc_path, "JPEGImages")
    ann_dir   = os.path.join(voc_path, "Annotations")
    split_txt = os.path.join(voc_path, "ImageSets", "Main", "trainval.txt")

    if not os.path.exists(VOC_ROOT):
        print(f"[ERROR] {VOC_ROOT}/ directory not found.")
        print("Please download Pascal VOC 2007 from:")
        print("  http://host.robots.ox.ac.uk/pascal/VOC/voc2007/")
        print("and extract VOCtrainval_06-Nov-2007.tar next to this script.")
        return False

    if not os.path.exists(split_txt):
        print(f"[ERROR] Split file not found: {split_txt}")
        return False

    with open(split_txt) as f:
        ids = [l.strip() for l in f if l.strip()]

    records = []
    for img_id in ids:
        img_path = os.path.join(img_dir, img_id + ".jpg")
        xml_path = os.path.join(ann_dir, img_id + ".xml")
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            boxes = parse_annotation(xml_path, orig_w, orig_h)
            if len(boxes) == 0:
                continue
            img_resized = img.resize((IMG_W, IMG_H), Image.BILINEAR)
            records.append((img_resized, boxes))
        except Exception as e:
            print(f"  [WARN] Skipping {img_id}: {e}")

    num_images = len(records)
    if num_images == 0:
        print("[ERROR] No valid records found. Check your VOCdevkit/ structure.")
        return False

    print(f"Found valid images in trainval split.")
    print(f"Writing valid images → {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "wb") as f:
        f.write(struct.pack("iiii", num_images, IMG_H, IMG_W, NUM_CLASSES))
        for idx, (img, boxes) in enumerate(records):
            f.write(struct.pack("i", len(boxes)))
            for box in boxes:
                f.write(struct.pack("fffff",
                    float(box[0]), box[1], box[2], box[3], box[4]))
            arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
            f.write(arr.tobytes())
            if (idx + 1) % 200 == 0 or (idx + 1) == num_images:
                print(f"\r  {idx+1}/{num_images} images written...", end="", flush=True)

    print(f"\nDone!  Saved → {OUTPUT_FILE}")
    return True


if __name__ == "__main__":
    if not build_binary():
        print("[ABORT] Dataset preparation failed.")
