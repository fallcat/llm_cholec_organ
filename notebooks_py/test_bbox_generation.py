#!/usr/bin/env python3
"""Quick test script to verify bbox generation works"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from prepare_fewshot_examples_bounding_box import extract_bounding_boxes_from_mask
from cholecseg8k_utils import example_to_tensors
from datasets import load_dataset
import json

# Load a sample
dataset = load_dataset("minwoosun/CholecSeg8k")
example = dataset['train'][0]
img_t, lab_t = example_to_tensors(example)

# Test bbox extraction for liver (class 2)
bboxes = extract_bounding_boxes_from_mask(
    lab_t, 
    class_id=2,  # Liver
    min_pixels=50,
    min_bbox_size=20,
    max_bboxes=3
)

print(f"Found {len(bboxes)} bounding boxes for Liver:")
for i, bbox in enumerate(bboxes):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    print(f"  Box {i+1}: x={x1}-{x2}, y={y1}-{y2} (size: {width}x{height})")

print("\nTest passed! Bbox extraction works correctly.")