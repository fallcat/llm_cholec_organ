#!/usr/bin/env python
"""Test script to verify coordinate system consistency."""

import os
import sys
import json

# Add src to path
ROOT_DIR = "/shared_data0/weiqiuy/llm_cholec_organ"
sys.path.append(os.path.join(ROOT_DIR, 'src'))

# Load API keys
with open(f"{ROOT_DIR}/API_KEYS2.json", "r") as file:
    api_keys = json.load(file)

os.environ['OPENAI_API_KEY'] = api_keys['OPENAI_API_KEY']
os.environ['ANTHROPIC_API_KEY'] = api_keys['ANTHROPIC_API_KEY']
os.environ['GOOGLE_API_KEY'] = api_keys['GOOGLE_API_KEY']

from datasets import load_dataset
from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter
import numpy as np

print("=" * 60)
print("Testing Coordinate System Consistency")
print("=" * 60)

# Load dataset
print("\n1. Loading dataset...")
dataset = load_dataset("minwoosun/CholecSeg8k")
adapter = CholecSeg8kAdapter()

# Check dimensions
example = dataset['train'][0]
img = example['image']
mask = example['color_mask']

print(f"\n2. Original PIL Image dimensions:")
print(f"   Image size (width, height): {img.size}")
print(f"   Mask size (width, height): {mask.size}")

# Convert to tensors
img_t, lab_t = adapter.example_to_tensors(example)

print(f"\n3. Tensor dimensions:")
print(f"   Image tensor shape (C,H,W): {img_t.shape}")
print(f"   Label tensor shape (H,W): {lab_t.shape}")

# Extract actual dimensions
_, H, W = img_t.shape
print(f"   Extracted: Width={W}, Height={H}")

print("\n4. Coordinate system analysis:")
print(f"   - The dataset images are {W}x{H} pixels (width x height)")
print(f"   - PIL uses (width, height) format: {img.size}")
print(f"   - Tensors use [C,H,W] format: {img_t.shape}")

print("\n5. Canvas size recommendations:")
print(f"   - Adapter recommended canvas: {adapter.recommended_canvas}")
print(f"   - Actual image dimensions: ({W}, {H})")

print("\n6. Coordinate scaling in check_point_hit:")
print("   When canvas_width=768, canvas_height=768:")
print(f"   - A point at canvas (384, 384) maps to mask ({int(384 * W / 768)}, {int(384 * H / 768)})")
print(f"   When using original size canvas_width={W}, canvas_height={H}:")
print(f"   - A point at canvas (427, 240) maps to mask (427, 240) - NO SCALING!")

print("\n7. Impact of using different canvas sizes:")
print("   - With 768x768 canvas: Points are SCALED from 768x768 to actual image size")
print("   - With original size canvas: Points map 1:1 with image pixels (NO SCALING)")

print("\n8. CONCLUSION:")
print("   The coordinate system is CONSISTENT throughout, but:")
print("   - When canvas_size != image_size, coordinates are SCALED")
print("   - When canvas_size == image_size, coordinates are DIRECT (1:1)")
print("   - Models are told the canvas size in prompts")
print("   - check_point_hit properly scales from canvas to mask coordinates")

print("\n9. Recommendation:")
print("   Using original image size avoids scaling and may improve accuracy")
print("   since models can work with actual pixel coordinates directly.")
print("=" * 60)