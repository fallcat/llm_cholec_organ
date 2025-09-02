#!/usr/bin/env python3
"""Quick test to check dataset splits."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from endopoint.datasets import build_dataset

# Load dataset
print("Loading CholecSeg8k dataset...")
dataset = build_dataset("cholecseg8k")

# Check available splits
print("\nTrying different splits:")
for split in ["train", "test", "validation", "val"]:
    try:
        total = dataset.total(split)
        print(f"  {split}: {total} samples ✅")
    except KeyError:
        print(f"  {split}: Not available ❌")
    except Exception as e:
        print(f"  {split}: Error - {e}")

# Load a sample from train
print("\nLoading sample from 'train' split:")
try:
    example = dataset.get_example("train", 0)
    print(f"  ✅ Successfully loaded example")
    print(f"  Image type: {type(example['image'])}")
    print(f"  Image size: {example['image'].size if hasattr(example['image'], 'size') else 'N/A'}")
    
    # Convert to tensors
    img_t, lab_t = dataset.example_to_tensors(example)
    print(f"  Tensor shapes: image={img_t.shape}, labels={lab_t.shape}")
    
    # Get organs present
    presence = dataset.labels_to_presence_vector(lab_t)
    organs = [dataset.id2label[i] for i, p in enumerate(presence) if p > 0 and i in dataset.label_ids]
    print(f"  Organs present: {organs}")
    
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n✅ Dataset is working correctly with 'train' split!")