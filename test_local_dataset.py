#!/usr/bin/env python
"""Test script for CholecSeg8k local dataset adapter."""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets import build_dataset


def test_local_adapter():
    """Test the local CholecSeg8k adapter."""
    
    # Build the local adapter
    print("Building local CholecSeg8k adapter...")
    dataset = build_dataset("cholecseg8k_local", data_dir="/shared_data0/weiqiuy/datasets/cholecseg8k")
    
    # Test basic properties
    print(f"Dataset tag: {dataset.dataset_tag}")
    print(f"Version: {dataset.version}")
    print(f"Number of classes: {len(dataset.label_ids)}")
    print(f"Recommended canvas: {dataset.recommended_canvas}")
    
    # Test data access
    splits = ["train", "validation", "test"]
    for split in splits:
        n = dataset.total(split)
        print(f"{split} split: {n} examples")
    
    # Test loading an example
    print("\nLoading first training example...")
    example = dataset.get_example("train", 0)
    print(f"  Image: {example['image'].size}")
    print(f"  Color mask: {example['color_mask'].size}")
    print(f"  Video ID: {example['video_id']}")
    print(f"  Frame ID: {example['frame_id']}")
    
    # Test tensor conversion
    print("\nConverting to tensors...")
    img_t, lab_t = dataset.example_to_tensors(example)
    print(f"  Image tensor: {img_t.shape}, dtype={img_t.dtype}, range=[{img_t.min():.2f}, {img_t.max():.2f}]")
    print(f"  Label tensor: {lab_t.shape}, dtype={lab_t.dtype}, unique values={len(lab_t.unique())}")
    
    # Test presence vector
    print("\nComputing presence vector...")
    presence = dataset.labels_to_presence_vector(lab_t)
    print(f"  Presence vector: {presence.shape}")
    present_classes = [dataset.id2label[i+1] for i, p in enumerate(presence) if p == 1]
    print(f"  Present classes: {present_classes}")
    
    # Test point sampling
    print("\nTesting point sampling...")
    for class_id in [2, 5, 10]:  # Liver, Grasper, Gallbladder
        point = dataset.sample_point_in_mask(lab_t, class_id, strategy="centroid")
        if point:
            print(f"  {dataset.id2label[class_id]}: point at {point}")
        else:
            print(f"  {dataset.id2label[class_id]}: not present")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_local_adapter()