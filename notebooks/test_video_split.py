#!/usr/bin/env python
"""Test script for video-based splitting in CholecSeg8kLocalAdapter."""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets import build_dataset

# Load the dataset with video-based splitting
print("Loading CholecSeg8k with video-based splits...")
dataset = build_dataset(
    "cholecseg8k_local", 
    data_dir="/shared_data0/weiqiuy/datasets/cholecseg8k"
)

print("\nDataset loaded successfully!")
print(f"Dataset: {dataset.dataset_tag}")
print(f"Train examples: {dataset.total('train')}")
print(f"Validation examples: {dataset.total('validation')}")
print(f"Test examples: {dataset.total('test')}")

# Get video split information
video_splits = dataset.get_video_splits()

print("\n" + "=" * 60)
print("VIDEO-BASED SPLITS:")
print("=" * 60)

for split_name in ['train', 'validation', 'test']:
    video_ids = video_splits[split_name]
    print(f"\n{split_name.upper()} Split:")
    print(f"  Number of videos: {len(video_ids)}")
    print(f"  Video IDs: {', '.join(sorted(video_ids))}")
    print(f"  Total frames: {dataset.total(split_name)}")

# Verify no overlap between splits
print("\n" + "=" * 60)
print("VERIFICATION:")
print("=" * 60)

all_videos = set()
overlap_found = False

for split_name, video_ids in video_splits.items():
    video_set = set(video_ids)
    overlap = all_videos.intersection(video_set)
    if overlap:
        print(f"ERROR: Videos {overlap} appear in multiple splits!")
        overlap_found = True
    all_videos.update(video_set)

if not overlap_found:
    print("✓ All videos are properly separated - no video appears in multiple splits")
    print(f"✓ Total unique videos: {len(all_videos)}")

# Test loading a few examples
print("\n" + "=" * 60)
print("SAMPLE DATA:")
print("=" * 60)

for split_name in ['train', 'validation', 'test']:
    if dataset.total(split_name) > 0:
        example = dataset.get_example(split_name, 0)
        print(f"\nFirst example from {split_name}:")
        print(f"  Video ID: {example['video_id']}")
        print(f"  Frame ID: {example['frame_id']}")
        print(f"  Image size: {example['image'].size}")