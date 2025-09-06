#!/usr/bin/env python3
"""Test that dataset splits match the original notebook implementations."""

import sys
import os
import glob
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from endopoint.datasets.cholec_organs import CholecOrgansAdapter, VIDEO_GLOBS_PUBLIC, VIDEO_GLOBS_PRIVATE
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter


def get_notebook_splits(data_dir, images_dir, video_globs, train_ratio=0.8, gen_seed=56, train_val_seed=0):
    """Reproduce the exact splitting logic from the original notebooks."""
    
    data_path = Path(data_dir)
    images_path = data_path / images_dir
    
    # Use torch generator for consistency with original code
    gen = torch.Generator()
    gen.manual_seed(gen_seed)
    
    # Split videos into train/test
    num_all = len(video_globs)
    num_train = int(num_all * train_ratio)
    perm = torch.randperm(num_all, generator=gen)
    
    train_video_indices = perm[:num_train]
    test_video_indices = perm[num_train:]
    
    # Collect image files for each split
    train_files = []
    test_files = []
    
    print("Collecting training video files...")
    for i in tqdm(train_video_indices.tolist(), desc="  Train videos"):
        pattern = str(images_path / video_globs[i])
        files = glob.glob(pattern)
        train_files.extend([os.path.basename(f) for f in files])
    
    print("Collecting test video files...")
    for i in tqdm(test_video_indices.tolist(), desc="  Test videos"):
        pattern = str(images_path / video_globs[i])
        files = glob.glob(pattern)
        test_files.extend([os.path.basename(f) for f in files])
    
    train_files = sorted(set(train_files))
    test_files = sorted(set(test_files))
    
    # Split train into train/val
    gen.manual_seed(train_val_seed)
    num_train_examples = len(train_files)
    num_train_split = int(num_train_examples * 0.9)  # 90% train, 10% val
    perm = torch.randperm(num_train_examples, generator=gen)
    
    train_files_shuffled = [train_files[i] for i in perm]
    final_train = train_files_shuffled[:num_train_split]
    final_val = train_files_shuffled[num_train_split:]
    
    return {
        'train': sorted(final_train),
        'validation': sorted(final_val),
        'test': sorted(test_files)
    }


def test_organs_splits():
    """Test CholecOrgans dataset splits match notebook."""
    print("\n" + "="*60)
    print("Testing CholecOrgans dataset splits...")
    print("="*60)
    
    data_dir = "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib"
    
    # Test public videos
    print("\nTesting PUBLIC video splits...")
    adapter = CholecOrgansAdapter(
        data_dir=data_dir,
        video_globs='public',
        gen_seed=56,
        train_val_seed=0
    )
    
    notebook_splits = get_notebook_splits(
        data_dir, "images", VIDEO_GLOBS_PUBLIC, 
        train_ratio=0.8, gen_seed=56, train_val_seed=0
    )
    
    # Get adapter file lists
    adapter_splits = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    for split in ['train', 'validation', 'test']:
        for idx in range(adapter.total(split)):
            example = adapter.get_example(split, idx)
            adapter_splits[split].append(example['filename'])
        adapter_splits[split] = sorted(adapter_splits[split])
    
    # Compare splits
    for split in ['train', 'validation', 'test']:
        notebook_set = set(notebook_splits[split])
        adapter_set = set(adapter_splits[split])
        
        # Filter to only files that exist in both (since adapter checks existence)
        organ_labels_dir = Path(data_dir) / "organ_labels"
        notebook_filtered = []
        for f in notebook_splits[split]:
            if (organ_labels_dir / f).exists():
                notebook_filtered.append(f)
        notebook_set = set(notebook_filtered)
        
        print(f"\n{split.upper()} split:")
        print(f"  Notebook: {len(notebook_set)} files")
        print(f"  Adapter:  {len(adapter_set)} files")
        
        if notebook_set == adapter_set:
            print(f"  ✓ Files match exactly!")
        else:
            only_notebook = notebook_set - adapter_set
            only_adapter = adapter_set - notebook_set
            if only_notebook:
                print(f"  ✗ Only in notebook: {len(only_notebook)} files")
                print(f"    Examples: {list(only_notebook)[:3]}")
            if only_adapter:
                print(f"  ✗ Only in adapter: {len(only_adapter)} files")
                print(f"    Examples: {list(only_adapter)[:3]}")


def test_gonogo_splits():
    """Test CholecGoNoGo dataset splits match notebook."""
    print("\n" + "="*60)
    print("Testing CholecGoNoGo dataset splits...")
    print("="*60)
    
    data_dir = "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib"
    
    # Test public videos
    print("\nTesting PUBLIC video splits...")
    adapter = CholecGoNoGoAdapter(
        data_dir=data_dir,
        video_globs='public',
        gen_seed=56,
        train_val_seed=0
    )
    
    notebook_splits = get_notebook_splits(
        data_dir, "images", VIDEO_GLOBS_PUBLIC,
        train_ratio=0.8, gen_seed=56, train_val_seed=0
    )
    
    # Get adapter file lists
    adapter_splits = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    for split in ['train', 'validation', 'test']:
        for idx in range(adapter.total(split)):
            example = adapter.get_example(split, idx)
            adapter_splits[split].append(example['filename'])
        adapter_splits[split] = sorted(adapter_splits[split])
    
    # Compare splits
    for split in ['train', 'validation', 'test']:
        notebook_set = set(notebook_splits[split])
        adapter_set = set(adapter_splits[split])
        
        # Filter to only files that exist in both (since adapter checks existence)
        gonogo_labels_dir = Path(data_dir) / "gonogo_labels"
        notebook_filtered = []
        for f in notebook_splits[split]:
            if (gonogo_labels_dir / f).exists():
                notebook_filtered.append(f)
        notebook_set = set(notebook_filtered)
        
        print(f"\n{split.upper()} split:")
        print(f"  Notebook: {len(notebook_set)} files")
        print(f"  Adapter:  {len(adapter_set)} files")
        
        if notebook_set == adapter_set:
            print(f"  ✓ Files match exactly!")
        else:
            only_notebook = notebook_set - adapter_set
            only_adapter = adapter_set - notebook_set
            if only_notebook:
                print(f"  ✗ Only in notebook: {len(only_notebook)} files")
                print(f"    Examples: {list(only_notebook)[:3]}")
            if only_adapter:
                print(f"  ✗ Only in adapter: {len(only_adapter)} files")
                print(f"    Examples: {list(only_adapter)[:3]}")


def test_seed_consistency():
    """Test that different seed values produce different splits."""
    print("\n" + "="*60)
    print("Testing seed consistency...")
    print("="*60)
    
    data_dir = "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib"
    
    # Create two adapters with different seeds
    adapter1 = CholecOrgansAdapter(
        data_dir=data_dir,
        video_globs='public',
        gen_seed=56,
        train_val_seed=0
    )
    
    adapter2 = CholecOrgansAdapter(
        data_dir=data_dir,
        video_globs='public',
        gen_seed=42,  # Different seed
        train_val_seed=0
    )
    
    # Compare test splits (should be different)
    test_files1 = set()
    test_files2 = set()
    
    for idx in range(min(adapter1.total('test'), 100)):  # Check first 100
        test_files1.add(adapter1.get_example('test', idx)['filename'])
        test_files2.add(adapter2.get_example('test', idx)['filename'])
    
    if test_files1 != test_files2:
        print("  ✓ Different gen_seed produces different test splits")
    else:
        print("  ✗ Different gen_seed produces same test splits (ERROR)")
    
    # Test train/val split with different seed
    adapter3 = CholecOrgansAdapter(
        data_dir=data_dir,
        video_globs='public',
        gen_seed=56,
        train_val_seed=1  # Different train/val seed
    )
    
    val_files1 = set()
    val_files3 = set()
    
    for idx in range(min(adapter1.total('validation'), 100)):  # Check first 100
        val_files1.add(adapter1.get_example('validation', idx)['filename'])
        val_files3.add(adapter3.get_example('validation', idx)['filename'])
    
    if val_files1 != val_files3:
        print("  ✓ Different train_val_seed produces different validation splits")
    else:
        print("  ✗ Different train_val_seed produces same validation splits (ERROR)")


def main():
    """Run all split consistency tests."""
    print("Starting dataset split consistency tests...")
    
    test_organs_splits()
    test_gonogo_splits()
    test_seed_consistency()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()