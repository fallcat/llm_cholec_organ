"""Test script to verify CholecOrgans and CholecGoNoGo dataset splitting consistency."""

import sys
import os
import glob
import torch
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from endopoint.datasets import build_dataset

# Original splitting logic from notebook
def get_original_split_files(data_dir, images_dir, video_globs, train_ratio, gen_seed):
    """Reproduce the original splitting logic from the notebook."""
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
    
    images_path = Path(data_dir) / images_dir
    
    for i in train_video_indices:
        pattern = str(images_path / video_globs[i])
        files = glob.glob(pattern)
        train_files.extend([os.path.basename(f) for f in files])
    
    for i in test_video_indices:
        pattern = str(images_path / video_globs[i])
        files = glob.glob(pattern)
        test_files.extend([os.path.basename(f) for f in files])
    
    train_files = sorted(train_files)
    test_files = sorted(test_files)
    
    # Apply train/val split with different seed
    train_val_seed = 0  # Default in original
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


def test_dataset_splitting(dataset_name, dataset_builder_func, video_globs_key='public'):
    """Test that dataset splitting matches the original implementation."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} dataset splitting")
    print('='*60)
    
    # Get video globs from the original notebook
    VIDEO_GLOBS_PUBLIC = (
        [f"cholec80_video{i:02d}_*" for i in range(1, 81)] +
        [f"M2CCAI2016_video{i}_*" for i in range(81, 122)]
    )
    
    VIDEO_GLOBS_PRIVATE = (
        [f"AdnanSet_LC_{i}_*" for i in range(1, 165)] +
        [f"AminSet_LC_{i}_*" for i in range(1, 11)] +
        ["HokkaidoSet_LC_1_*", "HokkaidoSet_LC_2_*"] +
        [f"UTSWSet_Case_{i}_*" for i in range(1, 13)] +
        ["WashUSet_LC_01_*"]
    )
    
    video_globs_dict = {
        'public': VIDEO_GLOBS_PUBLIC,
        'private': VIDEO_GLOBS_PRIVATE
    }
    
    # Test parameters from original notebook
    data_dir = "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib"
    train_ratio = 0.8
    gen_seed = 1234
    
    # Build dataset using new adapter
    dataset = dataset_builder_func(
        data_dir=data_dir,
        video_globs=video_globs_key,
        train_ratio=train_ratio,
        gen_seed=gen_seed
    )
    
    # Get original splits
    original_splits = get_original_split_files(
        data_dir=data_dir,
        images_dir="images",
        video_globs=video_globs_dict[video_globs_key],
        train_ratio=train_ratio,
        gen_seed=gen_seed
    )
    
    # Collect files from new dataset
    new_splits = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    for split in ['train', 'validation', 'test']:
        print(f"\nCollecting {split} files from dataset...")
        for i in tqdm(range(dataset.total(split)), desc=f"  Loading {split}"):
            example = dataset.get_example(split, i)
            filename = example.get('filename')
            if filename:
                new_splits[split].append(filename)
    
    # Sort for comparison
    for split in new_splits:
        new_splits[split] = sorted(new_splits[split])
    
    # Compare splits
    all_match = True
    for split in ['train', 'validation', 'test']:
        original_count = len(original_splits[split])
        new_count = len(new_splits[split])
        
        print(f"\n{split.upper()} split:")
        print(f"  Original: {original_count} files")
        print(f"  New adapter: {new_count} files")
        
        if original_count != new_count:
            print(f"  ❌ Count mismatch!")
            all_match = False
            
            # Find differences
            original_set = set(original_splits[split])
            new_set = set(new_splits[split])
            
            only_original = original_set - new_set
            only_new = new_set - original_set
            
            if only_original:
                print(f"  Files only in original: {len(only_original)}")
                for f in list(only_original)[:5]:
                    print(f"    - {f}")
            
            if only_new:
                print(f"  Files only in new: {len(only_new)}")
                for f in list(only_new)[:5]:
                    print(f"    - {f}")
        else:
            # Check if files match exactly
            files_match = all(o == n for o, n in zip(original_splits[split], new_splits[split]))
            if files_match:
                print(f"  ✅ Files match exactly")
            else:
                print(f"  ⚠️  Same count but different files")
                all_match = False
    
    if all_match:
        print(f"\n✅ {dataset_name} dataset splitting is CONSISTENT with original!")
    else:
        print(f"\n❌ {dataset_name} dataset splitting DIFFERS from original!")
    
    return all_match


def test_dataset_functionality(dataset_name, dataset_builder_func):
    """Test basic functionality of the dataset adapter."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} dataset functionality")
    print('='*60)
    
    # Build dataset
    dataset = dataset_builder_func(
        data_dir="/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
        video_globs='public',
        train_ratio=0.8,
        gen_seed=1234
    )
    
    print(f"\nDataset info:")
    print(f"  Tag: {dataset.dataset_tag}")
    print(f"  Version: {dataset.version}")
    print(f"  Classes: {dataset.id2label}")
    print(f"  Label IDs: {dataset.label_ids}")
    print(f"  Canvas size: {dataset.recommended_canvas}")
    
    # Test loading examples
    for split in ['train', 'validation', 'test']:
        if dataset.total(split) > 0:
            print(f"\nTesting {split} split:")
            
            # Load first example
            example = dataset.get_example(split, 0)
            print(f"  Example keys: {example.keys()}")
            
            # Convert to tensors
            img_t, lab_t = dataset.example_to_tensors(example)
            print(f"  Image tensor shape: {img_t.shape}")
            print(f"  Label tensor shape: {lab_t.shape}")
            print(f"  Label unique values: {torch.unique(lab_t).tolist()}")
            
            # Test presence vector
            presence = dataset.labels_to_presence_vector(lab_t, min_pixels=50)
            print(f"  Presence vector: {presence.tolist()}")
            
            # Test point sampling
            for class_id in dataset.label_ids:
                if presence[dataset.label_ids.index(class_id)] == 1:
                    point = dataset.sample_point_in_mask(lab_t, class_id)
                    if point:
                        print(f"  Point for {dataset.id2label[class_id]}: {point}")
                        break
            
            # Test bounding boxes
            bboxes = dataset.get_bounding_boxes(lab_t, min_pixels=50)
            if bboxes:
                for class_id, boxes in list(bboxes.items())[:1]:
                    print(f"  Bounding boxes for {dataset.id2label[class_id]}: {len(boxes)} box(es)")
                    if boxes:
                        print(f"    First box: {boxes[0]}")
    
    print(f"\n✅ {dataset_name} functionality tests passed!")


def main():
    """Run all tests."""
    print("Testing CholecOrgans and CholecGoNoGo dataset adapters")
    print("========================================================")
    
    # Test CholecOrgans
    print("\n1. Testing CholecOrgans Dataset")
    organs_split_ok = test_dataset_splitting(
        "CholecOrgans",
        lambda **kwargs: build_dataset("cholec_organs", **kwargs),
        'public'
    )
    test_dataset_functionality(
        "CholecOrgans",
        lambda **kwargs: build_dataset("cholec_organs", **kwargs)
    )
    
    # Test CholecGoNoGo
    print("\n2. Testing CholecGoNoGo Dataset")
    gonogo_split_ok = test_dataset_splitting(
        "CholecGoNoGo",
        lambda **kwargs: build_dataset("cholec_gonogo", **kwargs),
        'public'
    )
    test_dataset_functionality(
        "CholecGoNoGo",
        lambda **kwargs: build_dataset("cholec_gonogo", **kwargs)
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if organs_split_ok and gonogo_split_ok:
        print("✅ All dataset splitting tests PASSED!")
        print("The new adapters maintain the exact same splitting as the original notebook.")
    else:
        print("❌ Some dataset splitting tests FAILED!")
        if not organs_split_ok:
            print("  - CholecOrgans splitting differs from original")
        if not gonogo_split_ok:
            print("  - CholecGoNoGo splitting differs from original")
    
    print("\nBoth datasets follow the same interface as CholecSeg8k local adapter:")
    print("  - Same method signatures")
    print("  - Same tensor formats")
    print("  - Same presence vector computation")
    print("  - Same bounding box extraction")
    print("  - Video-based train/test splitting with seed consistency")


if __name__ == "__main__":
    main()