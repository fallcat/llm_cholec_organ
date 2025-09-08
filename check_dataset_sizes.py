#!/usr/bin/env python3
"""
Script to check actual image sizes and organ information for all three datasets.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
from endopoint.datasets.cholec_organs import CholecOrgansAdapter
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter


def check_dataset_info(dataset_name, adapter):
    """Check and print information about a dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print('='*60)
    
    try:
        # Get first example
        example = adapter.get_example_by_global_index(0)
        img_tensor, lab_tensor = adapter.example_to_tensors(example)
        
        # Image dimensions
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Image size (W x H): {img_tensor.shape[2]} x {img_tensor.shape[1]}")
        
        # Label dimensions
        print(f"Label tensor shape: {lab_tensor.shape}")
        
        # Organ classes
        if hasattr(adapter, 'organ_classes'):
            print(f"\nOrgan classes ({len(adapter.organ_classes)}):")
            for i, organ in enumerate(adapter.organ_classes, 1):
                print(f"  {i}. {organ}")
        
        # Check unique values in label mask
        unique_labels = np.unique(lab_tensor.numpy())
        print(f"\nUnique label values in first sample: {unique_labels.tolist()}")
        
        # Dataset size
        if hasattr(adapter, 'total'):
            try:
                train_size = adapter.total('train')
                val_size = adapter.total('val') if hasattr(adapter, 'val') else 0
                test_size = adapter.total('test')
                print(f"\nDataset sizes:")
                print(f"  Train: {train_size}")
                if val_size > 0:
                    print(f"  Val: {val_size}")
                print(f"  Test: {test_size}")
                print(f"  Total: {train_size + val_size + test_size}")
            except:
                pass
        
        # Check a few more samples to verify consistency
        print(f"\nChecking first 5 samples for size consistency:")
        sizes = []
        for i in range(min(5, adapter.total('test') if hasattr(adapter, 'total') else 5)):
            try:
                example = adapter.get_example_by_global_index(i)
                img_tensor, _ = adapter.example_to_tensors(example)
                size = (img_tensor.shape[2], img_tensor.shape[1])
                sizes.append(size)
                print(f"  Sample {i}: {size}")
            except Exception as e:
                print(f"  Sample {i}: Error - {e}")
                break
        
        # Check if all sizes are the same
        if len(set(sizes)) == 1:
            print(f"✓ All samples have consistent size: {sizes[0]}")
        else:
            print(f"⚠ Warning: Samples have different sizes: {set(sizes)}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*60)
    print("CHECKING DATASET INFORMATION")
    print("="*60)
    
    # 1. CholecSeg8k
    try:
        data_dir = "/shared_data0/weiqiuy/datasets/cholecseg8k"
        adapter = CholecSeg8kLocalAdapter(data_dir=data_dir)
        check_dataset_info("CholecSeg8k", adapter)
    except Exception as e:
        print(f"\nCholecSeg8k failed: {e}")
    
    # 2. CholecOrgans
    try:
        adapter = CholecOrgansAdapter()
        check_dataset_info("CholecOrgans", adapter)
    except Exception as e:
        print(f"\nCholecOrgans failed: {e}")
    
    # 3. CholecGoNoGo
    try:
        adapter = CholecGoNoGoAdapter()
        check_dataset_info("CholecGoNoGo", adapter)
    except Exception as e:
        print(f"\nCholecGoNoGo failed: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nExpected sizes according to notebook:")
    print("  - CholecSeg8k: 12 organs, image size (854, 480)")
    print("  - CholecOrgans: 3 organs, image size (640, 384)")
    print("  - CholecGoNoGo: 2 organs, image size (854, 480)")
    print("\nPlease compare with actual results above.")


if __name__ == "__main__":
    main()