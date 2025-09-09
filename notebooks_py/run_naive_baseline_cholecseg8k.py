#!/usr/bin/env python3
"""
Run naive baselines on CholecSeg8k dataset.
This will generate both Full Box and Random Box baselines.
"""

import sys
import os

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/notebooks_py')

# Import the baseline runner
from eval_bbox_naive_baseline import run_naive_baseline

def main():
    """Run both naive baselines for CholecSeg8k dataset."""
    
    print("=" * 80)
    print("RUNNING NAIVE BASELINES ON CHOLECSEG8K DATASET")
    print("=" * 80)
    print()
    
    # Dataset configuration
    dataset_name = "cholecseg8k"
    num_samples = 200  # Use 200 samples to match other evaluations
    
    # Run Full Box baseline
    print("\n1. Running FULL BOX baseline...")
    print("-" * 60)
    summary_full = run_naive_baseline(
        dataset_name=dataset_name,
        num_samples=num_samples,
        presence_mode="all",  # Always predict all organs present
        box_mode="full"  # Always use full image as bbox
    )
    
    # Run Random Box baseline
    print("\n2. Running RANDOM BOX baseline...")
    print("-" * 60)
    summary_random = run_naive_baseline(
        dataset_name=dataset_name,
        num_samples=num_samples,
        presence_mode="all",  # Always predict all organs present
        box_mode="random"  # Use random bounding boxes
    )
    
    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON - CHOLECSEG8K NAIVE BASELINES")
    print("=" * 80)
    
    if summary_full and 'metrics' in summary_full:
        metrics = summary_full['metrics']
        print(f"\nFull Box Baseline:")
        print(f"  Presence Accuracy: {metrics.get('presence_accuracy', 0)*100:.1f}%")
        print(f"  Mean IoU (bbox):   {metrics.get('mean_iou_bbox_to_bbox', 0):.3f}")
        print(f"  Mean IoU (mask):   {metrics.get('mean_iou_bbox_to_mask', 0):.3f}")
    
    if summary_random and 'metrics' in summary_random:
        metrics = summary_random['metrics']
        print(f"\nRandom Box Baseline:")
        print(f"  Presence Accuracy: {metrics.get('presence_accuracy', 0)*100:.1f}%")
        print(f"  Mean IoU (bbox):   {metrics.get('mean_iou_bbox_to_bbox', 0):.3f}")
        print(f"  Mean IoU (mask):   {metrics.get('mean_iou_bbox_to_mask', 0):.3f}")
    
    print("\n" + "=" * 80)
    print("NAIVE BASELINES COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()