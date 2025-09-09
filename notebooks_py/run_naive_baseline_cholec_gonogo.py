#!/usr/bin/env python3
"""
Run naive baselines on CholecGoNoGo dataset.
This will generate both Full Box and Random Box baselines.
"""

import sys
import os

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

# Import the baseline runner
from eval_bbox_naive_baseline import run_naive_baseline

def main():
    """Run both naive baselines for CholecGoNoGo dataset."""
    
    print("=" * 80)
    print("RUNNING NAIVE BASELINES ON CHOLEC_GONOGO DATASET")
    print("=" * 80)
    print()
    
    # Dataset configuration
    dataset_name = "cholec_gonogo"
    num_samples = 151  # Use 151 samples to match other models on CholecGoNoGo
    
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
    print("FINAL COMPARISON - CHOLEC_GONOGO NAIVE BASELINES")
    print("=" * 80)
    
    print("\nFull Box Baseline:")
    print(f"  Presence: {summary_full['metrics']['presence_accuracy']:.1%}")
    print(f"  IoU-B (bbox-to-bbox): {summary_full['metrics']['mean_iou_bbox_to_bbox']:.3f}")
    print(f"  IoU-M (bbox-to-mask): {summary_full['metrics']['mean_iou_bbox_to_mask']:.3f}")
    
    print("\nRandom Box Baseline:")
    print(f"  Presence: {summary_random['metrics']['presence_accuracy']:.1%}")
    print(f"  IoU-B (bbox-to-bbox): {summary_random['metrics']['mean_iou_bbox_to_bbox']:.3f}")
    print(f"  IoU-M (bbox-to-mask): {summary_random['metrics']['mean_iou_bbox_to_mask']:.3f}")
    
    print("\n✅ Naive baselines for CholecGoNoGo completed successfully!")
    print("Results are saved in: /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/")
    

if __name__ == "__main__":
    main()