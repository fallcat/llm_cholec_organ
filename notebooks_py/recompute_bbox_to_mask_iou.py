#!/usr/bin/env python3
"""
Recompute IoU metrics using bbox-to-mask instead of bbox-to-bbox.
Reads existing prediction files and recomputes IoU values.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter


def compute_bbox_to_mask_iou(bbox, mask):
    """Compute IoU between a bounding box and a segmentation mask."""
    # Create bbox mask
    bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within mask bounds
    h, w = mask.shape
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 > x1 and y2 > y1:
        bbox_mask[y1:y2, x1:x2] = 1
    
    # Compute intersection and union
    intersection = np.logical_and(bbox_mask, mask).sum()
    union = np.logical_or(bbox_mask, mask).sum()
    
    return float(intersection / union) if union > 0 else 0.0


def recompute_iou_for_file(pred_file, dataset_adapter, output_dir):
    """Recompute IoU for a single prediction file using bbox-to-mask."""
    
    with open(pred_file, 'r') as f:
        data = json.load(f)
    
    sample_idx = data['sample_idx']
    
    # Load the example
    example = dataset_adapter.get_example_by_global_index(sample_idx)
    
    # Get segmentation masks
    _, lab_tensor = dataset_adapter.example_to_tensors(example)
    full_mask = lab_tensor.numpy()
    
    # Recompute IoUs
    new_ious = []
    
    for organ_data in data['organs']:
        organ_id = organ_data['organ_id']
        
        if organ_data['ground_truth_present'] and organ_data['predicted_present']:
            # Get organ-specific mask
            organ_mask = (full_mask == organ_id).astype(np.uint8)
            
            if organ_data['predicted_bboxes']:
                # Use first predicted bbox
                pred_bbox = organ_data['predicted_bboxes'][0]
                
                # Compute bbox-to-mask IoU
                iou = compute_bbox_to_mask_iou(pred_bbox, organ_mask)
                new_ious.append(iou)
                
                # Update the organ data
                organ_data['iou_bbox_to_mask'] = iou
                # Keep original for comparison
                if 'iou' in organ_data:
                    organ_data['iou_bbox_to_bbox'] = organ_data['iou']
                organ_data['iou'] = iou  # Replace with new IoU
            else:
                new_ious.append(None)
        else:
            # No valid comparison
            if organ_data['ground_truth_present'] != organ_data['predicted_present']:
                new_ious.append(None)
    
    # Update main IoU list
    data['ious_bbox_to_mask'] = new_ious
    data['ious_bbox_to_bbox'] = data.get('ious', [])  # Keep original
    data['ious'] = new_ious  # Replace with new IoUs
    
    # Save updated file
    output_file = output_dir / pred_file.name
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return new_ious


def process_model_directory(model_dir, dataset_adapter, base_output_dir):
    """Process all test files for a model."""
    
    model_name = model_dir.name
    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_files = list(model_dir.glob("test_*.json"))
    
    if not test_files:
        print(f"  No test files found for {model_name}")
        return None
    
    all_ious = []
    
    for test_file in tqdm(test_files, desc=f"  Processing {model_name}", leave=False):
        ious = recompute_iou_for_file(test_file, dataset_adapter, output_dir)
        all_ious.extend([iou for iou in ious if iou is not None])
    
    # Compute metrics
    if all_ious:
        metrics = {
            'mean_iou_bbox_to_mask': np.mean(all_ious),
            'iou_at_50_bbox_to_mask': np.mean([iou >= 0.5 for iou in all_ious]),
            'num_predictions': len(all_ious)
        }
    else:
        metrics = {
            'mean_iou_bbox_to_mask': 0.0,
            'iou_at_50_bbox_to_mask': 0.0,
            'num_predictions': 0
        }
    
    return metrics


def main():
    # Setup paths
    results_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick")
    output_base = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick_mask_iou")
    
    # Load dataset adapter
    data_dir = "/shared_data0/weiqiuy/datasets/cholecseg8k"
    dataset_adapter = CholecSeg8kLocalAdapter(data_dir=data_dir)
    
    # Process each evaluation mode
    modes = ["zeroshot_combined", "zeroshot_separate", "fewshot_combined", "fewshot_separate"]
    
    all_results = {}
    
    for mode in modes:
        print(f"\nProcessing {mode}...")
        mode_dir = results_dir / mode
        
        if not mode_dir.exists():
            print(f"  {mode_dir} does not exist, skipping...")
            continue
        
        output_mode_dir = output_base / mode
        output_mode_dir.mkdir(parents=True, exist_ok=True)
        
        mode_results = {}
        
        for model_dir in mode_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            print(f"  Processing {model_dir.name}...")
            metrics = process_model_directory(model_dir, dataset_adapter, output_mode_dir)
            
            if metrics:
                mode_results[model_dir.name] = metrics
                print(f"    Mean IoU (bbox-to-mask): {metrics['mean_iou_bbox_to_mask']:.3f}")
                print(f"    IoU@0.5 (bbox-to-mask): {metrics['iou_at_50_bbox_to_mask']:.3f}")
        
        all_results[mode] = mode_results
        
        # Save summary for this mode
        summary_file = output_base / f"summary_{mode}.json"
        with open(summary_file, 'w') as f:
            json.dump(mode_results, f, indent=2)
        print(f"  Saved summary to {summary_file}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("BBOX-TO-MASK IOU RESULTS SUMMARY")
    print("="*80)
    
    for mode, results in all_results.items():
        if results:
            print(f"\n{mode}:")
            for model, metrics in results.items():
                print(f"  {model:40s}: IoU={metrics['mean_iou_bbox_to_mask']:.3f}, IoU@0.5={metrics['iou_at_50_bbox_to_mask']:.3f}")
    
    print("\n✅ Recomputation complete. Results saved to:", output_base)


if __name__ == "__main__":
    main()