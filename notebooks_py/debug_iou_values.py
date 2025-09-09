#!/usr/bin/env python3
"""Debug script to check IoU values in GoNoGoNet results."""

import json
import os
from pathlib import Path

def check_individual_files():
    """Check IoU values in individual test files."""
    
    gonogo_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet")
    
    print("=" * 80)
    print("CHECKING INDIVIDUAL TEST FILES")
    print("=" * 80)
    
    # Get all test files
    test_files = sorted(gonogo_dir.glob("test_*.json"))[-5:]  # Check last 5 files
    
    all_bbox_ious = []
    all_mask_ious = []
    
    for test_file in test_files:
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n{test_file.name}:")
        print(f"  Sample idx: {data.get('sample_idx', 'N/A')}")
        
        for org in data.get('organs', []):
            organ_name = org.get('organ_name', 'Unknown')
            bbox_iou = org.get('iou_bbox_to_bbox', None)
            mask_iou = org.get('iou_bbox_to_mask', None)
            
            print(f"  {organ_name}:")
            print(f"    bbox-to-bbox IoU: {bbox_iou}")
            print(f"    bbox-to-mask IoU: {mask_iou}")
            
            if bbox_iou is not None:
                all_bbox_ious.append(bbox_iou)
            if mask_iou is not None:
                all_mask_ious.append(mask_iou)
    
    print("\n" + "=" * 80)
    print("SUMMARY OF INDIVIDUAL FILES")
    print("=" * 80)
    
    if all_bbox_ious:
        print(f"Bbox-to-bbox IoU values found: {len(all_bbox_ious)}")
        print(f"  Mean: {sum(all_bbox_ious)/len(all_bbox_ious):.3f}")
        print(f"  Min: {min(all_bbox_ious):.3f}")
        print(f"  Max: {max(all_bbox_ious):.3f}")
    else:
        print("No bbox-to-bbox IoU values found!")
    
    if all_mask_ious:
        print(f"\nBbox-to-mask IoU values found: {len(all_mask_ious)}")
        print(f"  Mean: {sum(all_mask_ious)/len(all_mask_ious):.3f}")
        print(f"  Min: {min(all_mask_ious):.3f}")
        print(f"  Max: {max(all_mask_ious):.3f}")
    else:
        print("\nNo bbox-to-mask IoU values found!")


def check_metrics_file():
    """Check the aggregated metrics file."""
    
    metrics_file = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet/metrics.json")
    
    print("\n" + "=" * 80)
    print("CHECKING METRICS FILE")
    print("=" * 80)
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"Overall metrics:")
        print(f"  Presence accuracy: {metrics.get('presence_accuracy', 'N/A')}")
        print(f"  Mean bbox-to-bbox IoU: {metrics.get('mean_iou_bbox_to_bbox', 'N/A')}")
        print(f"  Mean bbox-to-mask IoU: {metrics.get('mean_iou_bbox_to_mask', 'N/A')}")
        
        print(f"\nPer-organ metrics:")
        for organ_name, organ_metrics in metrics.get('per_organ', {}).items():
            print(f"  {organ_name}:")
            print(f"    Mean bbox-to-bbox IoU: {organ_metrics.get('mean_iou_bbox_to_bbox', 'N/A')}")
            print(f"    Mean bbox-to-mask IoU: {organ_metrics.get('mean_iou_bbox_to_mask', 'N/A')}")
            print(f"    TP: {organ_metrics.get('tp', 0)}, FP: {organ_metrics.get('fp', 0)}")
    else:
        print(f"Metrics file not found: {metrics_file}")


def check_predictions_file():
    """Check the predictions.json file."""
    
    pred_file = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet/predictions.json")
    
    print("\n" + "=" * 80)
    print("CHECKING PREDICTIONS FILE")
    print("=" * 80)
    
    if pred_file.exists():
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
        
        print(f"Total predictions: {len(predictions)}")
        
        # Check first few predictions
        for i, pred in enumerate(predictions[:5]):
            print(f"\nPrediction {i}:")
            print(f"  Test idx: {pred.get('test_idx', 'N/A')}")
            print(f"  Organ: {pred.get('organ_name', 'N/A')}")
            print(f"  Has iou_bbox_to_bbox: {'iou_bbox_to_bbox' in pred}")
            print(f"  Has iou_bbox_to_mask: {'iou_bbox_to_mask' in pred}")
            if 'iou_bbox_to_bbox' in pred:
                print(f"    iou_bbox_to_bbox value: {pred['iou_bbox_to_bbox']}")
            if 'iou_bbox_to_mask' in pred:
                print(f"    iou_bbox_to_mask value: {pred['iou_bbox_to_mask']}")
    else:
        print(f"Predictions file not found: {pred_file}")


def test_aggregation():
    """Test if IoU values are being properly aggregated."""
    
    print("\n" + "=" * 80)
    print("TESTING AGGREGATION LOGIC")
    print("=" * 80)
    
    gonogo_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet")
    
    # Manually compute metrics from test files
    organ_ious = {
        "Go (Safe to Incise)": {"bbox": [], "mask": []},
        "NoGo (Unsafe to Incise)": {"bbox": [], "mask": []}
    }
    
    test_files = list(gonogo_dir.glob("test_*.json"))
    print(f"Found {len(test_files)} test files")
    
    for test_file in test_files:
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        for org in data.get('organs', []):
            organ_name = org.get('organ_name')
            if organ_name in organ_ious:
                if 'iou_bbox_to_bbox' in org and org['iou_bbox_to_bbox'] is not None:
                    organ_ious[organ_name]['bbox'].append(org['iou_bbox_to_bbox'])
                if 'iou_bbox_to_mask' in org and org['iou_bbox_to_mask'] is not None:
                    organ_ious[organ_name]['mask'].append(org['iou_bbox_to_mask'])
    
    print("\nManually computed metrics:")
    for organ_name, ious in organ_ious.items():
        print(f"\n{organ_name}:")
        if ious['bbox']:
            print(f"  Bbox IoU: {len(ious['bbox'])} values, mean = {sum(ious['bbox'])/len(ious['bbox']):.3f}")
        else:
            print(f"  Bbox IoU: No values")
        
        if ious['mask']:
            print(f"  Mask IoU: {len(ious['mask'])} values, mean = {sum(ious['mask'])/len(ious['mask']):.3f}")
        else:
            print(f"  Mask IoU: No values")


def main():
    """Run all debug checks."""
    
    print("DEBUGGING GONOGONET IOU VALUES")
    print("=" * 80)
    
    # 1. Check individual test files
    check_individual_files()
    
    # 2. Check metrics file
    check_metrics_file()
    
    # 3. Check predictions file
    check_predictions_file()
    
    # 4. Test aggregation
    test_aggregation()
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()