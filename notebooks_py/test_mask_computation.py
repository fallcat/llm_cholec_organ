#!/usr/bin/env python3
"""Test script to debug why bbox-to-mask IoU is 0 for new predictions."""

import sys
import numpy as np
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter
from endopoint.eval.bbox_evaluator import compute_bbox_to_mask_iou

def test_mask_values():
    """Test if masks are being loaded correctly."""
    
    print("=" * 80)
    print("TESTING MASK VALUES FOR CHOLEC_GONOGO")
    print("=" * 80)
    
    adapter = CholecGoNoGoAdapter()
    
    # Test indices that were freshly computed (had 0 IoU)
    test_indices = [63, 558]
    
    for test_idx in test_indices:
        print(f"\nTest index {test_idx}:")
        print("-" * 40)
        
        # Load example
        example = adapter.get_example_by_global_index(test_idx)
        _, lab_tensor = adapter.example_to_tensors(example)
        mask = lab_tensor.numpy()
        
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Unique values in mask: {sorted(set(mask.flatten()))}")
        
        # Check each organ
        for organ_id in [1, 2]:
            organ_mask = (mask == organ_id).astype(np.uint8)
            pixel_count = organ_mask.sum()
            
            print(f"\n  Organ {organ_id} (class {organ_id}):")
            print(f"    Pixels: {pixel_count}")
            
            if pixel_count > 0:
                # Get bounding box
                ys, xs = np.where(organ_mask > 0)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
                print(f"    Ground truth bbox: {bbox}")
                
                # Test compute_bbox_to_mask_iou with a sample bbox
                test_bbox = [[bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5]]  # Slightly larger bbox
                iou = compute_bbox_to_mask_iou(test_bbox, organ_mask)
                print(f"    Test IoU (slightly larger bbox): {iou:.3f}")
                
                # Test with exact bbox
                exact_iou = compute_bbox_to_mask_iou([bbox], organ_mask)
                print(f"    Test IoU (exact bbox): {exact_iou:.3f}")


def test_iou_function():
    """Test the compute_bbox_to_mask_iou function directly."""
    
    print("\n" + "=" * 80)
    print("TESTING compute_bbox_to_mask_iou FUNCTION")
    print("=" * 80)
    
    # Create a simple test mask
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[20:60, 30:70] = 1  # 40x40 rectangle
    
    print(f"\nTest mask shape: {test_mask.shape}")
    print(f"Test mask pixels: {test_mask.sum()}")
    
    # Test different bboxes
    test_cases = [
        ([30, 20, 70, 60], "Exact match"),
        ([25, 15, 75, 65], "Larger bbox"),
        ([35, 25, 65, 55], "Smaller bbox"),
        ([50, 40, 90, 80], "Partial overlap"),
        ([0, 0, 10, 10], "No overlap"),
        ([], "Empty bbox list"),
    ]
    
    for bbox, description in test_cases:
        if bbox:
            iou = compute_bbox_to_mask_iou([bbox], test_mask)
            print(f"  {description}: bbox={bbox}, IoU={iou:.3f}")
        else:
            iou = compute_bbox_to_mask_iou(bbox, test_mask)
            print(f"  {description}: IoU={iou:.3f}")


def check_gonogonet_predictions():
    """Check what GoNoGoNet is actually predicting."""
    
    print("\n" + "=" * 80)
    print("CHECKING GONOGONET PREDICTIONS")
    print("=" * 80)
    
    import json
    from pathlib import Path
    
    # Check the predictions that had 0 IoU
    result_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet")
    
    test_files = {
        63: result_dir / "test_00063.json",
        558: result_dir / "test_00558.json"
    }
    
    for test_idx, test_file in test_files.items():
        if test_file.exists():
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            print(f"\nTest {test_idx} predictions:")
            for organ in data.get('organs', []):
                organ_name = organ.get('organ_name')
                pred_bbox = organ.get('predicted_bboxes', [])
                gt_bbox = organ.get('ground_truth_bboxes', [])
                iou_bbox_to_mask = organ.get('iou_bbox_to_mask', 'N/A')
                
                print(f"  {organ_name}:")
                print(f"    Predicted bbox: {pred_bbox}")
                print(f"    Ground truth bbox: {gt_bbox}")
                print(f"    Stored IoU (bbox-to-mask): {iou_bbox_to_mask}")
                
                # Manually compute IoU if both bboxes exist
                if pred_bbox and gt_bbox:
                    # Load the actual mask
                    adapter = CholecGoNoGoAdapter()
                    example = adapter.get_example_by_global_index(test_idx)
                    _, lab_tensor = adapter.example_to_tensors(example)
                    mask = lab_tensor.numpy()
                    
                    organ_id = organ.get('organ_id')
                    organ_mask = (mask == organ_id).astype(np.uint8)
                    
                    if organ_mask.sum() > 0:
                        manual_iou = compute_bbox_to_mask_iou(pred_bbox, organ_mask)
                        print(f"    Manually computed IoU: {manual_iou:.3f}")
                        
                        if manual_iou > 0 and iou_bbox_to_mask == 0:
                            print(f"    ⚠️ ERROR: Manual IoU is {manual_iou:.3f} but stored IoU is 0!")


def main():
    """Run all tests."""
    
    print("DEBUGGING BBOX-TO-MASK IOU COMPUTATION")
    print("=" * 80)
    
    # Test 1: Check mask values
    test_mask_values()
    
    # Test 2: Test IoU function
    test_iou_function()
    
    # Test 3: Check GoNoGoNet predictions
    check_gonogonet_predictions()
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()