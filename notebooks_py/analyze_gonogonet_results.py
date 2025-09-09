#!/usr/bin/env python3
"""Analyze GoNoGoNet results to understand null bbox issues."""

import json
from pathlib import Path
import numpy as np

def analyze_gonogonet_results():
    """Analyze existing GoNoGoNet results."""
    
    print("=" * 80)
    print("ANALYZING GONOGONET RESULTS")
    print("=" * 80)
    
    # Result directory
    result_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet")
    
    # Collect statistics
    total_files = 0
    total_organs = 0
    present_with_bbox = 0
    present_without_bbox = 0
    not_present = 0
    
    go_zone_stats = {"present": 0, "with_bbox": 0, "without_bbox": 0}
    nogo_zone_stats = {"present": 0, "with_bbox": 0, "without_bbox": 0}
    
    missing_bbox_samples = []
    
    # Process each result file
    for result_file in sorted(result_dir.glob("test_*.json")):
        total_files += 1
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        sample_idx = data.get("sample_idx", -1)
        
        for organ in data.get("organs", []):
            total_organs += 1
            organ_name = organ.get("organ_name", "")
            organ_id = organ.get("organ_id", 0)
            pred_present = organ.get("predicted_present", 0)
            pred_bboxes = organ.get("predicted_bboxes", [])
            
            # Track statistics
            if pred_present:
                if pred_bboxes:
                    present_with_bbox += 1
                    if organ_id == 1:  # Go zone
                        go_zone_stats["with_bbox"] += 1
                    elif organ_id == 2:  # NoGo zone
                        nogo_zone_stats["with_bbox"] += 1
                else:
                    present_without_bbox += 1
                    missing_bbox_samples.append((sample_idx, organ_name))
                    if organ_id == 1:  # Go zone
                        go_zone_stats["without_bbox"] += 1
                    elif organ_id == 2:  # NoGo zone
                        nogo_zone_stats["without_bbox"] += 1
                
                if organ_id == 1:
                    go_zone_stats["present"] += 1
                elif organ_id == 2:
                    nogo_zone_stats["present"] += 1
            else:
                not_present += 1
    
    # Print statistics
    print(f"\nTotal files analyzed: {total_files}")
    print(f"Total organ predictions: {total_organs}")
    print()
    print("Overall Statistics:")
    print(f"  Present with bbox: {present_with_bbox} ({present_with_bbox/total_organs*100:.1f}%)")
    print(f"  Present WITHOUT bbox: {present_without_bbox} ({present_without_bbox/total_organs*100:.1f}%)")
    print(f"  Not present: {not_present} ({not_present/total_organs*100:.1f}%)")
    
    print("\nGo Zone Statistics:")
    print(f"  Total present: {go_zone_stats['present']}")
    print(f"  With bbox: {go_zone_stats['with_bbox']}")
    print(f"  WITHOUT bbox: {go_zone_stats['without_bbox']}")
    
    print("\nNoGo Zone Statistics:")
    print(f"  Total present: {nogo_zone_stats['present']}")
    print(f"  With bbox: {nogo_zone_stats['with_bbox']}")
    print(f"  WITHOUT bbox: {nogo_zone_stats['without_bbox']}")
    
    if missing_bbox_samples:
        print(f"\n⚠️ Found {len(missing_bbox_samples)} cases with missing bboxes:")
        for idx, organ in missing_bbox_samples[:10]:  # Show first 10
            print(f"  Sample {idx}: {organ}")
        if len(missing_bbox_samples) > 10:
            print(f"  ... and {len(missing_bbox_samples) - 10} more")
    
    # Check if there's a pattern in the missing indices
    if missing_bbox_samples:
        missing_indices = sorted(set(idx for idx, _ in missing_bbox_samples))
        print(f"\nSamples with missing bboxes: {missing_indices[:20]}")
    
    # Also check what test indices are covered
    covered_indices = []
    for result_file in result_dir.glob("test_*.json"):
        idx = int(result_file.stem.split('_')[1])
        covered_indices.append(idx)
    
    covered_indices = sorted(covered_indices)
    print(f"\nCovered test indices: {len(covered_indices)} samples")
    print(f"  Range: {min(covered_indices)} to {max(covered_indices)}")
    print(f"  First 10: {covered_indices[:10]}")
    print(f"  Last 10: {covered_indices[-10:]}")
    
    # Check for gaps
    expected_indices = set(range(151))
    missing_indices = expected_indices - set(covered_indices)
    if missing_indices:
        print(f"\n⚠️ Missing {len(missing_indices)} test indices:")
        print(f"  {sorted(missing_indices)[:20]}...")
    
    return present_without_bbox > 0  # Return True if there are missing bboxes


if __name__ == "__main__":
    has_issues = analyze_gonogonet_results()
    
    if has_issues:
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print("There are predictions marked as present but without bounding boxes.")
        print("This suggests the GoNoGoNet adapter might not be returning bboxes properly.")
        print("\nTo fix this:")
        print("1. Run: python complete_gonogonet_eval.py")
        print("2. Check the debug outputs in /shared_data0/weiqiuy/llm_cholec_organ/results/gonogonet_debug")
        print("3. Verify the GoNoGoNet adapter is correctly extracting bboxes from the model")