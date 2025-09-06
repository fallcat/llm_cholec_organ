#!/usr/bin/env python3
"""
Convert existing point-based few-shot plan to bounding box format.
This ensures the EXACT SAME examples are used, just with bbox annotations.
"""

import os
import sys
import json
from pathlib import Path
import torch
import numpy as np
from scipy import ndimage

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from cholecseg8k_utils import example_to_tensors, ID2LABEL, LABEL_IDS
from datasets import load_dataset

def extract_bounding_boxes_from_mask(
    lab_t: torch.Tensor, 
    class_id: int,
    min_pixels: int = 50,
    min_bbox_size: int = 20,
    max_bboxes: int = 3
):
    """Extract bounding boxes from a segmentation mask for a given class."""
    # Create binary mask for the target class
    mask = (lab_t == class_id).cpu().numpy().astype(np.uint8)
    
    if mask.sum() < min_pixels:
        return []
    
    # Find connected components
    labeled_mask, num_components = ndimage.label(mask)
    
    bboxes = []
    component_sizes = []
    
    # Extract bbox for each component
    for i in range(1, num_components + 1):
        component_mask = (labeled_mask == i)
        component_size = component_mask.sum()
        
        if component_size < min_pixels:
            continue
            
        # Find bbox coordinates
        rows, cols = np.where(component_mask)
        if len(rows) == 0 or len(cols) == 0:
            continue
            
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        
        # Check minimum size
        if (x2 - x1) < min_bbox_size or (y2 - y1) < min_bbox_size:
            continue
            
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        component_sizes.append(component_size)
    
    # Sort by component size (largest first) and limit to max_bboxes
    if len(bboxes) > max_bboxes:
        sorted_indices = np.argsort(component_sizes)[::-1][:max_bboxes]
        bboxes = [bboxes[i] for i in sorted_indices]
    
    return bboxes

def compute_single_bbox_from_mask(lab_t: torch.Tensor, class_id: int):
    """Compute a single bounding box that encompasses all pixels of a given class."""
    mask = (lab_t == class_id).cpu().numpy()
    
    if not mask.any():
        return None
    
    rows, cols = np.where(mask)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    
    return [int(x1), int(y1), int(x2), int(y2)]

def convert_point_plan_to_bbox(point_plan_file, output_file, dataset_split='train'):
    """
    Convert a point-based plan to bbox format using the EXACT SAME examples.
    """
    print(f"Loading point plan from: {point_plan_file}")
    with open(point_plan_file, 'r') as f:
        point_plan = json.load(f)
    
    print("Loading CholecSeg8k dataset...")
    dataset = load_dataset("minwoosun/CholecSeg8k")
    
    # Create new bbox plan structure
    bbox_plan = {
        "split": point_plan.get("split", dataset_split),
        "config": {
            "n_pos": point_plan.get("n_pos", 1),
            "n_neg_easy": point_plan.get("n_neg_easy", 1),
            "n_neg_hard": point_plan.get("n_neg_hard", 1),
            "min_pixels": point_plan.get("min_pixels", 50),
            "max_bboxes": 3,
            "min_bbox_size": 20,
            "seed": point_plan.get("seed", 44),
            "converted_from": str(point_plan_file)
        },
        "excluded_indices": point_plan.get("exclude_balanced", []),
        "plan": {}
    }
    
    print("Converting examples to bbox format...")
    
    # Convert each organ's examples
    for class_id_str, organ_info in point_plan["plan"].items():
        class_id = int(class_id_str)
        organ_name = organ_info.get("label", organ_info.get("name", ID2LABEL.get(class_id, f"Class {class_id}")))
        
        print(f"  Processing {organ_name}...")
        
        # Create new organ entry
        bbox_organ = {
            "name": organ_name,
            "positives": [],
            "negatives_easy": organ_info.get("neg_easy", organ_info.get("negatives_easy", [])),
            "negatives_hard": organ_info.get("neg_hard", organ_info.get("negatives_hard", [])),
            "pos_available": organ_info.get("pos_available", -1),
            "neg_available": organ_info.get("neg_available", -1)
        }
        
        # Convert positive examples - extract bboxes from the SAME images
        positives = organ_info.get("pos", organ_info.get("positives", []))
        for pos_item in positives:
            # Get the index
            if isinstance(pos_item, dict):
                idx = pos_item.get("index", pos_item.get("idx"))
                original_point = pos_item.get("point_original", pos_item.get("point"))
            else:
                idx = pos_item
                original_point = None
            
            if idx is None:
                continue
                
            # Load the image and extract bboxes
            example = dataset[dataset_split][idx]
            img_t, lab_t = example_to_tensors(example)
            
            # Extract bounding boxes
            bboxes = extract_bounding_boxes_from_mask(
                lab_t, class_id,
                min_pixels=50,
                min_bbox_size=20,
                max_bboxes=3
            )
            
            # If no valid bboxes from components, try single bbox
            if not bboxes:
                single_bbox = compute_single_bbox_from_mask(lab_t, class_id)
                if single_bbox:
                    bboxes = [single_bbox]
            
            if bboxes:
                bbox_organ["positives"].append({
                    "idx": idx,
                    "bboxes": bboxes,
                    "main_bbox": bboxes[0],
                    "original_point": original_point
                })
                print(f"    - Example {idx}: {len(bboxes)} bbox(es)")
        
        bbox_plan["plan"][class_id_str] = bbox_organ
    
    # Save the converted plan
    print(f"\nSaving bbox plan to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(bbox_plan, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Original point plan: {point_plan_file}")
    print(f"New bbox plan: {output_file}")
    print("\nSummary by organ:")
    for class_id_str, organ_info in bbox_plan["plan"].items():
        n_pos = len(organ_info["positives"])
        total_bboxes = sum(len(p["bboxes"]) for p in organ_info["positives"])
        print(f"  {organ_info['name']:30} {n_pos} positive(s), {total_bboxes} total bbox(es)")

if __name__ == "__main__":
    # Input and output files
    data_dir = Path(parent_dir) / "data_info" / "cholecseg8k"
    
    # Convert the seed 44 point plan
    point_file = data_dir / "fewshot_plan_train_pos1_nege1_negh1_seed44_excl100.json"
    bbox_file = data_dir / "fewshot_bbox_plan_converted_from_point_seed44.json"
    
    if not point_file.exists():
        print(f"ERROR: Point plan not found: {point_file}")
        sys.exit(1)
    
    convert_point_plan_to_bbox(point_file, bbox_file)
    print(f"\n✅ Done! New file created: {bbox_file}")