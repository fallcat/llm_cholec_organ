#!/usr/bin/env python3
"""Complete GoNoGoNet evaluation on CholecGoNoGo dataset with detailed logging."""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import traceback

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter
from endopoint.models.gonogonet_adapter import GoNoGoNetAdapter
from endopoint.prompts.registry import get_prompt_func


def compute_bbox_to_bbox_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Compute intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return float(inter_area / union_area) if union_area > 0 else 0.0


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


def evaluate_gonogonet_gonogo():
    """Complete evaluation of GoNoGoNet on CholecGoNoGo dataset."""
    
    print("=" * 80)
    print("GONOGONET EVALUATION ON CHOLEC_GONOGO - COMPLETE")
    print("=" * 80)
    
    # Setup
    dataset_adapter = CholecGoNoGoAdapter()
    model_adapter = GoNoGoNetAdapter(verbose=True)
    
    # Load test indices
    indices_file = Path("/shared_data0/weiqiuy/llm_cholec_organ/data_info/cholec_gonogo_balanced_200/balanced_test_indices_advanced_200.json")
    
    if indices_file.exists():
        with open(indices_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'indices' in data:
                test_indices = data['indices'][:151]  # Use 151 samples
            else:
                test_indices = data[:151]
    else:
        # Use first 151 test samples
        test_indices = list(range(151))
    
    print(f"Evaluating {len(test_indices)} test samples")
    print(f"Test indices: {test_indices[:5]}...{test_indices[-5:]}")
    
    # Output directory
    output_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create a debug directory for raw outputs
    debug_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/gonogonet_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check which files already exist
    existing_files = set()
    for f in output_dir.glob("test_*.json"):
        idx = int(f.stem.split('_')[1])
        existing_files.add(idx)
    
    print(f"Found {len(existing_files)} existing result files")
    
    # Process each test sample
    results = []
    organ_ids = [1, 2]  # Go zone, NoGo zone
    organ_names = ["Go (Safe to Incise)", "NoGo (Unsafe to Incise)"]
    
    for test_idx in tqdm(test_indices, desc="Evaluating GoNoGoNet"):
        # Skip if already processed
        if test_idx in existing_files:
            print(f"  Skipping {test_idx} (already processed)")
            continue
        
        try:
            # Load example
            example = dataset_adapter.get_example_by_global_index(test_idx)
            
            # Get ground truth
            img_tensor, lab_tensor = dataset_adapter.example_to_tensors(example)
            full_mask = lab_tensor.numpy()
            image = example.image
            
            # Prepare prompt for GoNoGoNet
            prompt_func = get_prompt_func("explain_qna")
            system_prompt, user_prompt = prompt_func(
                organ_list=organ_names,
                example_type="positive",  # Not used for detection
                num_examples=0  # Zero-shot
            )
            
            # Query the model
            query = [(user_prompt, image)]
            responses = model_adapter([query], system_prompt=system_prompt)
            response = responses[0]
            
            # Save raw response for debugging
            debug_file = debug_dir / f"raw_response_{test_idx:05d}_{timestamp}.json"
            with open(debug_file, 'w') as f:
                json.dump({
                    "test_idx": test_idx,
                    "prompt": user_prompt,
                    "raw_response": response
                }, f, indent=2)
            
            # Parse response
            try:
                result_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"  Error parsing JSON for sample {test_idx}: {e}")
                print(f"  Raw response: {response[:200]}...")
                continue
            
            # Process results for each organ
            sample_result = {
                "sample_idx": test_idx,
                "organs": []
            }
            
            for organ_id, organ_name in zip(organ_ids, organ_names):
                organ_data = {
                    "test_idx": test_idx,
                    "organ_id": organ_id,
                    "organ_name": organ_name
                }
                
                # Get ground truth
                organ_mask = (full_mask == organ_id).astype(np.uint8)
                gt_present = organ_mask.sum() > 0
                organ_data["ground_truth_present"] = int(gt_present)
                
                # Get ground truth bbox
                if gt_present:
                    ys, xs = np.where(organ_mask > 0)
                    if len(ys) > 0:
                        gt_bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
                        organ_data["ground_truth_bboxes"] = [gt_bbox]
                
                # Get prediction from response
                if organ_name in result_data:
                    pred_info = result_data[organ_name]
                    pred_present = pred_info.get("present", False)
                    pred_bbox = pred_info.get("bbox", None)
                    
                    organ_data["predicted_present"] = int(pred_present)
                    
                    if pred_bbox and pred_present:
                        organ_data["predicted_bboxes"] = [pred_bbox]
                        
                        # Compute IoU metrics if both GT and pred are present
                        if gt_present:
                            # Bbox-to-bbox IoU
                            iou_bbox = compute_bbox_to_bbox_iou(pred_bbox, gt_bbox)
                            organ_data["iou_bbox_to_bbox"] = iou_bbox
                            
                            # Bbox-to-mask IoU
                            iou_mask = compute_bbox_to_mask_iou(pred_bbox, organ_mask)
                            organ_data["iou_bbox_to_mask"] = iou_mask
                            
                            print(f"    {organ_name}: IoU-B={iou_bbox:.3f}, IoU-M={iou_mask:.3f}")
                    else:
                        # No bbox predicted
                        organ_data["predicted_bboxes"] = []
                        if pred_present:
                            print(f"    WARNING: {organ_name} predicted present but no bbox!")
                            # Log this case
                            with open(debug_dir / f"missing_bbox_{test_idx:05d}_{organ_id}.txt", 'w') as f:
                                f.write(f"Organ {organ_name} predicted present but no bbox\n")
                                f.write(f"Raw response:\n{response}\n")
                else:
                    # Organ not in response at all
                    organ_data["predicted_present"] = 0
                    organ_data["predicted_bboxes"] = []
                    print(f"    WARNING: {organ_name} not found in response!")
                
                # Check if model included mask
                if organ_name in result_data and "mask" in result_data[organ_name]:
                    organ_data["predicted_mask"] = True
                else:
                    organ_data["predicted_mask"] = False
                
                sample_result["organs"].append(organ_data)
            
            # Save result
            result_file = output_dir / f"test_{test_idx:05d}.json"
            with open(result_file, 'w') as f:
                json.dump(sample_result, f, indent=2)
            
            results.append(sample_result)
            
        except Exception as e:
            print(f"Error processing sample {test_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    # Count total files now
    final_count = len(list(output_dir.glob("test_*.json")))
    print(f"Total result files: {final_count}/{len(test_indices)}")
    
    if results:
        # Compute aggregate metrics from new results
        all_ious_bbox = []
        all_ious_mask = []
        all_presence_correct = []
        
        for result in results:
            for organ in result["organs"]:
                # Presence accuracy
                if "ground_truth_present" in organ and "predicted_present" in organ:
                    correct = organ["ground_truth_present"] == organ["predicted_present"]
                    all_presence_correct.append(int(correct))
                
                # IoU metrics
                if "iou_bbox_to_bbox" in organ:
                    all_ious_bbox.append(organ["iou_bbox_to_bbox"])
                if "iou_bbox_to_mask" in organ:
                    all_ious_mask.append(organ["iou_bbox_to_mask"])
        
        if all_presence_correct:
            print(f"\nNew samples processed: {len(results)}")
            print(f"Presence accuracy (new): {np.mean(all_presence_correct):.3f}")
            if all_ious_bbox:
                print(f"Mean IoU-B (new): {np.mean(all_ious_bbox):.3f}")
            if all_ious_mask:
                print(f"Mean IoU-M (new): {np.mean(all_ious_mask):.3f}")
    
    print(f"\nDebug outputs saved to: {debug_dir}")
    print(f"Results saved to: {output_dir}")
    
    # Save summary
    summary = {
        "dataset": "cholec_gonogo",
        "model": "gonogonet",
        "num_samples": final_count,
        "timestamp": timestamp,
        "debug_dir": str(debug_dir)
    }
    
    with open(output_dir / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ GoNoGoNet evaluation completed successfully!")


if __name__ == "__main__":
    evaluate_gonogonet_gonogo()