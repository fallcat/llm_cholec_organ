#!/usr/bin/env python3
"""
Evaluation script that saves predicted masks from GoNoGoNet and CholeNet.
Masks are saved as numpy arrays alongside the JSON results.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import traceback
from PIL import Image
import base64
import io

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
from endopoint.datasets.cholec_organs import CholecOrgansAdapter
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter
from endopoint.models.gonogonet_adapter import GoNoGoNetAdapter
from endopoint.models.cholenet_adapter import CholeNetAdapter
from endopoint.prompts.registry import get_prompt_func


def decode_base64_mask(mask_base64):
    """Decode base64 encoded mask back to numpy array."""
    # Decode base64 to bytes
    mask_bytes = base64.b64decode(mask_base64)
    
    # Load as PIL Image
    mask_img = Image.open(io.BytesIO(mask_bytes))
    
    # Convert to numpy array
    mask_array = np.array(mask_img)
    
    return mask_array


def compute_mask_to_mask_iou(pred_mask, gt_mask):
    """Compute IoU between two masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


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


def evaluate_model_with_masks(model_name="gonogonet", dataset_name="cholec_gonogo", num_samples=10):
    """
    Evaluate a model and save predicted masks.
    
    Args:
        model_name: "gonogonet" or "cholenet"
        dataset_name: "cholecseg8k", "cholec_organs", or "cholec_gonogo"
        num_samples: Number of samples to evaluate
    """
    
    print("=" * 80)
    print(f"EVALUATING {model_name.upper()} ON {dataset_name.upper()} WITH MASK SAVING")
    print("=" * 80)
    
    # Load dataset adapter
    if dataset_name == "cholecseg8k":
        dataset_adapter = CholecSeg8kLocalAdapter()
        organ_ids = list(range(1, 13))  # 12 organs
    elif dataset_name == "cholec_organs":
        dataset_adapter = CholecOrgansAdapter()
        organ_ids = [1, 2, 3]  # Liver, Gallbladder, Hepatocystic Triangle
    elif dataset_name == "cholec_gonogo":
        dataset_adapter = CholecGoNoGoAdapter()
        organ_ids = [1, 2]  # Go zone, NoGo zone
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get organ names
    organ_names = [dataset_adapter.ID2LABEL[oid] for oid in organ_ids]
    print(f"Organs to detect: {organ_names}")
    
    # Load model adapter
    if model_name == "gonogonet":
        model_adapter = GoNoGoNetAdapter(verbose=True, return_masks=True)
    elif model_name == "cholenet":
        model_adapter = CholeNetAdapter(verbose=True, return_masks=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load test indices
    indices_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
    indices_file = indices_dir / "balanced_test_indices_advanced_200.json"
    
    if indices_file.exists():
        with open(indices_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'indices' in data:
                test_indices = data['indices'][:num_samples]
            else:
                test_indices = data[:num_samples]
    else:
        test_indices = list(range(num_samples))
    
    print(f"Evaluating {len(test_indices)} test samples: {test_indices[:5]}...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/{model_name}_{dataset_name}_masks_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create mask subdirectory
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Mask directory: {mask_dir}")
    
    # Process each sample
    results = []
    
    for test_idx in tqdm(test_indices, desc=f"Evaluating {model_name}"):
        try:
            # Load example
            example = dataset_adapter.get_example_by_global_index(test_idx)
            
            # Get ground truth
            img_tensor, lab_tensor = dataset_adapter.example_to_tensors(example)
            full_mask_gt = lab_tensor.numpy()
            image = example.image
            
            # Prepare prompt
            prompt_func = get_prompt_func("explain_qna")
            system_prompt, user_prompt = prompt_func(
                organ_list=organ_names,
                example_type="positive",
                num_examples=0  # Zero-shot
            )
            
            # Query the model
            query = [(user_prompt, image)]
            responses = model_adapter([query], system_prompt=system_prompt)
            response = responses[0]
            
            # Parse response
            try:
                result_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"  Error parsing JSON for sample {test_idx}: {e}")
                continue
            
            # Initialize sample result
            sample_result = {
                "sample_idx": test_idx,
                "model": model_name,
                "dataset": dataset_name,
                "timestamp": timestamp,
                "organs": []
            }
            
            # Check if full mask is provided
            full_pred_mask = None
            if "_full_mask" in result_data:
                mask_info = result_data["_full_mask"]
                if "encoded" in mask_info:
                    # Decode the full segmentation mask
                    full_pred_mask = decode_base64_mask(mask_info["encoded"])
                    
                    # Save full predicted mask
                    mask_file = mask_dir / f"pred_mask_{test_idx:05d}_full.npy"
                    np.save(mask_file, full_pred_mask)
                    sample_result["full_mask_file"] = str(mask_file.relative_to(output_dir))
                    
                    print(f"  Saved full mask for sample {test_idx}: shape {full_pred_mask.shape}")
            
            # Also save ground truth mask
            gt_mask_file = mask_dir / f"gt_mask_{test_idx:05d}.npy"
            np.save(gt_mask_file, full_mask_gt)
            sample_result["gt_mask_file"] = str(gt_mask_file.relative_to(output_dir))
            
            # Process each organ
            for organ_id, organ_name in zip(organ_ids, organ_names):
                organ_data = {
                    "organ_id": organ_id,
                    "organ_name": organ_name
                }
                
                # Get ground truth for this organ
                organ_mask_gt = (full_mask_gt == organ_id).astype(np.uint8)
                gt_present = organ_mask_gt.sum() > 0
                organ_data["ground_truth_present"] = int(gt_present)
                
                # Get ground truth bbox if present
                if gt_present:
                    ys, xs = np.where(organ_mask_gt > 0)
                    if len(ys) > 0:
                        gt_bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
                        organ_data["ground_truth_bbox"] = gt_bbox
                
                # Get prediction from response
                if organ_name in result_data:
                    pred_info = result_data[organ_name]
                    pred_present = pred_info.get("present", False)
                    pred_bbox = pred_info.get("bbox", None)
                    
                    organ_data["predicted_present"] = int(pred_present)
                    
                    if pred_bbox and pred_present:
                        organ_data["predicted_bbox"] = pred_bbox
                        
                        # Compute bbox-based IoU metrics
                        if gt_present:
                            iou_bbox = compute_bbox_to_bbox_iou(pred_bbox, gt_bbox)
                            organ_data["iou_bbox_to_bbox"] = iou_bbox
                            
                            iou_bbox_mask = compute_bbox_to_mask_iou(pred_bbox, organ_mask_gt)
                            organ_data["iou_bbox_to_mask"] = iou_bbox_mask
                    
                    # Check for organ-specific mask
                    if "mask" in pred_info:
                        # Decode organ mask
                        organ_pred_mask = decode_base64_mask(pred_info["mask"])
                        
                        # Save individual organ mask
                        organ_mask_file = mask_dir / f"pred_mask_{test_idx:05d}_organ_{organ_id}.npy"
                        np.save(organ_mask_file, organ_pred_mask)
                        organ_data["mask_file"] = str(organ_mask_file.relative_to(output_dir))
                        
                        # Compute mask-to-mask IoU if ground truth present
                        if gt_present:
                            # Convert to binary (in case it's 0-255)
                            organ_pred_mask_binary = (organ_pred_mask > 0).astype(np.uint8)
                            iou_mask = compute_mask_to_mask_iou(organ_pred_mask_binary, organ_mask_gt)
                            organ_data["iou_mask_to_mask"] = iou_mask
                            print(f"    {organ_name}: Mask IoU = {iou_mask:.3f}")
                    
                    # If we have full mask but no individual mask, extract from full mask
                    elif full_pred_mask is not None:
                        # Extract organ from full mask
                        organ_pred_mask = (full_pred_mask == organ_id).astype(np.uint8)
                        
                        if organ_pred_mask.sum() > 0:  # Only save if organ is present in mask
                            # Save extracted organ mask
                            organ_mask_file = mask_dir / f"pred_mask_{test_idx:05d}_organ_{organ_id}_extracted.npy"
                            np.save(organ_mask_file, organ_pred_mask)
                            organ_data["mask_file"] = str(organ_mask_file.relative_to(output_dir))
                            
                            # Compute mask-to-mask IoU
                            if gt_present:
                                iou_mask = compute_mask_to_mask_iou(organ_pred_mask, organ_mask_gt)
                                organ_data["iou_mask_to_mask"] = iou_mask
                                print(f"    {organ_name}: Extracted Mask IoU = {iou_mask:.3f}")
                else:
                    # Organ not in response
                    organ_data["predicted_present"] = 0
                    
                    # Check if organ is in full mask anyway
                    if full_pred_mask is not None:
                        organ_pred_mask = (full_pred_mask == organ_id).astype(np.uint8)
                        if organ_pred_mask.sum() > 0:
                            print(f"    WARNING: {organ_name} in mask but not in JSON response!")
                
                sample_result["organs"].append(organ_data)
            
            # Save result JSON
            result_file = output_dir / f"result_{test_idx:05d}.json"
            with open(result_file, 'w') as f:
                json.dump(sample_result, f, indent=2)
            
            results.append(sample_result)
            
        except Exception as e:
            print(f"Error processing sample {test_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Compute summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    total_organs = 0
    correct_presence = 0
    total_bbox_ious = []
    total_mask_ious = []
    
    for result in results:
        for organ in result["organs"]:
            total_organs += 1
            
            # Presence accuracy
            if "ground_truth_present" in organ and "predicted_present" in organ:
                if organ["ground_truth_present"] == organ["predicted_present"]:
                    correct_presence += 1
            
            # Collect IoU values
            if "iou_bbox_to_bbox" in organ:
                total_bbox_ious.append(organ["iou_bbox_to_bbox"])
            if "iou_mask_to_mask" in organ:
                total_mask_ious.append(organ["iou_mask_to_mask"])
    
    if total_organs > 0:
        print(f"Samples processed: {len(results)}")
        print(f"Total organ predictions: {total_organs}")
        print(f"Presence accuracy: {correct_presence/total_organs:.3f}")
        
        if total_bbox_ious:
            print(f"Mean bbox-to-bbox IoU: {np.mean(total_bbox_ious):.3f}")
        
        if total_mask_ious:
            print(f"Mean mask-to-mask IoU: {np.mean(total_mask_ious):.3f}")
            print(f"Number of masks saved: {len(total_mask_ious)}")
    
    # Save summary
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": len(results),
        "timestamp": timestamp,
        "presence_accuracy": correct_presence/total_organs if total_organs > 0 else 0,
        "mean_bbox_iou": float(np.mean(total_bbox_ious)) if total_bbox_ious else None,
        "mean_mask_iou": float(np.mean(total_mask_ious)) if total_mask_ious else None,
        "num_masks_saved": len(total_mask_ious),
        "output_dir": str(output_dir)
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Masks saved to: {mask_dir}")
    
    return summary


def main():
    """Run evaluation with mask saving for specified models and datasets."""
    
    # Parse command line arguments or use defaults
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate models with mask saving")
    parser.add_argument("--model", choices=["gonogonet", "cholenet", "both"], 
                       default="both", help="Model to evaluate")
    parser.add_argument("--dataset", choices=["cholec_gonogo", "cholec_organs", "cholecseg8k", "all"],
                       default="cholec_gonogo", help="Dataset to use")
    parser.add_argument("--samples", type=int, default=10, 
                       help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Determine which models and datasets to evaluate
    if args.model == "both":
        models = ["gonogonet", "cholenet"]
    else:
        models = [args.model]
    
    if args.dataset == "all":
        datasets = ["cholec_gonogo", "cholec_organs", "cholecseg8k"]
    else:
        datasets = [args.dataset]
    
    # Run evaluations
    all_summaries = []
    
    for model in models:
        for dataset in datasets:
            # Skip invalid combinations
            if model == "gonogonet" and dataset == "cholecseg8k":
                print(f"Skipping {model} on {dataset} (GoNoGoNet doesn't detect individual organs)")
                continue
            
            print(f"\n{'='*80}")
            print(f"Evaluating {model} on {dataset}")
            print(f"{'='*80}\n")
            
            try:
                summary = evaluate_model_with_masks(
                    model_name=model,
                    dataset_name=dataset,
                    num_samples=args.samples
                )
                all_summaries.append(summary)
            except Exception as e:
                print(f"Error evaluating {model} on {dataset}: {e}")
                traceback.print_exc()
    
    # Print final summary
    if all_summaries:
        print("\n" + "=" * 80)
        print("ALL EVALUATIONS COMPLETE")
        print("=" * 80)
        
        for summary in all_summaries:
            print(f"\n{summary['model']} on {summary['dataset']}:")
            print(f"  Samples: {summary['num_samples']}")
            print(f"  Presence accuracy: {summary['presence_accuracy']:.3f}")
            if summary['mean_bbox_iou'] is not None:
                print(f"  Mean bbox IoU: {summary['mean_bbox_iou']:.3f}")
            if summary['mean_mask_iou'] is not None:
                print(f"  Mean mask IoU: {summary['mean_mask_iou']:.3f}")
                print(f"  Masks saved: {summary['num_masks_saved']}")
            print(f"  Output: {summary['output_dir']}")


if __name__ == "__main__":
    main()