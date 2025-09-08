#!/usr/bin/env python3
"""
Naive Baseline for Bounding Box Evaluation
Always predicts the entire image as the bounding box for any detected organ.
This establishes a lower bound for IoU performance.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
from endopoint.datasets.cholec_organs import CholecOrgansAdapter


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


def run_naive_baseline(dataset_name="cholecseg8k", num_samples=200, presence_mode="perfect", box_mode="full"):
    """
    Run naive baseline evaluation.
    
    Args:
        dataset_name: "cholecseg8k", "cholec_organs", or "cholec_gonogo"
        num_samples: Number of samples to evaluate
        presence_mode: How to determine organ presence
            - "perfect": Use ground truth presence (oracle) - CHEATING!
            - "all": Always predict all organs are present
            - "random": Random 50% chance for each organ
        box_mode: How to generate bounding boxes
            - "full": Always predict entire image as bounding box
            - "random": Generate random bounding boxes (x2 > x1, y2 > y1)
    """
    
    print("=" * 80)
    print(f"NAIVE BASELINE - {dataset_name.upper()}")
    print("=" * 80)
    print(f"Box strategy: {box_mode}")
    print(f"Presence mode: {presence_mode}")
    print(f"Samples: {num_samples}")
    print()
    
    # Load dataset
    if dataset_name == "cholecseg8k":
        data_dir = "/shared_data0/weiqiuy/datasets/cholecseg8k"
        dataset_adapter = CholecSeg8kLocalAdapter(data_dir=data_dir)
        organ_ids = list(range(1, 13))  # 12 organs
        image_width = 854
        image_height = 480
    elif dataset_name == "cholec_organs":
        dataset_adapter = CholecOrgansAdapter()
        organ_ids = [1, 2, 3]  # Liver, Gallbladder, Hepatocystic Triangle
        image_width = 640
        image_height = 384
    elif dataset_name == "cholec_gonogo":
        # Import the GoNoGo adapter
        from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter
        dataset_adapter = CholecGoNoGoAdapter()
        organ_ids = [1, 2]  # Go zone, No-go zone
        image_width = 640  # FIXED: Was incorrectly 854
        image_height = 384  # FIXED: Was incorrectly 480
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load test indices
    indices_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
    indices_file = indices_dir / "balanced_test_indices_advanced_200.json"
    
    if indices_file.exists():
        with open(indices_file, 'r') as f:
            data = json.load(f)
            # Handle different formats: dict with 'indices' key or direct list
            if isinstance(data, dict) and 'indices' in data:
                test_indices = data['indices']
            else:
                test_indices = data
            test_indices = test_indices[:num_samples]
    else:
        # Use first N test samples
        total_test = dataset_adapter.total("test")
        test_indices = list(range(min(num_samples, total_test)))
    
    print(f"Selected test indices: {test_indices[:5]}{'...' if len(test_indices) > 5 else ''}")
    
    # Initialize random generator for reproducible random boxes
    if box_mode == "random":
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    # Results storage
    all_results = []
    all_presence_correct = []
    all_ious_bbox = []
    all_ious_mask = []
    
    # Process each sample
    for idx in tqdm(test_indices, desc="Evaluating naive baseline"):
        # Load example
        example = dataset_adapter.get_example_by_global_index(idx)
        
        # Get ground truth
        _, lab_tensor = dataset_adapter.example_to_tensors(example)
        full_mask = lab_tensor.numpy()
        
        # Determine ground truth presence for each organ
        gt_presence = []
        gt_bboxes = {}
        
        for organ_id in organ_ids:
            organ_mask = (full_mask == organ_id).astype(np.uint8)
            is_present = organ_mask.sum() > 0
            gt_presence.append(1 if is_present else 0)
            
            if is_present:
                # Compute ground truth bbox
                ys, xs = np.where(organ_mask > 0)
                if len(ys) > 0:
                    gt_bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
                    gt_bboxes[organ_id] = gt_bbox
        
        # Determine predicted presence based on mode
        if presence_mode == "perfect":
            pred_presence = gt_presence.copy()
        elif presence_mode == "all":
            pred_presence = [1] * len(organ_ids)
        elif presence_mode == "random":
            np.random.seed(idx)  # Reproducible randomness
            pred_presence = [np.random.randint(0, 2) for _ in organ_ids]
        else:
            raise ValueError(f"Unknown presence mode: {presence_mode}")
        
        # Compute metrics
        sample_result = {
            "sample_idx": idx,
            "y_true": gt_presence,
            "y_pred": pred_presence,
            "organs": []
        }
        
        # Presence accuracy
        presence_correct = [1 if gt == pred else 0 for gt, pred in zip(gt_presence, pred_presence)]
        all_presence_correct.extend(presence_correct)
        
        # IoU metrics for each organ
        for i, organ_id in enumerate(organ_ids):
            organ_data = {
                "organ_id": organ_id,
                "ground_truth_present": gt_presence[i],
                "predicted_present": pred_presence[i],
            }
            
            # Only compute IoU if both GT and pred say present
            if gt_presence[i] == 1 and pred_presence[i] == 1:
                # Generate the predicted bbox based on box_mode
                if box_mode == "full":
                    pred_bbox = [0, 0, image_width, image_height]
                elif box_mode == "random":
                    # Generate random box with x2 > x1 and y2 > y1
                    # Use combination of sample idx and organ_id for unique seed per organ per sample
                    local_seed = idx * 100 + organ_id
                    local_rng = np.random.RandomState(local_seed)
                    
                    # Random x coordinates (ensure x2 > x1)
                    x1 = local_rng.randint(0, image_width - 1)
                    x2 = local_rng.randint(x1 + 1, image_width + 1)  # x2 must be > x1
                    
                    # Random y coordinates (ensure y2 > y1) 
                    y1 = local_rng.randint(0, image_height - 1)
                    y2 = local_rng.randint(y1 + 1, image_height + 1)  # y2 must be > y1
                    
                    pred_bbox = [x1, y1, x2, y2]
                else:
                    raise ValueError(f"Unknown box mode: {box_mode}")
                
                # Bbox-to-bbox IoU
                if organ_id in gt_bboxes:
                    iou_bbox = compute_bbox_to_bbox_iou(pred_bbox, gt_bboxes[organ_id])
                    organ_data["iou_bbox_to_bbox"] = iou_bbox
                    all_ious_bbox.append(iou_bbox)
                
                # Bbox-to-mask IoU
                organ_mask = (full_mask == organ_id).astype(np.uint8)
                iou_mask = compute_bbox_to_mask_iou(pred_bbox, organ_mask)
                organ_data["iou_bbox_to_mask"] = iou_mask
                all_ious_mask.append(iou_mask)
                
                organ_data["predicted_bbox"] = pred_bbox
                organ_data["ground_truth_bbox"] = gt_bboxes.get(organ_id)
            
            sample_result["organs"].append(organ_data)
        
        all_results.append(sample_result)
    
    # Compute aggregate metrics
    presence_accuracy = np.mean(all_presence_correct) if all_presence_correct else 0.0
    mean_iou_bbox = np.mean(all_ious_bbox) if all_ious_bbox else 0.0
    mean_iou_mask = np.mean(all_ious_mask) if all_ious_mask else 0.0
    iou_at_50_bbox = np.mean([iou >= 0.5 for iou in all_ious_bbox]) if all_ious_bbox else 0.0
    iou_at_50_mask = np.mean([iou >= 0.5 for iou in all_ious_mask]) if all_ious_mask else 0.0
    
    # Save results - use the same directory structure as other models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output directory based on dataset
    if dataset_name == "cholecseg8k":
        base_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick")
    elif dataset_name == "cholec_organs":
        base_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick")
    elif dataset_name == "cholec_gonogo":
        base_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create subdirectory for the evaluation mode
    # Using "zeroshot_combined" as the mode since naive baseline doesn't use few-shot
    mode_dir = base_dir / "zeroshot_combined"
    
    # Model name includes both presence mode and box mode
    model_name = f"naive_baseline_{presence_mode}_{box_mode}"
    output_dir = mode_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save to a timestamped directory for easy access
    backup_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/results/naive_baseline_{dataset_name}_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual results to main directory (all samples)
    for i, result in enumerate(all_results):
        # Format filename to match other models: test_XXXXX.json
        filename = f"test_{test_indices[i]:05d}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Also save first 10 to backup directory for inspection
    for i, result in enumerate(all_results[:10]):
        with open(backup_dir / f"test_{test_indices[i]:05d}.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save summary in the format expected by the main evaluation pipeline
    summary = {
        "model": f"naive_baseline_{presence_mode}_{box_mode}",
        "num_samples": num_samples,
        "detection_mode": "combined",  # Naive baseline is always "combined"
        "use_fewshot": False,  # Naive baseline doesn't use few-shot
        "evaluation": "Zero-shot Combined",
        "presence_mode": presence_mode,
        "box_mode": box_mode,
        "timestamp": timestamp,
        "metrics": {
            "presence_accuracy": presence_accuracy,
            "mean_iou_bbox_to_bbox": mean_iou_bbox,
            "mean_iou_bbox_to_mask": mean_iou_mask,
            "iou_at_50_bbox_to_bbox": iou_at_50_bbox,
            "iou_at_50_bbox_to_mask": iou_at_50_mask,
            "elapsed_seconds": 0.0  # Naive baseline is instant
        }
    }
    
    # Save summary to main directory
    with open(output_dir / "summary_combined_zeroshot.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save to backup directory
    with open(backup_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "=" * 80)
    print("NAIVE BASELINE RESULTS")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Presence mode: {presence_mode}")
    print(f"Box mode: {box_mode}")
    print(f"Samples evaluated: {num_samples}")
    print(f"Results saved to:")
    print(f"  Main: {output_dir}")
    print(f"  Backup: {backup_dir}")
    print()
    print("📊 METRICS:")
    print("-" * 60)
    print(f"Presence Accuracy: {presence_accuracy:.1%}")
    print(f"Bbox-to-Bbox IoU:  {mean_iou_bbox:.3f}")
    print(f"Bbox-to-Mask IoU:  {mean_iou_mask:.3f}")
    print(f"IoU@0.5 (BBox):    {iou_at_50_bbox:.1%}")
    print(f"IoU@0.5 (Mask):    {iou_at_50_mask:.1%}")
    print(f"Total predictions: {len(all_ious_bbox)}")
    print("-" * 60)
    
    return summary


def main():
    """Run naive baseline for datasets."""
    
    # Configuration from environment or defaults
    DATASET = os.environ.get('EVAL_DATASET', 'all')  # "cholecseg8k", "cholec_organs", "cholec_gonogo", "all"
    NUM_SAMPLES = int(os.environ.get('EVAL_NUM_SAMPLES', '200'))
    PRESENCE_MODE = os.environ.get('EVAL_PRESENCE_MODE', 'all')  # "perfect", "all", or "random"
    BOX_MODE = os.environ.get('EVAL_BOX_MODE', 'full')  # "full" or "random"
    
    if DATASET == 'all':
        datasets = ['cholecseg8k', 'cholec_organs', 'cholec_gonogo']
    elif DATASET == 'both':
        # Legacy support for "both"
        datasets = ['cholecseg8k', 'cholec_organs']
    else:
        datasets = [DATASET]
    
    all_summaries = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Running naive baseline for {dataset_name}")
        print(f"{'='*80}\n")
        
        summary = run_naive_baseline(
            dataset_name=dataset_name,
            num_samples=NUM_SAMPLES,
            presence_mode=PRESENCE_MODE,
            box_mode=BOX_MODE
        )
        
        all_summaries[dataset_name] = summary
    
    # Print comparison if both datasets
    if len(all_summaries) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON ACROSS DATASETS")
        print("=" * 80)
        
        for dataset_name, summary in all_summaries.items():
            metrics = summary['metrics']
            print(f"\n{dataset_name.upper()}:")
            print(f"  Presence: {metrics['presence_accuracy']:.1%}")
            print(f"  IoU (BBox): {metrics['mean_iou_bbox_to_bbox']:.3f}")
            print(f"  IoU (Mask): {metrics['mean_iou_bbox_to_mask']:.3f}")


if __name__ == "__main__":
    main()