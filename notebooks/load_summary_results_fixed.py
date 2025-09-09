#!/usr/bin/env python3
"""Fixed version of load_summary_results that handles corrupted/empty JSON files."""

import json
import numpy as np
from pathlib import Path

def bootstrap_std(data, n_bootstrap=1000, seed=42):
    """Compute bootstrap standard deviation of the mean."""
    np.random.seed(seed)
    data = np.array(data)
    n = len(data)
    
    if n == 0:
        return 0.0
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    return np.std(bootstrap_means)

def load_summary_results_fixed(dataset="CholecSeg8k", mode="zeroshot_combined", compute_bootstrap=True):
    """Load results from individual test files and compute metrics with bootstrap confidence intervals.
    This version handles corrupted/empty JSON files gracefully.
    
    Args:
        dataset: Dataset name ("CholecSeg8k", "CholecOrgans", or "CholecGoNoGo")
        mode: Evaluation mode (e.g., "zeroshot_combined")
        compute_bootstrap: If True, compute bootstrap standard deviations
    """
    
    # Define result directories for all three datasets
    results_dirs = {
        "CholecSeg8k": Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick"),
        "CholecOrgans": Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick"),
        "CholecGoNoGo": Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick")
    }
    
    # Model name mapping
    MODEL_NAME_MAPPING = {
        "naive_baseline_all_full": "Naive (Full Box)",
        "naive_baseline_all_random": "Naive (Random Box)",
        "gpt-4.1": "GPT-4.1",
        "gemini-2.0-flash": "Gemini-2.0-Flash",
        "claude-sonnet-4-20250514": "Claude-Sonnet-4",
        "llava-hf_llava-v1.6-mistral-7b-hf": "Llava-v1.6-Mistral-7B",
        "Qwen_Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "mistralai_Pixtral-12B-2409": "Pixtral-12B",
        "peskavlp": "PeskaVLP",
        "raso": "RASO",
        "cholenet": "CholeNet",
        "gonogonet": "GoNoGoNet"
    }
    
    if dataset not in results_dirs:
        print(f"Warning: Unknown dataset {dataset}")
        return {}
    
    results_dir = results_dirs[dataset]
    mode_dir = results_dir / mode
    results = {}
    
    if not mode_dir.exists():
        print(f"Warning: {mode_dir} does not exist")
        return results
    
    for model_dir in mode_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        if model_name not in MODEL_NAME_MAPPING:
            continue
        display_name = MODEL_NAME_MAPPING.get(model_name, model_name)
        
        # Load all test result files
        test_files = list(model_dir.glob("test_*.json"))
        
        if not test_files:
            print(f"No test files found for {model_name} in {dataset}")
            continue
        
        all_y_true = []
        all_y_pred = []
        all_ious_bbox = []  # bbox-to-bbox IoU
        all_ious_mask = []  # bbox-to-mask IoU
        all_binary_correct = []  # For bootstrap of presence accuracy
        
        # Track skipped files
        skipped_files = []
        
        for test_file in test_files:
            try:
                # Check if file is empty
                if test_file.stat().st_size == 0:
                    skipped_files.append((test_file.name, "Empty file"))
                    continue
                
                with open(test_file, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        skipped_files.append((test_file.name, "Empty content"))
                        continue
                    
                    data = json.loads(content)
                
                # Collect presence accuracy data
                if 'y_true' in data and 'y_pred' in data:
                    all_y_true.extend(data['y_true'])
                    all_y_pred.extend(data['y_pred'])
                    # Store binary correctness for each prediction
                    all_binary_correct.extend([1 if yt == yp else 0 
                                              for yt, yp in zip(data['y_true'], data['y_pred'])])
                
                # Collect BOTH IoU types from individual organ data
                for organ in data.get('organs', []):
                    # Bbox-to-bbox IoU - check multiple possible key names
                    if 'iou_bbox_to_bbox' in organ and organ['iou_bbox_to_bbox'] is not None:
                        all_ious_bbox.append(organ['iou_bbox_to_bbox'])
                    elif 'iou' in organ and organ['iou'] is not None:
                        all_ious_bbox.append(organ['iou'])
                    
                    # Bbox-to-mask IoU
                    if 'iou_bbox_to_mask' in organ and organ['iou_bbox_to_mask'] is not None:
                        all_ious_mask.append(organ['iou_bbox_to_mask'])
                        
            except json.JSONDecodeError as e:
                skipped_files.append((test_file.name, f"JSON error: {str(e)[:50]}"))
                continue
            except Exception as e:
                skipped_files.append((test_file.name, f"Error: {str(e)[:50]}"))
                continue
        
        # Report skipped files if any
        if skipped_files:
            print(f"Warning: Skipped {len(skipped_files)} files for {display_name} in {dataset}:")
            for filename, reason in skipped_files[:3]:  # Show first 3
                print(f"  - {filename}: {reason}")
            if len(skipped_files) > 3:
                print(f"  ... and {len(skipped_files) - 3} more")
        
        # Compute metrics
        metrics = {}
        
        # Presence accuracy with bootstrap
        if all_binary_correct:
            metrics['presence_accuracy'] = np.mean(all_binary_correct)
            if compute_bootstrap:
                metrics['presence_accuracy_std'] = bootstrap_std(all_binary_correct)
        
        # Bbox-to-bbox IoU with bootstrap
        if all_ious_bbox:
            metrics['mean_iou_bbox'] = np.mean(all_ious_bbox)
            metrics['iou_at_50_bbox'] = np.mean([iou >= 0.5 for iou in all_ious_bbox])
            
            if compute_bootstrap:
                metrics['mean_iou_bbox_std'] = bootstrap_std(all_ious_bbox)
                iou_at_50_binary = [1 if iou >= 0.5 else 0 for iou in all_ious_bbox]
                metrics['iou_at_50_bbox_std'] = bootstrap_std(iou_at_50_binary)
        
        # Bbox-to-mask IoU with bootstrap
        if all_ious_mask:
            metrics['mean_iou_mask'] = np.mean(all_ious_mask)
            metrics['iou_at_50_mask'] = np.mean([iou >= 0.5 for iou in all_ious_mask])
            
            if compute_bootstrap:
                metrics['mean_iou_mask_std'] = bootstrap_std(all_ious_mask)
                iou_at_50_binary = [1 if iou >= 0.5 else 0 for iou in all_ious_mask]
                metrics['iou_at_50_mask_std'] = bootstrap_std(iou_at_50_binary)
        
        results[display_name] = metrics
        
        # Report successful loading
        valid_files = len(test_files) - len(skipped_files)
        print(f"Loaded {valid_files} files for {display_name} in {dataset} (both IoU types)")
    
    return results


# Test the function
if __name__ == "__main__":
    print("Testing fixed load_summary_results function...")
    print("="*80)
    
    # Test on CholecGoNoGo which has the empty file issue
    results = load_summary_results_fixed("CholecGoNoGo", "zeroshot_combined", compute_bootstrap=True)
    
    print(f"\nSuccessfully loaded results for {len(results)} models")
    for model, metrics in results.items():
        print(f"\n{model}:")
        if 'presence_accuracy' in metrics:
            print(f"  Presence accuracy: {metrics['presence_accuracy']:.3f}")
        if 'mean_iou_bbox' in metrics:
            print(f"  Bbox IoU: {metrics['mean_iou_bbox']:.3f}")
        if 'mean_iou_mask' in metrics:
            print(f"  Mask IoU: {metrics['mean_iou_mask']:.3f}")