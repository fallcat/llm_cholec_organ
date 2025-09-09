#!/usr/bin/env python3
"""
Standalone Per-Organ Performance Analysis Script

This script analyzes model performance on individual organs across datasets.
It includes robust error handling and flexible data structure parsing.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Configuration
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

results_dirs = {
    "CholecSeg8k": Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick"),
    "CholecOrgans": Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick"),
    "CholecGoNoGo": Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick")
}

def load_per_organ_results(dataset="CholecSeg8k", mode="zeroshot_combined"):
    """Load per-organ performance results with robust error handling."""
    
    print(f"Loading per-organ results for {dataset}...")
    
    if dataset not in results_dirs:
        print(f"Warning: Unknown dataset {dataset}")
        return {}
    
    results_dir = results_dirs[dataset]
    mode_dir = results_dir / mode
    per_organ_results = {}
    
    if not mode_dir.exists():
        print(f"Warning: {mode_dir} does not exist")
        return per_organ_results
    
    for model_dir in mode_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        if model_name not in MODEL_NAME_MAPPING:
            continue
        display_name = MODEL_NAME_MAPPING.get(model_name, model_name)
        
        print(f"  Processing {display_name}...")
        
        # Load all test result files
        test_files = list(model_dir.glob("test_*.json"))
        
        if not test_files:
            print(f"    No test files found")
            continue
        
        # Organize by organ
        organ_data = defaultdict(lambda: {'y_true': [], 'y_pred': [], 'ious_bbox': [], 'ious_mask': []})
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"    Warning: Could not load {test_file}: {e}")
                continue
            
            # Process each organ - handle different data structures
            organs_list = data.get('organs', [])
            if not organs_list:
                continue
                
            for organ in organs_list:
                # Get organ name - handle different possible field names
                organ_name = None
                for name_field in ['organ_name', 'name', 'organ']:
                    if name_field in organ:
                        organ_name = organ[name_field]
                        break
                
                if not organ_name:
                    print(f"    Warning: No organ name in {test_file}, keys: {list(organ.keys())}")
                    continue
                
                gt_present = organ.get('ground_truth_present', 0)
                pred_present = organ.get('predicted_present', 0)
                
                organ_data[organ_name]['y_true'].append(gt_present)
                organ_data[organ_name]['y_pred'].append(pred_present)
                
                # Add IoU values if both GT and prediction are present
                if gt_present and pred_present:
                    # Try different IoU field names
                    iou_bbox = None
                    for iou_field in ['iou_bbox_to_bbox', 'iou', 'bbox_iou']:
                        if iou_field in organ and organ[iou_field] is not None:
                            iou_bbox = organ[iou_field]
                            break
                    
                    if iou_bbox is not None:
                        organ_data[organ_name]['ious_bbox'].append(iou_bbox)
                    
                    # Try different mask IoU field names
                    iou_mask = None
                    for iou_field in ['iou_bbox_to_mask', 'bbox_to_mask_iou', 'mask_iou']:
                        if iou_field in organ and organ[iou_field] is not None:
                            iou_mask = organ[iou_field]
                            break
                    
                    if iou_mask is not None:
                        organ_data[organ_name]['ious_mask'].append(iou_mask)
        
        # Compute per-organ metrics
        model_organ_metrics = {}
        for organ_name, data in organ_data.items():
            metrics = {}
            
            # Confusion matrix and accuracy metrics
            if data['y_true'] and data['y_pred']:
                y_true = np.array(data['y_true'])
                y_pred = np.array(data['y_pred'])
                
                # Compute confusion matrix components
                tp = np.sum((y_true == 1) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                # Store confusion matrix metrics
                metrics['tp'] = int(tp)
                metrics['tn'] = int(tn)
                metrics['fp'] = int(fp)
                metrics['fn'] = int(fn)
                
                # Calculate accuracy
                total = len(y_true)
                metrics['accuracy'] = (tp + tn) / total if total > 0 else 0.0
                metrics['presence_acc'] = metrics['accuracy']  # Keep for backward compatibility
                metrics['n_samples'] = total
                
                # Additional useful metrics
                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
            else:
                # No data available
                metrics.update({
                    'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                    'accuracy': 0.0, 'presence_acc': 0.0, 'n_samples': 0,
                    'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
                })
            
            # IoU metrics - CLIP models don't have IoU
            display_name_for_check = display_name
            if display_name in ["PeskaVLP", "RASO"]:
                # CLIP-based models don't do localization
                metrics['iou_bbox'] = None  # Use None to indicate N/A
                metrics['iou_mask'] = None
                metrics['n_iou_samples'] = 0
            else:
                metrics['iou_bbox'] = np.mean(data['ious_bbox']) if data['ious_bbox'] else 0.0
                metrics['iou_mask'] = np.mean(data['ious_mask']) if data['ious_mask'] else 0.0
                metrics['n_iou_samples'] = len(data['ious_bbox'])
            
            model_organ_metrics[organ_name] = metrics
        
        if model_organ_metrics:
            per_organ_results[display_name] = model_organ_metrics
            print(f"    Found {len(model_organ_metrics)} organs")
    
    return per_organ_results

def create_per_organ_heatmap(per_organ_results, dataset_name, metric='presence_acc'):
    """Create a heatmap showing model performance by organ."""
    
    # Define the same model order as in main_table_results_new.ipynb
    MODEL_ORDER = [
        # Baselines
        "Naive (Full Box)",
        "Naive (Random Box)",
        # Commercial LVLMs
        "Claude-Sonnet-4",
        "GPT-4.1",
        "Gemini-2.0-Flash",
        # Open-Source LVLMs
        "Llava-v1.6-Mistral-7B",
        "Pixtral-12B",
        "Qwen2.5-VL-7B",
        # CLIP-based Models
        "PeskaVLP",
        "RASO",
        # Task-Specific Models
        "CholeNet",
        "GoNoGoNet"
    ]
    
    # Get all unique organs
    all_organs = set()
    for model, organ_data in per_organ_results.items():
        all_organs.update(organ_data.keys())
    
    if not all_organs:
        print(f"Warning: No data to plot for {dataset_name} {metric}")
        return None
    
    all_organs = sorted(all_organs)
    
    # Filter MODEL_ORDER to only include models we have data for
    all_models = [m for m in MODEL_ORDER if m in per_organ_results]
    
    if not all_models:
        print(f"Warning: No models with data found for {dataset_name}")
        return None
    
    # Create matrix
    matrix = []
    for model in all_models:
        row = []
        for organ in all_organs:
            if model in per_organ_results and organ in per_organ_results[model]:
                value = per_organ_results[model][organ][metric]
                # Convert None to NaN for CLIP models' IoU metrics
                if value is None:
                    value = np.nan
            else:
                value = np.nan
            row.append(value)
        matrix.append(row)
    
    # Create heatmap
    figsize = (max(8, len(all_organs)), max(6, len(all_models) * 0.5))
    plt.figure(figsize=figsize)
    
    # Create mask for NaN values
    mask = np.isnan(matrix)
    
    # Choose appropriate colormap and range
    if metric == 'presence_acc':
        cmap = 'RdYlBu_r'
        vmin, vmax = 0, 1
    else:
        cmap = 'viridis'
        vmin, vmax = 0, None
    
    sns.heatmap(matrix, 
                xticklabels=all_organs, 
                yticklabels=all_models,
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                vmin=vmin, 
                vmax=vmax,
                mask=mask,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    # Format the title based on the metric
    if metric == 'accuracy':
        title_metric = 'Presence Accuracy'
    elif metric == 'iou_bbox':
        title_metric = 'Bbox-to-Bbox IoU'
    elif metric == 'iou_mask':
        title_metric = 'Bbox-to-Mask IoU'
    else:
        title_metric = metric.replace("_", " ").title()
    
    plt.title(f'{dataset_name} - {title_metric} by Model and Organ')
    plt.xlabel('Organ')
    plt.ylabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot in notebooks/images/ directory in both formats
    notebooks_images_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/notebooks/images")
    notebooks_images_dir.mkdir(parents=True, exist_ok=True)
    
    output_file_png = f"{dataset_name}_{metric}_heatmap.png"
    output_file_pdf = f"{dataset_name}_{metric}_heatmap.pdf"
    
    output_path_png = notebooks_images_dir / output_file_png
    output_path_pdf = notebooks_images_dir / output_file_pdf
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    
    print(f"Saved heatmap: {output_path_png}")
    print(f"Saved heatmap: {output_path_pdf}")
    
    return plt.gcf()

def print_confusion_matrix_metrics(per_organ_results, dataset_name):
    """Print detailed confusion matrix metrics (TP, FN, TN, FP, accuracy) for each organ and model."""
    
    # Define model order to match main table (same order as in main_table_results_new.ipynb)
    MODEL_ORDER = [
        # Baselines
        "Naive (Full Box)",
        "Naive (Random Box)",
        # Commercial LVLMs
        "Claude-Sonnet-4",
        "GPT-4.1", 
        "Gemini-2.0-Flash",
        # Open-Source LVLMs
        "Llava-v1.6-Mistral-7B",
        "Pixtral-12B",
        "Qwen2.5-VL-7B",
        # CLIP-based Models
        "PeskaVLP",
        "RASO",
        # Task-Specific Models
        "CholeNet",
        "GoNoGoNet"
    ]
    
    # CLIP-based models that don't do localization
    CLIP_MODELS = ["PeskaVLP", "RASO"]
    
    # Get all unique organs
    all_organs = set()
    for model, organ_data in per_organ_results.items():
        all_organs.update(organ_data.keys())
    
    all_organs = sorted(all_organs)
    
    if not all_organs:
        print(f"No organs found for {dataset_name}")
        return
    
    print(f"\n{dataset_name} - Detailed Confusion Matrix Metrics:")
    print("=" * 120)
    
    for organ in all_organs:
        print(f"\n{organ.upper()}:")
        print("-" * 120)
        print(f"{'Model':<25} {'TP':<4} {'TN':<4} {'FP':<4} {'FN':<4} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'N':<4}")
        print("-" * 120)
        
        # Get model metrics for this organ in the specified order
        for model in MODEL_ORDER:
            if model in per_organ_results and organ in per_organ_results[model]:
                metrics = per_organ_results[model][organ]
                
                tp = metrics.get('tp', 0)
                tn = metrics.get('tn', 0)
                fp = metrics.get('fp', 0)
                fn = metrics.get('fn', 0)
                acc = metrics.get('accuracy', 0.0)
                prec = metrics.get('precision', 0.0)
                rec = metrics.get('recall', 0.0)
                f1 = metrics.get('f1_score', 0.0)
                n = metrics.get('n_samples', 0)
                
                print(f"{model:<25} {tp:<4} {tn:<4} {fp:<4} {fn:<4} {acc:<6.3f} {prec:<6.3f} {rec:<6.3f} {f1:<6.3f} {n:<4}")

def print_top_models_per_organ(per_organ_results, dataset_name, metric='presence_acc', top_k=3):
    """Print the top k models for each organ."""
    
    # CLIP-based models that don't do localization
    CLIP_MODELS = ["PeskaVLP", "RASO"]
    
    # Get all unique organs
    all_organs = set()
    for model, organ_data in per_organ_results.items():
        all_organs.update(organ_data.keys())
    
    all_organs = sorted(all_organs)
    
    if not all_organs:
        print(f"No organs found for {dataset_name}")
        return
    
    print(f"\n{dataset_name} - Top {top_k} models per organ ({metric.replace('_', ' ').title()}):")
    print("=" * 80)
    
    for organ in all_organs:
        # Get model scores for this organ
        organ_scores = []
        for model, organ_data in per_organ_results.items():
            if organ in organ_data:
                score = organ_data[organ][metric]
                # Skip CLIP models for IoU metrics
                if metric in ['iou_bbox', 'iou_mask'] and model in CLIP_MODELS:
                    continue
                # Skip if score is None (for CLIP models)
                if score is None:
                    continue
                n_samples = organ_data[organ].get('n_samples', 0)
                organ_scores.append((score, model, n_samples))
        
        # Sort by score (descending)
        organ_scores.sort(reverse=True)
        
        if organ_scores:
            print(f"\n{organ}:")
            for i, (score, model, n_samples) in enumerate(organ_scores[:top_k]):
                print(f"  {i+1}. {model}: {score:.3f} (n={n_samples})")
            
            if len(organ_scores) > top_k:
                print(f"  ... and {len(organ_scores) - top_k} more models")

def main():
    """Run the per-organ analysis for all three datasets."""
    
    print("=" * 80)
    print("PER-ORGAN CONFUSION MATRIX ANALYSIS - ALL DATASETS")
    print("=" * 80)
    
    # Analyze all three datasets
    datasets = ["CholecSeg8k", "CholecOrgans", "CholecGoNoGo"]
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset.upper()}")
        print('='*60)
        
        # Load per-organ results
        per_organ_results = load_per_organ_results(dataset, "zeroshot_combined")
        
        if not per_organ_results:
            print(f"No per-organ results found for {dataset}")
            continue
        
        # Print summary
        n_models = len(per_organ_results)
        all_organs = set()
        for model_data in per_organ_results.values():
            all_organs.update(model_data.keys())
        n_organs = len(all_organs)
        
        print(f"\nSummary: {n_models} models, {n_organs} organs")
        print(f"Organs: {', '.join(sorted(all_organs))}")
        
        # Print detailed confusion matrix metrics
        print_confusion_matrix_metrics(per_organ_results, dataset)
        
        # Create heatmaps for different metrics
        # 1. Accuracy/Presence heatmap
        try:
            fig = create_per_organ_heatmap(per_organ_results, dataset, 'accuracy')
            if fig is not None:
                plt.close(fig)  # Close to save memory
        except Exception as e:
            print(f"Error creating accuracy heatmap: {e}")
        
        # 2. Bbox-to-bbox IoU heatmap
        try:
            fig = create_per_organ_heatmap(per_organ_results, dataset, 'iou_bbox')
            if fig is not None:
                plt.close(fig)  # Close to save memory
        except Exception as e:
            print(f"Error creating bbox IoU heatmap: {e}")
        
        # 3. Bbox-to-mask IoU heatmap
        try:
            fig = create_per_organ_heatmap(per_organ_results, dataset, 'iou_mask')
            if fig is not None:
                plt.close(fig)  # Close to save memory
        except Exception as e:
            print(f"Error creating mask IoU heatmap: {e}")
        
        # Print top models per organ for accuracy
        print_top_models_per_organ(per_organ_results, dataset, 'accuracy', top_k=3)
    
    print("\n" + "="*80)
    print("PER-ORGAN CONFUSION MATRIX ANALYSIS COMPLETE")
    print("="*80)
    print("Results show TP (True Positives), TN (True Negatives), FP (False Positives), FN (False Negatives)")
    print("Acc (Accuracy), Prec (Precision), Rec (Recall), F1 (F1-Score), N (Total Samples)")
    print("\nHeatmaps saved to: /shared_data0/weiqiuy/llm_cholec_organ/notebooks/images/")

if __name__ == "__main__":
    main()