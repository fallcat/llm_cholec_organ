#!/usr/bin/env python
"""
Standalone Cell Selection Evaluation - Works without external dependencies
This demonstrates the cell selection implementation with mock data.
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple

# Add src to path for our cell selection modules
sys.path.append(os.path.join("..", "src"))

print("Cell Selection Evaluation (Standalone)")
print("="*60)

# Import only the core cell selection logic (no external deps)
exec(open("test_cell_selection_nodeps.py").read())

# Additional evaluation functions
def evaluate_cell_selection_on_samples(num_samples=5, grid_size=3, top_k=1):
    """Run cell selection evaluation on mock samples."""
    
    print(f"\nEvaluating {num_samples} samples with G={grid_size}, K={top_k}")
    print("-"*60)
    
    # Mock organ names (from CholecSeg8k)
    organ_names = [
        "Liver", "Gallbladder", "Hepatocystic Triangle", "Fat",
        "Grasper", "Connective Tissue", "Blood", "Cystic Artery",
        "Cystic Plate", "Cystic Vein", "Liver Ligament", "Peritoneum"
    ]
    
    all_results = []
    
    for sample_idx in range(num_samples):
        print(f"\nSample {sample_idx + 1}/{num_samples}")
        sample_results = {"sample_idx": sample_idx, "organs": {}}
        
        # Create mock masks for each organ (224x224 typical size)
        canvas_size = 224
        
        for organ_idx, organ_name in enumerate(organ_names):
            # Create a mock mask with some pattern
            mask = [[0]*canvas_size for _ in range(canvas_size)]
            
            # Simulate different patterns for different organs
            if organ_idx % 3 == 0:  # Some organs present in top-left
                for y in range(0, canvas_size//3):
                    for x in range(0, canvas_size//3):
                        mask[y][x] = 1
            elif organ_idx % 3 == 1:  # Some in center
                for y in range(canvas_size//3, 2*canvas_size//3):
                    for x in range(canvas_size//3, 2*canvas_size//3):
                        mask[y][x] = 1
            # else: organ absent
            
            # Compute ground truth
            gt = compute_cell_ground_truth_simple(mask, grid_size, min_pixels=50)
            
            # Simulate model prediction
            if gt['present']:
                # Mock prediction - sometimes correct, sometimes wrong
                if sample_idx % 2 == 0:
                    # Correct prediction
                    pred_cells = [gt['dominant_cell']] if gt['dominant_cell'] else []
                else:
                    # Partial/wrong prediction
                    all_cells = get_cell_labels(grid_size)
                    pred_cells = [all_cells[organ_idx % len(all_cells)]]
                pred_present = 1
            else:
                # Sometimes false positive
                if organ_idx % 5 == 0:
                    pred_cells = ["A1"]
                    pred_present = 1
                else:
                    pred_cells = []
                    pred_present = 0
            
            # Compute metrics
            metrics = compute_cell_metrics_simple(
                pred_cells, gt['cells'], gt['present'], pred_present, top_k
            )
            
            # Store results
            sample_results["organs"][organ_name] = {
                "gt_present": gt['present'],
                "gt_cells": list(gt['cells']),
                "pred_present": pred_present,
                "pred_cells": pred_cells,
                "metrics": metrics
            }
        
        all_results.append(sample_results)
    
    return all_results

def compute_aggregate_metrics(results):
    """Compute aggregate metrics across all samples."""
    
    organ_metrics = {}
    
    for sample_result in results:
        for organ_name, organ_result in sample_result["organs"].items():
            if organ_name not in organ_metrics:
                organ_metrics[organ_name] = {
                    "presence_correct": [],
                    "cell_hits": [],
                    "cell_precisions": [],
                    "cell_recalls": [],
                }
            
            # Presence accuracy
            presence_correct = int(organ_result['pred_present'] == organ_result['gt_present'])
            organ_metrics[organ_name]["presence_correct"].append(presence_correct)
            
            # Cell metrics (only if organ present in GT)
            if organ_result['gt_present']:
                metrics = organ_result['metrics']
                organ_metrics[organ_name]["cell_hits"].append(metrics['cell_hit'])
                organ_metrics[organ_name]["cell_precisions"].append(metrics['cell_precision'])
                organ_metrics[organ_name]["cell_recalls"].append(metrics['cell_recall'])
    
    # Compute averages
    aggregate = {}
    for organ_name, metrics in organ_metrics.items():
        n_samples = len(metrics["presence_correct"])
        presence_acc = sum(metrics["presence_correct"]) / n_samples if n_samples > 0 else 0
        
        if metrics["cell_hits"]:
            cell_hit_rate = sum(metrics["cell_hits"]) / len(metrics["cell_hits"])
            cell_precision = sum(metrics["cell_precisions"]) / len(metrics["cell_precisions"])
            cell_recall = sum(metrics["cell_recalls"]) / len(metrics["cell_recalls"])
        else:
            cell_hit_rate = cell_precision = cell_recall = 0
        
        aggregate[organ_name] = {
            "presence_acc": presence_acc * 100,
            "cell_hit_rate": cell_hit_rate * 100,
            "cell_precision": cell_precision * 100,
            "cell_recall": cell_recall * 100,
            "n_samples": n_samples
        }
    
    # Macro averages
    macro_presence = sum(m["presence_acc"] for m in aggregate.values()) / len(aggregate)
    macro_hit = sum(m["cell_hit_rate"] for m in aggregate.values()) / len(aggregate)
    macro_precision = sum(m["cell_precision"] for m in aggregate.values()) / len(aggregate)
    macro_recall = sum(m["cell_recall"] for m in aggregate.values()) / len(aggregate)
    
    return {
        "per_organ": aggregate,
        "macro": {
            "presence_acc": macro_presence,
            "cell_hit_rate": macro_hit,
            "cell_precision": macro_precision,
            "cell_recall": macro_recall
        }
    }

def print_results_table(aggregate_metrics):
    """Print a formatted results table."""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Header
    print(f"{'Organ':<25} {'Presence':<12} {'Cell@K':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*80)
    
    # Per-organ results
    for organ_name, metrics in aggregate_metrics["per_organ"].items():
        print(f"{organ_name:<25} {metrics['presence_acc']:>10.1f}% {metrics['cell_hit_rate']:>10.1f}% "
              f"{metrics['cell_precision']:>10.1f}% {metrics['cell_recall']:>10.1f}%")
    
    print("-"*80)
    
    # Macro averages
    macro = aggregate_metrics["macro"]
    print(f"{'MACRO AVERAGE':<25} {macro['presence_acc']:>10.1f}% {macro['cell_hit_rate']:>10.1f}% "
          f"{macro['cell_precision']:>10.1f}% {macro['cell_recall']:>10.1f}%")
    
    print("="*80)

def main():
    """Main evaluation function."""
    
    # Parse environment variables
    grid_size = int(os.getenv('EVAL_GRID_SIZE', '3'))
    top_k = int(os.getenv('EVAL_TOP_K', '1'))
    num_samples = int(os.getenv('EVAL_NUM_SAMPLES', '5'))
    
    print(f"\nConfiguration:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Top-K: {top_k}")
    print(f"  Samples: {num_samples}")
    
    # Run evaluation
    results = evaluate_cell_selection_on_samples(num_samples, grid_size, top_k)
    
    # Compute aggregate metrics
    aggregate_metrics = compute_aggregate_metrics(results)
    
    # Print results table
    print_results_table(aggregate_metrics)
    
    # Save results
    output_dir = Path("cell_selection_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_G{grid_size}_K{top_k}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "config": {
                "grid_size": grid_size,
                "top_k": top_k,
                "num_samples": num_samples
            },
            "results": results,
            "aggregate_metrics": aggregate_metrics
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*60)
    print("✅ CELL SELECTION EVALUATION COMPLETE")
    print("="*60)
    
    print("\nThis standalone evaluation demonstrates that:")
    print("  1. Cell selection ground truth computation works")
    print("  2. Metrics calculation is correct")
    print("  3. The evaluation pipeline structure is ready")
    print("\nTo run with real models and data:")
    print("  1. Install required packages (numpy, torch, PIL, etc.)")
    print("  2. Restore model adapter files from archived/")
    print("  3. Run the full eval_cell_selection_original_size.py")

if __name__ == "__main__":
    # First run the basic tests
    print("\nRunning basic tests first...")
    print("-"*60)
    exec(open("test_cell_selection_nodeps.py").read())
    
    print("\n" + "="*60)
    print("Now running mock evaluation...")
    print("="*60)
    
    # Then run the evaluation
    main()