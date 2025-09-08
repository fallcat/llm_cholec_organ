#!/usr/bin/env python3
"""
Run CholeNet and GoNoGoNet evaluation on all datasets and generate summary table.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

# Configuration
NUM_SAMPLES = 20  # Adjust for testing
DATASETS = ["cholecseg8k", "cholec_organs", "cholec_gonogo"]
MODELS = ["cholenet", "gonogonet"]

# Expected capabilities
CAPABILITIES = {
    "cholenet": {
        "cholecseg8k": "Partial (3/13 organs: Liver, Gallbladder, Hepatocystic Triangle)",
        "cholec_organs": "Native (all 3 organs)",
        "cholec_gonogo": "Cross-mapped (Hepatocystic Triangle → Go Zone)"
    },
    "gonogonet": {
        "cholecseg8k": "None (cannot detect organs)",
        "cholec_organs": "Cross-mapped (Go Zone → Hepatocystic Triangle)",
        "cholec_gonogo": "Native (Go/NoGo zones)"
    }
}

def run_evaluation(dataset, model, num_samples=NUM_SAMPLES):
    """Run a single evaluation and return results."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model} on {dataset} ({num_samples} samples)")
    print(f"Expected: {CAPABILITIES[model][dataset]}")
    print('='*60)
    
    env = os.environ.copy()
    env.update({
        'EVAL_DATASET': dataset,
        'EVAL_MODEL': model,
        'EVAL_NUM_SAMPLES': str(num_samples),
        'EVAL_USE_CACHE': 'true',
        'EVAL_PERSISTENT_DIR': 'true',
        'EVAL_DETECTION_MODE': 'combined'
    })
    
    try:
        result = subprocess.run(
            ['python', 'notebooks_py/eval_bbox_unified.py'],
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse output for metrics
        output = result.stdout + result.stderr
        metrics = parse_metrics(output)
        
        if metrics:
            print(f"✓ Presence Accuracy: {metrics.get('presence_acc', 'N/A'):.1f}%")
            print(f"✓ Bbox-to-Bbox IoU: {metrics.get('bbox_iou', 'N/A'):.3f}")
            print(f"✓ Bbox-to-Mask IoU: {metrics.get('mask_iou', 'N/A'):.3f}")
        else:
            print("✗ Evaluation failed or no metrics found")
            
        return metrics
        
    except subprocess.TimeoutExpired:
        print("✗ Evaluation timed out")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def parse_metrics(output):
    """Parse metrics from evaluation output."""
    metrics = {}
    
    # Look for metric lines in output
    lines = output.split('\n')
    for line in lines:
        if 'Presence Accuracy:' in line:
            try:
                acc = float(line.split(':')[1].strip().replace('%', ''))
                metrics['presence_acc'] = acc
            except:
                pass
        elif 'Bbox-to-Bbox IoU:' in line:
            try:
                iou = float(line.split(':')[1].strip())
                metrics['bbox_iou'] = iou
            except:
                pass
        elif 'Bbox-to-Mask IoU:' in line:
            try:
                iou = float(line.split(':')[1].strip())
                metrics['mask_iou'] = iou
            except:
                pass
        elif 'Evaluation Time:' in line:
            try:
                time = line.split(':')[1].strip()
                metrics['time'] = time
            except:
                pass
    
    return metrics if metrics else None

def generate_summary_table(results):
    """Generate a summary table of all results."""
    # Create DataFrame
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            key = f"{model}_{dataset}"
            if key in results and results[key]:
                row = {
                    'Model': model.upper(),
                    'Dataset': dataset.replace('_', ' ').title(),
                    'Capability': CAPABILITIES[model][dataset],
                    'Presence Acc (%)': results[key].get('presence_acc', 0),
                    'BBox IoU': results[key].get('bbox_iou', 0),
                    'Mask IoU': results[key].get('mask_iou', 0),
                }
            else:
                row = {
                    'Model': model.upper(),
                    'Dataset': dataset.replace('_', ' ').title(),
                    'Capability': CAPABILITIES[model][dataset],
                    'Presence Acc (%)': 0,
                    'BBox IoU': 0,
                    'Mask IoU': 0,
                }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Print table
    print("\n" + "="*100)
    print("EVALUATION SUMMARY TABLE")
    print("="*100)
    print(df.to_string(index=False))
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"cholenet_gonogo_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    return df

def main():
    """Run all evaluations and generate summary."""
    print("="*80)
    print("CholeNet and GoNoGoNet Comprehensive Evaluation")
    print("="*80)
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Samples per evaluation: {NUM_SAMPLES}")
    
    # Run all evaluations
    results = {}
    
    for model in MODELS:
        print(f"\n{'#'*80}")
        print(f"# {model.upper()} EVALUATION")
        print(f"{'#'*80}")
        
        for dataset in DATASETS:
            key = f"{model}_{dataset}"
            metrics = run_evaluation(dataset, model, NUM_SAMPLES)
            if metrics:
                results[key] = metrics
            else:
                print(f"Warning: No results for {model} on {dataset}")
    
    # Generate summary
    summary_df = generate_summary_table(results)
    
    # Print final notes
    print("\n" + "="*80)
    print("NOTES:")
    print("="*80)
    print("1. CholeNet detects 3 organs: Liver, Gallbladder, Hepatocystic Triangle")
    print("2. GoNoGoNet detects 2 zones: Go Zone (safe), NoGo Zone (unsafe)")
    print("3. Cross-dataset mappings:")
    print("   - CholeNet on GoNoGo: Hepatocystic Triangle → Go Zone")
    print("   - GoNoGoNet on Organs: Go Zone → Hepatocystic Triangle")
    print("4. Background class is excluded from evaluation in all datasets")
    
    return results

if __name__ == '__main__':
    results = main()