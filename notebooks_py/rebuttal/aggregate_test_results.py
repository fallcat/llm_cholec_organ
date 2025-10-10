#!/usr/bin/env python3
"""
Aggregate results from test_all_models.sh runs
Reads the summary JSON files and creates a comparison table
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Optional

def load_model_results(results_dir: Path, model: str, split: str, frames: int, seed: int) -> Dict:
    """Load results for a specific model and configuration."""
    # Convert model name slashes to underscores for directory names
    model_dir = model.replace('/', '_')
    
    result_dir = results_dir / f"bbox_cholecseg8k_local_{split}{frames}_seed{seed}"
    model_result_dir = result_dir / "zeroshot_combined" / model_dir
    
    if not model_result_dir.exists():
        return {
            'presence_accuracy': None,
            'iou_bbox': None, 
            'iou_mask': None,
            'exists': False
        }
    
    # Find all test files
    test_files = list(model_result_dir.glob("test_*.json"))
    
    if not test_files:
        return {
            'presence_accuracy': None,
            'iou_bbox': None, 
            'iou_mask': None,
            'exists': False
        }
    
    # Aggregate metrics from all test files
    total_correct_presence = 0
    total_organs = 0
    total_iou_bbox = 0
    total_iou_count = 0
    
    for test_file in test_files:
        with open(test_file, 'r') as f:
            data = json.load(f)
            
        # Count presence accuracy
        for organ in data['organs']:
            total_organs += 1
            if organ['ground_truth_present'] == organ['predicted_present']:
                total_correct_presence += 1
            
            # Count IoU for present organs
            if organ['ground_truth_present'] == 1 and organ['predicted_present'] == 1:
                total_iou_bbox += organ['iou_bbox_to_bbox']
                total_iou_count += 1
    
    presence_accuracy = total_correct_presence / total_organs if total_organs > 0 else 0
    iou_bbox = total_iou_bbox / total_iou_count if total_iou_count > 0 else 0
    
    return {
        'presence_accuracy': presence_accuracy,
        'iou_bbox': iou_bbox,
        'iou_mask': None,  # Not computed for now
        'exists': True
    }

def aggregate_test_results(
    results_root: str = "/shared_data0/weiqiuy/llm_cholec_organ/results",
    frames: int = 10,
    seed: int = 42,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate test results from all models."""
    
    results_dir = Path(results_root)
    
    # Define all models to check
    models = {
        'API LVLMs': ['gpt-4.1', 'gemini-2.0-flash'],
        'Local LVLMs': ['Qwen/Qwen2.5-VL-7B-Instruct', 'mistralai/Pixtral-12B-2409'],
        'CLIP Models': ['raso', 'peskavlp'],
        'Segmentation': ['cholenet', 'gonogonet']
    }
    
    splits = ['balanced', 'unbalanced']
    
    # Collect results
    results = []
    for category, model_list in models.items():
        for model in model_list:
            row = {
                'Category': category,
                'Model': model
            }
            
            for split in splits:
                metrics = load_model_results(results_dir, model, split, frames, seed)
                
                if metrics['exists']:
                    presence = metrics['presence_accuracy']
                    iou_bbox = metrics['iou_bbox']
                    
                    # Format metrics
                    if presence is not None:
                        row[f'{split}_presence'] = f"{presence:.1%}"
                    else:
                        row[f'{split}_presence'] = "N/A"
                    
                    if iou_bbox is not None:
                        row[f'{split}_iou'] = f"{iou_bbox:.3f}"
                    else:
                        row[f'{split}_iou'] = "N/A"
                    
                    row[f'{split}_status'] = "✓"
                else:
                    row[f'{split}_presence'] = "-"
                    row[f'{split}_iou'] = "-"
                    row[f'{split}_status'] = "✗"
            
            results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print(f"Test Results Summary (frames={frames}, seed={seed})")
    print("="*80)
    
    # Format for display
    display_cols = ['Category', 'Model', 
                   'balanced_status', 'balanced_presence', 'balanced_iou',
                   'unbalanced_status', 'unbalanced_presence', 'unbalanced_iou']
    
    display_df = df[display_cols].copy()
    display_df.columns = ['Category', 'Model', 
                          'Bal✓', 'Bal Pres%', 'Bal IoU',
                          'Unb✓', 'Unb Pres%', 'Unb IoU']
    
    print(display_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "-"*80)
    total_models = len(df)
    balanced_pass = (df['balanced_status'] == '✓').sum()
    unbalanced_pass = (df['unbalanced_status'] == '✓').sum()
    both_pass = ((df['balanced_status'] == '✓') & (df['unbalanced_status'] == '✓')).sum()
    
    print(f"Total models tested: {total_models}")
    print(f"Balanced split passed: {balanced_pass}/{total_models}")
    print(f"Unbalanced split passed: {unbalanced_pass}/{total_models}")
    print(f"Both splits passed: {both_pass}/{total_models}")
    
    # Failed models
    failed_balanced = df[df['balanced_status'] == '✗']['Model'].tolist()
    failed_unbalanced = df[df['unbalanced_status'] == '✗']['Model'].tolist()
    
    if failed_balanced:
        print(f"\nFailed on balanced: {', '.join(failed_balanced)}")
    if failed_unbalanced:
        print(f"Failed on unbalanced: {', '.join(failed_unbalanced)}")
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Aggregate test results')
    parser.add_argument('--results-root', type=str, 
                       default='/shared_data0/weiqiuy/llm_cholec_organ/results',
                       help='Root directory for results')
    parser.add_argument('--frames', type=int, default=10,
                       help='Number of frames used in test')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed used in test')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Optional CSV output file')
    
    args = parser.parse_args()
    
    df = aggregate_test_results(
        results_root=args.results_root,
        frames=args.frames,
        seed=args.seed,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main()