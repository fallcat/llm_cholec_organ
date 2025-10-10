#!/usr/bin/env python3
"""
Aggregate results across multiple seeds for balanced/unbalanced evaluation.

This script computes mean, std, and 95% CI across seeds for each metric.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def load_results(results_dir: Path, split: str, num_samples: int, seed: int) -> Dict[str, Any]:
    """Load results for a specific configuration."""
    results = {}
    
    # Construct directory name
    if split == "balanced":
        dir_name = f"bbox_cholecseg8k_local_balanced{num_samples}_seed{seed}"
    else:
        dir_name = f"bbox_cholecseg8k_local_unbalanced{num_samples}_seed{seed}"
    
    result_dir = results_dir / dir_name
    
    if not result_dir.exists():
        print(f"Warning: Directory not found: {result_dir}")
        return results
    
    # Load all summary files
    for summary_file in result_dir.glob("summary_*.json"):
        with open(summary_file, 'r') as f:
            data = json.load(f)
            
        # Extract configuration from filename
        parts = summary_file.stem.split('_')
        detection_mode = parts[1]  # combined or separate
        fewshot_mode = parts[2]    # zeroshot or fewshot
        
        key = f"{detection_mode}_{fewshot_mode}"
        results[key] = data
    
    return results


def aggregate_metrics(all_results: List[Dict[str, Any]], metric_key: str) -> Dict[str, float]:
    """Aggregate a specific metric across seeds."""
    values = []
    for result in all_results:
        if 'metrics' in result and metric_key in result['metrics']:
            values.append(result['metrics'][metric_key])
    
    if not values:
        return {'mean': np.nan, 'std': np.nan, 'ci95': np.nan, 'count': 0}
    
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0
    ci95 = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0
    
    return {
        'mean': mean,
        'std': std,
        'ci95': ci95,
        'count': len(values),
        'min': np.min(values),
        'max': np.max(values)
    }


def create_summary_table(results_by_seed: Dict[int, Dict[str, Any]], 
                        split: str, 
                        model: str) -> pd.DataFrame:
    """Create summary table for a model across seeds."""
    rows = []
    
    # Define scenarios
    scenarios = [
        ('combined_zeroshot', 'Combined', 'Zero-shot'),
        ('combined_fewshot', 'Combined', 'Few-shot'),
        ('separate_zeroshot', 'Separate', 'Zero-shot'),
        ('separate_fewshot', 'Separate', 'Few-shot')
    ]
    
    # Metrics to aggregate
    metrics = [
        ('presence_accuracy', 'Presence Acc'),
        ('mean_iou_bbox_to_bbox', 'Bbox IoU'),
        ('mean_iou_bbox_to_mask', 'Mask IoU')
    ]
    
    for scenario_key, detection_mode, fewshot_mode in scenarios:
        # Collect results for this scenario across seeds
        scenario_results = []
        for seed, seed_results in results_by_seed.items():
            if scenario_key in seed_results:
                scenario_results.append(seed_results[scenario_key])
        
        if not scenario_results:
            continue
        
        row = {
            'Split': split,
            'Model': model,
            'Detection': detection_mode,
            'Few-shot': fewshot_mode,
        }
        
        # Aggregate each metric
        for metric_key, metric_name in metrics:
            agg = aggregate_metrics(scenario_results, metric_key)
            row[f'{metric_name} Mean'] = agg['mean']
            row[f'{metric_name} Std'] = agg['std']
            row[f'{metric_name} CI95'] = agg['ci95']
            row[f'{metric_name} N'] = agg['count']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed bbox results")
    parser.add_argument("--results-dir", type=str, 
                        default="/shared_data0/weiqiuy/llm_cholec_organ/results",
                        help="Directory containing result folders")
    parser.add_argument("--split-modes", type=str, default="both",
                        help="Split modes to aggregate: balanced, unbalanced, or both")
    parser.add_argument("--seeds", type=str, default="42,7,2025,518,1337",
                        help="Comma-separated seeds")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all found)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output CSV file (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Parse inputs
    results_dir = Path(args.results_dir)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    if args.split_modes == "both":
        splits = ["balanced", "unbalanced"]
    else:
        splits = [args.split_modes]
    
    print(f"Aggregating results for:")
    print(f"  Splits: {splits}")
    print(f"  Seeds: {seeds}")
    print(f"  Samples: {args.num_samples}")
    print()
    
    # Collect all results
    all_tables = []
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Split: {split.upper()}")
        print(f"{'='*60}")
        
        # Determine which models to process
        if args.models:
            models = [m.strip() for m in args.models.split(",")]
        else:
            # Auto-detect models from first seed
            first_dir = results_dir / f"bbox_cholecseg8k_local_{split}{args.num_samples}_seed{seeds[0]}"
            if first_dir.exists():
                # Look for model directories
                model_dirs = set()
                for subdir in first_dir.iterdir():
                    if subdir.is_dir() and subdir.name.endswith(('_zeroshot', '_fewshot')):
                        for model_dir in subdir.iterdir():
                            if model_dir.is_dir():
                                model_dirs.add(model_dir.name)
                models = sorted(model_dirs)
            else:
                print(f"Warning: Could not find {first_dir} to auto-detect models")
                models = []
        
        print(f"Models found: {models}")
        
        for model in models:
            print(f"\nProcessing: {model}")
            
            # Load results for all seeds
            results_by_seed = {}
            for seed in seeds:
                seed_results = load_results(results_dir, split, args.num_samples, seed)
                if seed_results:
                    # Filter to just this model
                    filtered = {}
                    for key, data in seed_results.items():
                        if data.get('model') == model:
                            filtered[key] = data
                    if filtered:
                        results_by_seed[seed] = filtered
            
            if not results_by_seed:
                print(f"  No results found for {model}")
                continue
            
            print(f"  Found results for {len(results_by_seed)} seeds")
            
            # Create summary table
            table = create_summary_table(results_by_seed, split, model)
            if not table.empty:
                all_tables.append(table)
                
                # Print summary
                print("\n  Summary:")
                for _, row in table.iterrows():
                    print(f"    {row['Detection']} {row['Few-shot']}:")
                    print(f"      Presence: {row['Presence Acc Mean']:.1%} ± {row['Presence Acc CI95']:.1%}")
                    print(f"      Bbox IoU: {row['Bbox IoU Mean']:.3f} ± {row['Bbox IoU CI95']:.3f}")
                    print(f"      Mask IoU: {row['Mask IoU Mean']:.3f} ± {row['Mask IoU CI95']:.3f}")
    
    # Save combined results
    if all_tables:
        combined_df = pd.concat(all_tables, ignore_index=True)
        
        # Determine output file
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = results_dir / f"aggregate_bbox_results_{timestamp}.csv"
        
        combined_df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Saved aggregated results to: {output_file}")
        
        # Also save as Excel for easier viewing
        excel_file = output_file.with_suffix('.xlsx')
        with pd.ExcelWriter(excel_file) as writer:
            combined_df.to_excel(writer, sheet_name='Aggregated Results', index=False)
            
            # Add formatting
            worksheet = writer.sheets['Aggregated Results']
            
            # Auto-adjust column widths
            for idx, col in enumerate(combined_df.columns):
                max_len = max(
                    combined_df[col].astype(str).map(len).max(),
                    len(col)
                ) + 2
                worksheet.set_column(idx, idx, max_len)
        
        print(f"Also saved as Excel: {excel_file}")
        
        # Print final summary statistics
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        # Group by split and compute overall stats
        for split in combined_df['Split'].unique():
            split_df = combined_df[combined_df['Split'] == split]
            print(f"\n{split.upper()}:")
            
            # Overall mean across all models and scenarios
            presence_mean = split_df['Presence Acc Mean'].mean()
            bbox_mean = split_df['Bbox IoU Mean'].mean()
            mask_mean = split_df['Mask IoU Mean'].mean()
            
            print(f"  Overall Presence Acc: {presence_mean:.1%}")
            print(f"  Overall Bbox IoU: {bbox_mean:.3f}")
            print(f"  Overall Mask IoU: {mask_mean:.3f}")
            
            # Best performing configuration
            best_idx = split_df['Presence Acc Mean'].idxmax()
            best_row = split_df.loc[best_idx]
            print(f"  Best config: {best_row['Model']} - {best_row['Detection']} {best_row['Few-shot']}")
            print(f"    Presence: {best_row['Presence Acc Mean']:.1%}")
    
    else:
        print("\nNo results found to aggregate")


if __name__ == "__main__":
    main()