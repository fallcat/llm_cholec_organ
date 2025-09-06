#!/usr/bin/env python3
"""Generate 200 balanced test examples for each dataset.

This script creates balanced test sets with 200 examples each for:
- CholecSeg8k (12 organ classes)
- CholecOrgans (3 organ classes)  
- CholecGoNoGo (2 Go/NoGo classes)
"""

import sys
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from endopoint.datasets import build_dataset
from endopoint.fewshot import UnifiedFewShotSelector


def generate_balanced_test_set(
    dataset_name: str,
    n_test_samples: int = 200,
    seed: int = 42,
    output_base_dir: str = "/shared_data0/weiqiuy/llm_cholec_organ/data_info"
) -> Dict[str, Any]:
    """Generate balanced test set for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        n_test_samples: Number of test samples to select (200 default)
        seed: Random seed for reproducibility
        output_base_dir: Base directory for outputs
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*70)
    print(f"Generating balanced test set for: {dataset_name}")
    print("="*70)
    
    # Dataset configurations with PUBLIC as default
    dataset_configs = {
        "cholecseg8k_local": {
            "data_dir": "/shared_data0/weiqiuy/datasets/cholecseg8k"
        },
        "cholec_organs": {
            "data_dir": "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
            "video_globs": "public",  # Default to public
            "gen_seed": 56,
            "train_val_seed": 0
        },
        "cholec_gonogo": {
            "data_dir": "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
            "video_globs": "public",  # Default to public
            "gen_seed": 56,
            "train_val_seed": 0
        }
    }
    
    # Load dataset
    dataset_config = dataset_configs[dataset_name]
    dataset = build_dataset(dataset_name, **dataset_config)
    
    print(f"\n📊 Dataset Info:")
    print(f"  Tag: {dataset.dataset_tag}")
    print(f"  Train: {dataset.total('train')} examples")
    print(f"  Classes: {len(dataset.label_ids)}")
    
    # Print class distribution
    print(f"\n  Classes:")
    for cid in dataset.label_ids:
        print(f"    {cid}: {dataset.id2label[cid]}")
    
    # Create output directory
    output_dir = Path(output_base_dir) / f"{dataset_name}_balanced_200"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unified selector
    selector = UnifiedFewShotSelector(
        dataset=dataset,
        output_dir=output_dir,
        n_test_samples=n_test_samples,
        n_pos_examples=1,
        n_neg_absent=1,
        n_neg_wrong=1,
        min_pixels=50,
        seed=seed,
        cache_enabled=True
    )
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Step 1: Compute presence matrix for ALL training samples
    print(f"\n🔄 Computing presence matrix for ALL training samples...")
    Y = selector.compute_presence_matrix("train", max_samples=None)  # Process all
    print(f"  Presence matrix shape: {Y.shape}")
    
    # Calculate class statistics
    class_counts = Y.sum(axis=0)
    class_percentages = (class_counts / Y.shape[0]) * 100
    
    print(f"\n📊 Class distribution in training set:")
    for i, cid in enumerate(dataset.label_ids):
        name = dataset.id2label[cid]
        count = class_counts[i]
        pct = class_percentages[i]
        print(f"  {name:30} {count:6d} samples ({pct:5.1f}%)")
    
    # Identify rare and common classes
    rare_threshold = 20  # Classes with less than 20% presence
    rare_classes = []
    common_classes = []
    
    for i, cid in enumerate(dataset.label_ids):
        if class_percentages[i] < rare_threshold:
            rare_classes.append(dataset.id2label[cid])
        else:
            common_classes.append(dataset.id2label[cid])
    
    if rare_classes:
        print(f"\n🔴 Rare classes (<{rare_threshold}%): {rare_classes}")
    if common_classes:
        print(f"🟢 Common classes (>={rare_threshold}%): {common_classes}")
    
    # Step 2: Select balanced test set with advanced algorithm
    print(f"\n🔄 Selecting {n_test_samples} balanced test samples...")
    
    # Auto-configure parameters based on dataset
    n_classes = len(dataset.label_ids)
    rare_top_k = min(4, max(1, n_classes // 3))
    min_quota_rare = max(10, min(30, n_test_samples // 10))  # At least 10-30 samples for rare classes
    
    test_indices, selection_info = selector.select_balanced_test_set(
        Y=Y,
        split="train",
        use_advanced=True,
        rare_top_k=rare_top_k,
        min_quota_rare=min_quota_rare,
        max_cap_frac=0.70
    )
    
    print(f"  Selected {len(test_indices)} test samples")
    
    # Analyze selected distribution
    Y_selected = Y[test_indices]
    selected_counts = Y_selected.sum(axis=0)
    selected_percentages = (selected_counts / len(test_indices)) * 100
    
    print(f"\n📊 Selected test set distribution:")
    for i, cid in enumerate(dataset.label_ids):
        name = dataset.id2label[cid]
        orig_pct = class_percentages[i]
        sel_count = selected_counts[i]
        sel_pct = selected_percentages[i]
        improvement = sel_pct - orig_pct
        
        # Color code the improvement
        if improvement > 5:
            marker = "⬆️"  # Boosted
        elif improvement < -5:
            marker = "⬇️"  # Reduced
        else:
            marker = "➡️"  # Similar
            
        print(f"  {name:30} {sel_count:3d} samples ({sel_pct:5.1f}%) "
              f"{marker} (was {orig_pct:5.1f}%)")
    
    # Calculate balance score (standard deviation of percentages)
    orig_std = np.std(class_percentages)
    selected_std = np.std(selected_percentages)
    balance_improvement = ((orig_std - selected_std) / orig_std) * 100
    
    print(f"\n📈 Balance Metrics:")
    print(f"  Original StdDev: {orig_std:.2f}%")
    print(f"  Selected StdDev: {selected_std:.2f}%")
    print(f"  Balance Improvement: {balance_improvement:.1f}%")
    
    # Save detailed results
    results = {
        "dataset_name": dataset_name,
        "n_classes": n_classes,
        "n_train_total": Y.shape[0],
        "n_test_selected": len(test_indices),
        "test_indices": test_indices,
        "class_names": [dataset.id2label[cid] for cid in dataset.label_ids],
        "original_distribution": {
            "counts": class_counts.tolist(),
            "percentages": class_percentages.tolist()
        },
        "selected_distribution": {
            "counts": selected_counts.tolist(),
            "percentages": selected_percentages.tolist()
        },
        "balance_metrics": {
            "original_stddev": float(orig_std),
            "selected_stddev": float(selected_std),
            "improvement_percent": float(balance_improvement)
        },
        "selection_info": selection_info,
        "seed": seed
    }
    
    # Save results to JSON
    results_file = output_dir / f"balanced_test_200_summary.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Also save just the indices for easy loading
    indices_file = output_dir / f"balanced_test_indices_200.json"
    with open(indices_file, 'w') as f:
        json.dump({"indices": test_indices, "seed": seed}, f, indent=2)
    print(f"💾 Indices saved to: {indices_file}")
    
    return results


def main():
    """Generate balanced test sets for all datasets."""
    print("="*70)
    print("🎯 Generating 200 Balanced Test Examples for Each Dataset")
    print("="*70)
    
    datasets = [
        "cholecseg8k_local",
        "cholec_organs",
        "cholec_gonogo"
    ]
    
    all_results = []
    
    for dataset_name in datasets:
        try:
            results = generate_balanced_test_set(
                dataset_name=dataset_name,
                n_test_samples=200,
                seed=42
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("✨ Summary of All Balanced Test Sets")
    print("="*70)
    
    print(f"\n{'Dataset':<20} {'Classes':<10} {'Test Size':<12} {'Balance Improvement'}")
    print("-"*65)
    
    for res in all_results:
        improvement = res['balance_metrics']['improvement_percent']
        color = "🟢" if improvement > 10 else "🟡" if improvement > 0 else "🔴"
        print(f"{res['dataset_name']:<20} {res['n_classes']:<10} "
              f"{res['n_test_selected']:<12} {color} {improvement:+.1f}%")
    
    print("\n📁 Output directories created:")
    for res in all_results:
        print(f"  • data_info/{res['dataset_name']}_balanced_200/")
    
    print("\n✅ All balanced test sets generated successfully!")
    print("Each dataset now has 200 carefully balanced test examples.")


if __name__ == "__main__":
    main()