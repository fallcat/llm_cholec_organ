#!/usr/bin/env python3
"""
Generate few-shot examples for both separate and combined evaluation modes.

This script generates:
1. Separate few-shot examples for each organ (traditional approach)
2. Combined few-shot examples using greedy set cover (efficient approach)

Usage:
    python generate_few_shot_sep_comb.py --dataset cholecseg8k_local
    
Environment variables:
    DATASET_NAME: Name of the dataset (cholecseg8k_local, cholec_organs, cholec_gonogo)
    FORCE_REGENERATE: If set to "true", force regeneration of cached files
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any, Tuple

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets import build_dataset
from endopoint.fewshot import UnifiedFewShotSelector
from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter


def select_examples_from_presence_matrix(
    presence_matrix: np.ndarray,
    required_cols: Optional[Iterable[int]] = None,
    max_examples: Optional[int] = None,
    prefer_row_with_more_gain_then_density: bool = True,
) -> Dict[str, Any]:
    """
    Greedy set-cover over a binary presence matrix.
    
    Args:
        presence_matrix: shape (N_examples, N_organs), entries in {0,1}
        required_cols: optional iterable of column indices to cover
        max_examples: optional hard cap on number of selected rows
        prefer_row_with_more_gain_then_density: tie-break preference
        
    Returns:
        dict with selected_indices, covered_cols, uncovered_cols, etc.
    """
    if presence_matrix.dtype != np.bool_:
        pm = presence_matrix.astype(bool)
    else:
        pm = presence_matrix

    n_rows, n_cols = pm.shape
    if required_cols is None:
        coverable = np.where(pm.any(axis=0))[0]
        required = np.array(coverable, dtype=int)
    else:
        required = np.array(list(required_cols), dtype=int)

    # Identify impossible columns
    col_has_any = pm.any(axis=0)
    impossible = required[~col_has_any[required]] if len(required) > 0 else np.array([], dtype=int)

    remaining_mask = np.zeros(n_cols, dtype=bool)
    remaining_mask[required] = True
    if len(impossible) > 0:
        remaining_mask[impossible] = False

    selected: List[int] = []
    covered = np.zeros(n_cols, dtype=bool)

    # Precompute row densities
    pm_req = pm[:, required] if len(required) else pm[:, []]
    row_req_counts = pm_req.sum(axis=1) if pm_req.size else np.zeros(n_rows, dtype=int)

    first_gain_report: List[Tuple[int, int]] = []

    while remaining_mask.any():
        gain_vec = (pm[:, remaining_mask]).sum(axis=1)
        if len(first_gain_report) == 0:
            for i in range(n_rows):
                first_gain_report.append((i, int(gain_vec[i])))

        best_idx = -1
        best_gain = 0
        best_tiebreak = -1

        for i in range(n_rows):
            g = int(gain_vec[i])
            if g <= 0:
                continue
            
            if prefer_row_with_more_gain_then_density:
                tie = (g > best_gain) or (g == best_gain and row_req_counts[i] > best_tiebreak) or (
                    g == best_gain and row_req_counts[i] == best_tiebreak and i < best_idx
                )
                if tie:
                    best_idx = i
                    best_gain = g
                    best_tiebreak = int(row_req_counts[i])
            else:
                tie = (g > best_gain) or (g == best_gain and row_req_counts[i] < best_tiebreak) or (
                    g == best_gain and row_req_counts[i] == best_tiebreak and i < best_idx
                )
                if tie:
                    best_idx = i
                    best_gain = g
                    best_tiebreak = int(row_req_counts[i])

        if best_idx == -1:
            break

        selected.append(best_idx)
        covered |= pm[best_idx]
        remaining_mask = np.zeros(n_cols, dtype=bool)
        
        for c in required:
            if c in impossible:
                continue
            if not covered[c]:
                remaining_mask[c] = True

        if max_examples is not None and len(selected) >= max_examples:
            break

    uncovered = [c for c in required if (c not in set(impossible)) and (not covered[c])]
    
    return {
        "selected_indices": selected,
        "covered_cols": [c for c in required if covered[c]],
        "uncovered_cols": list(uncovered),
        "impossible_cols": list(impossible),
        "selected_count": len(selected),
        "coverage_vector": covered,
        "per_row_gain": first_gain_report,
    }


def generate_separate_examples(dataset_name: str, output_dir: Path, force_regenerate: bool = False):
    """Generate separate few-shot examples for each organ."""
    
    print("\n" + "="*70)
    print(f"Generating SEPARATE few-shot examples for: {dataset_name}")
    print("="*70)
    
    # Dataset configurations
    configs = {
        "cholecseg8k_local": {
            "data_dir": "/shared_data0/weiqiuy/datasets/cholecseg8k"
        },
        "cholec_organs": {
            "data_dir": "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
            "video_globs": "public",
            "gen_seed": 56,
            "train_val_seed": 0
        },
        "cholec_gonogo": {
            "data_dir": "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
            "video_globs": "public",
            "gen_seed": 56,
            "train_val_seed": 0
        }
    }
    
    dataset_config = configs.get(dataset_name, {})
    
    # Load dataset
    dataset = build_dataset(dataset_name, **dataset_config)
    
    print(f"\n📊 Dataset Info:")
    print(f"  Tag: {dataset.dataset_tag}")
    print(f"  Train: {dataset.total('train')} examples")
    print(f"  Classes: {len(dataset.label_ids)}")
    
    # Create selector
    if force_regenerate:
        cache_file = output_dir / "balanced_test_indices_advanced_200.json"
        if cache_file.exists():
            print(f"⚠️ Removing old cache file: {cache_file.name}")
            cache_file.unlink()
    
    selector = UnifiedFewShotSelector(
        dataset=dataset,
        output_dir=output_dir,
        n_test_samples=200,
        n_pos_examples=1,
        n_neg_absent=1,
        n_neg_wrong=1,
        min_pixels=50,
        seed=42,
        cache_enabled=True
    )
    
    # Run the pipeline to generate separate examples
    results = selector.run_balanced_selection_pipeline(
        split="train",
        visualize=False,
        save_summary=True
    )
    
    print(f"\n✅ Generated separate few-shot examples:")
    print(f"  Pointing examples: {len(results.get('pointing_examples', []))}")
    print(f"  Bounding box examples: {len(results.get('bbox_examples', []))}")
    print(f"  Saved to: {output_dir}")
    
    return results


def generate_combined_examples(dataset_name: str, output_dir: Path, max_examples: Optional[int] = None, min_examples: int = 3):
    """Generate combined few-shot examples using greedy set cover.
    
    Args:
        dataset_name: Name of the dataset
        output_dir: Output directory path
        max_examples: Maximum number of examples to select (None for no limit)
        min_examples: Minimum number of examples to select (default: 3)
    """
    
    print("\n" + "="*70)
    print(f"Generating COMBINED few-shot examples for: {dataset_name}")
    print("="*70)
    
    # Load presence matrix
    presence_matrix_file = output_dir / "presence_matrix_train.npy"
    if not presence_matrix_file.exists():
        print(f"❌ Presence matrix not found: {presence_matrix_file}")
        print("   Please run separate example generation first.")
        return None
    
    presence_matrix = np.load(presence_matrix_file)
    print(f"✅ Loaded presence matrix: {presence_matrix.shape}")
    
    # Dataset configurations
    configs = {
        "cholecseg8k_local": {
            "data_dir": "/shared_data0/weiqiuy/datasets/cholecseg8k"
        },
        "cholec_organs": {
            "data_dir": "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
            "video_globs": "public",
            "gen_seed": 56,
            "train_val_seed": 0
        },
        "cholec_gonogo": {
            "data_dir": "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib",
            "video_globs": "public",
            "gen_seed": 56,
            "train_val_seed": 0
        }
    }
    
    dataset_config = configs.get(dataset_name, {})
    
    # Build dataset and adapter based on dataset name
    dataset = build_dataset(dataset_name, **dataset_config)
    
    # For cholecseg8k_local, we have the full adapter implementation
    if dataset_name == "cholecseg8k_local":
        adapter = CholecSeg8kLocalAdapter(data_dir="/shared_data0/weiqiuy/datasets/cholecseg8k")
        has_bbox_support = True
    else:
        # For other datasets, use the generic dataset which may have limited bbox support
        adapter = dataset
        # Check if the adapter has the required methods
        has_bbox_support = hasattr(adapter, 'example_to_tensors') and hasattr(adapter, 'get_bounding_boxes')
        
        if not has_bbox_support:
            print(f"⚠️ Dataset {dataset_name} adapter doesn't have full bounding box support")
            print("   Will use pointing examples instead")
    
    # Get organ names
    label_names = []
    for class_id in dataset.label_ids:
        label_name = dataset.id2label[class_id]
        label_names.append(label_name)
    
    print(f"✅ Dataset has {len(label_names)} organs")
    
    # Use greedy cover to select examples
    print(f"\n🔄 Running greedy set cover algorithm...")
    result = select_examples_from_presence_matrix(
        presence_matrix,
        required_cols=None,
        max_examples=max_examples
    )
    
    selected_indices = result["selected_indices"]
    uncovered_cols = result["uncovered_cols"]
    
    # Ensure we have at least min_examples for better few-shot learning
    if len(selected_indices) < min_examples:
        print(f"📝 Greedy cover found {len(selected_indices)} images, but using minimum of {min_examples}")
        
        # Add more random examples (using seed for reproducibility)
        np.random.seed(42)  # Fixed seed for reproducibility
        
        # Get all indices not already selected
        all_indices = list(range(presence_matrix.shape[0]))
        available_indices = [i for i in all_indices if i not in selected_indices]
        
        # Check if any available indices have at least one organ
        valid_indices = []
        for idx in available_indices:
            if presence_matrix[idx].any():  # Has at least one organ
                valid_indices.append(idx)
        
        # Randomly select additional examples
        n_needed = min_examples - len(selected_indices)
        if len(valid_indices) >= n_needed:
            additional_indices = np.random.choice(valid_indices, n_needed, replace=False)
            selected_indices.extend(additional_indices.tolist())
            print(f"   Added {n_needed} random examples for diversity")
        else:
            # If not enough valid indices, just add what we can
            selected_indices.extend(valid_indices)
            print(f"   Added {len(valid_indices)} additional examples (all available)")
    
    print(f"✅ Selected {len(selected_indices)} images total")
    print(f"   Indices: {selected_indices}")
    
    if uncovered_cols:
        uncovered_organs = [label_names[i] for i in uncovered_cols]
        print(f"\n⚠️ Warning: {len(uncovered_organs)} organs not covered:")
        for organ in uncovered_organs:
            print(f"     - {organ}")
    else:
        print(f"✅ All {len(label_names)} organs covered!")
    
    # Create combined examples
    combined_examples = []
    
    if has_bbox_support:
        print(f"\n🔄 Creating combined few-shot examples with bounding boxes...")
        
        for i, idx in enumerate(selected_indices, 1):
            print(f"\nProcessing image {i}/{len(selected_indices)}: index {idx}")
            
            # Get the example from dataset
            example = adapter.get_example('train', idx)
            
            # Convert to tensors to get label tensor
            img_t, lab_t = adapter.example_to_tensors(example)
            
            # Get bounding boxes for ALL organs in this image
            all_bboxes_dict = adapter.get_bounding_boxes(lab_t, min_pixels=50)
            
            # Create combined bounding boxes
            combined_bboxes = {}
            organs_present = []
            
            for organ_class_id, boxes_list in all_bboxes_dict.items():
                organ_name = adapter.id2label[organ_class_id]
                organs_present.append(organ_name)
                
                combined_bboxes[organ_name] = {
                    "bboxes": boxes_list,
                    "num_regions": len(boxes_list)
                }
                
                print(f"  ✓ {organ_name}: {len(boxes_list)} region(s)")
            
            combined_examples.append({
                "idx": int(idx),
                "frame_id": example.get('frame_id', f"train_{idx}"),
                "video_id": example.get('video_id', 'unknown'),
                "organs_present": organs_present,
                "num_organs": len(organs_present),
                "bboxes": combined_bboxes
            })
    else:
        # For datasets without bbox support, create simpler combined examples
        print(f"\n🔄 Creating combined few-shot examples (pointing mode)...")
        
        for i, idx in enumerate(selected_indices, 1):
            print(f"\nProcessing image {i}/{len(selected_indices)}: index {idx}")
            
            # Get which organs are present based on presence matrix
            organs_present = []
            organ_points = {}
            
            for j, organ_name in enumerate(label_names):
                if presence_matrix[idx, j] == 1:
                    organs_present.append(organ_name)
                    # For pointing mode, we don't have actual coordinates
                    # This would need to be filled in during evaluation
                    organ_points[organ_name] = {
                        "present": True,
                        "point": None  # To be determined during evaluation
                    }
                    
            print(f"  Present organs ({len(organs_present)}): {', '.join(organs_present)}")
            
            combined_examples.append({
                "idx": int(idx),
                "frame_id": f"train_{idx}",
                "organs_present": organs_present,
                "num_organs": len(organs_present),
                "points": organ_points
            })
    
    print(f"\n✅ Created {len(combined_examples)} combined few-shot examples")
    
    # Save combined plan
    if has_bbox_support:
        output_file = output_dir / "fewshot_plan_bbox_combined_greedy.json"
        task_type = "bounding_box"
    else:
        output_file = output_dir / "fewshot_plan_pointing_combined_greedy.json"
        task_type = "pointing"
    
    total_organs_covered = set()
    for ex in combined_examples:
        total_organs_covered.update(ex['organs_present'])
    
    save_data = {
        "metadata": {
            "creation_method": "greedy_set_cover",
            "task_type": task_type,
            "description": f"Combined few-shot examples for {task_type} detection",
            "total_organs": len(label_names),
            "num_examples": len(combined_examples),
            "organs_covered": len(total_organs_covered),
            "uncovered_organs": [label_names[i] for i in uncovered_cols] if uncovered_cols else [],
            "dataset": dataset_name,
            "max_examples": max_examples,
            "min_examples": min_examples,
            "has_bbox_support": has_bbox_support,
            "greedy_cover_results": {
                "selected_indices": selected_indices,
                "covered_cols": [int(c) for c in result["covered_cols"]],
                "uncovered_cols": [int(c) for c in result["uncovered_cols"]],
                "impossible_cols": [int(c) for c in result["impossible_cols"]]
            }
        },
        "organ_names": label_names,
        "examples": combined_examples,
        "coverage_analysis": {
            "organs_per_image": {
                f"image_{ex['idx']}": ex['num_organs'] for ex in combined_examples
            },
            "organ_frequency": {
                organ: sum(1 for ex in combined_examples if organ in ex['organs_present']) 
                for organ in label_names
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n💾 Saved combined few-shot plan to: {output_file.name}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  - Selected {len(combined_examples)} images (greedy set cover)")
    print(f"  - Covers {len(total_organs_covered)}/{len(label_names)} organs")
    print(f"  - Efficiency: {(1 - len(combined_examples)/len(label_names))*100:.0f}% reduction vs per-organ")
    
    return save_data


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Generate few-shot examples for organ detection')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (or use DATASET_NAME env var)')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration of cached files')
    parser.add_argument('--max-combined', type=int, default=None,
                        help='Maximum number of combined examples (default: no limit)')
    parser.add_argument('--min-combined', type=int, default=3,
                        help='Minimum number of combined examples (default: 3)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['separate', 'combined', 'both'],
                        help='Generation mode')
    
    args = parser.parse_args()
    
    # Get dataset name from args or environment
    dataset_name = args.dataset or os.environ.get('DATASET_NAME')
    if not dataset_name:
        print("❌ Error: Please specify dataset name via --dataset or DATASET_NAME env var")
        print("   Options: cholecseg8k_local, cholec_organs, cholec_gonogo")
        sys.exit(1)
    
    # Check for force regenerate from env
    force_regenerate = args.force or os.environ.get('FORCE_REGENERATE', '').lower() == 'true'
    
    # Set output directory
    output_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 Processing dataset: {dataset_name}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Force regenerate: {force_regenerate}")
    print(f"📋 Mode: {args.mode}")
    
    # Generate separate examples
    if args.mode in ['separate', 'both']:
        try:
            separate_results = generate_separate_examples(
                dataset_name, output_dir, force_regenerate
            )
        except Exception as e:
            print(f"❌ Error generating separate examples: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate combined examples for all datasets
    if args.mode in ['combined', 'both']:
        try:
            combined_results = generate_combined_examples(
                dataset_name, output_dir, args.max_combined, args.min_combined
            )
        except Exception as e:
            print(f"❌ Error generating combined examples: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✨ Generation complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    # List generated files
    print("\n📄 Generated files:")
    for file in sorted(output_dir.glob("*.json")):
        size_kb = file.stat().st_size / 1024
        print(f"   - {file.name} ({size_kb:.1f} KB)")
    
    for file in sorted(output_dir.glob("*.npy")):
        size_kb = file.stat().st_size / 1024
        print(f"   - {file.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()