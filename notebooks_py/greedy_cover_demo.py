#!/usr/bin/env python3
"""
Demonstration of greedy set cover for few-shot example selection.

This script shows how to use the greedy cover algorithm to select a minimal
set of training examples that cover all organ classes in the dataset.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Iterable, Any, Tuple


def select_examples_from_presence_matrix(
    presence_matrix: np.ndarray,
    required_cols: Optional[Iterable[int]] = None,
    max_examples: Optional[int] = None,
    prefer_row_with_more_gain_then_density: bool = True,
) -> Dict[str, Any]:
    """
    Greedy set-cover over a binary presence matrix.

    Args:
        presence_matrix: shape (N_examples, N_organs), entries in {0,1} (or any truthy/falsey).
        required_cols: optional iterable of column indices to cover. If None, use all columns
                       that have at least one positive across rows.
        max_examples: optional hard cap on number of selected rows.
        prefer_row_with_more_gain_then_density: tie-break preference.

    Returns:
        dict with keys:
            - selected_indices: list[int] of chosen row indices
            - covered_cols: list[int] of covered column indices
            - uncovered_cols: list[int] of required but uncovered column indices
            - impossible_cols: list[int] of columns with no 1s in any row
            - selected_count: int
            - coverage_vector: 1D bool array for covered columns
            - per_row_gain: list[Tuple[int,int]] (row_index, first-iteration gain)
    """
    if presence_matrix.dtype != np.bool_:
        pm = presence_matrix.astype(bool)
    else:
        pm = presence_matrix

    n_rows, n_cols = pm.shape
    if required_cols is None:
        # Only columns that are coverable (have at least one True)
        coverable = np.where(pm.any(axis=0))[0]
        required = np.array(coverable, dtype=int)
    else:
        required = np.array(list(required_cols), dtype=int)

    # Identify impossible columns (all False across rows)
    col_has_any = pm.any(axis=0)
    impossible = required[~col_has_any[required]] if len(required) > 0 else np.array([], dtype=int)

    remaining_mask = np.zeros(n_cols, dtype=bool)
    remaining_mask[required] = True
    # Remove impossible from remaining
    if len(impossible) > 0:
        remaining_mask[impossible] = False

    selected: List[int] = []
    covered = np.zeros(n_cols, dtype=bool)

    # Precompute row densities (how many 1s in required columns)
    pm_req = pm[:, required] if len(required) else pm[:, []]
    row_req_counts = pm_req.sum(axis=1) if pm_req.size else np.zeros(n_rows, dtype=int)

    # Keep for reporting: first-iteration gain
    first_gain_report: List[Tuple[int, int]] = []

    while remaining_mask.any():
        # Gain per row = how many *currently* uncovered required cols this row covers
        gain_vec = (pm[:, remaining_mask]).sum(axis=1)
        if len(first_gain_report) == 0:
            for i in range(n_rows):
                first_gain_report.append((i, int(gain_vec[i])))

        best_idx = -1
        best_gain = 0
        best_tiebreak = -1  # used depending on preference

        for i in range(n_rows):
            g = int(gain_vec[i])
            if g <= 0:
                continue
            # Tie-breaks
            if prefer_row_with_more_gain_then_density:
                # primarily by gain, then by row density over required cols, then by earlier index
                tie = (g > best_gain) or (g == best_gain and row_req_counts[i] > best_tiebreak) or (
                    g == best_gain and row_req_counts[i] == best_tiebreak and i < best_idx
                )
                if tie:
                    best_idx = i
                    best_gain = g
                    best_tiebreak = int(row_req_counts[i])
            else:
                # primarily by gain, then favor sparser row (smaller density), then earlier index
                tie = (g > best_gain) or (g == best_gain and row_req_counts[i] < best_tiebreak) or (
                    g == best_gain and row_req_counts[i] == best_tiebreak and i < best_idx
                )
                if tie:
                    best_idx = i
                    best_gain = g
                    best_tiebreak = int(row_req_counts[i])

        if best_idx == -1:
            # no further progress possible
            break

        selected.append(best_idx)
        # Update covered/remaining
        covered |= pm[best_idx]
        remaining_mask = np.zeros(n_cols, dtype=bool)
        # still need required columns that aren't yet covered and aren't impossible
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


def main():
    """Main demo function."""
    
    print("=" * 80)
    print("GREEDY SET COVER DEMO FOR FEW-SHOT SELECTION")
    print("=" * 80)
    print()
    
    # Load the presence matrix
    dataset_name = "cholecseg8k_local"
    data_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
    
    presence_matrix_file = data_dir / "presence_matrix_train.npy"
    if not presence_matrix_file.exists():
        print(f"❌ Presence matrix not found: {presence_matrix_file}")
        print("   Please run the unified fewshot selector first to generate it.")
        return
    
    presence_matrix = np.load(presence_matrix_file)
    print(f"✅ Loaded presence matrix: {presence_matrix.shape}")
    
    # Load organ labels
    import sys
    sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')
    from endopoint.datasets import build_dataset
    
    dataset = build_dataset(dataset_name, data_dir="/shared_data0/weiqiuy/datasets/cholecseg8k")
    
    # Get label names
    label_names = []
    for class_id in dataset.label_ids:
        label_name = dataset.id2label[class_id]
        label_names.append(label_name)
    
    print(f"✅ Dataset has {len(label_names)} organs:")
    for i, name in enumerate(label_names):
        count = presence_matrix[:, i].sum()
        pct = count / len(presence_matrix) * 100
        print(f"   {i:2d}. {name:30} {count:5d} samples ({pct:5.1f}%)")
    print()
    
    # Example 1: Select minimum examples covering all organs
    print("=" * 80)
    print("EXAMPLE 1: Minimum Set to Cover All Organs")
    print("=" * 80)
    
    result = select_examples_from_presence_matrix(
        presence_matrix,
        required_cols=None,  # Cover all columns that have at least one 1
        max_examples=None     # No limit
    )
    
    selected_indices = result["selected_indices"]
    uncovered_cols = result["uncovered_cols"]
    impossible_cols = result["impossible_cols"]
    
    print(f"✅ Selected {len(selected_indices)} images to cover all organs")
    print(f"   Indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
    
    if uncovered_cols:
        print(f"⚠️  Uncovered columns: {uncovered_cols}")
    if impossible_cols:
        print(f"⚠️  Impossible columns (no examples): {impossible_cols}")
    
    # Show coverage per selected image
    print("\n📊 Coverage per selected image:")
    for i, idx in enumerate(selected_indices[:5], 1):
        organs_present = []
        for j, name in enumerate(label_names):
            if presence_matrix[idx, j]:
                organs_present.append(name)
        print(f"   {i}. Image {idx:4d}: {len(organs_present)} organs - {', '.join(organs_present[:3])}...")
    
    if len(selected_indices) > 5:
        print(f"   ... and {len(selected_indices) - 5} more images")
    
    # Example 2: Select with a cap on number of examples
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Best Coverage with Maximum 8 Examples")
    print("=" * 80)
    
    result_capped = select_examples_from_presence_matrix(
        presence_matrix,
        required_cols=None,
        max_examples=8  # Cap at 8 examples
    )
    
    selected_capped = result_capped["selected_indices"]
    covered_cols = result_capped["covered_cols"]
    uncovered_capped = result_capped["uncovered_cols"]
    
    print(f"✅ Selected {len(selected_capped)} images (capped at 8)")
    print(f"   Indices: {selected_capped}")
    
    # Show which organs are covered
    covered_organs = [label_names[i] for i in covered_cols]
    print(f"\n✅ Covered {len(covered_organs)}/{len(label_names)} organs:")
    for organ in covered_organs:
        print(f"   ✓ {organ}")
    
    if uncovered_capped:
        uncovered_organs = [label_names[i] for i in uncovered_capped if i < len(label_names)]
        print(f"\n⚠️  Uncovered organs ({len(uncovered_organs)}):")
        for organ in uncovered_organs:
            print(f"   ✗ {organ}")
    
    # Example 3: Focus on rare organs
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Focus on Rare Organs (< 20% prevalence)")
    print("=" * 80)
    
    # Identify rare organs
    organ_prevalence = presence_matrix.mean(axis=0)
    rare_threshold = 0.2
    rare_cols = [i for i, prev in enumerate(organ_prevalence) if prev < rare_threshold]
    rare_organs = [label_names[i] for i in rare_cols]
    
    print(f"🔴 Rare organs (< {rare_threshold*100:.0f}% prevalence):")
    for col_idx in rare_cols:
        organ = label_names[col_idx]
        prev = organ_prevalence[col_idx] * 100
        print(f"   {organ:30} {prev:5.1f}%")
    
    # Select examples that cover rare organs
    result_rare = select_examples_from_presence_matrix(
        presence_matrix,
        required_cols=rare_cols,  # Only cover rare organs
        max_examples=5  # Limit to 5 examples
    )
    
    selected_rare = result_rare["selected_indices"]
    print(f"\n✅ Selected {len(selected_rare)} images to cover rare organs")
    print(f"   Indices: {selected_rare}")
    
    # Show coverage
    for idx in selected_rare:
        covered_rare = [label_names[i] for i in rare_cols if presence_matrix[idx, i]]
        all_organs = [label_names[i] for i in range(len(label_names)) if presence_matrix[idx, i]]
        print(f"\n   Image {idx}:")
        print(f"     Rare organs: {', '.join(covered_rare)}")
        print(f"     All organs ({len(all_organs)}): {', '.join(all_organs[:5])}...")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_file = data_dir / "greedy_cover_results.json"
    
    save_data = {
        "presence_matrix_shape": list(presence_matrix.shape),
        "organ_names": label_names,
        "examples": {
            "all_organs": {
                "selected_indices": [int(i) for i in selected_indices],
                "num_selected": len(selected_indices),
                "uncovered_cols": [int(i) for i in uncovered_cols],
                "impossible_cols": [int(i) for i in impossible_cols]
            },
            "capped_8": {
                "selected_indices": [int(i) for i in selected_capped],
                "num_selected": len(selected_capped),
                "covered_cols": [int(i) for i in covered_cols],
                "uncovered_cols": [int(i) for i in uncovered_capped]
            },
            "rare_organs": {
                "rare_threshold": rare_threshold,
                "rare_organs": rare_organs,
                "selected_indices": [int(i) for i in selected_rare],
                "num_selected": len(selected_rare)
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"💾 Results saved to: {output_file.name}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    print("\n✨ Key Insights:")
    print(f"  1. Minimum images needed for full coverage: {len(selected_indices)}")
    print(f"  2. Coverage with 8 images: {len(covered_organs)}/{len(label_names)} organs")
    print(f"  3. Images needed for rare organs: {len(selected_rare)}")
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()