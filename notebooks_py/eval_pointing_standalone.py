#!/usr/bin/env python
"""
Standalone Pointing Evaluation with Comprehensive Metrics
This version loads results from existing JSON files and calculates comprehensive metrics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Constants from CholecSeg8k
ID2LABEL = {
    0: "Black Background", 1: "Abdominal Wall", 2: "Liver", 3: "Gastrointestinal Tract",
    4: "Fat", 5: "Grasper", 6: "Connective Tissue", 7: "Blood", 8: "Cystic Duct",
    9: "L-hook Electrocautery", 10: "Gallbladder", 11: "Hepatic Vein", 12: "Liver Ligament"
}
LABEL_IDS = [k for k in sorted(ID2LABEL) if k != 0]


def format_percentage(x: Optional[float]) -> str:
    """Format a percentage value."""
    return f"{x*100:6.2f}%" if x is not None else "   --  "


def calculate_comprehensive_metrics(
    records: List[Dict],
) -> Tuple[List[Dict], Dict, int]:
    """Calculate comprehensive pointing metrics.
    
    Args:
        records: List of evaluation records with y_true, y_pred, hits fields
        
    Returns:
        Tuple of (per-organ metrics, totals, number of examples)
    """
    K = len(LABEL_IDS)
    
    # Initialize counters
    tp = np.zeros(K, dtype=np.int64)
    tn = np.zeros(K, dtype=np.int64)
    fp = np.zeros(K, dtype=np.int64)
    fn = np.zeros(K, dtype=np.int64)
    
    # Gated metrics
    tpg = np.zeros(K, dtype=np.int64)
    tng = np.zeros(K, dtype=np.int64)
    fpg = np.zeros(K, dtype=np.int64)
    fng = np.zeros(K, dtype=np.int64)
    
    hit_present = np.zeros(K, dtype=np.int64)
    present_n = np.zeros(K, dtype=np.int64)
    n_examples = 0
    
    for obj in records:
        y_true = np.asarray(obj["y_true"], dtype=int)
        y_pred = np.asarray(obj["y_pred"], dtype=int)
        hits = np.asarray(obj.get("hits", np.zeros(K)), dtype=int)
        
        # Standard confusion matrix
        tp += ((y_true == 1) & (y_pred == 1))
        tn += ((y_true == 0) & (y_pred == 0))
        fp += ((y_true == 0) & (y_pred == 1))
        fn += ((y_true == 1) & (y_pred == 0))
        
        # Gated predictions
        pred_g = ((y_pred == 1) & (hits == 1))
        tpg += ((y_true == 1) & (pred_g == 1))
        tng += ((y_true == 0) & (pred_g == 0))
        fpg += ((y_true == 0) & (pred_g == 1))
        fng += ((y_true == 1) & (pred_g == 0))
        
        # Hitting statistics
        hit_present += ((hits == 1) & (y_true == 1))
        present_n += (y_true == 1)
        
        n_examples += 1
    
    absent_n = (tn + fp)
    total_n = present_n + absent_n
    
    def safe_div(a, b):
        return float(a / b) if b > 0 else None
    
    # Build per-organ metric rows
    rows = []
    for i, label_id in enumerate(LABEL_IDS):
        rows.append({
            "label_id": label_id,
            "label": ID2LABEL[label_id],
            "TP": int(tp[i]),
            "FN": int(fn[i]),
            "TN": int(tn[i]),
            "FP": int(fp[i]),
            "Present_n": int(present_n[i]),
            "Absent_n": int(absent_n[i]),
            "Total": int(total_n[i]),
            "PresenceAcc": safe_div(tp[i] + tn[i], total_n[i]),
            "Hit@Point|Present": safe_div(hit_present[i], present_n[i]),
            "Gated_TP": int(tpg[i]),
            "Gated_FN": int(fng[i]),
            "Gated_TN": int(tng[i]),
            "Gated_FP": int(fpg[i]),
            "GatedAcc": safe_div(tpg[i] + tng[i], total_n[i]),
        })
    
    # Calculate totals
    totals = {
        "TP": int(tp.sum()),
        "FN": int(fn.sum()),
        "TN": int(tn.sum()),
        "FP": int(fp.sum()),
        "Present_n": int(present_n.sum()),
        "Absent_n": int(absent_n.sum()),
        "Total": int(total_n.sum()),
        "Gated_TP": int(tpg.sum()),
        "Gated_FN": int(fng.sum()),
        "Gated_TN": int(tng.sum()),
        "Gated_FP": int(fpg.sum()),
    }
    
    return rows, totals, n_examples


def print_metrics_table(
    rows: List[Dict],
    totals: Dict,
    n_examples: int,
    model_name: str,
    prompt_name: str,
    split: str = "train",
):
    """Print formatted metrics table."""
    print(f"\nModel: {model_name} | Prompt: {prompt_name} | Split: {split} | Examples used: {n_examples}")
    
    header = (
        "ID  Label                     TP   FN   TN   FP   "
        "Pres  Abs   Tot   PresenceAcc   Hit@Pt|Pres   gTP  gFN  gTN  gFP   GatedAcc"
    )
    print(header)
    
    for r in rows:
        print(
            f"{r['label_id']:>2}  {r['label']:<24}  "
            f"{r['TP']:>3}  {r['FN']:>3}  {r['TN']:>3}  {r['FP']:>3}   "
            f"{r['Present_n']:>4} {r['Absent_n']:>4} {r['Total']:>5}   "
            f"{format_percentage(r['PresenceAcc'])}    {format_percentage(r['Hit@Point|Present'])}   "
            f"{r['Gated_TP']:>3}  {r['Gated_FN']:>3}  {r['Gated_TN']:>3}  {r['Gated_FP']:>3}   "
            f"{format_percentage(r['GatedAcc'])}"
        )
    
    # Calculate macro averages
    pres_accs = [r["PresenceAcc"] for r in rows if r["PresenceAcc"] is not None]
    gated_accs = [r["GatedAcc"] for r in rows if r["GatedAcc"] is not None]
    hit_rates = [r["Hit@Point|Present"] for r in rows if r["Hit@Point|Present"] is not None]
    
    macro_presence = np.mean(pres_accs) if pres_accs else 0.0
    macro_gated = np.mean(gated_accs) if gated_accs else 0.0
    macro_hit = np.mean(hit_rates) if hit_rates else 0.0
    
    print("\nTotals across organs:")
    print(
        f"TP={totals['TP']}  FN={totals['FN']}  TN={totals['TN']}  FP={totals['FP']}  "
        f"Present={totals['Present_n']}  Absent={totals['Absent_n']}  Total={totals['Total']}"
    )
    print(
        f"Macro PresenceAcc={format_percentage(macro_presence)}   "
        f"Macro Hit@Point|Present={format_percentage(macro_hit)}   "
        f"Macro GatedAcc={format_percentage(macro_gated)}"
    )


def load_pointing_results(
    root_dir: str,
    prompt_name: str,
    model_name: str,
    split: str = "train",
    results_dir_name: str = "results/pointing",
    samples_subdir: str = "cholecseg8k_pointing",
) -> List[Dict]:
    """Load pointing results from JSON files.
    
    Args:
        root_dir: Root directory path
        prompt_name: Prompt/evaluation name
        model_name: Model name
        split: Dataset split
        results_dir_name: Results directory name
        samples_subdir: Samples subdirectory name
        
    Returns:
        List of result records
    """
    sample_dir = Path(root_dir) / results_dir_name / prompt_name / model_name / samples_subdir
    
    if not sample_dir.exists():
        print(f"⚠️  Directory not found: {sample_dir}")
        return []
    
    records = []
    json_files = sorted(sample_dir.glob(f"{split}_*.json"))
    
    if not json_files:
        print(f"⚠️  No JSON files found in {sample_dir}")
        return []
    
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                obj = json.load(f)
            
            # Validate structure
            if not isinstance(obj.get("y_true"), list) or not isinstance(obj.get("y_pred"), list):
                continue
                
            # Add hits if missing
            if "hits" not in obj:
                obj["hits"] = [0] * len(LABEL_IDS)
            
            records.append(obj)
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
            continue
    
    return records


def evaluate_from_files(
    root_dir: str = "..",
    models: List[str] = None,
    prompts: List[str] = None,
    split: str = "train",
):
    """Evaluate results from existing JSON files.
    
    Args:
        root_dir: Root directory
        models: List of model names
        prompts: List of prompt names
        split: Dataset split
    """
    if models is None:
        models = [
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "gemini-2.0-flash-exp"
        ]
    
    if prompts is None:
        prompts = ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]
    
    print("="*80)
    print("COMPREHENSIVE POINTING EVALUATION FROM FILES")
    print("="*80)
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        for prompt_name in prompts:
            # Load results
            records = load_pointing_results(
                root_dir, prompt_name, model_name, split
            )
            
            if not records:
                print(f"\n⚠️  No results found for {prompt_name}")
                continue
            
            # Calculate metrics
            rows, totals, n_examples = calculate_comprehensive_metrics(records)
            
            # Print table
            print_metrics_table(
                rows, totals, n_examples,
                model_name, prompt_name, split
            )


def main():
    """Main function."""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python eval_pointing_standalone.py [root_dir]")
            print("\nThis script loads existing pointing results and calculates comprehensive metrics.")
            print("\nArguments:")
            print("  root_dir: Root directory containing results (default: ..)")
            return
        
        root_dir = sys.argv[1]
    else:
        root_dir = ".."
    
    # Run evaluation
    evaluate_from_files(root_dir)


if __name__ == "__main__":
    main()