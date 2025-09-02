#!/usr/bin/env python
"""
Analyze Pointing Results from Enhanced Evaluator
This script loads and analyzes results from the new enhanced evaluator format.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


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


def find_latest_results_dir(base_dir: str = "../results") -> Optional[Path]:
    """Find the most recent pointing results directory.
    
    Args:
        base_dir: Base results directory
        
    Returns:
        Path to latest results directory or None
    """
    base_path = Path(base_dir)
    
    # Look for directories with pattern pointing_YYYYMMDD_HHMMSS
    pointing_dirs = sorted([d for d in base_path.glob("pointing_*") if d.is_dir()])
    
    if not pointing_dirs:
        print(f"No pointing result directories found in {base_dir}")
        return None
    
    # Return the most recent one
    return pointing_dirs[-1]


def load_results_from_pickle(pickle_path: Path) -> Dict:
    """Load results from pickle file.
    
    Args:
        pickle_path: Path to raw_results.pkl
        
    Returns:
        Results dictionary
    """
    try:
        with open(pickle_path, "rb") as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return {}


def load_results_from_json_dir(
    results_dir: Path,
    prompt_name: str,
    model_name: str,
    split: str = "train",
) -> List[Dict]:
    """Load results from JSON files in enhanced evaluator format.
    
    Args:
        results_dir: Root results directory
        prompt_name: Prompt name (e.g., "zero_shot", "fewshot_standard")
        model_name: Model name
        split: Dataset split
        
    Returns:
        List of result records
    """
    # Check different possible paths
    possible_paths = [
        results_dir / prompt_name / model_name / "cholecseg8k_pointing",
        results_dir / prompt_name / model_name,
        results_dir / model_name / prompt_name,
    ]
    
    for sample_dir in possible_paths:
        if sample_dir.exists():
            break
    else:
        print(f"⚠️  No results directory found for {model_name}/{prompt_name}")
        return []
    
    records = []
    json_files = sorted(sample_dir.glob(f"{split}_*.json"))
    
    if not json_files:
        print(f"⚠️  No JSON files found in {sample_dir}")
        return []
    
    print(f"  Found {len(json_files)} JSON files in {sample_dir.relative_to(results_dir)}")
    
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
            print(f"⚠️  Error loading {json_file.name}: {e}")
            continue
    
    return records


def calculate_comprehensive_metrics(
    records: List[Dict],
) -> Tuple[List[Dict], Dict, int]:
    """Calculate comprehensive pointing metrics.
    
    Args:
        records: List of evaluation records
        
    Returns:
        Tuple of (per-organ metrics, totals, number of examples)
    """
    if not records:
        return [], {}, 0
    
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
            "F1": safe_div(2 * tp[i], 2 * tp[i] + fp[i] + fn[i]),
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
    f1_scores = [r["F1"] for r in rows if r["F1"] is not None]
    
    macro_presence = np.mean(pres_accs) if pres_accs else 0.0
    macro_gated = np.mean(gated_accs) if gated_accs else 0.0
    macro_hit = np.mean(hit_rates) if hit_rates else 0.0
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    print("\nTotals across organs:")
    print(
        f"TP={totals['TP']}  FN={totals['FN']}  TN={totals['TN']}  FP={totals['FP']}  "
        f"Present={totals['Present_n']}  Absent={totals['Absent_n']}  Total={totals['Total']}"
    )
    print(
        f"Macro PresenceAcc={format_percentage(macro_presence)}   "
        f"Macro Hit@Point|Present={format_percentage(macro_hit)}   "
        f"Macro GatedAcc={format_percentage(macro_gated)}   "
        f"Macro F1={format_percentage(macro_f1)}"
    )


def analyze_results_directory(
    results_dir: Path,
    models: List[str] = None,
    prompts: List[str] = None,
    split: str = "train",
):
    """Analyze results from a specific results directory.
    
    Args:
        results_dir: Path to results directory
        models: List of model names to analyze
        prompts: List of prompt names to analyze
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
    print(f"ANALYZING RESULTS FROM: {results_dir}")
    print("="*80)
    
    # First, try to load from pickle if available
    pickle_path = results_dir / "raw_results.pkl"
    if pickle_path.exists():
        print(f"\n📦 Found pickle file: {pickle_path}")
        results = load_results_from_pickle(pickle_path)
        
        if results:
            print(f"  Loaded results for models: {list(results.keys())}")
            
            # Check if results have enhanced format
            for model_name in results:
                if model_name in models:
                    model_results = results[model_name]
                    for eval_type in model_results:
                        if "records" in model_results[eval_type]:
                            # Enhanced format with records
                            records = model_results[eval_type]["records"]
                            rows, totals, n_examples = calculate_comprehensive_metrics(records)
                            
                            print(f"\n{'='*60}")
                            print(f"From pickle: {model_name} - {eval_type}")
                            print(f"{'='*60}")
                            
                            print_metrics_table(
                                rows, totals, n_examples,
                                model_name, eval_type, split
                            )
    
    # Also check for JSON files
    print("\n" + "="*80)
    print("CHECKING FOR JSON FILES")
    print("="*80)
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        for prompt_name in prompts:
            # Load results from JSON files
            records = load_results_from_json_dir(
                results_dir, prompt_name, model_name, split
            )
            
            if not records:
                print(f"  ⚠️  No results found for {prompt_name}")
                continue
            
            # Calculate metrics
            rows, totals, n_examples = calculate_comprehensive_metrics(records)
            
            # Print table
            print_metrics_table(
                rows, totals, n_examples,
                model_name, prompt_name, split
            )
    
    # Check for summary files
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        print(f"\n📊 Summary CSV available at: {summary_path}")
    
    comparison_path = results_dir / "metrics_comparison.txt"
    if comparison_path.exists():
        print(f"📝 Metrics comparison available at: {comparison_path}")


def main():
    """Main function."""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python eval_pointing_analyze.py [results_dir|--latest]")
            print("\nThis script analyzes pointing evaluation results.")
            print("\nArguments:")
            print("  results_dir: Path to specific results directory")
            print("  --latest: Analyze the most recent results (default)")
            print("\nExamples:")
            print("  python eval_pointing_analyze.py --latest")
            print("  python eval_pointing_analyze.py ../results/pointing_20250901_041511")
            return
        
        if sys.argv[1] == "--latest":
            results_dir = find_latest_results_dir()
            if results_dir is None:
                print("No results directories found!")
                return
        else:
            results_dir = Path(sys.argv[1])
            if not results_dir.exists():
                print(f"Error: Directory not found: {results_dir}")
                return
    else:
        # Default to latest results
        results_dir = find_latest_results_dir()
        if results_dir is None:
            print("No results directories found!")
            print("Run: python eval_pointing_analyze.py --help for usage")
            return
    
    # Run analysis
    analyze_results_directory(results_dir)


if __name__ == "__main__":
    main()