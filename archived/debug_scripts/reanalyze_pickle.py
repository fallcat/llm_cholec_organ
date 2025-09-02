#!/usr/bin/env python
"""
Re-analyze pickle results to extract comprehensive metrics.
"""

import pickle
import numpy as np
from pathlib import Path
import sys


# Constants
ID2LABEL = {
    0: "Black Background", 1: "Abdominal Wall", 2: "Liver", 3: "Gastrointestinal Tract",
    4: "Fat", 5: "Grasper", 6: "Connective Tissue", 7: "Blood", 8: "Cystic Duct",
    9: "L-hook Electrocautery", 10: "Gallbladder", 11: "Hepatic Vein", 12: "Liver Ligament"
}
LABEL_IDS = [k for k in sorted(ID2LABEL) if k != 0]


def format_pct(x):
    return f"{x*100:6.2f}%" if x is not None else "   --  "


def analyze_pickle_results(pickle_path):
    """Analyze results from pickle file."""
    
    print(f"Loading: {pickle_path}")
    
    try:
        with open(pickle_path, "rb") as f:
            all_results = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return
    
    print(f"Models found: {list(all_results.keys())}")
    print("")
    
    for model_name, model_results in all_results.items():
        print("="*80)
        print(f"MODEL: {model_name}")
        print("="*80)
        
        for eval_type, eval_data in model_results.items():
            print(f"\n{eval_type}:")
            print("-" * 40)
            
            if "results" in eval_data:
                # Has individual sample results
                results = eval_data["results"]
                n_samples = len(results)
                
                # Initialize counters for comprehensive metrics
                K = len(LABEL_IDS)
                tp = np.zeros(K, dtype=np.int64)
                tn = np.zeros(K, dtype=np.int64)
                fp = np.zeros(K, dtype=np.int64)
                fn = np.zeros(K, dtype=np.int64)
                
                # Process each sample
                for result in results:
                    y_true = np.asarray(result["y_true"])
                    y_pred = np.asarray(result["y_pred"])
                    
                    tp += ((y_true == 1) & (y_pred == 1))
                    tn += ((y_true == 0) & (y_pred == 0))
                    fp += ((y_true == 0) & (y_pred == 1))
                    fn += ((y_true == 1) & (y_pred == 0))
                
                # Calculate per-organ metrics
                print(f"Samples: {n_samples}")
                print("\nPer-Organ Metrics:")
                print("ID  Organ                     TP  FN  TN  FP  Accuracy  Precision  Recall    F1")
                print("-" * 80)
                
                organ_accs = []
                organ_f1s = []
                
                for i, label_id in enumerate(LABEL_IDS):
                    organ_name = ID2LABEL[label_id]
                    
                    total = tp[i] + tn[i] + fp[i] + fn[i]
                    acc = (tp[i] + tn[i]) / total if total > 0 else 0
                    prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
                    rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    
                    organ_accs.append(acc)
                    organ_f1s.append(f1)
                    
                    print(f"{label_id:2}  {organ_name:<24}  "
                          f"{tp[i]:2}  {fn[i]:2}  {tn[i]:2}  {fp[i]:2}  "
                          f"{format_pct(acc)}  {format_pct(prec)}  {format_pct(rec)}  {format_pct(f1)}")
                
                # Overall metrics
                print("\nOverall Metrics:")
                print(f"Total TP={tp.sum()}  FN={fn.sum()}  TN={tn.sum()}  FP={fp.sum()}")
                
                micro_acc = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum())
                macro_acc = np.mean(organ_accs)
                macro_f1 = np.mean(organ_f1s)
                
                print(f"Micro Accuracy (overall): {format_pct(micro_acc)}")
                print(f"Macro Accuracy (avg):     {format_pct(macro_acc)}")
                print(f"Macro F1 (avg):          {format_pct(macro_f1)}")
                
            elif "metrics" in eval_data:
                # Has aggregated metrics
                metrics = eval_data["metrics"]
                print(f"Overall Accuracy: {metrics.get('overall_accuracy', 'N/A'):.3f}")
                print(f"Avg F1: {metrics.get('avg_f1', 'N/A'):.3f}")
                
                if "organ_metrics" in metrics:
                    print("\nPer-Organ Summary:")
                    for organ_name, organ_metrics in metrics["organ_metrics"].items():
                        print(f"  {organ_name}: F1={organ_metrics['f1']:.3f}")
            else:
                print("  No detailed results available")
    
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    # Print comparison table
    print("\nModel                          Eval Type                Accuracy    F1")
    print("-" * 75)
    
    for model_name in all_results:
        for eval_type in all_results[model_name]:
            if "metrics" in all_results[model_name][eval_type]:
                metrics = all_results[model_name][eval_type]["metrics"]
                acc = metrics.get("overall_accuracy", 0)
                f1 = metrics.get("avg_f1", 0)
                print(f"{model_name:<30} {eval_type:<24} {format_pct(acc)}  {format_pct(f1)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python reanalyze_pickle.py <path_to_raw_results.pkl>")
        print("\nExample:")
        print("  python reanalyze_pickle.py ../results/pointing_20250901_041511/raw_results.pkl")
        return
    
    pickle_path = Path(sys.argv[1])
    if not pickle_path.exists():
        print(f"Error: File not found: {pickle_path}")
        return
    
    analyze_pickle_results(pickle_path)


if __name__ == "__main__":
    main()