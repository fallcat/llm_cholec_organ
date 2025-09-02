#!/usr/bin/env python3
"""Script to load and analyze metrics from JSON output."""

import json
import sys
from pathlib import Path
from typing import Dict, List

def load_metrics_json(json_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def print_model_comparison(data: Dict):
    """Print a comparison of models from JSON data."""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    models = data["metadata"]["models"]
    
    # Collect all evaluation types
    eval_types = set()
    for model in models:
        if model in data["results"]:
            eval_types.update(data["results"][model].keys())
    
    eval_types = sorted(eval_types)
    
    # Print comparison table
    for eval_type in eval_types:
        print(f"\n📊 {eval_type.replace('_', ' ').title()}:")
        print("-" * 60)
        
        # Headers
        print(f"{'Model':<40} {'N':<5} {'Macro F1':<10} {'Micro F1':<10} {'Gated Acc':<10}")
        print("-" * 60)
        
        for model in models:
            if model in data["results"] and eval_type in data["results"][model]:
                result = data["results"][model][eval_type]
                n_examples = result.get("n_examples", 0)
                
                if "aggregate" in result:
                    agg = result["aggregate"]
                    macro_f1 = agg.get("macro_f1", 0)
                    micro_f1 = agg.get("micro_f1", 0)
                    gated_acc = agg.get("macro_gated_acc", 0)
                    
                    # Truncate long model names
                    model_display = model if len(model) <= 38 else model[:35] + "..."
                    
                    print(f"{model_display:<40} {n_examples:<5} {macro_f1:<10.3f} {micro_f1:<10.3f} {gated_acc:<10.3f}")

def print_organ_performance(data: Dict, model: str, eval_type: str):
    """Print per-organ performance for a specific model and evaluation type."""
    if model not in data["results"] or eval_type not in data["results"][model]:
        print(f"No data found for {model} - {eval_type}")
        return
    
    result = data["results"][model][eval_type]
    
    print(f"\n📋 Per-Organ Performance: {model} - {eval_type}")
    print("-" * 80)
    
    if "per_organ" in result:
        print(f"{'Organ':<25} {'TP':<5} {'FN':<5} {'TN':<5} {'FP':<5} {'Acc':<8} {'Hit@P':<8} {'Gated':<8} {'F1':<8}")
        print("-" * 80)
        
        for organ in result["per_organ"]:
            label = organ.get("label", "Unknown")
            tp = organ.get("TP", 0)
            fn = organ.get("FN", 0)
            tn = organ.get("TN", 0)
            fp = organ.get("FP", 0)
            acc = organ.get("PresenceAcc", 0)
            hit = organ.get("Hit@Point|Present", 0)
            gated = organ.get("GatedAcc", 0)
            f1 = organ.get("F1", 0)
            
            label_display = label if len(label) <= 23 else label[:20] + "..."
            
            print(f"{label_display:<25} {tp:<5} {fn:<5} {tn:<5} {fp:<5} "
                  f"{acc:<8.3f} {hit:<8.3f} {gated:<8.3f} {f1:<8.3f}")

def find_best_performers(data: Dict):
    """Find the best performing model for each metric."""
    print("\n" + "="*80)
    print("BEST PERFORMERS")
    print("="*80)
    
    metrics_to_check = ["macro_f1", "micro_f1", "macro_gated_acc", "macro_presence_acc"]
    
    for metric in metrics_to_check:
        best_model = None
        best_eval = None
        best_value = -1
        
        for model in data["results"]:
            for eval_type in data["results"][model]:
                if "aggregate" in data["results"][model][eval_type]:
                    agg = data["results"][model][eval_type]["aggregate"]
                    value = agg.get(metric, -1)
                    
                    if value > best_value:
                        best_value = value
                        best_model = model
                        best_eval = eval_type
        
        if best_model:
            metric_display = metric.replace("_", " ").replace("macro", "Macro").replace("micro", "Micro")
            print(f"\n🏆 {metric_display}: {best_value:.3f}")
            print(f"   Model: {best_model}")
            print(f"   Evaluation: {best_eval}")

def main():
    """Main function to analyze metrics JSON."""
    # Check if JSON file path is provided
    if len(sys.argv) < 2:
        # Look for most recent results
        results_dir = Path("results")
        if results_dir.exists():
            # Find directories starting with "pointing_original_"
            pointing_dirs = sorted([d for d in results_dir.iterdir() 
                                  if d.is_dir() and d.name.startswith("pointing_original_")],
                                 reverse=True)
            
            for dir in pointing_dirs:
                json_file = dir / "metrics_comparison.json"
                if json_file.exists():
                    print(f"Using most recent results: {json_file}")
                    json_path = str(json_file)
                    break
            else:
                print("No metrics_comparison.json found in recent results.")
                print("Usage: python analyze_metrics_json.py [path_to_metrics_comparison.json]")
                sys.exit(1)
        else:
            print("No results directory found.")
            print("Usage: python analyze_metrics_json.py [path_to_metrics_comparison.json]")
            sys.exit(1)
    else:
        json_path = sys.argv[1]
    
    # Load JSON data
    try:
        data = load_metrics_json(json_path)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON file: {e}")
        sys.exit(1)
    
    # Print analysis
    print(f"\n📂 Loaded metrics from: {json_path}")
    print(f"📅 Timestamp: {data['metadata']['timestamp']}")
    print(f"🖼️  Canvas size: {data['metadata']['canvas_width']}x{data['metadata']['canvas_height']}")
    print(f"🔬 Models evaluated: {len(data['metadata']['models'])}")
    
    # Print model comparison
    print_model_comparison(data)
    
    # Find best performers
    find_best_performers(data)
    
    # Example: Print detailed organ performance for best model
    if data["metadata"]["models"]:
        first_model = data["metadata"]["models"][0]
        if first_model in data["results"]:
            first_eval = list(data["results"][first_model].keys())[0]
            print_organ_performance(data, first_model, first_eval)

if __name__ == "__main__":
    main()