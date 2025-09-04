#!/usr/bin/env python3
"""
Generate LaTeX code for creating example figures directly in LaTeX.
This avoids the need for matplotlib or other dependencies.

Usage:
    python notebooks_py/generate_latex_figures.py
"""

import json
from pathlib import Path
import random

def load_results_for_samples(results_dir, eval_type, num_samples=6):
    """Load results for a few samples across all models."""
    
    models = ["gpt-4.1", "gemini-2.0-flash", "claude-sonnet-4-20250514"]
    model_names = {
        "gpt-4.1": "GPT-4.1",
        "gemini-2.0-flash": "Gemini-2.0",
        "claude-sonnet-4-20250514": "Claude-4"
    }
    
    # Find common samples
    common_samples = None
    for model in models:
        model_path = Path(results_dir) / eval_type / model / "cholecseg8k_pointing"
        if model_path.exists():
            files = set([f.name for f in model_path.glob("train_*.json") 
                        if "metrics" not in f.name])
            if common_samples is None:
                common_samples = files
            else:
                common_samples = common_samples.intersection(files)
    
    if not common_samples:
        return []
    
    # Take random samples
    sample_files = sorted(list(common_samples))[:num_samples]
    
    # Organ names
    organ_names = [
        "Abdominal Wall", "Liver", "Gastrointestinal Tract", "Fat",
        "Grasper", "Connective Tissue", "Blood", "Cystic Duct",
        "L-hook Electrocautery", "Gallbladder", "Hepatic Vein", "Liver Ligament"
    ]
    
    results = []
    for sample_file in sample_files:
        sample_id = sample_file.replace('train_', '').replace('.json', '')
        sample_results = {"id": sample_id, "organs": []}
        
        # Load data for each model
        model_data = {}
        for model in models:
            model_path = Path(results_dir) / eval_type / model / "cholecseg8k_pointing" / sample_file
            if model_path.exists():
                try:
                    with open(model_path, 'r') as f:
                        data = json.load(f)
                        model_data[model] = data
                except:
                    pass
        
        if len(model_data) < 2:
            continue
        
        # Process each organ
        first_model_data = list(model_data.values())[0]
        y_true = first_model_data.get('y_true', [])
        
        for i in range(min(len(y_true), len(organ_names))):
            if y_true[i] == 1:  # Organ is present
                organ_result = {"name": organ_names[i], "results": {}}
                
                for model, data in model_data.items():
                    y_pred = data.get('y_pred', [])
                    hits = data.get('hits', [])
                    
                    if i < len(y_pred):
                        if y_pred[i] == 1:
                            if i < len(hits) and hits[i]:
                                organ_result["results"][model_names[model]] = "✓"
                            else:
                                organ_result["results"][model_names[model]] = "✗"
                        else:
                            organ_result["results"][model_names[model]] = "—"
                    else:
                        organ_result["results"][model_names[model]] = "?"
                
                sample_results["organs"].append(organ_result)
        
        if sample_results["organs"]:
            results.append(sample_results)
    
    return results

def generate_latex_figure(results, eval_type, output_file):
    """Generate LaTeX code for a figure showing results."""
    
    latex = []
    latex.append("% Generated figure for " + eval_type)
    latex.append("\\begin{figure}[h]")
    latex.append("\\centering")
    
    # Create a tabular environment for the results
    latex.append("\\begin{tabular}{|l|l|c|c|c|}")
    latex.append("\\hline")
    latex.append("\\textbf{Sample} & \\textbf{Organ} & \\textbf{GPT-4.1} & \\textbf{Gemini-2.0} & \\textbf{Claude-4} \\\\")
    latex.append("\\hline")
    
    for sample in results[:4]:  # Limit to 4 samples for space
        first = True
        for organ in sample["organs"][:3]:  # Limit to 3 organs per sample
            if first:
                latex.append(f"{sample['id']} & {organ['name'][:20]} & "
                           f"{organ['results'].get('GPT-4.1', '?')} & "
                           f"{organ['results'].get('Gemini-2.0', '?')} & "
                           f"{organ['results'].get('Claude-4', '?')} \\\\")
                first = False
            else:
                latex.append(f" & {organ['name'][:20]} & "
                           f"{organ['results'].get('GPT-4.1', '?')} & "
                           f"{organ['results'].get('Gemini-2.0', '?')} & "
                           f"{organ['results'].get('Claude-4', '?')} \\\\")
        latex.append("\\hline")
    
    latex.append("\\end{tabular}")
    
    eval_name = eval_type.replace('_', ' ').replace('fewshot', 'few-shot').title()
    latex.append(f"\\caption{{Organ localization results from {eval_name} evaluation. "
                f"✓ = correct localization, ✗ = incorrect, — = not detected.}}")
    latex.append(f"\\label{{fig:examples_{eval_type}}}")
    latex.append("\\end{figure}")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(latex))
    
    return len(results)

def main():
    results_dir = Path("results/pointing_original")
    output_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ_paper/images/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eval_types = [
        ("zero_shot", "zero-shot"),
        ("fewshot_standard", "few-shot-standard"),
        ("fewshot_hard_negatives", "few-shot-hard-negatives")
    ]
    
    all_figures = []
    
    for eval_type, eval_name in eval_types:
        print(f"\nProcessing {eval_type}...")
        
        # Load results
        results = load_results_for_samples(results_dir, eval_type, num_samples=8)
        
        if results:
            # Generate LaTeX figure
            output_file = output_dir / f"figure_{eval_type}.tex"
            n = generate_latex_figure(results, eval_type, output_file)
            print(f"  Generated LaTeX figure with {n} samples: {output_file.name}")
            all_figures.append(f"\\input{{images/examples/figure_{eval_type}.tex}}")
        else:
            print(f"  No results found for {eval_type}")
    
    # Create a master file that includes all figures
    master_file = output_dir / "all_example_figures.tex"
    with open(master_file, 'w') as f:
        f.write("% Include all example figures\n")
        f.write("% Add this to your appendix\n\n")
        for fig in all_figures:
            f.write(fig + "\n\n")
    
    print(f"\n✓ Done! Generated LaTeX figures in: {output_dir}")
    print(f"\nTo include all figures, add to your appendix:")
    print(f"\\input{{images/examples/all_example_figures.tex}}")

if __name__ == "__main__":
    main()