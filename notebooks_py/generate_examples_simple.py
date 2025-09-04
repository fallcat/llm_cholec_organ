#!/usr/bin/env python3
"""
Simplified script to generate example visualizations for paper.
Creates a LaTeX figure showing prediction results.

Usage:
    python notebooks_py/generate_examples_simple.py
"""

import json
import os
from pathlib import Path
import random

def create_latex_figure(results_dir, output_file, num_examples=6):
    """Create LaTeX code for showing examples."""
    
    # Model display names
    model_names = {
        "gpt-4.1": "GPT-4.1",
        "gemini-2.0-flash": "Gemini-2.0", 
        "claude-sonnet-4-20250514": "Claude-4"
    }
    
    # First get a list of common samples across all models
    common_samples = None
    for model_dir in ["gpt-4.1", "gemini-2.0-flash", "claude-sonnet-4-20250514"]:
        model_path = Path(results_dir) / "zero_shot" / model_dir / "cholecseg8k_pointing"
        if model_path.exists():
            train_files = set([f.name for f in model_path.glob("train_*.json") if "metrics" not in f.name])
            if common_samples is None:
                common_samples = train_files
            else:
                common_samples = common_samples.intersection(train_files)
    
    print(f"Debug: Found {len(common_samples) if common_samples else 0} common sample files")
    
    # Now load these common samples for each model
    samples = []
    common_samples = list(common_samples)[:10] if common_samples else []  # Take first 10
    
    for model_dir in ["gpt-4.1", "gemini-2.0-flash", "claude-sonnet-4-20250514"]:
        model_path = Path(results_dir) / "zero_shot" / model_dir / "cholecseg8k_pointing"
        
        print(f"Debug: Loading {len(common_samples)} files for {model_dir}")
        if model_path.exists():
            for train_filename in common_samples:
                train_file = model_path / train_filename
                if "metrics" not in str(train_file):
                    try:
                        with open(train_file, 'r') as f:
                            data = json.load(f)
                            
                            # The data has y_true and y_pred as lists
                            if isinstance(data, dict) and 'sample_idx' in data:
                                sample_idx = data['sample_idx']
                                
                                # Check if we have the detailed structure with rows
                                if 'rows' in data and isinstance(data['rows'], list):
                                    # Also need y_true, y_pred, and hits lists
                                    y_true = data.get('y_true', [])
                                    y_pred = data.get('y_pred', [])
                                    hits = data.get('hits', [])
                                    
                                    # Each row represents an organ
                                    for i, row in enumerate(data['rows']):
                                        if isinstance(row, dict) and 'organ' in row:
                                            sample_info = {
                                                'sample_idx': sample_idx,
                                                'organ': row['organ'],
                                                'model': model_dir,
                                                'present_true': y_true[i] if i < len(y_true) else 0,
                                                'present_pred': y_pred[i] if i < len(y_pred) else 0,
                                                'hit_point': hits[i] if i < len(hits) else False
                                            }
                                            samples.append(sample_info)
                                else:
                                    print(f"Debug: No rows found in {train_file}, keys: {list(data.keys())[:5]}")
                    except Exception as e:
                        print(f"Error reading {train_file}: {e}")
                        continue
    
    # Group by sample_idx and organ
    examples = {}
    for sample in samples:
        key = (sample['sample_idx'], sample['organ'])
        if key not in examples:
            examples[key] = {'sample_idx': sample['sample_idx'], 
                           'organ': sample['organ'],
                           'results': {}}
        examples[key]['results'][sample['model']] = {
            'present_pred': sample['present_pred'],
            'hit_point': sample['hit_point'],
            'present_true': sample['present_true']
        }
    
    print(f"Debug: Found {len(samples)} total samples")
    print(f"Debug: Found {len(examples)} unique (sample, organ) pairs")
    
    # Select interesting examples
    selected = []
    for key, ex in examples.items():
        # Check if organ is present in ground truth
        first_model_result = list(ex['results'].values())[0]
        if first_model_result['present_true'] == 1:
            # Only keep if we have results from at least 2 models
            if len(ex['results']) >= 2:
                selected.append(ex)
            
    print(f"Debug: Selected {len(selected)} examples with organ present and multiple models")
    
    # Take first N examples
    selected = selected[:num_examples]
    
    # Generate LaTeX table
    latex = []
    latex.append("% Example results table")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Sample organ detection and localization results from zero-shot evaluation.}")
    latex.append("\\label{tab:examples}")
    latex.append("\\begin{tabular}{llccc}")
    latex.append("\\toprule")
    latex.append("Sample & Organ & GPT-4.1 & Gemini-2.0 & Claude-4 \\\\")
    latex.append("\\midrule")
    
    for ex in selected:
        organ = ex['organ'].replace("_", " ")
        sample = f"{ex['sample_idx']:04d}"
        
        results = []
        for model in ["gpt-4.1", "gemini-2.0-flash", "claude-sonnet-4-20250514"]:
            if model in ex['results']:
                res = ex['results'][model]
                if res['present_pred'] == 1:
                    if res['hit_point']:
                        results.append("✓")  # Correct localization
                    else:
                        results.append("✗")  # Wrong localization
                else:
                    results.append("—")  # Not detected
            else:
                results.append("?")
        
        latex.append(f"{sample} & {organ} & {results[0]} & {results[1]} & {results[2]} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    latex.append("% Legend: ✓ = correct localization, ✗ = incorrect localization, — = not detected")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write("\n".join(latex))
    
    return len(selected)

def main():
    results_dir = "results/pointing_original"
    output_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ_paper/images/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "examples_table.tex"
    
    print("Generating LaTeX table of examples...")
    n = create_latex_figure(results_dir, output_file, num_examples=10)
    print(f"✓ Created table with {n} examples: {output_file}")
    
    print("\nTo include in paper:")
    print(f"\\input{{images/examples/examples_table.tex}}")

if __name__ == "__main__":
    main()