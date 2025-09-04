#!/usr/bin/env python3
"""
Generate PDF examples using matplotlib without complex dependencies.
Creates visual examples showing model predictions.

Usage:
    python notebooks_py/generate_examples_pdf.py
"""

import json
import os
from pathlib import Path
import random

# Try to import matplotlib - if not available, create a placeholder
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will create placeholder PDFs")

def create_placeholder_pdf(output_path):
    """Create a placeholder PDF when matplotlib is not available."""
    # Create a simple text file as placeholder
    with open(str(output_path).replace('.pdf', '.txt'), 'w') as f:
        f.write(f"Placeholder for {output_path.name}\n")
        f.write("Install matplotlib to generate actual figures:\n")
        f.write("pip install matplotlib\n")
    print(f"Created placeholder: {output_path.name}")

def load_sample_data(results_dir, eval_type, sample_files, num_samples=8):
    """Load sample data from result files."""
    
    models = ["gpt-4.1", "gemini-2.0-flash", "claude-sonnet-4-20250514"]
    samples = []
    
    for sample_file in sample_files[:num_samples]:
        sample_data = {'file': sample_file, 'models': {}}
        
        for model in models:
            model_path = Path(results_dir) / eval_type / model / "cholecseg8k_pointing" / sample_file
            if model_path.exists():
                try:
                    with open(model_path, 'r') as f:
                        data = json.load(f)
                        sample_data['models'][model] = data
                except:
                    pass
        
        if len(sample_data['models']) >= 2:  # At least 2 models have data
            samples.append(sample_data)
    
    return samples

def create_grid_figure_simple(samples, output_path, eval_type):
    """Create a simplified grid figure showing results."""
    
    if not HAS_MATPLOTLIB:
        create_placeholder_pdf(output_path)
        return
    
    num_samples = min(len(samples), 8)
    if num_samples == 0:
        print(f"No samples to plot for {eval_type}")
        return
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 2.5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    model_names = {
        "gpt-4.1": "GPT-4.1",
        "gemini-2.0-flash": "Gemini-2.0",
        "claude-sonnet-4-20250514": "Claude-4"
    }
    
    # Organ names mapping (simplified)
    organ_names = [
        "Abdominal Wall", "Liver", "Gastrointestinal Tract", "Fat",
        "Grasper", "Connective Tissue", "Blood", "Cystic Duct",
        "L-hook Electrocautery", "Gallbladder", "Hepatic Vein", "Liver Ligament"
    ]
    
    for row, sample in enumerate(samples[:num_samples]):
        sample_idx = sample['file'].replace('train_', '').replace('.json', '')
        
        # Column 0: Sample info
        ax = axes[row, 0]
        ax.axis('off')
        ax.text(0.5, 0.5, f"Sample\n{sample_idx}", 
                ha='center', va='center', fontsize=10, weight='bold')
        
        # Columns 1-3: Model results
        for col, (model_key, model_name) in enumerate(model_names.items(), 1):
            ax = axes[row, col]
            ax.axis('off')
            
            if model_key in sample['models']:
                data = sample['models'][model_key]
                
                # Create a simple visualization of results
                y_true = data.get('y_true', [])
                y_pred = data.get('y_pred', [])
                hits = data.get('hits', [])
                
                # Create a simple grid showing organs
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 12)
                
                for i in range(min(len(y_true), 12)):
                    y_pos = 11.5 - i
                    
                    # Draw organ name
                    organ_name = organ_names[i] if i < len(organ_names) else f"Organ {i}"
                    ax.text(0.05, y_pos, organ_name[:15], fontsize=6, va='center')
                    
                    # Draw result indicator
                    if i < len(y_true) and y_true[i] == 1:  # Organ is present
                        if i < len(y_pred) and y_pred[i] == 1:  # Model predicted present
                            if i < len(hits) and hits[i]:  # Correct localization
                                ax.plot(0.8, y_pos, 'go', markersize=8)  # Green circle
                                ax.text(0.85, y_pos, '✓', fontsize=8, color='green', va='center')
                            else:  # Wrong localization
                                ax.plot(0.8, y_pos, 'rx', markersize=8)  # Red X
                                ax.text(0.85, y_pos, '✗', fontsize=8, color='red', va='center')
                        else:  # Not detected
                            ax.text(0.8, y_pos, '—', fontsize=8, color='gray', va='center')
            
            # Add title for first row
            if row == 0:
                ax.set_title(model_names.get(model_key, model_key), fontsize=9, weight='bold')
    
    # Overall title
    eval_name = eval_type.replace('_', ' ').replace('fewshot', 'few-shot')
    plt.suptitle(f'Organ Detection and Localization Examples ({eval_name})', 
                 fontsize=12, weight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
                   markersize=8, label='Correct'),
        plt.Line2D([0], [0], marker='x', color='r', markersize=8, 
                   label='Incorrect', markeredgewidth=2),
        plt.Line2D([0], [0], marker='_', color='gray', markersize=8, 
                   label='Not detected', markeredgewidth=2)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")

def main():
    results_dir = Path("results/pointing_original")
    output_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ_paper/images/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get common sample files across all models
    eval_types = ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]
    
    for eval_type in eval_types:
        print(f"\nProcessing {eval_type}...")
        
        # Find common samples
        common_samples = None
        for model in ["gpt-4.1", "gemini-2.0-flash", "claude-sonnet-4-20250514"]:
            model_path = results_dir / eval_type / model / "cholecseg8k_pointing"
            if model_path.exists():
                files = set([f.name for f in model_path.glob("train_*.json") 
                            if "metrics" not in f.name])
                if common_samples is None:
                    common_samples = files
                else:
                    common_samples = common_samples.intersection(files)
        
        if not common_samples:
            print(f"  No common samples found for {eval_type}")
            continue
        
        print(f"  Found {len(common_samples)} common samples")
        
        # Load sample data
        sample_files = sorted(list(common_samples))[:10]
        samples = load_sample_data(results_dir, eval_type, sample_files)
        
        if samples:
            # Create grid figure
            output_file = output_dir / f"examples_grid_{eval_type}.pdf"
            create_grid_figure_simple(samples, output_file, eval_type)
        else:
            print(f"  No valid samples to visualize for {eval_type}")
    
    print("\n✓ Done! Generated example figures in:")
    print(f"  {output_dir}")

if __name__ == "__main__":
    main()