#!/usr/bin/env python
"""Generate LaTeX tables and PDF figures for paper from metrics results.

Usage:
    python generate_paper_figures.py /path/to/results/folder /path/to/output/base
    
    Example:
    python generate_paper_figures.py results/pointing_original /shared_data0/weiqiuy/llm_cholec_organ_paper
    
This will create:
    - /shared_data0/weiqiuy/llm_cholec_organ_paper/figures/*.tex (LaTeX files)
    - /shared_data0/weiqiuy/llm_cholec_organ_paper/images/pointing/*.pdf (Figure files)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from endopoint.vis.latex_tables import (
    generate_metrics_latex_table,
    generate_comparison_latex_table
)
from endopoint.vis.bar_charts import (
    generate_model_performance_bars,
    generate_metric_comparison_bars,
    generate_all_models_comparison
)


def load_metrics_data(results_dir: Path) -> Dict[str, Any]:
    """Load metrics data from results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary with metrics data organized by evaluation type and model
    """
    metrics_data = {}
    
    # Always load from individual metrics_summary files for completeness
    eval_types = ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]
    
    for eval_type in eval_types:
        eval_dir = results_dir / eval_type
        if not eval_dir.exists():
            continue
        
        metrics_data[eval_type] = {}
        
        # Find all metrics_summary files
        for summary_file in eval_dir.rglob("metrics_summary_*.json"):
            # Get model name from path
            model_parts = summary_file.relative_to(eval_dir).parts[:-2]
            model_name = "/".join(model_parts)
            
            with open(summary_file, 'r') as f:
                data = json.load(f)
                metrics_data[eval_type][model_name] = data
    
    return metrics_data


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables and PDF figures for paper")
    parser.add_argument("results_dir", help="Path to results directory (e.g., results/pointing_original)")
    parser.add_argument("output_base", help="Base path for output (e.g., /path/to/paper)")
    parser.add_argument("--experiment-name", default="pointing", help="Name for the experiment subdirectory")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_base = Path(args.output_base)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Setup output directories
    figures_dir = output_base / "figures"
    images_dir = output_base / "images" / args.experiment_name
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Loading metrics from: {results_dir}")
    print(f"📝 Output directories:")
    print(f"   - LaTeX: {figures_dir}")
    print(f"   - Images: {images_dir}")
    print("")
    
    # Load metrics data
    print("📊 Loading metrics data...")
    metrics_data = load_metrics_data(results_dir)
    
    if not metrics_data:
        print("❌ No metrics data found")
        sys.exit(1)
    
    # Print what we found
    for eval_type, models in metrics_data.items():
        print(f"  {eval_type}: {len(models)} models")
    
    # Generate LaTeX tables
    print("\n📄 Generating LaTeX tables...")
    
    # Main comprehensive table
    latex_table = generate_metrics_latex_table(
        metrics_data,
        caption="Comprehensive evaluation metrics for organ pointing task. Models are evaluated using zero-shot, few-shot, and few-shot with hard negatives approaches. Best values are \\textbf{bold}, second best are \\textit{italicized} for each metric.",
        label=f"tab:{args.experiment_name}_metrics"
    )
    
    table_file = figures_dir / f"{args.experiment_name}_metrics_table.tex"
    with open(table_file, 'w') as f:
        f.write(latex_table)
    print(f"  ✓ Created: {table_file}")
    
    # Generate comparison tables for each evaluation type
    for eval_type in ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]:
        if eval_type in metrics_data:
            eval_name = eval_type.replace("_", " ").replace("fewshot", "few-shot")
            latex_table = generate_comparison_latex_table(
                metrics_data,
                eval_type=eval_type,
                caption=f"Model performance comparison using {eval_name} evaluation. Best values are \\textbf{{bold}}, second best are \\textit{{italicized}}.",
                label=f"tab:{args.experiment_name}_{eval_type}"
            )
            
            table_file = figures_dir / f"{args.experiment_name}_{eval_type}_table.tex"
            with open(table_file, 'w') as f:
                f.write(latex_table)
            print(f"  ✓ Created: {table_file}")
    
    # Generate bar charts
    print("\n📊 Generating bar charts...")
    
    # Individual model performance charts
    all_models = set()
    for eval_data in metrics_data.values():
        all_models.update(eval_data.keys())
    
    for model_name in sorted(all_models):
        # Create safe filename
        safe_name = model_name.replace("/", "_").replace(" ", "_").replace(".", "_")
        output_path = images_dir / f"{safe_name}_performance.pdf"
        
        try:
            generate_model_performance_bars(
                model_name=model_name,
                metrics_data=metrics_data,
                output_path=output_path
            )
            print(f"  ✓ Created: {output_path.name}")
        except Exception as e:
            print(f"  ⚠️ Failed to create chart for {model_name}: {e}")
    
    # Comparison charts for each metric
    metrics_to_compare = [
        ("Presence Accuracy", "macro_presence_acc"),
        ("Hit at Point Given Present", "macro_hit_rate"),
        ("Gated Accuracy", "macro_gated_acc"),
        ("F1 Score", "macro_f1")
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        for eval_type in ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]:
            if eval_type in metrics_data:
                safe_metric = metric_name.replace(" ", "_").replace("@", "at").lower()
                safe_eval = eval_type.replace("_", "-")
                output_path = images_dir / f"comparison_{safe_metric}_{safe_eval}.pdf"
                
                try:
                    generate_metric_comparison_bars(
                        metric_name=metric_name,
                        metric_key=metric_key,
                        metrics_data=metrics_data,
                        output_path=output_path,
                        eval_type=eval_type
                    )
                    print(f"  ✓ Created: {output_path.name}")
                except Exception as e:
                    print(f"  ⚠️ Failed to create comparison for {metric_name} ({eval_type}): {e}")
    
    # Overall comparison chart
    output_path = images_dir / "overall_comparison.pdf"
    try:
        generate_all_models_comparison(
            metrics_data=metrics_data,
            output_path=output_path
        )
        print(f"  ✓ Created: {output_path.name}")
    except Exception as e:
        print(f"  ⚠️ Failed to create overall comparison: {e}")
    
    # Generate main LaTeX document that includes everything
    print("\n📄 Generating main LaTeX document...")
    
    main_tex = generate_main_latex_document(
        figures_dir=figures_dir,
        images_dir=images_dir,
        experiment_name=args.experiment_name,
        metrics_data=metrics_data
    )
    
    main_file = figures_dir / f"{args.experiment_name}_main.tex"
    with open(main_file, 'w') as f:
        f.write(main_tex)
    print(f"  ✓ Created: {main_file}")
    
    print("\n✨ Generation complete!")
    print(f"\nTo use in your paper:")
    print(f"1. Include tables: \\input{{{figures_dir.relative_to(output_base)}/{args.experiment_name}_metrics_table.tex}}")
    print(f"2. Include figures: \\includegraphics{{{images_dir.relative_to(output_base)}/model_performance.pdf}}")
    print(f"3. Or compile the main document: {main_file}")


def generate_main_latex_document(
    figures_dir: Path,
    images_dir: Path,
    experiment_name: str,
    metrics_data: Dict[str, Any]
) -> str:
    """Generate main LaTeX document that references all tables and figures."""
    
    lines = []
    
    # Document preamble
    lines.extend([
        "\\documentclass[11pt]{article}",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\usepackage{amsmath}",
        "\\usepackage{array}",
        "\\usepackage{multirow}",
        "\\usepackage{caption}",
        "\\usepackage{subcaption}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{hyperref}",
        "",
        "\\title{Organ Pointing Task: Evaluation Results}",
        "\\author{Generated Report}",
        "\\date{\\today}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\section{Overview}",
        "This document presents comprehensive evaluation results for the organ pointing task in laparoscopic cholecystectomy procedures.",
        "",
        "\\section{Metrics Description}",
        "\\begin{itemize}",
        "\\item \\textbf{Presence Accuracy}: Binary classification accuracy for detecting organ presence/absence",
        "\\item \\textbf{Hit@Point|Present}: Pointing accuracy when organ is correctly detected as present",
        "\\item \\textbf{Gated Accuracy}: Combined accuracy requiring both correct detection and pointing",
        "\\item \\textbf{F1 Score}: Harmonic mean of precision and recall for presence detection",
        "\\end{itemize}",
        "",
        "\\section{Results Tables}",
        ""
    ])
    
    # Include main table
    table_path = figures_dir / f"{experiment_name}_metrics_table.tex"
    if table_path.exists():
        lines.append(f"\\input{{{table_path.stem}}}")
        lines.append("")
    
    # Include comparison tables
    lines.append("\\clearpage")
    lines.append("\\section{Detailed Comparisons}")
    lines.append("")
    
    for eval_type in ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]:
        table_path = figures_dir / f"{experiment_name}_{eval_type}_table.tex"
        if table_path.exists():
            lines.append(f"\\input{{{table_path.stem}}}")
            lines.append("")
    
    # Include figures
    lines.append("\\clearpage")
    lines.append("\\section{Performance Visualizations}")
    lines.append("")
    
    # Overall comparison
    lines.append("\\begin{figure}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\includegraphics[width=\\textwidth]{{{images_dir}/overall_comparison.pdf}}")
    lines.append("\\caption{Overall performance comparison across all models and evaluation methods.}")
    lines.append("\\label{fig:overall_comparison}")
    lines.append("\\end{figure}")
    lines.append("")
    
    # Individual model charts
    lines.append("\\clearpage")
    lines.append("\\section{Individual Model Performance}")
    lines.append("")
    
    all_models = set()
    for eval_data in metrics_data.values():
        all_models.update(eval_data.keys())
    
    for i, model_name in enumerate(sorted(all_models), 1):
        safe_name = model_name.replace("/", "_").replace(" ", "_").replace(".", "_")
        pdf_path = images_dir / f"{safe_name}_performance.pdf"
        
        if pdf_path.exists():
            # Format model name for display
            model_display = model_name.replace("_", "\\_")
            
            lines.append("\\begin{figure}[htbp]")
            lines.append("\\centering")
            lines.append(f"\\includegraphics[width=0.8\\textwidth]{{{pdf_path}}}")
            lines.append(f"\\caption{{Performance metrics for {model_display} across different evaluation methods.}}")
            lines.append(f"\\label{{fig:model_{i}}}")
            lines.append("\\end{figure}")
            lines.append("")
            
            if i % 2 == 0:
                lines.append("\\clearpage")
    
    lines.append("\\end{document}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()