"""LaTeX table generation for metrics visualization."""

from pathlib import Path
from typing import Dict, Any, List
import json


def format_model_name_latex(model_name: str) -> str:
    """Format model name for LaTeX display."""
    # Escape underscores and special characters
    model_name = model_name.replace("_", "\\_")
    model_name = model_name.replace("/", "/\\-")  # Allow line breaks at slashes
    
    # Shorten long model names
    replacements = {
        "llava-hf/\\-llava-v1.6-mistral-7b-hf": "LLaVA-v1.6",
        "mistralai/\\-Pixtral-12B-2409": "Pixtral-12B",
        "Qwen/\\-Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL",
        "claude-sonnet-4-20250514": "Claude-Sonnet-4",
        "gemini-2.0-flash": "Gemini-2.0",
        "gpt-4.1": "GPT-4.1",
        "deepseek-ai/\\-deepseek-vl2": "DeepSeek-VL2"
    }
    
    for old, new in replacements.items():
        if old in model_name:
            return new
    
    return model_name


def generate_metrics_latex_table(
    metrics_data: Dict[str, Any],
    caption: str = "Comprehensive evaluation metrics for organ pointing task. Best values are \\textbf{bold}, second best are \\textit{italicized} for each metric.",
    label: str = "tab:pointing_metrics"
) -> str:
    """Generate LaTeX table from metrics data.
    
    Args:
        metrics_data: Dictionary with model results
        caption: Table caption
        label: LaTeX label for referencing
        
    Returns:
        LaTeX table as string
    """
    
    lines = []
    
    # Start table - use table* for two-column papers
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll@{\\hspace{1em}}rr@{\\hspace{1em}}rr}")
    lines.append("\\toprule")
    
    # Header with column groups
    lines.append("& & \\multicolumn{2}{c}{\\textbf{Presence}} & \\multicolumn{2}{c}{\\textbf{Localization}} \\\\")
    lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
    lines.append("\\textbf{Model} & \\textbf{Method} & \\textbf{Acc.} & \\textbf{F1} & \\textbf{Hit@Pt|Pres} & \\textbf{Gated Acc.} \\\\")
    lines.append("& & (\\%) & (\\%) & (\\%) & (\\%) \\\\")
    lines.append("\\midrule")
    
    # Collect and sort models using the same ordering as figures
    all_models = set()
    for eval_type_data in metrics_data.values():
        all_models.update(eval_type_data.keys())
    
    # Order models: APIs first, then open-source
    preferred_order = [
        "gpt-4.1",
        "gemini-2.0-flash", 
        "claude-sonnet-4-20250514",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "mistralai/Pixtral-12B-2409",
        "deepseek-ai/deepseek-vl2"
    ]
    
    ordered_models = []
    # Add models in preferred order if they exist
    for pref_model in preferred_order:
        for model in all_models:
            if pref_model in model or model in pref_model:
                if model not in ordered_models:
                    ordered_models.append(model)
                break
    
    # Add any remaining models not in preferred order
    for model in all_models:
        if model not in ordered_models:
            ordered_models.append(model)
    
    # First, collect all metrics to find best and second best
    all_metrics = {
        'presence_acc': {},
        'f1': {},
        'hit_rate': {},
        'gated_acc': {}
    }
    
    for eval_type in ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]:
        if eval_type not in metrics_data:
            continue
        for model in ordered_models:
            if model not in metrics_data[eval_type]:
                continue
            macro = metrics_data[eval_type][model].get('macro_metrics', {})
            key = f"{model}_{eval_type}"
            all_metrics['presence_acc'][key] = macro.get('macro_presence_acc', 0) * 100
            all_metrics['f1'][key] = macro.get('macro_f1', 0) * 100
            all_metrics['hit_rate'][key] = macro.get('macro_hit_rate', 0) * 100
            all_metrics['gated_acc'][key] = macro.get('macro_gated_acc', 0) * 100
    
    # Find best and second best for each metric
    best_second = {}
    for metric_name, values in all_metrics.items():
        sorted_vals = sorted(values.values(), reverse=True)
        if len(sorted_vals) >= 1:
            best_second[metric_name] = {
                'best': sorted_vals[0] if sorted_vals else 0,
                'second': sorted_vals[1] if len(sorted_vals) > 1 else 0
            }
    
    # Process each model
    current_model = None
    for model in ordered_models:
        model_display = format_model_name_latex(model)
        
        # Add model rows for each evaluation type
        for eval_idx, eval_type in enumerate(["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]):
            if eval_type not in metrics_data or model not in metrics_data[eval_type]:
                continue
            
            metrics = metrics_data[eval_type][model]
            macro = metrics.get('macro_metrics', {})
            
            # Format method name
            method_map = {
                "zero_shot": "Zero-shot",
                "fewshot_standard": "Few-shot",
                "fewshot_hard_negatives": "Few-shot (hard neg.)"
            }
            method_display = method_map.get(eval_type, eval_type)
            
            # Get raw values
            pres_acc_val = macro.get('macro_presence_acc', 0) * 100
            f1_val = macro.get('macro_f1', 0) * 100
            hit_rate_val = macro.get('macro_hit_rate', 0) * 100
            gated_acc_val = macro.get('macro_gated_acc', 0) * 100
            
            # Format with bold/italic for best/second best
            def format_value(val, metric_name):
                formatted = f"{val:.1f}"
                if abs(val - best_second[metric_name]['best']) < 0.01:
                    return f"\\textbf{{{formatted}}}"
                elif abs(val - best_second[metric_name]['second']) < 0.01:
                    return f"\\textit{{{formatted}}}"
                return formatted
            
            pres_acc = format_value(pres_acc_val, 'presence_acc')
            f1_score = format_value(f1_val, 'f1')
            hit_rate = format_value(hit_rate_val, 'hit_rate')
            gated_acc = format_value(gated_acc_val, 'gated_acc')
            
            # Only show model name on first row (reordered columns)
            if eval_idx == 0:
                lines.append(f"{model_display} & {method_display} & {pres_acc} & {f1_score} & {hit_rate} & {gated_acc} \\\\")
            else:
                lines.append(f" & {method_display} & {pres_acc} & {f1_score} & {hit_rate} & {gated_acc} \\\\")
        
        # Add small separator between models
        if model != ordered_models[-1]:
            lines.append("\\addlinespace[0.5em]")
    
    # End table
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    
    # Add footnote-style explanation without tablenotes package
    lines.append("")
    lines.append("\\vspace{0.5em}")
    lines.append("\\parbox{\\textwidth}{%")
    lines.append("\\footnotesize")
    lines.append("\\textbf{Metrics:}")
    lines.append("\\textbf{Presence} - \\textit{Acc.}: Binary classification accuracy for organ presence/absence detection; \\textit{F1}: Harmonic mean of precision and recall.")
    lines.append("\\textbf{Localization} - \\textit{Hit@Pt|Pres}: Pointing accuracy when organ is correctly detected as present; \\textit{Gated Acc.}: Combined accuracy requiring both correct detection and pointing.")
    lines.append("}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_comparison_latex_table(
    metrics_data: Dict[str, Any],
    eval_type: str = "zero_shot",
    caption: str = None,
    label: str = None
) -> str:
    """Generate LaTeX comparison table for a specific evaluation type.
    
    Args:
        metrics_data: Dictionary with model results
        eval_type: Type of evaluation to show
        caption: Table caption (auto-generated if None)
        label: LaTeX label (auto-generated if None)
        
    Returns:
        LaTeX table as string
    """
    
    if caption is None:
        eval_name = eval_type.replace("_", " ").replace("fewshot", "few-shot")
        caption = f"Performance comparison of models using {eval_name} evaluation. Best values are \\textbf{{bold}}, second best are \\textit{{italicized}}."
    
    if label is None:
        label = f"tab:comparison_{eval_type}"
    
    lines = []
    
    # Start table - use table* for two-column papers
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\begin{tabular}{l@{\\hspace{1em}}rr@{\\hspace{1em}}rr}")
    lines.append("\\toprule")
    
    # Header with column groups
    lines.append("& \\multicolumn{2}{c}{\\textbf{Presence}} & \\multicolumn{2}{c}{\\textbf{Localization}} \\\\")
    lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    lines.append("\\textbf{Model} & \\textbf{Acc.} & \\textbf{F1} & \\textbf{Hit@Pt|Pres} & \\textbf{Gated Acc.} \\\\")
    lines.append("& (\\%) & (\\%) & (\\%) & (\\%) \\\\")
    lines.append("\\midrule")
    
    if eval_type in metrics_data:
        # Order models: APIs first, then open-source (same as main table)
        preferred_order = [
            "gpt-4.1",
            "gemini-2.0-flash", 
            "claude-sonnet-4-20250514",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "mistralai/Pixtral-12B-2409",
            "deepseek-ai/deepseek-vl2"
        ]
        
        # Get all models for this eval type
        all_models = list(metrics_data[eval_type].keys())
        
        # Order according to preference
        ordered_models = []
        for pref_model in preferred_order:
            for model in all_models:
                if pref_model in model or model in pref_model:
                    if model not in ordered_models:
                        ordered_models.append(model)
                    break
        
        # Add any remaining models
        for model in all_models:
            if model not in ordered_models:
                ordered_models.append(model)
        
        # Create sorted list with metrics
        models_sorted = [(model, metrics_data[eval_type][model]) for model in ordered_models]
        
        # Collect all values to find best and second best
        all_values = {
            'presence_acc': [],
            'f1': [],
            'hit_rate': [],
            'gated_acc': []
        }
        
        for model, metrics in models_sorted:
            macro = metrics.get('macro_metrics', {})
            all_values['presence_acc'].append(macro.get('macro_presence_acc', 0) * 100)
            all_values['f1'].append(macro.get('macro_f1', 0) * 100)
            all_values['hit_rate'].append(macro.get('macro_hit_rate', 0) * 100)
            all_values['gated_acc'].append(macro.get('macro_gated_acc', 0) * 100)
        
        # Find best and second best for each metric
        best_second = {}
        for metric_name, values in all_values.items():
            sorted_vals = sorted(values, reverse=True)
            best_second[metric_name] = {
                'best': sorted_vals[0] if sorted_vals else 0,
                'second': sorted_vals[1] if len(sorted_vals) > 1 else 0
            }
        
        # Format and output rows
        for i, (model, metrics) in enumerate(models_sorted):
            model_display = format_model_name_latex(model)
            macro = metrics.get('macro_metrics', {})
            
            # Get raw values
            pres_acc_val = macro.get('macro_presence_acc', 0) * 100
            f1_val = macro.get('macro_f1', 0) * 100
            hit_rate_val = macro.get('macro_hit_rate', 0) * 100
            gated_acc_val = macro.get('macro_gated_acc', 0) * 100
            
            # Format with bold/italic for best/second best
            def format_value(val, metric_name):
                formatted = f"{val:.1f}"
                if abs(val - best_second[metric_name]['best']) < 0.01:
                    return f"\\textbf{{{formatted}}}"
                elif abs(val - best_second[metric_name]['second']) < 0.01:
                    return f"\\textit{{{formatted}}}"
                return formatted
            
            pres_acc = format_value(pres_acc_val, 'presence_acc')
            f1_score = format_value(f1_val, 'f1')
            hit_rate = format_value(hit_rate_val, 'hit_rate')
            gated_acc = format_value(gated_acc_val, 'gated_acc')
            
            # Reordered columns: Acc, F1, Hit@Pt, Gated
            lines.append(f"{model_display} & {pres_acc} & {f1_score} & {hit_rate} & {gated_acc} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)