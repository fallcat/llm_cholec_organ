"""Bar chart generation for model performance visualization."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set matplotlib to use a non-interactive backend
mpl.use('Agg')

# Configure matplotlib for publication-quality figures
# Use DejaVu Sans which is available on most systems
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 14  # Increased from 11
plt.rcParams['axes.labelsize'] = 16  # Increased from 12
plt.rcParams['axes.titlesize'] = 18  # Increased from 12
plt.rcParams['xtick.labelsize'] = 14  # Increased from 10
plt.rcParams['ytick.labelsize'] = 14  # Increased from 10
plt.rcParams['legend.fontsize'] = 14  # Increased from 10
plt.rcParams['figure.titlesize'] = 20  # Increased from 14


def get_model_order():
    """Get the preferred order for models (APIs first, then open-source)."""
    return [
        "gpt-4.1",
        "gemini-2.0-flash", 
        "claude-sonnet-4-20250514",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "mistralai/Pixtral-12B-2409"
    ]


def order_models(models):
    """Order models according to preferred sequence."""
    preferred_order = get_model_order()
    ordered = []
    
    # Add models in preferred order if they exist
    for pref_model in preferred_order:
        for model in models:
            if pref_model in model or model in pref_model:
                if model not in ordered:
                    ordered.append(model)
                break
    
    # Add any remaining models not in preferred order
    for model in models:
        if model not in ordered:
            ordered.append(model)
    
    return ordered


def generate_model_performance_bars(
    model_name: str,
    metrics_data: Dict[str, Any],
    output_path: Path,
    figsize: tuple = (10, 6)
) -> None:
    """Generate grouped bar chart for a single model's performance across evaluation types.
    
    Args:
        model_name: Name of the model
        metrics_data: Dictionary with all evaluation results
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    
    # Evaluation types and their display names
    eval_types = ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]
    eval_labels = ["Zero-shot", "Few-shot", "Few-shot\n(hard neg.)"]
    
    # Metrics to plot (same order as table columns)
    metrics = ["Presence Acc.", "F1", "Hit@Pt|Pres", "Gated Acc."]
    
    # Collect data for this model
    data = []
    for eval_type in eval_types:
        if eval_type in metrics_data and model_name in metrics_data[eval_type]:
            macro = metrics_data[eval_type][model_name].get('macro_metrics', {})
            data.append([
                macro.get('macro_presence_acc', 0) * 100,
                macro.get('macro_f1', 0) * 100,
                macro.get('macro_hit_rate', 0) * 100,
                macro.get('macro_gated_acc', 0) * 100
            ])
        else:
            data.append([0, 0, 0, 0])  # Missing data
    
    data = np.array(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bar settings
    x = np.arange(len(metrics))
    width = 0.25
    
    # Colors for each evaluation type
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Create bars
    for i, (eval_label, color) in enumerate(zip(eval_labels, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[i], width, label=eval_label, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if there's data
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=12)  # Increased from 9
    
    # Add stars to the highest bar for each metric
    for metric_idx in range(len(metrics)):
        # Find the highest value for this metric across all evaluation types
        metric_values = [data[i][metric_idx] for i in range(len(eval_types))]
        max_value = max(metric_values)
        
        if max_value > 0:
            # Find which eval type has the max value
            max_eval_idx = metric_values.index(max_value)
            
            # Calculate correct x position for the center of the bar
            bar_center_x = x[metric_idx] + (max_eval_idx - 1) * width
            
            # Add a star above the highest bar
            ax.annotate('★',
                       xy=(bar_center_x, max_value),
                       xytext=(0, 5),  # Small offset above bar
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=18,
                       color='black',
                       weight='bold')
    
    # Customize plot
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Performance (%)')
    
    # Format model name for title
    model_display = model_name.replace("_", " ").replace("/", " ")
    if len(model_display) > 40:
        # Shorten long names
        if "llava" in model_display.lower():
            model_display = "LLaVA-v1.6"
        elif "pixtral" in model_display.lower():
            model_display = "Pixtral-12B"
        elif "qwen" in model_display.lower():
            model_display = "Qwen2.5-VL"
    
    ax.set_title(f'Performance Metrics: {model_display}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower left')  # Changed to lower left
    ax.set_ylim(0, 105)  # Set y-axis from 0 to 105%
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_metric_comparison_bars(
    metric_name: str,
    metric_key: str,
    metrics_data: Dict[str, Any],
    output_path: Path,
    eval_type: str = "zero_shot",
    figsize: tuple = (14, 8)  # Increased size
) -> None:
    """Generate bar chart comparing all models for a specific metric.
    
    Args:
        metric_name: Display name of the metric
        metric_key: Key in macro_metrics dict
        metrics_data: Dictionary with all evaluation results
        output_path: Path to save the figure
        eval_type: Evaluation type to compare
        figsize: Figure size
    """
    
    if eval_type not in metrics_data:
        return
    
    # Collect data with proper ordering
    model_dict = {}
    
    for model_name, model_data in metrics_data[eval_type].items():
        macro = model_data.get('macro_metrics', {})
        value = macro.get(metric_key, 0) * 100
        model_dict[model_name] = value
    
    # Order models according to preference
    ordered_model_names = order_models(list(model_dict.keys()))
    
    models = []
    values = []
    for model_name in ordered_model_names:
        # Format model name for display
        if "llava" in model_name.lower():
            model_display = "LLaVA"
        elif "pixtral" in model_name.lower():
            model_display = "Pixtral"
        elif "qwen" in model_name.lower():
            model_display = "Qwen2.5"
        elif "claude" in model_name.lower():
            model_display = "Claude"
        elif "gemini" in model_name.lower():
            model_display = "Gemini"
        elif "gpt" in model_name.lower():
            model_display = "GPT-4.1"
        else:
            model_display = model_name.replace("_", " ").split("/")[-1][:15]
        
        models.append(model_display)
        values.append(model_dict[model_name])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars with different colors for APIs vs open-source
    x = np.arange(len(models))
    colors = []
    for model_name in ordered_model_names:
        if any(api in model_name.lower() for api in ['gpt', 'gemini', 'claude']):
            colors.append('#1f77b4')  # Blue for APIs
        else:
            colors.append('#2ca02c')  # Green for open-source
    
    bars = ax.bar(x, values, color=colors, alpha=0.8)
    
    # Add a vertical line to separate APIs from open-source (after first 3 models)
    if len(models) > 3:
        ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Add star to the highest bar
    if values:
        max_value = max(values)
        max_idx = values.index(max_value)
        ax.annotate('★',
                   xy=(x[max_idx], max_value),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=20,
                   color='black',
                   weight='bold')
    
    # Customize plot
    ax.set_xlabel('Models')
    ax.set_ylabel(f'{metric_name} (%)')
    ax.set_title(f'{metric_name} Comparison ({eval_type.replace("_", " ").title()})')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, max(values) * 1.1 if values else 100)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_models_comparison(
    metrics_data: Dict[str, Any],
    output_path: Path,
    figsize: tuple = (16, 10)  # Increased size
) -> None:
    """Generate comprehensive comparison chart for all models and methods.
    
    Args:
        metrics_data: Dictionary with all evaluation results
        output_path: Path to save the figure
        figsize: Figure size
    """
    
    # Prepare data
    all_models = set()
    for eval_data in metrics_data.values():
        all_models.update(eval_data.keys())
    models = order_models(list(all_models))  # Use ordered models
    
    # Format model names
    model_labels = []
    for model in models:
        if "llava" in model.lower():
            model_labels.append("LLaVA")
        elif "pixtral" in model.lower():
            model_labels.append("Pixtral")
        elif "qwen" in model.lower():
            model_labels.append("Qwen2.5")
        elif "claude" in model.lower():
            model_labels.append("Claude")
        elif "gemini" in model.lower():
            model_labels.append("Gemini")
        elif "gpt" in model.lower():
            model_labels.append("GPT-4.1")
        else:
            model_labels.append(model.split("/")[-1][:10])
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Metrics in same order as table columns
    metrics = [
        ("Presence Accuracy", "macro_presence_acc"),
        ("F1 Score", "macro_f1"),
        ("Hit@Point|Present", "macro_hit_rate"),
        ("Gated Accuracy", "macro_gated_acc")
    ]
    
    eval_types = ["zero_shot", "fewshot_standard", "fewshot_hard_negatives"]
    eval_labels = ["Zero-shot", "Few-shot", "Few-shot (hard neg.)"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (metric_name, metric_key) in enumerate(metrics):
        ax = axes[idx]
        
        x = np.arange(len(models))
        width = 0.25
        
        all_values_by_eval = []
        for i, (eval_type, eval_label, color) in enumerate(zip(eval_types, eval_labels, colors)):
            values = []
            for model in models:
                if eval_type in metrics_data and model in metrics_data[eval_type]:
                    macro = metrics_data[eval_type][model].get('macro_metrics', {})
                    values.append(macro.get(metric_key, 0) * 100)
                else:
                    values.append(0)
            
            all_values_by_eval.append(values)
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=eval_label if idx == 0 else "", 
                  color=color, alpha=0.8)
        
        # Add a single star to the overall highest bar across all models and methods
        overall_max = 0
        best_model_idx = 0
        best_eval_idx = 0
        
        for model_idx in range(len(models)):
            for eval_idx in range(len(eval_types)):
                value = all_values_by_eval[eval_idx][model_idx]
                if value > overall_max:
                    overall_max = value
                    best_model_idx = model_idx
                    best_eval_idx = eval_idx
        
        if overall_max > 0:
            # Calculate correct x position for the center of the best bar
            bar_center_x = x[best_model_idx] + (best_eval_idx - 1) * width
            
            # Add a star above the highest bar
            ax.annotate('★',
                       xy=(bar_center_x, overall_max),
                       xytext=(0, 2),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=12,
                       color='black',
                       weight='bold')
        
        ax.set_title(metric_name)
        ax.set_ylabel('Performance (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        if idx == 0:  # First subplot (Presence Accuracy)
            ax.legend(loc='lower left')
    
    plt.suptitle('Comprehensive Model Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()