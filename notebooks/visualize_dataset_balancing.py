#!/usr/bin/env python3
"""
Visualizations to demonstrate dataset balancing for CholecSeg8k
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_balance_data(dataset_name="cholecseg8k_local"):
    """Load the balance data from the pipeline summary."""
    data_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
    summary_file = data_dir / "pipeline_summary.json"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Pipeline summary not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    return data

def create_balance_visualizations(dataset_name="cholecseg8k_local", save_dir=None):
    """Create comprehensive visualizations showing dataset balancing."""
    
    # Load data
    data = load_balance_data(dataset_name)
    balance_data = data['balance_analysis']
    
    # Extract organ names and distributions
    organ_names = list(balance_data['original_distribution'].keys())
    
    # Get percentages
    original_pct = np.array([balance_data['original_distribution'][org]['percentage'] for org in organ_names])
    balanced_pct = np.array([balance_data['selected_distribution'][org]['percentage'] for org in organ_names])
    
    # Get counts
    original_counts = np.array([balance_data['original_distribution'][org]['count'] for org in organ_names])
    balanced_counts = np.array([balance_data['selected_distribution'][org]['count'] for org in organ_names])
    
    # Sort by original percentage for better visualization
    sort_idx = np.argsort(original_pct)
    organ_names_sorted = [organ_names[i] for i in sort_idx]
    original_pct_sorted = original_pct[sort_idx]
    balanced_pct_sorted = balanced_pct[sort_idx]
    
    # If save_dir not specified, use notebooks/images
    if save_dir is None:
        save_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/notebooks/images")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Before/After Bar Chart Comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(organ_names_sorted))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_pct_sorted, width, label='Original', color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, balanced_pct_sorted, width, label='Balanced', color='lightgreen', alpha=0.8)
    
    ax1.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% rare threshold')
    ax1.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='30% min quota')
    
    ax1.set_xlabel('Organ Classes')
    ax1.set_ylabel('Presence (%)')
    ax1.set_title('Distribution: Before vs After Balancing')
    ax1.set_xticks(x)
    ax1.set_xticklabels(organ_names_sorted, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement Heatmap
    ax2 = plt.subplot(2, 3, 2)
    improvement = balanced_pct - original_pct
    improvement_matrix = improvement.reshape(1, -1)
    
    sns.heatmap(improvement_matrix[:, sort_idx], 
                xticklabels=organ_names_sorted,
                yticklabels=['Change'],
                cmap='RdYlGn', center=0, 
                annot=True, fmt='+.1f',
                cbar_kws={'label': 'Percentage Point Change'},
                ax=ax2)
    ax2.set_title('Representation Change After Balancing')
    
    # 3. Distribution Spread (Box Plot)
    ax3 = plt.subplot(2, 3, 3)
    box_data = pd.DataFrame({
        'Original': original_pct,
        'Balanced': balanced_pct
    })
    
    bp = ax3.boxplot([original_pct, balanced_pct], 
                      labels=['Original', 'Balanced'],
                      patch_artist=True)
    
    # Color the boxes
    colors = ['coral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax3.set_ylabel('Presence Percentage')
    ax3.set_title('Distribution Spread Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    ax3.text(1, np.max(original_pct) + 5, 
             f'StdDev: {np.std(original_pct):.1f}%', 
             ha='center', fontsize=9)
    ax3.text(2, np.max(balanced_pct) + 5, 
             f'StdDev: {np.std(balanced_pct):.1f}%', 
             ha='center', fontsize=9)
    
    # 4. Balance Metrics
    ax4 = plt.subplot(2, 3, 4)
    metrics = data['balance_comparison']['metrics']
    
    metric_names = ['Original\nStdDev', 'Balanced\nStdDev', 'Balance\nImprovement']
    metric_values = [
        metrics['original_stddev'],
        metrics['selected_stddev'],
        metrics['balance_improvement_pct']
    ]
    colors_metrics = ['coral', 'lightgreen', 'gold']
    
    bars = ax4.bar(metric_names, metric_values, color=colors_metrics, alpha=0.8)
    ax4.set_ylabel('Value (%)')
    ax4.set_title('Balance Metrics Summary')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 5. Rare vs Common Classes
    ax5 = plt.subplot(2, 3, 5)
    
    # Categorize organs
    rare_mask = original_pct < 20
    rare_organs = [org for org, is_rare in zip(organ_names, rare_mask) if is_rare]
    common_organs = [org for org, is_rare in zip(organ_names, rare_mask) if not is_rare]
    
    # Calculate average improvement for rare vs common
    rare_improvement = np.mean([balanced_pct[i] - original_pct[i] 
                                for i, is_rare in enumerate(rare_mask) if is_rare])
    common_improvement = np.mean([balanced_pct[i] - original_pct[i] 
                                  for i, is_rare in enumerate(rare_mask) if not is_rare])
    
    categories = ['Rare Classes\n(<20%)', 'Common Classes\n(≥20%)']
    improvements = [rare_improvement, common_improvement]
    colors_cat = ['darkred', 'darkgreen']
    
    bars = ax5.bar(categories, improvements, color=colors_cat, alpha=0.7)
    ax5.set_ylabel('Average Improvement (pp)')
    ax5.set_title('Improvement by Class Category')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + 0.5 if height > 0 else height - 1,
                f'{val:+.1f}pp', ha='center', fontsize=10, fontweight='bold')
    
    # Add counts
    ax5.text(0, -5, f'n={len(rare_organs)}', ha='center', fontsize=9, style='italic')
    ax5.text(1, -5, f'n={len(common_organs)}', ha='center', fontsize=9, style='italic')
    
    # 6. Sample Efficiency (if greedy cover data exists)
    ax6 = plt.subplot(2, 3, 6)
    
    # Check if greedy cover file exists
    greedy_file = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200/fewshot_plan_bbox_combined_greedy.json")
    
    if greedy_file.exists():
        with open(greedy_file, 'r') as f:
            greedy_data = json.load(f)
        
        methods = ['Per-Organ\nApproach', 'Greedy Cover\nOptimized']
        num_examples = [len(organ_names), len(greedy_data['examples'])]
        colors_eff = ['coral', 'lightgreen']
        
        bars = ax6.bar(methods, num_examples, color=colors_eff, alpha=0.8)
        ax6.set_ylabel('Number of Few-Shot Examples')
        ax6.set_title('Few-Shot Efficiency Improvement')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and percentage reduction
        for bar, val in zip(bars, num_examples):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                    f'{val}', ha='center', fontsize=11, fontweight='bold')
        
        reduction = (1 - num_examples[1]/num_examples[0]) * 100
        ax6.text(0.5, max(num_examples) * 0.5, 
                f'{reduction:.0f}%\nreduction', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        ax6.text(0.5, 0.5, 'Greedy Cover\ndata not available', 
                ha='center', va='center', transform=ax6.transAxes,
                fontsize=12, style='italic')
        ax6.set_title('Few-Shot Efficiency')
    
    # Overall title
    fig.suptitle(f'Dataset Balancing Analysis - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = save_dir / f"{dataset_name}_balance_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_file}")
    
    # Also save as PDF
    output_pdf = save_dir / f"{dataset_name}_balance_visualization.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✅ Saved PDF to: {output_pdf}")
    
    plt.show()
    
    return fig

def create_coverage_matrix_visualization(dataset_name="cholecseg8k_local", save_dir=None):
    """Create a visualization of the coverage matrix for selected samples."""
    
    # Load presence matrix
    data_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
    presence_matrix = np.load(data_dir / "presence_matrix_train.npy")
    
    # Load balanced indices
    with open(data_dir / "balanced_test_indices_advanced_200.json", 'r') as f:
        indices_data = json.load(f)
        if isinstance(indices_data, dict):
            test_indices = indices_data.get('indices', indices_data.get('test_indices', []))
        else:
            test_indices = indices_data
    
    # Load organ names from summary
    with open(data_dir / "pipeline_summary.json", 'r') as f:
        summary = json.load(f)
        organ_names = list(summary['balance_analysis']['original_distribution'].keys())
    
    # Create coverage matrix for first 50 samples
    n_samples_to_show = min(50, len(test_indices))
    coverage_matrix = np.zeros((n_samples_to_show, len(organ_names)))
    
    for i, idx in enumerate(test_indices[:n_samples_to_show]):
        coverage_matrix[i] = presence_matrix[idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(coverage_matrix.T, 
                xticklabels=range(1, n_samples_to_show + 1),
                yticklabels=organ_names,
                cmap='YlOrRd',
                cbar_kws={'label': 'Present in Sample'},
                ax=ax)
    
    ax.set_xlabel('Sample Index (from balanced test set)', fontsize=12)
    ax.set_ylabel('Organ Class', fontsize=12)
    ax.set_title(f'Organ Coverage Matrix - First {n_samples_to_show} Balanced Samples', 
                fontsize=14, fontweight='bold')
    
    # Add coverage statistics on the right
    coverage_pct = (coverage_matrix.sum(axis=0) / n_samples_to_show * 100)
    for i, (organ, pct) in enumerate(zip(organ_names, coverage_pct)):
        ax.text(n_samples_to_show + 0.5, i + 0.5, f'{pct:.0f}%', 
               ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save if directory specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / f"{dataset_name}_coverage_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved coverage matrix to: {output_file}")
    
    plt.show()
    return fig

if __name__ == "__main__":
    # Create visualizations for CholecSeg8k
    print("Creating balance visualizations for CholecSeg8k...")
    fig1 = create_balance_visualizations("cholecseg8k_local")
    
    print("\nCreating coverage matrix visualization...")
    fig2 = create_coverage_matrix_visualization("cholecseg8k_local", 
                                                save_dir="/shared_data0/weiqiuy/llm_cholec_organ/notebooks/images")
    
    print("\n✅ All visualizations complete!")