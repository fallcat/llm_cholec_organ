#!/usr/bin/env python3
"""Fixed visualization code for dataset balancing"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('default')
sns.set_style("whitegrid")

# Load the balance data
dataset_name = "cholecseg8k_local"
data_dir = Path(f"/shared_data0/weiqiuy/llm_cholec_organ/data_info/{dataset_name}_balanced_200")
summary_file = data_dir / "pipeline_summary.json"

with open(summary_file, 'r') as f:
    data = json.load(f)

# Extract organ names and distributions from the correct structure
class_dist = data['class_distribution']['selected']
organ_names = list(class_dist.keys())

# Get original and selected percentages
original_pct = np.array([class_dist[org]['original_pct'] for org in organ_names])
balanced_pct = np.array([class_dist[org]['selected_pct'] for org in organ_names])
original_counts = np.array([class_dist[org]['original_count'] for org in organ_names])
balanced_counts = np.array([class_dist[org]['selected_count'] for org in organ_names])

# Sort by original percentage for better visualization
sort_idx = np.argsort(original_pct)
organ_names_sorted = [organ_names[i] for i in sort_idx]
original_pct_sorted = original_pct[sort_idx]
balanced_pct_sorted = balanced_pct[sort_idx]

# Create comprehensive figure
fig = plt.figure(figsize=(18, 10))

# 1. Before/After Bar Chart
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(organ_names_sorted))
width = 0.35

bars1 = ax1.bar(x - width/2, original_pct_sorted, width, label='Original', color='coral', alpha=0.8)
bars2 = ax1.bar(x + width/2, balanced_pct_sorted, width, label='Balanced', color='lightgreen', alpha=0.8)

ax1.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% rare threshold')
ax1.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='30% min quota')

ax1.set_xlabel('Organ Classes', fontsize=10)
ax1.set_ylabel('Presence (%)', fontsize=10)
ax1.set_title('Distribution: Before vs After Balancing', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(organ_names_sorted, rotation=45, ha='right', fontsize=8)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Improvement Heatmap
ax2 = plt.subplot(2, 3, 2)
improvement = balanced_pct - original_pct
improvement_matrix = improvement.reshape(1, -1)

sns.heatmap(improvement_matrix[:, sort_idx], 
            xticklabels=[name[:15] for name in organ_names_sorted],  # Truncate long names
            yticklabels=['Change'],
            cmap='RdYlGn', center=0, 
            annot=True, fmt='+.1f',
            cbar_kws={'label': 'Percentage Point Change'},
            ax=ax2,
            annot_kws={'size': 8})
ax2.set_title('Representation Change After Balancing', fontsize=12, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)

# 3. Distribution Spread (Box Plot)
ax3 = plt.subplot(2, 3, 3)

bp = ax3.boxplot([original_pct, balanced_pct], 
                  labels=['Original', 'Balanced'],
                  patch_artist=True,
                  widths=0.6)

# Color the boxes
colors = ['coral', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax3.set_ylabel('Presence Percentage', fontsize=10)
ax3.set_title('Distribution Spread Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add statistics
metrics = data['balance_metrics']
ax3.text(1, np.max(original_pct) + 5, 
         f'StdDev: {metrics["original_stddev"]:.1f}%', 
         ha='center', fontsize=9)
ax3.text(2, np.max(balanced_pct) + 5, 
         f'StdDev: {metrics["selected_stddev"]:.1f}%', 
         ha='center', fontsize=9)

# 4. Balance Metrics
ax4 = plt.subplot(2, 3, 4)

metric_names = ['Original\nStdDev', 'Balanced\nStdDev', 'Balance\nImprovement']
metric_values = [
    metrics['original_stddev'],
    metrics['selected_stddev'],
    metrics['balance_improvement_pct']
]
colors_metrics = ['coral', 'lightgreen', 'gold']

bars = ax4.bar(metric_names, metric_values, color=colors_metrics, alpha=0.8)
ax4.set_ylabel('Value (%)', fontsize=10)
ax4.set_title('Balance Metrics Summary', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, metric_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 1,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

# 5. Rare vs Common Classes
ax5 = plt.subplot(2, 3, 5)

# Get rare classes from data
rare_classes = data.get('rare_classes', [])
rare_mask = np.array([org in rare_classes for org in organ_names])

# Calculate average improvement for rare vs common
rare_improvements = [class_dist[org]['improvement'] for org in organ_names if org in rare_classes]
common_improvements = [class_dist[org]['improvement'] for org in organ_names if org not in rare_classes]

rare_avg = np.mean(rare_improvements) if rare_improvements else 0
common_avg = np.mean(common_improvements) if common_improvements else 0

categories = [f'Rare Classes\n(<20%)\nn={len(rare_improvements)}', 
              f'Common Classes\n(≥20%)\nn={len(common_improvements)}']
improvements_avg = [rare_avg, common_avg]
colors_cat = ['darkred', 'darkgreen']

bars = ax5.bar(categories, improvements_avg, color=colors_cat, alpha=0.7)
ax5.set_ylabel('Average Improvement (pp)', fontsize=10)
ax5.set_title('Improvement by Class Category', fontsize=12, fontweight='bold')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax5.grid(True, alpha=0.3, axis='y')

# Add annotations
for bar, val in zip(bars, improvements_avg):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2, height + 0.5 if height > 0 else height - 1,
            f'{val:+.1f}pp', ha='center', fontsize=10, fontweight='bold')

# 6. Class Distribution Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Create table data
table_data = []
for org in organ_names_sorted[:8]:  # Show top 8 for space
    orig = f"{class_dist[org]['original_pct']:.1f}%"
    selected = f"{class_dist[org]['selected_pct']:.1f}%"
    change = f"{class_dist[org]['improvement']:+.1f}%"
    table_data.append([org[:20], orig, selected, change])

table = ax6.table(cellText=table_data,
                  colLabels=['Organ', 'Original', 'Balanced', 'Change'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.4, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Style the header
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code the cells based on change
for i in range(1, len(table_data) + 1):
    change_val = float(table_data[i-1][3].replace('%', '').replace('+', ''))
    if change_val > 10:
        table[(i, 3)].set_facecolor('#90EE90')
    elif change_val < -10:
        table[(i, 3)].set_facecolor('#FFB6C1')

ax6.set_title('Top Changes in Class Distribution', fontsize=12, fontweight='bold', pad=20)

# Overall title
fig.suptitle(f'Dataset Balancing Analysis - {dataset_name.upper()}\n'
            f'Test Samples: {data["n_test_selected"]} | '
            f'Classes: {data["n_classes"]} | '
            f'Training Set: {data["n_train_total"]} samples', 
            fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# Save figure
save_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/notebooks/images")
save_dir.mkdir(parents=True, exist_ok=True)

output_file = save_dir / f"{dataset_name}_balance_visualization.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization to: {output_file}")

output_pdf = save_dir / f"{dataset_name}_balance_visualization.pdf"
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✅ Saved PDF to: {output_pdf}")

plt.show()

# Print summary statistics
print("\n📊 Balance Statistics Summary:")
print("=" * 50)
print(f"Dataset: {dataset_name}")
print(f"Original StdDev: {metrics['original_stddev']:.1f}%")
print(f"Balanced StdDev: {metrics['selected_stddev']:.1f}%")
print(f"Balance Improvement: {metrics['balance_improvement_pct']:.1f}%")
print(f"\nRare classes (<20%): {', '.join(rare_classes)}")
print(f"These were boosted to minimum 30% representation")
print("=" * 50)