#!/usr/bin/env python3
import json
import glob
from pathlib import Path
import pandas as pd
from datetime import datetime

# Find all naive baseline result directories
result_dirs = sorted(glob.glob("/shared_data0/weiqiuy/llm_cholec_organ/results/naive_baseline_*/"))

all_results = []

for result_dir in result_dirs:
    result_path = Path(result_dir)
    summary_file = result_path / "summary.json"
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            
            # Extract key information
            result = {
                'dataset': data['dataset'],
                'presence_mode': data['presence_mode'],
                'num_samples': data['num_samples'],
                'presence_accuracy': data['metrics']['presence_accuracy'],
                'iou_bbox_to_bbox': data['metrics']['mean_iou_bbox_to_bbox'],
                'iou_bbox_to_mask': data['metrics']['mean_iou_bbox_to_mask'],
                'iou50_bbox': data['metrics']['iou_at_50_bbox_to_bbox'],
                'iou50_mask': data['metrics']['iou_at_50_bbox_to_mask'],
                'num_predictions': data['metrics']['num_predictions'],
                'timestamp': data['timestamp']
            }
            all_results.append(result)

# Create DataFrame
df = pd.DataFrame(all_results)

# Sort by dataset and presence mode
df = df.sort_values(['dataset', 'presence_mode'])

# Display results
print("\n" + "="*100)
print("NAIVE BASELINE RESULTS SUMMARY")
print("="*100)
print("\nAll results:")
print(df.to_string(index=False))

# Group by dataset for comparison
print("\n" + "="*100)
print("BY DATASET COMPARISON")
print("="*100)

for dataset in df['dataset'].unique():
    dataset_df = df[df['dataset'] == dataset]
    print(f"\n{dataset.upper()}:")
    print("-" * 60)
    
    for _, row in dataset_df.iterrows():
        print(f"\nPresence mode: {row['presence_mode']}")
        print(f"  Presence Accuracy: {row['presence_accuracy']:.1%}")
        print(f"  Bbox-to-Bbox IoU:  {row['iou_bbox_to_bbox']:.3f}")
        print(f"  Bbox-to-Mask IoU:  {row['iou_bbox_to_mask']:.3f}")
        print(f"  IoU@0.5 (BBox):    {row['iou50_bbox']:.1%}")
        print(f"  IoU@0.5 (Mask):    {row['iou50_mask']:.1%}")

# Save summary to CSV
csv_file = 'naive_baseline_summary.csv'
df.to_csv(csv_file, index=False)
print(f"\n\nSummary saved to: {csv_file}")

# Create LaTeX table
print("\n" + "="*100)
print("LATEX TABLE")
print("="*100)

# Pivot for better presentation
pivot_df = df.pivot_table(
    index='presence_mode',
    columns='dataset',
    values=['presence_accuracy', 'iou_bbox_to_bbox', 'iou_bbox_to_mask'],
    aggfunc='first'
)

latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Naive Baseline Results: Always predicting entire image as bounding box}
\\begin{tabular}{lcccccc}
\\toprule
& \\multicolumn{3}{c}{CholecSeg8k} & \\multicolumn{3}{c}{CholecOrgans} \\\\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}
Presence Mode & Acc & IoU-B & IoU-M & Acc & IoU-B & IoU-M \\\\
\\midrule
"""

for mode in ['perfect', 'all', 'random']:
    if mode in pivot_df.index:
        row_data = []
        row_data.append(mode.capitalize())
        
        for dataset in ['cholecseg8k', 'cholec_organs']:
            if dataset in pivot_df.columns.get_level_values(1):
                acc = pivot_df.loc[mode, ('presence_accuracy', dataset)]
                iou_b = pivot_df.loc[mode, ('iou_bbox_to_bbox', dataset)]
                iou_m = pivot_df.loc[mode, ('iou_bbox_to_mask', dataset)]
                
                row_data.append(f"{acc:.1%}" if not pd.isna(acc) else "—")
                row_data.append(f"{iou_b:.3f}" if not pd.isna(iou_b) else "—")
                row_data.append(f"{iou_m:.3f}" if not pd.isna(iou_m) else "—")
            else:
                row_data.extend(["—", "—", "—"])
        
        latex_table += " & ".join(row_data) + " \\\\\n"

latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""

print(latex_table)

# Save LaTeX table
with open('naive_baseline_table.tex', 'w') as f:
    f.write(latex_table)
print(f"\nLaTeX table saved to: naive_baseline_table.tex")
