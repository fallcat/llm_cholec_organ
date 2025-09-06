#!/usr/bin/env python
"""
Prepare Few-Shot Examples for Organ Detection Evaluation
This script creates balanced test sets and few-shot example plans using the refactored modules

Key features:
- Selects 200 balanced test samples (with 20% minimum quota for rare classes)
- Builds few-shot training examples from remaining data
- Supports both pointing and bounding box tasks
- Creates visualizations of the selected examples
"""

# Cell 1: Setup and imports
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Import the refactored modules
from endopoint.datasets import build_dataset
from endopoint.fewshot import UnifiedFewShotSelector

print("✅ Modules imported")

# Cell 2: Configuration
CONFIG = {
    "n_test_samples": 200,     # Number of balanced test samples (updated from 100 to 200)
    "n_pos_examples": 1,       # Positive examples per organ
    "n_neg_absent": 1,         # Negative examples where organ is absent  
    "n_neg_wrong": 1,          # Negative examples with wrong location/bbox
    "min_pixels": 50,          # Minimum pixels for organ presence
    "seed": 42,                # Random seed for reproducibility
    "output_dir": Path(ROOT_DIR) / "data_info" / "cholecseg8k_balanced_200"
}

print(f"Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Cell 3: Load dataset using new adapter
print("\n📊 Loading CholecSeg8k dataset...")
dataset = build_dataset(
    "cholecseg8k_local",
    data_dir="/shared_data0/weiqiuy/datasets/cholecseg8k"
)
train_size = dataset.total('train')
print(f"✅ Dataset loaded: {train_size} training samples")
print(f"  Classes: {len(dataset.label_ids)} organs")

# Cell 4: Create unified selector
print("\n🔧 Creating UnifiedFewShotSelector...")
selector = UnifiedFewShotSelector(
    dataset=dataset,
    output_dir=CONFIG["output_dir"],
    n_test_samples=CONFIG["n_test_samples"],
    n_pos_examples=CONFIG["n_pos_examples"],
    n_neg_absent=CONFIG["n_neg_absent"],
    n_neg_wrong=CONFIG["n_neg_wrong"],
    min_pixels=CONFIG["min_pixels"],
    seed=CONFIG["seed"],
    cache_enabled=True
)

print(f"📁 Output directory: {selector.output_dir}")

# Cell 5: Run the complete pipeline with balance analysis
print("\n🚀 Running balanced selection pipeline...")
results = selector.run_balanced_selection_pipeline(
    split="train",
    visualize=True,      # Print detailed analysis
    save_summary=True    # Save summary to file
)

print(f"\n✨ Pipeline complete!")
print(f"  Test samples selected: {len(results['test_indices'])}")
print(f"  Balance improvement: {results['balance_comparison']['metrics']['balance_improvement_pct']:.1f}%")

# Save test indices explicitly
test_indices_file = CONFIG["output_dir"] / f"balanced_test_indices_{CONFIG['n_test_samples']}.json"
with open(test_indices_file, 'w') as f:
    json.dump({
        'indices': results['test_indices'],
        'n_samples': len(results['test_indices']),
        'seed': CONFIG['seed'],
        'timestamp': datetime.now().isoformat()
    }, f, indent=2)
print(f"💾 Test indices saved to: {test_indices_file}")

# Cell 6: Analyze organ distribution
print("\n📊 Organ presence statistics in selected test set:")

# Get presence matrix and analyze
Y = selector._presence_matrix  # Already computed in pipeline
Y_test = Y[results['test_indices']]
organ_counts = Y_test.sum(axis=0)
test_samples = len(results['test_indices'])

for i, class_id in enumerate(dataset.label_ids):
    organ_name = dataset.id2label[class_id]
    count = organ_counts[i]
    percentage = (count / test_samples) * 100
    print(f"  {organ_name:25} {count:5d} samples ({percentage:5.1f}%)")

# Cell 7: Build few-shot plans for different tasks
print("\n📝 Building few-shot example plans...")

# 1. Pointing task plan (if supported)
if selector.supports_pointing:
    print("\n🎯 Building pointing task plan...")
    pointing_plan = selector.build_fewshot_plan(
        split="train",
        task_type="pointing",
        excluded_indices=results['test_indices']
    )
    
    pointing_file = CONFIG["output_dir"] / f"fewshot_plan_pointing_n{CONFIG['n_test_samples']}.json"
    with open(pointing_file, 'w') as f:
        json.dump(pointing_plan, f, indent=2)
    print(f"  ✅ Saved: {pointing_file.name}")
    
    # Print summary
    total_pos = sum(len(info['positives']) for info in pointing_plan['plan'].values())
    total_neg_absent = sum(len(info['negatives_absent']) for info in pointing_plan['plan'].values())
    total_neg_wrong = sum(len(info.get('negatives_wrong_point', [])) for info in pointing_plan['plan'].values())
    print(f"  Total examples: {total_pos} positive, {total_neg_absent} negative (absent), {total_neg_wrong} negative (wrong)")
else:
    print("  ⏭️ Skipping pointing task (not supported)")
    pointing_plan = None

# 2. Bounding box task plan (if supported)
if selector.supports_bbox:
    print("\n📦 Building bounding box task plan...")
    bbox_plan = selector.build_fewshot_plan(
        split="train",
        task_type="bbox",
        excluded_indices=results['test_indices']
    )
    
    bbox_file = CONFIG["output_dir"] / f"fewshot_plan_bbox_n{CONFIG['n_test_samples']}.json"
    with open(bbox_file, 'w') as f:
        json.dump(bbox_plan, f, indent=2)
    print(f"  ✅ Saved: {bbox_file.name}")
    
    # Print summary
    total_pos = sum(len(info['positives']) for info in bbox_plan['plan'].values())
    total_neg_absent = sum(len(info['negatives_absent']) for info in bbox_plan['plan'].values())
    total_neg_wrong = sum(len(info.get('negatives_wrong_bbox', [])) for info in bbox_plan['plan'].values())
    total_multi = sum(
        sum(1 for pos in info['positives'] if pos.get('num_regions', 1) > 1)
        for info in bbox_plan['plan'].values()
    )
    print(f"  Total examples: {total_pos} positive, {total_neg_absent} negative (absent), {total_neg_wrong} negative (wrong)")
    if total_multi > 0:
        print(f"  Multi-region examples: {total_multi}")
else:
    print("  ⏭️ Skipping bounding box task (not supported)")
    bbox_plan = None

# Cell 8: Visualize few-shot examples
print("\n📊 Visualizing few-shot examples...")

def visualize_fewshot_examples(dataset, plan, organ_names_to_show=None, max_organs=4):
    """
    Visualize few-shot examples for selected organs.
    Shows positive examples with annotations and negative examples.
    """
    if plan is None:
        print("No plan available for visualization")
        return None
        
    if organ_names_to_show is None:
        # Show first few organs with both positives and negatives
        organ_names_to_show = []
        for class_id_str, info in plan["plan"].items():
            if len(info["positives"]) > 0 and len(info["negatives_absent"]) > 0:
                organ_names_to_show.append(info["name"])
                if len(organ_names_to_show) >= max_organs:
                    break
    
    n_organs = len(organ_names_to_show)
    if n_organs == 0:
        print("No organs to visualize")
        return None
        
    # Adjust figure size based on number of organs
    fig_height = min(4 * n_organs, 20)  # Cap at reasonable height
    fig = plt.figure(figsize=(18, fig_height))
    
    for organ_idx, organ_name in enumerate(organ_names_to_show):
        # Find the organ in the plan
        organ_info = None
        for class_id_str, info in plan["plan"].items():
            if info["name"] == organ_name:
                organ_info = info
                class_id = int(class_id_str)
                break
        
        if organ_info is None:
            continue
        
        # Determine what to show based on task type
        has_points = 'negatives_wrong_point' in organ_info
        has_bboxes = 'negatives_wrong_bbox' in organ_info
        
        # Create subplot for this organ
        n_pos = min(2, len(organ_info["positives"]))
        n_neg_absent = min(2, len(organ_info["negatives_absent"]))
        n_neg_wrong = min(2, len(organ_info.get("negatives_wrong_point" if has_points else "negatives_wrong_bbox", [])))
        n_cols = n_pos + n_neg_absent + n_neg_wrong
        
        if n_cols == 0:
            continue
        
        col_idx = 0
        
        # Plot positive examples
        for i in range(n_pos):
            ax = plt.subplot(n_organs, 6, organ_idx * 6 + col_idx + 1)
            pos_data = organ_info["positives"][i]
            idx = pos_data["idx"]
            
            # Load and display image
            example = dataset.get_example(plan["split"], idx)
            img = example["image"]
            ax.imshow(img)
            
            # Add annotations based on task type
            if has_points and "point" in pos_data:
                point = pos_data["point"]
                ax.plot(point[0], point[1], 'r*', markersize=15, markeredgewidth=2, 
                       markeredgecolor='white', label='Organ location')
                circle = patches.Circle((point[0], point[1]), 30, linewidth=2, 
                                       edgecolor='red', facecolor='none', alpha=0.7)
                ax.add_patch(circle)
            elif has_bboxes and "bboxes" in pos_data:
                for bbox in pos_data["bboxes"]:
                    x1, y1, x2, y2 = bbox
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor='green',
                                           facecolor='none', alpha=0.7)
                    ax.add_patch(rect)
            
            ax.set_title(f"✅ Positive {i+1}\n(Has {organ_name})\nidx: {idx}", fontsize=9, color='green')
            ax.axis('off')
            col_idx += 1
        
        # Plot negative absent examples
        for i in range(n_neg_absent):
            ax = plt.subplot(n_organs, 6, organ_idx * 6 + col_idx + 1)
            idx = organ_info["negatives_absent"][i]
            
            # Load and display image
            example = dataset.get_example(plan["split"], idx)
            img = example["image"]
            ax.imshow(img)
            
            ax.set_title(f"❌ Neg Absent {i+1}\n(No {organ_name})\nidx: {idx}", fontsize=9, color='orange')
            ax.axis('off')
            col_idx += 1
        
        # Plot negative wrong examples
        neg_wrong_key = "negatives_wrong_point" if has_points else "negatives_wrong_bbox"
        if neg_wrong_key in organ_info:
            for i in range(n_neg_wrong):
                ax = plt.subplot(n_organs, 6, organ_idx * 6 + col_idx + 1)
                wrong_data = organ_info[neg_wrong_key][i]
                idx = wrong_data["idx"]
                
                # Load and display image
                example = dataset.get_example(plan["split"], idx)
                img = example["image"]
                ax.imshow(img)
                
                # Show wrong annotation
                if has_points and "wrong_point" in wrong_data:
                    point = wrong_data["wrong_point"]
                    ax.plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=2,
                           label='Wrong location')
                    circle = patches.Circle((point[0], point[1]), 25, linewidth=2,
                                          edgecolor='red', facecolor='none', alpha=0.7, linestyle='--')
                    ax.add_patch(circle)
                elif has_bboxes and "wrong_bbox" in wrong_data:
                    x1, y1, x2, y2 = wrong_data["wrong_bbox"]
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor='red',
                                           facecolor='none', alpha=0.7, linestyle='--')
                    ax.add_patch(rect)
                
                ax.set_title(f"⚠️ Neg Wrong {i+1}\n(Has {organ_name}, wrong loc)\nidx: {idx}", 
                           fontsize=9, color='red')
                ax.axis('off')
                col_idx += 1
        
        # Add organ name on the left
        fig.text(0.02, 0.9 - organ_idx / n_organs - 0.05, organ_name, 
                fontsize=12, fontweight='bold', rotation=0)
    
    task_type = "Pointing" if has_points else "Bounding Box"
    plt.suptitle(f"Few-Shot Examples Visualization ({task_type} Task)\n"
                 f"(Showing {n_organs} organs with positives and negatives)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    
    # Save figure
    task_suffix = "pointing" if has_points else "bbox"
    viz_path = CONFIG["output_dir"] / f"fewshot_examples_visualization_{task_suffix}.png"
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    print(f"✅ Visualization saved to: {viz_path}")
    
    plt.show()
    return fig

# Visualize pointing examples
if pointing_plan:
    print("\n📸 Visualizing pointing task examples...")
    organs_to_show = ["Liver", "Gallbladder", "Gastrointestinal Tract", "Grasper"]
    fig_pointing = visualize_fewshot_examples(dataset, pointing_plan, organs_to_show, max_organs=4)

# Visualize bbox examples
if bbox_plan:
    print("\n📸 Visualizing bounding box task examples...")
    organs_to_show = ["Liver", "Gallbladder", "Fat", "L-hook Electrocautery"]
    fig_bbox = visualize_fewshot_examples(dataset, bbox_plan, organs_to_show, max_organs=4)

# Cell 9: Create summary statistics visualization
print("\n📊 Creating summary statistics...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Class distribution in test set
ax = axes[0]
organ_names = [dataset.id2label[cid][:15] for cid in dataset.label_ids]
x = np.arange(len(organ_names))

test_percentages = (organ_counts / test_samples) * 100
bars = ax.bar(x, test_percentages, color='steelblue', alpha=0.7)
ax.set_xlabel('Organ')
ax.set_ylabel('Presence (%)')
ax.set_title(f'Test Set Distribution (n={CONFIG["n_test_samples"]})')
ax.set_xticks(x)
ax.set_xticklabels(organ_names, rotation=45, ha='right')
ax.axhline(y=100/len(organ_names), color='red', linestyle='--', alpha=0.5, 
          label=f'Ideal balanced ({100/len(organ_names):.1f}%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Original vs Selected distribution comparison
ax = axes[1]
original_dist = results['balance_comparison']['classes']
orig_percentages = []
selected_percentages = []

for cid in dataset.label_ids:
    name = dataset.id2label[cid]
    if name in original_dist:
        orig_percentages.append(original_dist[name]['original_pct'])
        selected_percentages.append(original_dist[name]['selected_pct'])

width = 0.35
bars1 = ax.bar(x - width/2, orig_percentages, width, label='Original', color='lightcoral', alpha=0.7)
bars2 = ax.bar(x + width/2, selected_percentages, width, label='Selected', color='darkgreen', alpha=0.7)

ax.set_xlabel('Organ')
ax.set_ylabel('Presence (%)')
ax.set_title('Original vs Selected Distribution')
ax.set_xticks(x)
ax.set_xticklabels(organ_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Few-shot examples count
ax = axes[2]
if bbox_plan:  # Use bbox_plan as it typically has all examples
    pos_counts = []
    neg_counts = []
    
    for cid in dataset.label_ids:
        cid_str = str(cid)
        if cid_str in bbox_plan['plan']:
            info = bbox_plan['plan'][cid_str]
            pos_counts.append(len(info['positives']))
            neg_absent = len(info['negatives_absent'])
            neg_wrong = len(info.get('negatives_wrong_bbox', []))
            neg_counts.append(neg_absent + neg_wrong)
        else:
            pos_counts.append(0)
            neg_counts.append(0)
    
    width = 0.35
    bars1 = ax.bar(x - width/2, pos_counts, width, label='Positive', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, neg_counts, width, label='Negative', color='red', alpha=0.7)
    
    ax.set_xlabel('Organ')
    ax.set_ylabel('Number of Examples')
    ax.set_title('Few-Shot Examples per Organ')
    ax.set_xticks(x)
    ax.set_xticklabels(organ_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Dataset Preparation Summary', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save statistics figure
stats_path = CONFIG["output_dir"] / "preparation_statistics.png"
plt.savefig(stats_path, dpi=100, bbox_inches='tight')
print(f"✅ Statistics saved to: {stats_path}")
plt.show()

# Cell 10: Create final summary report
print("\n📝 Creating summary report...")

summary = {
    "timestamp": datetime.now().isoformat(),
    "dataset": "CholecSeg8k",
    "configuration": {
        "n_test_samples": CONFIG["n_test_samples"],
        "n_pos_examples": CONFIG["n_pos_examples"],
        "n_neg_absent": CONFIG["n_neg_absent"],
        "n_neg_wrong": CONFIG["n_neg_wrong"],
        "min_pixels": CONFIG["min_pixels"],
        "seed": CONFIG["seed"]
    },
    "results": {
        "test_set": {
            "n_samples": len(results['test_indices']),
            "balance_improvement": results['balance_comparison']['metrics']['balance_improvement_pct'],
            "original_stddev": results['balance_comparison']['metrics']['original_stddev'],
            "selected_stddev": results['balance_comparison']['metrics']['selected_stddev']
        },
        "few_shot": {
            "pointing_supported": selector.supports_pointing,
            "bbox_supported": selector.supports_bbox,
            "total_classes": len(dataset.label_ids)
        }
    },
    "files_created": {
        "test_indices": str(test_indices_file),
        "pointing_plan": str(pointing_file) if pointing_plan else None,
        "bbox_plan": str(bbox_file) if bbox_plan else None,
        "pipeline_summary": str(CONFIG["output_dir"] / "pipeline_summary.json")
    },
    "statistics": {
        "total_train_samples": train_size,
        "organ_distribution_test": {
            dataset.id2label[dataset.label_ids[i]]: int(organ_counts[i]) 
            for i in range(len(dataset.label_ids))
        }
    }
}

summary_file = CONFIG["output_dir"] / "preparation_summary.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"✅ Summary saved to: {summary_file}")

# Cell 11: Display final summary
print("\n" + "="*60)
print("✨ Few-shot example preparation complete!")
print("="*60)
print(f"\nDataset: CholecSeg8k")
print(f"Output directory: {CONFIG['output_dir']}")
print(f"\n📊 Results:")
print(f"  • Test set: {CONFIG['n_test_samples']} balanced samples")
print(f"  • Balance improvement: {results['balance_comparison']['metrics']['balance_improvement_pct']:.1f}%")
print(f"  • Min quota for rare classes: 20% ({int(0.20 * CONFIG['n_test_samples'])} samples)")

if pointing_plan:
    print(f"\n🎯 Pointing task:")
    print(f"  • {CONFIG['n_pos_examples']} positive example per organ")
    print(f"  • {CONFIG['n_neg_absent']} negative (absent) example per organ")
    print(f"  • {CONFIG['n_neg_wrong']} negative (wrong location) example per organ")

if bbox_plan:
    print(f"\n📦 Bounding box task:")
    print(f"  • {CONFIG['n_pos_examples']} positive example per organ")
    print(f"  • {CONFIG['n_neg_absent']} negative (absent) example per organ")
    print(f"  • {CONFIG['n_neg_wrong']} negative (wrong bbox) example per organ")

print(f"\n📁 Files created:")
print(f"  1. Test indices: {test_indices_file.name}")
if pointing_plan:
    print(f"  2. Pointing plan: {pointing_file.name}")
if bbox_plan:
    print(f"  3. Bbox plan: {bbox_file.name}")
print(f"  4. Pipeline summary: pipeline_summary.json")
print(f"  5. Preparation summary: {summary_file.name}")

print("\n🎯 Next steps:")
print("  1. Use the test indices for consistent evaluation")
print("  2. Use the few-shot plans for prompting LLMs")
print("  3. Compare zero-shot vs few-shot performance")
print("  4. Analyze which organs benefit most from few-shot examples")