#!/usr/bin/env python3
"""
Generate example visualizations for the paper appendix.
Shows ground truth masks with predicted points from different models.

Usage:
    python notebooks_py/generate_paper_examples.py \
        --results-dir results/pointing_original \
        --output-dir /shared_data0/weiqiuy/llm_cholec_organ_paper/images/examples \
        --num-examples 8
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import torch
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter, ID2LABEL
from datasets import load_dataset

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def load_predictions(results_dir: Path, model_name: str, eval_type: str = "zero_shot") -> Dict:
    """Load prediction results for a model."""
    # Map model names to directory names
    model_dir_map = {
        "gpt-4.1": "gpt-4.1",
        "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "llava-hf/llava-v1.6-mistral-7b-hf": "llava-hf",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen",
        "mistralai/Pixtral-12B-2409": "mistralai"
    }
    
    model_dir = model_dir_map.get(model_name, model_name)
    
    # Try to find individual result files
    base_path = results_dir / eval_type / model_dir / "cholecseg8k_pointing"
    
    if not base_path.exists():
        # Try alternate path
        base_path = results_dir / eval_type / model_dir
        if base_path.exists():
            # Find cholecseg8k_pointing subdirectory
            for subdir in base_path.glob("*/cholecseg8k_pointing"):
                base_path = subdir
                break
    
    if not base_path.exists():
        print(f"Warning: No results directory found for {model_name} at {base_path}")
        return {}
    
    # Load all train_*.json files
    pred_map = {}
    train_files = list(base_path.glob("train_*.json"))
    
    if not train_files:
        print(f"Warning: No train files found for {model_name} in {base_path}")
        return {}
    
    for results_file in train_files:
        # Skip the metrics summary file
        if "metrics_summary" in str(results_file):
            continue
            
        with open(results_file, 'r') as f:
            data = json.load(f)
            # Handle both single item and list formats
            if not isinstance(data, list):
                data = [data]
            
            for item in data:
                sample_idx = item['sample_idx']
                if sample_idx not in pred_map:
                    pred_map[sample_idx] = {}
                organ = item['y_true']['organ']
                pred_map[sample_idx][organ] = item
    
    print(f"  Loaded {len(pred_map)} samples for {model_name} from {len(train_files)} files")
    return pred_map


def select_interesting_examples(
    dataset,
    adapter: CholecSeg8kAdapter,
    predictions: Dict[str, Dict],
    num_examples: int = 8,
    seed: int = 42
) -> List[int]:
    """Select interesting examples that show variety of organs and outcomes."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Categories of interest
    categories = {
        'perfect': [],      # All models correct
        'mixed': [],        # Some models correct
        'challenging': [],  # All models miss or incorrect
        'rare_organs': [],  # Examples with rare organs
        'multi_organ': []   # Examples with many organs
    }
    
    # Get the first model's predictions as reference
    first_model = list(predictions.keys())[0]
    sample_indices = list(predictions[first_model].keys())
    
    for sample_idx in sample_indices[:200]:  # Check first 200 samples
        # Count correct predictions across models
        correct_count = 0
        has_rare = False
        organ_count = 0
        
        # Get ground truth for this sample
        example = dataset['train'][sample_idx]
        masks = adapter.parse_color_mask_to_class_id(example['color_mask'])  # Parse color mask
        
        for organ_idx in range(1, 13):  # Skip background
            organ_name = ID2LABEL[organ_idx]
            organ_mask = (masks == organ_idx).astype(np.float32)
            
            if organ_mask.sum() > 50:  # Organ is present
                organ_count += 1
                
                # Check if it's a rare organ
                if organ_name in ['L-hook Electrocautery', 'Specimen Bag', 'Clip', 'Bipolar Forceps']:
                    has_rare = True
                
                # Check predictions for this organ
                for model_name in predictions:
                    if sample_idx in predictions[model_name] and organ_name in predictions[model_name][sample_idx]:
                        pred = predictions[model_name][sample_idx][organ_name]
                        if pred.get('hits', {}).get('hit_point', False):
                            correct_count += 1
        
        # Categorize the example
        total_preds = len(predictions) * organ_count if organ_count > 0 else 1
        accuracy = correct_count / total_preds if total_preds > 0 else 0
        
        if accuracy > 0.8:
            categories['perfect'].append(sample_idx)
        elif 0.2 < accuracy < 0.8:
            categories['mixed'].append(sample_idx)
        else:
            categories['challenging'].append(sample_idx)
        
        if has_rare:
            categories['rare_organs'].append(sample_idx)
        if organ_count >= 4:
            categories['multi_organ'].append(sample_idx)
    
    # Select diverse examples
    selected = []
    
    # Try to get at least one from each category
    for category, indices in categories.items():
        if indices and len(selected) < num_examples:
            # Take up to 2 from each category
            n_take = min(2, len(indices), num_examples - len(selected))
            selected.extend(random.sample(indices, n_take))
    
    # Fill remaining with random samples if needed
    while len(selected) < num_examples:
        remaining = [idx for idx in sample_indices[:200] if idx not in selected]
        if remaining:
            selected.append(random.choice(remaining))
        else:
            break
    
    return selected[:num_examples]


def create_example_figure(
    dataset,
    adapter: CholecSeg8kAdapter,
    sample_idx: int,
    predictions: Dict[str, Dict],
    model_display_names: Dict[str, str],
    output_path: Path
):
    """Create a single example figure showing image, GT, and predictions."""
    
    # Load the example
    example = dataset['train'][sample_idx]
    image = np.array(example['image'])
    masks = adapter.parse_color_mask_to_class_id(example['color_mask'])
    
    # Determine which organs are present
    present_organs = []
    organ_masks = {}
    
    for organ_idx in range(1, 13):  # Skip background
        organ_name = ID2LABEL[organ_idx]
        organ_mask = (masks == organ_idx).astype(np.float32)
        
        if organ_mask.sum() > 50:  # Organ is present
            present_organs.append(organ_name)
            organ_masks[organ_name] = organ_mask
    
    # Create figure with subplots for each model
    num_models = len(predictions)
    fig, axes = plt.subplots(1, num_models + 2, figsize=(3 * (num_models + 2), 3))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth with organ masks
    axes[1].imshow(image, alpha=0.7)
    
    # Create colored overlay for each organ
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    for i, (organ_name, mask) in enumerate(organ_masks.items()):
        color = colors[i % 12][:3]
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = [*color, 0.4]
        axes[1].imshow(colored_mask)
    
    axes[1].set_title("Ground Truth", fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    # Model predictions
    for idx, (model_name, model_preds) in enumerate(predictions.items()):
        ax = axes[idx + 2]
        ax.imshow(image)
        
        # Plot predicted points for each organ
        if sample_idx in model_preds:
            for organ_name in present_organs:
                if organ_name in model_preds[sample_idx]:
                    pred = model_preds[sample_idx][organ_name]
                    y_pred = pred.get('y_pred', {})
                    
                    if y_pred.get('present', 0) == 1 and 'point_canvas' in y_pred:
                        x, y = y_pred['point_canvas']
                        
                        # Check if point is in mask (hit or miss)
                        is_hit = pred.get('hits', {}).get('hit_point', False)
                        
                        # Plot point
                        color = 'green' if is_hit else 'red'
                        marker = 'o' if is_hit else 'x'
                        ax.plot(x, y, marker=marker, color=color, markersize=8, 
                               markeredgewidth=2, markeredgecolor='white')
        
        # Model name
        display_name = model_display_names.get(model_name, model_name)
        ax.set_title(display_name, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=8, label='Correct', markeredgewidth=2, markeredgecolor='white'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                  markersize=8, label='Incorrect', markeredgewidth=2, markeredgecolor='white')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
              frameon=False, bbox_to_anchor=(0.5, -0.05))
    
    # Add sample info
    organ_list = ", ".join(present_organs[:3])
    if len(present_organs) > 3:
        organ_list += f" (+{len(present_organs)-3} more)"
    fig.suptitle(f"Sample {sample_idx}: {organ_list}", fontsize=11, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def create_grid_figure(
    dataset,
    adapter: CholecSeg8kAdapter,
    sample_indices: List[int],
    predictions: Dict[str, Dict],
    model_display_names: Dict[str, str],
    output_path: Path
):
    """Create a grid figure showing multiple examples."""
    
    num_examples = len(sample_indices)
    num_models = len(predictions)
    
    # Create grid: rows for examples, columns for image + GT + models
    fig, axes = plt.subplots(num_examples, num_models + 2, 
                             figsize=(2.5 * (num_models + 2), 2.5 * num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for row, sample_idx in enumerate(sample_indices):
        # Load the example
        example = dataset['train'][sample_idx]
        image = np.array(example['image'])
        masks = adapter.parse_color_mask_to_class_id(example['color_mask'])
        
        # Get present organs
        present_organs = []
        organ_masks = {}
        
        for organ_idx in range(1, 13):
            organ_name = ID2LABEL[organ_idx]
            organ_mask = (masks == organ_idx).astype(np.float32)
            
            if organ_mask.sum() > 50:
                present_organs.append(organ_name)
                organ_masks[organ_name] = organ_mask
        
        # Original image
        axes[row, 0].imshow(image)
        if row == 0:
            axes[row, 0].set_title("Input", fontsize=9, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Ground truth
        axes[row, 1].imshow(image, alpha=0.7)
        
        # Overlay organ masks with colored overlays instead of contours
        colors = plt.cm.tab20(np.linspace(0, 1, 12))
        for i, (organ_name, mask) in enumerate(organ_masks.items()):
            color = colors[i % 12][:3]
            
            # Create colored overlay for the mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0] = [*color, 0.3]  # Semi-transparent
            axes[row, 1].imshow(colored_mask)
        
        if row == 0:
            axes[row, 1].set_title("GT", fontsize=9, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Model predictions
        for col, (model_name, model_preds) in enumerate(predictions.items()):
            ax = axes[row, col + 2]
            ax.imshow(image)
            
            # Count hits and misses
            hits = 0
            misses = 0
            
            if sample_idx in model_preds:
                for organ_name in present_organs:
                    if organ_name in model_preds[sample_idx]:
                        pred = model_preds[sample_idx][organ_name]
                        y_pred = pred.get('y_pred', {})
                        
                        if y_pred.get('present', 0) == 1 and 'point_canvas' in y_pred:
                            x, y = y_pred['point_canvas']
                            
                            # Scale coordinates if needed (original is 224x224)
                            if image.shape[0] != 224:
                                scale_y = image.shape[0] / 224
                                scale_x = image.shape[1] / 224
                                x = int(x * scale_x)
                                y = int(y * scale_y)
                            
                            # Check if point is in mask
                            is_hit = False
                            if organ_name in organ_masks:
                                mask = organ_masks[organ_name]
                                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                                    is_hit = mask[y, x] > 0
                            
                            # Plot point
                            if is_hit:
                                ax.plot(x, y, 'o', color='lime', markersize=6,
                                       markeredgewidth=1.5, markeredgecolor='white')
                                hits += 1
                            else:
                                ax.plot(x, y, 'x', color='red', markersize=6,
                                       markeredgewidth=1.5, markeredgecolor='white')
                                misses += 1
            
            # Add hit/miss count as text
            if hits > 0 or misses > 0:
                text = f"✓{hits}"
                if misses > 0:
                    text += f" ✗{misses}"
                ax.text(0.95, 0.05, text, transform=ax.transAxes,
                       fontsize=8, color='white', ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
            
            if row == 0:
                display_name = model_display_names.get(model_name, model_name)
                ax.set_title(display_name, fontsize=9, fontweight='bold')
            ax.axis('off')
        
        # Add organ labels on the left
        organ_text = ", ".join(present_organs[:2])
        if len(present_organs) > 2:
            organ_text += f" +{len(present_organs)-2}"
        axes[row, 0].text(-0.1, 0.5, organ_text, transform=axes[row, 0].transAxes,
                         fontsize=8, ha='right', va='center', rotation=0)
    
    # Add overall legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                  markersize=6, label='Correct', markeredgewidth=1.5, markeredgecolor='white'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                  markersize=6, label='Incorrect', markeredgewidth=1.5, markeredgecolor='white')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle("Organ Pointing Examples", fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.02)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate example visualizations for paper")
    parser.add_argument("--results-dir", type=str, default="results/pointing_original",
                       help="Directory containing evaluation results")
    parser.add_argument("--output-dir", type=str, 
                       default="/shared_data0/weiqiuy/llm_cholec_organ_paper/images/examples",
                       help="Output directory for generated figures")
    parser.add_argument("--num-examples", type=int, default=8,
                       help="Number of examples to generate")
    parser.add_argument("--models", type=str, nargs="+",
                       default=["gpt-4.1", "gemini-2.0-flash", "claude-sonnet-4-20250514"],
                       help="Model names to include")
    parser.add_argument("--eval-type", type=str, default="zero_shot",
                       choices=["zero_shot", "fewshot_standard", "fewshot_hard_negatives"],
                       help="Evaluation type to visualize")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for example selection")
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model display names
    model_display_names = {
        "gpt-4.1": "GPT-4.1",
        "gemini-2.0-flash": "Gemini-2.0",
        "claude-sonnet-4-20250514": "Claude-4",
        "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5",
        "mistralai/Pixtral-12B-2409": "Pixtral"
    }
    
    print(f"Loading dataset...")
    dataset = load_dataset("minwoosun/CholecSeg8k")
    adapter = CholecSeg8kAdapter()
    print(f"✓ Dataset loaded")
    
    print(f"Loading predictions from {results_dir}...")
    predictions = {}
    for model_name in args.models:
        preds = load_predictions(results_dir, model_name, args.eval_type)
        if preds:
            predictions[model_name] = preds
            print(f"  Loaded {len(preds)} samples for {model_name}")
        else:
            print(f"  WARNING: No predictions found for {model_name}")
    
    if not predictions:
        print("ERROR: No predictions loaded!")
        sys.exit(1)
    
    print(f"\nSelecting {args.num_examples} interesting examples...")
    selected_indices = select_interesting_examples(
        dataset, adapter, predictions, args.num_examples, args.seed
    )
    print(f"Selected samples: {selected_indices}")
    
    # Generate grid figure (main figure for paper)
    print(f"\nGenerating grid figure...")
    grid_output = output_dir / f"examples_grid_{args.eval_type}.pdf"
    create_grid_figure(
        dataset, adapter, selected_indices, predictions,
        model_display_names, grid_output
    )
    print(f"  Saved to {grid_output}")
    
    # Also generate individual figures for each example (for flexibility)
    print(f"\nGenerating individual example figures...")
    for i, sample_idx in enumerate(selected_indices):
        output_path = output_dir / f"example_{i+1}_sample{sample_idx}_{args.eval_type}.pdf"
        create_example_figure(
            dataset, adapter, sample_idx, predictions,
            model_display_names, output_path
        )
        print(f"  Example {i+1}: {output_path.name}")
    
    print(f"\n✓ Generated {args.num_examples} examples in {output_dir}")
    print("\nTo include in LaTeX:")
    print(f"\\includegraphics[width=\\textwidth]{{{grid_output.relative_to(output_dir.parent)}}}")


if __name__ == "__main__":
    main()