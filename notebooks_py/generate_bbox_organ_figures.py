#!/usr/bin/env python3
"""
Generate bounding box visualizations for specific organs across models.
Shows ground truth and predicted bounding boxes for:
- Gastrointestinal Tract
- Gallbladder  
- L-hook Electrocautery
"""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import sys
sys.path.append('/shared_data0/weiqiuy/llm_cholec_organ/src')
from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter

# Configuration
DATASET_NAME = "cholecseg8k"
RESULTS_DIR = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick/zeroshot_combined")
BASE_OUTPUT_DIR = Path("/shared_data0/weiqiuy/llm_cholec_organ/notebooks/images/bbox_examples")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target organs
TARGET_ORGANS = ["Gastrointestinal Tract", "Gallbladder", "L-hook Electrocautery"]

# Model mapping
MODEL_MAPPING = {
    "gpt-4.1": "GPT",
    "gemini-2.0-flash": "Gemini", 
    "claude-sonnet-4-20250514": "Claude",
    "llava-hf_llava-v1.6-mistral-7b-hf": "Llava",
    "mistralai_Pixtral-12B-2409": "Pixtral",
    "Qwen_Qwen2.5-VL-7B-Instruct": "Qwen",
    "peskavlp": "PeskaVLP",
    "raso": "RASO",
    "cholenet": "CholeNet",
    "gonogonet": "GoNoGoNet"
}

# Colors for visualization
GREEN = (49, 163, 84)  # Ground truth
RED = (214, 39, 40)     # Prediction
BLUE = (31, 119, 180)   # Alternative

def load_dataset_adapter():
    """Load the local CholecSeg8k dataset adapter."""
    print("Loading CholecSeg8k local dataset...")
    adapter = CholecSeg8kLocalAdapter(
        data_dir="/shared_data0/weiqiuy/datasets/cholecseg8k"
    )
    return adapter

def load_image(adapter, idx):
    """Load original image from dataset using global index."""
    # The indices in the result files ARE global indices
    # We use them directly with get_example_by_global_index
    try:
        print(f"  Loading image for global index: {idx}")
        example = adapter.get_example_by_global_index(idx)
        return example['image']
    except Exception as e:
        print(f"Error loading image for global index {idx}: {e}")
        print(f"  Error details: {str(e)}")
        # Try to get any valid test example as fallback
        try:
            test_indices = adapter.get_test_indices()
            if test_indices:
                fallback_idx = test_indices[0]
                print(f"  Using fallback: test index {fallback_idx}")
                example = adapter.get_example_by_global_index(fallback_idx)
                return example['image']
        except:
            pass
        raise ValueError(f"Could not load image for index {idx}")

def find_best_examples():
    """Find test examples that have all three target organs present."""
    candidates = defaultdict(lambda: {"organs": set(), "models": set()})
    
    # Scan through test files to find examples with target organs
    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        if model_name not in MODEL_MAPPING:
            continue
            
        for test_file in model_dir.glob("test_*.json"):
            try:
                with open(test_file, 'r') as f:
                    data = json.load(f)
                
                idx = data.get('sample_idx', int(test_file.stem.split('_')[1]))
                
                # Check which target organs are present in ground truth
                bboxes_true = data.get('bboxes_true', {})
                for organ_name in TARGET_ORGANS:
                    if organ_name in bboxes_true and bboxes_true[organ_name]:
                        candidates[idx]["organs"].add(organ_name)
                        candidates[idx]["models"].add(model_name)
                            
            except Exception as e:
                continue
    
    # Find examples with all three organs
    best_examples = []
    for idx, info in candidates.items():
        if len(info["organs"]) == len(TARGET_ORGANS):
            best_examples.append((idx, len(info["models"])))
    
    # Sort by number of models that have data
    best_examples.sort(key=lambda x: x[1], reverse=True)
    
    if best_examples:
        return best_examples[0][0]  # Return index with most model coverage
    elif candidates:
        # If no example has all three, pick one with most organs
        sorted_candidates = sorted(candidates.items(), 
                                 key=lambda x: (len(x[1]["organs"]), len(x[1]["models"])),
                                 reverse=True)
        return sorted_candidates[0][0]
    else:
        # Fallback to first available test file
        return 1

def load_organ_data(idx, model_name, organ_name):
    """Load bounding box data for a specific organ from a model's results."""
    model_dir = RESULTS_DIR / model_name
    test_file = model_dir / f"test_{idx:05d}.json"
    
    if not test_file.exists():
        return None, None
    
    try:
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        # Get bounding boxes directly from bboxes_true and bboxes_pred
        bboxes_true = data.get('bboxes_true', {})
        bboxes_pred = data.get('bboxes_pred', {})
        
        gt_bbox = None
        pred_bbox = None
        
        # Get ground truth bbox
        if organ_name in bboxes_true and bboxes_true[organ_name]:
            # Take the first bbox if there are multiple
            gt_bbox = bboxes_true[organ_name][0]
        
        # Get predicted bbox
        if organ_name in bboxes_pred and bboxes_pred[organ_name]:
            # Take the first bbox if there are multiple
            pred_bbox = bboxes_pred[organ_name][0]
        
        return gt_bbox, pred_bbox
                
    except Exception as e:
        print(f"Error loading {test_file}: {e}")
        
    return None, None

def draw_bbox(draw, bbox, color, width=3, label=None):
    """Draw a bounding box on the image."""
    if bbox is None or len(bbox) != 4:
        return
    
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    for i in range(width):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
    
    # Add label if provided
    if label:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = None
        
        # Draw label background
        if font:
            bbox_text = draw.textbbox((x1, y1-22), label, font=font)
        else:
            bbox_text = (x1, y1-22, x1+len(label)*8, y1-5)
        
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1+2, y1-20), label, fill="white", font=font)

def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes."""
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def generate_organ_images(adapter, example_idx):
    """Generate visualization images for each organ-model pair."""
    
    # Create output directory for this specific example
    output_dir = BASE_OUTPUT_DIR / f"{DATASET_NAME}_{example_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {output_dir}")
    
    base_img = load_image(adapter, example_idx)
    
    for organ_name in TARGET_ORGANS:
        print(f"\nGenerating images for {organ_name}...")
        
        # Generate ground truth image
        gt_img = base_img.copy()
        draw = ImageDraw.Draw(gt_img)
        
        # Get ground truth bbox from any model (they should all have same GT)
        for model_name in MODEL_MAPPING.keys():
            gt_bbox, _ = load_organ_data(example_idx, model_name, organ_name)
            if gt_bbox:
                draw_bbox(draw, gt_bbox, GREEN, width=12, label="GT")  # 4x thicker (was 3)
                break
        
        # Save ground truth image
        organ_safe = organ_name.replace(" ", "_").replace("-", "_")
        gt_path = output_dir / f"GT_{organ_safe}.png"
        gt_img.save(gt_path)
        print(f"  Saved GT: {gt_path.name}")
        
        # Generate prediction images for each model
        for model_name, display_name in MODEL_MAPPING.items():
            pred_img = base_img.copy()
            draw = ImageDraw.Draw(pred_img)
            
            gt_bbox, pred_bbox = load_organ_data(example_idx, model_name, organ_name)
            
            # Draw ground truth in green
            if gt_bbox:
                draw_bbox(draw, gt_bbox, GREEN, width=8)  # 4x thicker (was 2)
            
            # Draw prediction in red
            if pred_bbox:
                draw_bbox(draw, pred_bbox, RED, width=12)  # 4x thicker (was 3)
                
                # Calculate and add IoU
                if gt_bbox:
                    iou = calculate_iou(gt_bbox, pred_bbox)
                    # Add IoU text
                    draw.rectangle([pred_img.width-80, 5, pred_img.width-5, 25], fill=(0,0,0,180))
                    draw.text((pred_img.width-75, 7), f"IoU: {iou:.2f}", fill="white")
            
            # Save prediction image
            pred_path = output_dir / f"{display_name}_{organ_safe}.png"
            pred_img.save(pred_path)
            print(f"  Saved {display_name}: {pred_path.name}")

def generate_latex_table(example_idx):
    """Generate LaTeX table code for the visualizations."""
    
    # Organ names for column headers
    organ_headers = [o.replace(" ", "_").replace("-", "_") for o in TARGET_ORGANS]
    
    # Folder name for this example
    folder = f"{DATASET_NAME}_{example_idx}"
    
    # Build LaTeX
    header = " & " + " & ".join(TARGET_ORGANS) + r" \\"
    rows = []
    
    # Ground Truth row
    gt_cells = [rf"\includegraphics[width=0.28\linewidth]{{images/bbox_examples/{folder}/GT_{o}.png}}" 
                for o in organ_headers]
    rows.append(rf"Ground Truth & " + " & ".join(gt_cells) + r" \\")
    
    # Model rows
    for display_name in MODEL_MAPPING.values():
        cells = [rf"\includegraphics[width=0.28\linewidth]{{images/bbox_examples/{folder}/{display_name}_{o}.png}}" 
                 for o in organ_headers]
        rows.append(rf"{display_name} & " + " & ".join(cells) + r" \\")
    
    latex = r"""% Requires: \usepackage{graphicx}, \usepackage{booktabs}
\begin{table*}[t]
\centering
\resizebox{\textwidth}{!}{%
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{l|ccc}
\toprule
Model & """ + " & ".join(TARGET_ORGANS) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
} % end resizebox
\caption{Bounding box predictions for three representative organs. Green boxes show ground truth, red boxes show model predictions. IoU values are displayed in the top-right corner when both GT and prediction are present.}
\label{tab:bbox-organ-comparison}
\end{table*}
"""
    
    # Save LaTeX file in the example folder
    output_dir = BASE_OUTPUT_DIR / f"{DATASET_NAME}_{example_idx}"
    latex_path = output_dir / f"bbox_table_{example_idx}.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {latex_path}")
    print("\nLaTeX code:")
    print(latex)

def main():
    """Main function to generate all visualizations."""
    
    # Load dataset adapter first
    adapter = load_dataset_adapter()
    
    # Process multiple examples
    examples = [
        (840, "has all three target organs in ground truth"),
        (4081, "best IoU for Gastrointestinal Tract (0.75)")
    ]
    
    for example_idx, description in examples:
        print("=" * 80)
        print(f"Processing example {example_idx}: {description}")
        print("=" * 80)
        
        print("\nGenerating visualization images...")
        generate_organ_images(adapter, example_idx)
        
        print("\nGenerating LaTeX table...")
        generate_latex_table(example_idx)
        
        output_folder = BASE_OUTPUT_DIR / f"{DATASET_NAME}_{example_idx}"
        print(f"\nDone! Images saved to: {output_folder}")
        print()

if __name__ == "__main__":
    main()