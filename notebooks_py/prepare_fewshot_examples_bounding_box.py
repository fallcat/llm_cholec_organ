#!/usr/bin/env python
"""
Prepare Few-Shot Examples with Bounding Boxes for Organ Detection Evaluation
This script creates balanced test sets and few-shot example plans with bounding boxes
instead of single points for organ localization.
"""

# Cell 1: Setup and imports
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from scipy import ndimage
from typing import List, Tuple, Optional, Dict

# Add src to path - fix the path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Import the few-shot selection module
from few_shot_selection import (
    build_presence_matrix,
    select_balanced_indices, 
    save_balanced_indices,
    load_balanced_indices,
    load_fewshot_plan
)

from datasets import load_dataset
import numpy as np

print("✓ Modules imported")

# Cell 2: Configuration
CONFIG = {
    "n_test_samples": 100,  # Number of balanced test samples
    "n_pos_examples": 1,    # Positive examples per organ
    "n_neg_easy": 1,        # Easy negative examples per organ  
    "n_neg_hard": 1,        # Hard negative examples per organ
    "min_pixels": 50,       # Minimum pixels for organ presence
    "min_bbox_size": 20,    # Minimum bbox width/height
    "max_bboxes": 3,        # Maximum number of bboxes per organ (for disconnected segments)
    "seed": 42,             # Random seed for reproducibility
    "output_dir": Path(parent_dir) / "data_info" / "cholecseg8k"
}

print(f"Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Cell 3: Bounding box extraction functions
def extract_bounding_boxes_from_mask(
    lab_t: torch.Tensor, 
    class_id: int,
    min_pixels: int = 50,
    min_bbox_size: int = 20,
    max_bboxes: int = 3
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes from a segmentation mask for a given class.
    Returns boxes for disconnected segments.
    
    Args:
        lab_t: Label tensor (H, W) with class IDs
        class_id: Target organ class ID
        min_pixels: Minimum pixels for a valid segment
        min_bbox_size: Minimum width/height for a valid bbox
        max_bboxes: Maximum number of bboxes to return
        
    Returns:
        List of bounding boxes as (x1, y1, x2, y2) tuples
    """
    # Create binary mask for the target class
    mask = (lab_t == class_id).cpu().numpy().astype(np.uint8)
    
    if mask.sum() < min_pixels:
        return []
    
    # Find connected components
    labeled_mask, num_components = ndimage.label(mask)
    
    bboxes = []
    component_sizes = []
    
    # Extract bbox for each component
    for i in range(1, num_components + 1):
        component_mask = (labeled_mask == i)
        component_size = component_mask.sum()
        
        if component_size < min_pixels:
            continue
            
        # Find bbox coordinates
        rows, cols = np.where(component_mask)
        if len(rows) == 0 or len(cols) == 0:
            continue
            
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        
        # Check minimum size
        if (x2 - x1) < min_bbox_size or (y2 - y1) < min_bbox_size:
            continue
            
        bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        component_sizes.append(component_size)
    
    # Sort by component size (largest first) and limit to max_bboxes
    if len(bboxes) > max_bboxes:
        sorted_indices = np.argsort(component_sizes)[::-1][:max_bboxes]
        bboxes = [bboxes[i] for i in sorted_indices]
    
    return bboxes


def compute_single_bbox_from_mask(lab_t: torch.Tensor, class_id: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute a single bounding box that encompasses all pixels of a given class.
    
    Args:
        lab_t: Label tensor (H, W) with class IDs
        class_id: Target organ class ID
        
    Returns:
        Bounding box as (x1, y1, x2, y2) or None if no pixels found
    """
    mask = (lab_t == class_id).cpu().numpy()
    
    if not mask.any():
        return None
    
    rows, cols = np.where(mask)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    
    return (int(x1), int(y1), int(x2), int(y2))


def sample_near_miss_bbox(
    lab_t: torch.Tensor,
    class_id: int,
    distance_range: Tuple[int, int] = (10, 30)
) -> Optional[Tuple[int, int, int, int]]:
    """
    Create a 'near-miss' bounding box that is close to but doesn't overlap with the organ.
    
    Args:
        lab_t: Label tensor
        class_id: Target organ class ID
        distance_range: Min and max distance from organ boundary
        
    Returns:
        Near-miss bbox as (x1, y1, x2, y2) or None
    """
    # Get the true bbox first
    true_bbox = compute_single_bbox_from_mask(lab_t, class_id)
    if true_bbox is None:
        return None
    
    x1, y1, x2, y2 = true_bbox
    width = x2 - x1
    height = y2 - y1
    H, W = lab_t.shape
    
    # Create dilated mask to find nearby regions
    mask = (lab_t == class_id).cpu().numpy().astype(np.uint8)
    dilated_mask = ndimage.binary_dilation(mask, iterations=distance_range[1])
    eroded_dilated = ndimage.binary_erosion(dilated_mask, iterations=distance_range[0])
    
    # Find region that is close but not overlapping
    near_region = eroded_dilated & ~mask
    
    if not near_region.any():
        return None
    
    # Sample a random position in the near region
    near_pixels = np.where(near_region)
    if len(near_pixels[0]) == 0:
        return None
    
    idx = np.random.choice(len(near_pixels[0]))
    center_y = near_pixels[0][idx]
    center_x = near_pixels[1][idx]
    
    # Create a bbox with similar size but offset position
    # Add some random variation to size (±20%)
    size_var = np.random.uniform(0.8, 1.2)
    new_width = int(width * size_var)
    new_height = int(height * size_var)
    
    # Create bbox centered at the sampled point
    nm_x1 = max(0, center_x - new_width // 2)
    nm_y1 = max(0, center_y - new_height // 2)
    nm_x2 = min(W - 1, nm_x1 + new_width)
    nm_y2 = min(H - 1, nm_y1 + new_height)
    
    # Ensure the near-miss bbox doesn't overlap with the true organ
    true_mask = mask[nm_y1:nm_y2+1, nm_x1:nm_x2+1]
    if true_mask.any():
        # Shift the bbox to avoid overlap
        if center_x < x1:  # Near-miss is to the left
            nm_x2 = min(x1 - distance_range[0], nm_x2)
            nm_x1 = nm_x2 - new_width
        elif center_x > x2:  # Near-miss is to the right
            nm_x1 = max(x2 + distance_range[0], nm_x1)
            nm_x2 = nm_x1 + new_width
        elif center_y < y1:  # Near-miss is above
            nm_y2 = min(y1 - distance_range[0], nm_y2)
            nm_y1 = nm_y2 - new_height
        else:  # Near-miss is below
            nm_y1 = max(y2 + distance_range[0], nm_y1)
            nm_y2 = nm_y1 + new_height
    
    # Final bounds check
    nm_x1 = max(0, nm_x1)
    nm_y1 = max(0, nm_y1)
    nm_x2 = min(W - 1, nm_x2)
    nm_y2 = min(H - 1, nm_y2)
    
    if nm_x2 <= nm_x1 or nm_y2 <= nm_y1:
        return None
    
    return (int(nm_x1), int(nm_y1), int(nm_x2), int(nm_y2))


# Cell 4: Build few-shot plan with bounding boxes (standalone)
def build_fewshot_plan_with_bboxes(
    dataset,
    split: str = "train",
    balanced_indices: List[int] = None,
    n_pos: int = 1,
    n_neg_easy: int = 1,
    n_neg_hard: int = 0,
    min_pixels: int = 50,
    seed: int = 42,
    max_bboxes_per_organ: int = 3,
    min_bbox_size: int = 20
) -> Dict:
    """
    Build a few-shot learning plan with bounding boxes for organ localization.
    Uses the same seed as the point-based version to select the same examples.
    
    Args:
        dataset: The dataset
        split: Dataset split to use
        balanced_indices: Indices to exclude (test set)
        n_pos: Number of positive examples per organ
        n_neg_easy: Number of easy negative examples
        n_neg_hard: Number of hard negative examples
        min_pixels: Minimum pixels for organ presence
        seed: Random seed (use same as point version for consistency)
        max_bboxes_per_organ: Maximum bboxes to extract per organ
        min_bbox_size: Minimum bbox size
        
    Returns:
        Plan with bboxes for organ localization
    """
    from cholecseg8k_utils import (
        ID2LABEL, LABEL_IDS, 
        example_to_tensors, 
        labels_to_presence_vector
    )
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load cached presence matrix if available, otherwise build it
    # Get parent directory for paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir_local = os.path.dirname(script_dir)
    cache_file = Path(parent_dir_local) / "data_info" / "cholecseg8k" / f"presence_matrix_{split}_8080.npz"
    indices_file = Path(parent_dir_local) / "data_info" / "cholecseg8k" / f"presence_indices_{split}_8080.json"
    
    if cache_file.exists() and indices_file.exists():
        # Use cached presence matrix
        data = np.load(cache_file)
        Y_pool = data['Y']
        with open(indices_file, 'r') as f:
            pool_indices = json.load(f)['indices']
    else:
        # Build presence matrix for available pool
        Y_pool, pool_indices = build_presence_matrix(
            dataset, split, indices=None, min_pixels=min_pixels
        )
    
    # Exclude balanced test set indices
    if balanced_indices:
        # Create mapping for efficient lookup
        pool_idx_map = {idx: i for i, idx in enumerate(pool_indices)}
        # Filter indices and matrix
        filtered_rows = []
        filtered_indices = []
        for idx in pool_indices:
            if idx not in balanced_indices:
                filtered_indices.append(idx)
                filtered_rows.append(pool_idx_map[idx])
        Y_pool = Y_pool[filtered_rows]
        pool_indices = filtered_indices
    
    # Confuser mapping for hard negatives (same as point version)
    CONFUSER_MAP = {
        2: [4, 6], 3: [1, 10], 4: [2, 6], 5: [9], 6: [2, 4],
        7: [8], 8: [7, 11], 9: [5], 10: [3], 11: [8], 12: [6]
    }
    
    plan = {}
    
    for class_idx, class_id in enumerate(LABEL_IDS):
        organ_name = ID2LABEL[class_id]
        
        # Find positive and negative examples
        pos_mask = Y_pool[:, class_idx] == 1
        neg_mask = Y_pool[:, class_idx] == 0
        
        pos_indices = [pool_indices[i] for i in np.where(pos_mask)[0]]
        neg_indices = [pool_indices[i] for i in np.where(neg_mask)[0]]
        
        # Select diverse positive examples with bboxes
        selected_pos = []
        used_videos = set()
        
        for _ in range(min(n_pos, len(pos_indices))):
            candidates = []
            for idx in pos_indices:
                if idx not in [p["idx"] for p in selected_pos]:
                    video_id = f"video{idx // 1000:02d}"
                    if video_id not in used_videos or len(candidates) < 5:
                        candidates.append(idx)
            
            if not candidates:
                break
                
            selected_idx = int(np.random.choice(candidates))  # Convert to int
            used_videos.add(f"video{selected_idx // 1000:02d}")
            
            # Extract bounding boxes for this organ
            example = dataset[split][selected_idx]
            img_t, lab_t = example_to_tensors(example)
            
            # Get bounding boxes for all segments
            bboxes = extract_bounding_boxes_from_mask(
                lab_t, class_id,
                min_pixels=min_pixels,
                min_bbox_size=min_bbox_size,
                max_bboxes=max_bboxes_per_organ
            )
            
            # If no valid bboxes from components, try single bbox
            if not bboxes:
                single_bbox = compute_single_bbox_from_mask(lab_t, class_id)
                if single_bbox:
                    bboxes = [single_bbox]
            
            if bboxes:
                selected_pos.append({
                    "idx": selected_idx,
                    "bboxes": bboxes,
                    "main_bbox": bboxes[0]  # Primary bbox (largest component)
                })
        
        # Select easy negative examples (same as point version)
        selected_neg_easy = []
        if neg_indices:
            selected_neg_easy = [int(idx) for idx in np.random.choice(
                neg_indices, 
                size=min(n_neg_easy, len(neg_indices)),
                replace=False
            )]
        
        # Select hard negative examples (with confuser organs)
        selected_neg_hard = []
        if n_neg_hard > 0 and class_id in CONFUSER_MAP:
            confuser_ids = CONFUSER_MAP[class_id]
            hard_candidates = []
            
            for idx in neg_indices:
                for conf_id in confuser_ids:
                    if conf_id in LABEL_IDS:
                        conf_idx = LABEL_IDS.index(conf_id)
                        pool_idx = pool_indices.index(idx) if idx in pool_indices else -1
                        if pool_idx >= 0 and Y_pool[pool_idx, conf_idx] == 1:
                            hard_candidates.append(idx)
                            break
            
            if hard_candidates:
                selected_neg_hard = [int(idx) for idx in np.random.choice(
                    hard_candidates,
                    size=min(n_neg_hard, len(hard_candidates)),
                    replace=False
                )]
        
        plan[str(class_id)] = {
            "name": organ_name,
            "positives": selected_pos,
            "negatives_easy": selected_neg_easy,
            "negatives_hard": selected_neg_hard,
            "pos_available": len(pos_indices),
            "neg_available": len(neg_indices)
        }
    
    return {
        "split": split,
        "config": {
            "n_pos": n_pos,
            "n_neg_easy": n_neg_easy,
            "n_neg_hard": n_neg_hard,
            "min_pixels": min_pixels,
            "max_bboxes": max_bboxes_per_organ,
            "min_bbox_size": min_bbox_size,
            "seed": seed
        },
        "excluded_indices": balanced_indices if balanced_indices else [],
        "plan": plan
    }




def save_bbox_fewshot_plan(plan: Dict, output_dir: Path) -> Path:
    """Save bounding box few-shot plan to JSON file."""
    config = plan["config"]
    
    # Build filename based on configuration
    parts = [f"fewshot_bbox_plan", plan["split"]]
    parts.append(f"pos{config['n_pos']}")
    
    if config.get("n_neg_easy", 0) > 0:
        parts.append(f"nege{config['n_neg_easy']}")
    if config.get("n_neg_hard", 0) > 0:
        parts.append(f"negh{config['n_neg_hard']}")
    if config.get("n_near_miss", 0) > 0:
        parts.append(f"nm{config['n_near_miss']}")
    if config.get("max_bboxes"):
        parts.append(f"maxbb{config['max_bboxes']}")
        
    parts.append(f"mp{config['min_pixels']}")
    parts.append(f"seed{config['seed']}")
    
    if plan.get("excluded_indices"):
        parts.append(f"excl{len(plan['excluded_indices'])}")
    
    filename = "_".join(parts) + ".json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(plan, f, indent=2)
    
    return filepath


# Cell 5: Load dataset
print("\n📊 Loading CholecSeg8k dataset...")
dataset = load_dataset("minwoosun/CholecSeg8k")
train_size = len(dataset['train'])
print(f"✓ Dataset loaded: {train_size} training samples")

# Cell 6: Build presence matrix
print("\n🔨 Building presence matrix for all training samples...")
print("This may take a few minutes...")

# Check if presence matrix is already cached
cache_file = CONFIG["output_dir"] / f"presence_matrix_train_{train_size}.npz"
indices_file = CONFIG["output_dir"] / f"presence_indices_train_{train_size}.json"

if cache_file.exists() and indices_file.exists():
    print("Loading cached presence matrix...")
    data = np.load(cache_file)
    Y = data['Y']
    with open(indices_file, 'r') as f:
        all_indices = json.load(f)['indices']
    print(f"✓ Loaded cached presence matrix: shape {Y.shape}")
else:
    Y, all_indices = build_presence_matrix(
        dataset, 
        split="train", 
        indices=None,  # Use all samples
        min_pixels=CONFIG["min_pixels"]
    )
    
    # Save for future use
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, Y=Y)
    with open(indices_file, 'w') as f:
        json.dump({'indices': all_indices}, f)
    print(f"✓ Built presence matrix: shape {Y.shape}")
    print(f"  Cached to: {cache_file}")

# Cell 7: Analyze organ distribution
print("\n📊 Organ presence statistics:")
from cholecseg8k_utils import ID2LABEL, LABEL_IDS

organ_counts = Y.sum(axis=0)
total_samples = Y.shape[0]

for i, class_id in enumerate(LABEL_IDS):
    organ_name = ID2LABEL[class_id]
    count = organ_counts[i]
    percentage = (count / total_samples) * 100
    print(f"  {organ_name:25} {count:5d} samples ({percentage:5.1f}%)")

# Cell 8: Load the SAME balanced test set as the point version
print(f"\n🎯 Loading balanced test samples...")

# Use the EXACT SAME file as prepare_fewshot_examples.py and eval_pointing_original_size.py
balanced_file = CONFIG["output_dir"] / f"balanced_indices_train_{CONFIG['n_test_samples']}.json"

if balanced_file.exists():
    print(f"✓ Using existing balanced indices: {balanced_file.name}")
    print("  Loading indices (same as point version)...")
    balanced_indices = load_balanced_indices(balanced_file)
    print(f"  Loaded {len(balanced_indices)} test samples")
else:
    print(f"❌ ERROR: Balanced indices file not found: {balanced_file}")
    print("  Please run prepare_fewshot_examples.py first to create the balanced test set")
    raise FileNotFoundError(f"Required file not found: {balanced_file}")

# Analyze balance
Y_balanced = Y[[all_indices.index(i) for i in balanced_indices]]
balanced_counts = Y_balanced.sum(axis=0)

print("Balanced set organ distribution:")
for i, class_id in enumerate(LABEL_IDS):
    organ_name = ID2LABEL[class_id]
    count = balanced_counts[i]
    percentage = (count / CONFIG["n_test_samples"]) * 100
    print(f"  {organ_name:25} {count:3d} samples ({percentage:5.1f}%)")

# Cell 9: Build few-shot plan with bounding boxes
print("\n📦 Building few-shot plan with bounding boxes...")
print("  Note: Using same seeds as point-based version for consistent example selection")

# Check if file already exists
expected_filename = f"fewshot_bbox_plan_train_pos{CONFIG['n_pos_examples']}_nege{CONFIG['n_neg_easy']}_negh{CONFIG['n_neg_hard']}_maxbb{CONFIG['max_bboxes']}_seed{CONFIG['seed'] + 2}_excl{CONFIG['n_test_samples']}.json"
bbox_plan_file = CONFIG["output_dir"] / expected_filename

if bbox_plan_file.exists():
    print(f"✓ Bbox few-shot plan already exists: {bbox_plan_file.name}")
    print("  Loading existing plan...")
    with open(bbox_plan_file, 'r') as f:
        plan_bbox = json.load(f)
else:
    print("  Building new bbox plan...")
    print(f"  Seed: {CONFIG['seed'] + 2} (same as point version for consistency)")
    
    plan_bbox = build_fewshot_plan_with_bboxes(
        dataset,
        split="train",
        balanced_indices=balanced_indices,
        n_pos=CONFIG["n_pos_examples"],
        n_neg_easy=CONFIG["n_neg_easy"],
        n_neg_hard=CONFIG["n_neg_hard"],
        min_pixels=CONFIG["min_pixels"],
        seed=CONFIG["seed"] + 2,  # Same seed as point version
        max_bboxes_per_organ=CONFIG["max_bboxes"],
        min_bbox_size=CONFIG["min_bbox_size"]
    )
    
    # Save the plan
    plan_bbox_file = save_bbox_fewshot_plan(plan_bbox, CONFIG["output_dir"])
    print(f"✓ Saved bbox plan to: {bbox_plan_file}")

# Cell 10: Note about near-miss bboxes
print("\n📦 Note about near-miss bounding boxes:")
print("  Near-miss bboxes are challenging to generate automatically")
print("  They require bboxes that are close to but don't overlap with organs")
print("  This is more complex than near-miss points")
print("  For now, using standard hard negatives with confuser organs")

# The standard bbox plan already includes hard negatives with confuser organs
# which serve a similar purpose to near-miss examples
plan_nearmiss_bbox = None

# Print summary
print("\nBbox plan summary:")
for class_id_str, info in plan_bbox["plan"].items():
    name = info["name"]
    n_pos = len(info["positives"])
    n_neg_e = len(info["negatives_easy"])
    n_neg_h = len(info.get("negatives_hard", []))
    
    # Count total bboxes across positive examples
    total_bboxes = sum(len(p.get("bboxes", [])) for p in info["positives"])
    
    print(f"  {name:25} pos={n_pos} (bboxes={total_bboxes}), neg_easy={n_neg_e}, neg_hard={n_neg_h}")

# Cell 11: Visualize bounding box examples
print("\n📊 Visualizing bounding box few-shot examples...")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def visualize_bbox_fewshot_examples(dataset, plan, organ_names_to_show=None, max_organs=4):
    """
    Visualize few-shot examples with bounding boxes.
    """
    if organ_names_to_show is None:
        # Show first few organs with examples
        organ_names_to_show = []
        for class_id_str, info in plan["plan"].items():
            if len(info["positives"]) > 0:
                organ_names_to_show.append(info["name"])
                if len(organ_names_to_show) >= max_organs:
                    break
    
    n_organs = len(organ_names_to_show)
    fig_height = min(4 * n_organs, 20)
    fig = plt.figure(figsize=(18, fig_height))
    
    for organ_idx, organ_name in enumerate(organ_names_to_show):
        # Find the organ in the plan
        organ_info = None
        for class_id_str, info in plan["plan"].items():
            if info["name"] == organ_name:
                organ_info = info
                break
        
        if organ_info is None:
            continue
        
        n_pos = min(2, len(organ_info["positives"]))
        n_neg_easy = min(2, len(organ_info["negatives_easy"]))
        n_neg_hard = min(2, len(organ_info.get("negatives_hard", [])))
        
        col_idx = 0
        
        # Plot positive examples with bboxes
        for i in range(n_pos):
            ax = plt.subplot(n_organs, 6, organ_idx * 6 + col_idx + 1)
            pos_data = organ_info["positives"][i]
            idx = pos_data["idx"]
            
            # Load and display image
            example = dataset[plan["split"]][idx]
            img = example["image"]
            ax.imshow(img)
            
            # Draw bounding boxes
            if "bboxes" in pos_data:
                for j, bbox in enumerate(pos_data["bboxes"]):
                    x1, y1, x2, y2 = bbox
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='lime' if j == 0 else 'yellow',
                        facecolor='none', alpha=0.8
                    )
                    ax.add_patch(rect)
                    # Label the primary bbox
                    if j == 0:
                        ax.text(x1, y1-5, 'Main', color='lime', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            elif "bbox" in pos_data:
                x1, y1, x2, y2 = pos_data["bbox"]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='lime',
                    facecolor='none', alpha=0.8
                )
                ax.add_patch(rect)
            
            n_boxes = len(pos_data.get("bboxes", [pos_data.get("bbox", [])]))
            ax.set_title(f"✅ Positive {i+1}\n{n_boxes} bbox(es)\nidx: {idx}", 
                        fontsize=9, color='green')
            ax.axis('off')
            col_idx += 1
        
        # Plot easy negative examples
        for i in range(n_neg_easy):
            ax = plt.subplot(n_organs, 6, organ_idx * 6 + col_idx + 1)
            idx = organ_info["negatives_easy"][i]
            
            example = dataset[plan["split"]][idx]
            img = example["image"]
            ax.imshow(img)
            
            ax.set_title(f"❌ Easy Neg {i+1}\n(No {organ_name})\nidx: {idx}", 
                        fontsize=9, color='orange')
            ax.axis('off')
            col_idx += 1
        
        # Plot hard negative examples
        for i in range(n_neg_hard):
            ax = plt.subplot(n_organs, 6, organ_idx * 6 + col_idx + 1)
            idx = organ_info["negatives_hard"][i]
            
            example = dataset[plan["split"]][idx]
            img = example["image"]
            ax.imshow(img)
            
            ax.set_title(f"⚠️ Hard Neg {i+1}\n(Confuser present)\nidx: {idx}", 
                        fontsize=9, color='red')
            ax.axis('off')
            col_idx += 1
        
        # Add organ name on the left
        fig.text(0.02, 0.9 - organ_idx / n_organs - 0.05, organ_name, 
                fontsize=12, fontweight='bold', rotation=0)
    
    plt.suptitle(f"Bounding Box Few-Shot Examples\n"
                 f"(Green = main bbox, Yellow = additional segments)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    
    # Save figure
    viz_path = CONFIG["output_dir"] / "fewshot_bbox_visualization.png"
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    print(f"✓ Visualization saved to: {viz_path}")
    
    plt.show()
    return fig

# Visualize bbox examples
print("\nVisualizing bounding box few-shot examples:")
organs_to_show = ["Liver", "Gallbladder", "Fat", "Grasper"]
fig = visualize_bbox_fewshot_examples(
    dataset, 
    plan_bbox,
    organ_names_to_show=organs_to_show,
    max_organs=4
)

# Cell 12: Visualize near-miss bboxes
def visualize_nearmiss_bbox_examples(dataset, plan, organ_names_to_show=None, max_organs=4):
    """
    Visualize near-miss bbox examples showing true vs near-miss bboxes.
    """
    from cholecseg8k_utils import example_to_tensors
    
    if organ_names_to_show is None:
        organ_names_to_show = []
        for class_id_str, info in plan["plan"].items():
            if len(info.get("near_miss", [])) > 0:
                organ_names_to_show.append(info["name"])
                if len(organ_names_to_show) >= max_organs:
                    break
    
    n_organs = min(len(organ_names_to_show), max_organs)
    fig = plt.figure(figsize=(16, 5 * n_organs))
    
    for organ_idx, organ_name in enumerate(organ_names_to_show[:n_organs]):
        organ_info = None
        for class_id_str, info in plan["plan"].items():
            if info["name"] == organ_name:
                organ_info = info
                break
        
        if organ_info is None or len(organ_info.get("near_miss", [])) == 0:
            continue
        
        n_examples = min(3, len(organ_info["near_miss"]))
        
        for ex_idx in range(n_examples):
            nm_data = organ_info["near_miss"][ex_idx]
            idx = nm_data["idx"]
            nm_bbox = nm_data["bbox"]
            true_bbox = nm_data.get("true_bbox")
            
            example = dataset[plan["split"]][idx]
            img = example["image"]
            
            ax = plt.subplot(n_organs, n_examples, organ_idx * n_examples + ex_idx + 1)
            ax.imshow(img)
            
            # Draw true bbox (green)
            if true_bbox:
                x1, y1, x2, y2 = true_bbox
                rect_true = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='lime',
                    facecolor='none', alpha=0.8,
                    label='True bbox'
                )
                ax.add_patch(rect_true)
            
            # Draw near-miss bbox (red)
            if nm_bbox:
                x1, y1, x2, y2 = nm_bbox
                rect_nm = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red',
                    facecolor='none', alpha=0.8,
                    linestyle='--',
                    label='Near-miss bbox'
                )
                ax.add_patch(rect_nm)
            
            ax.set_title(f"{organ_name}\nNear-miss {ex_idx+1} (idx: {idx})", fontsize=10)
            ax.axis('off')
            
            if ex_idx == 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    plt.suptitle("Near-miss Bounding Boxes for Localization\n"
                 "Green = Correct bbox | Red (dashed) = Near-miss bbox",
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    viz_path = CONFIG["output_dir"] / "nearmiss_bbox_visualization.png"
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    print(f"✓ Near-miss bbox visualization saved to: {viz_path}")
    
    plt.show()
    return fig

if plan_nearmiss_bbox and "near_miss" in plan_nearmiss_bbox.get("plan", {}).get(str(LABEL_IDS[0]), {}):
    print("\n🔍 Visualizing near-miss bounding boxes:")
    fig_nm = visualize_nearmiss_bbox_examples(
        dataset,
        plan_nearmiss_bbox,
        organ_names_to_show=["Liver", "Gallbladder", "Fat", "Grasper"],
        max_organs=4
    )

# Cell 13: Summary statistics
print("\n" + "="*60)
print("✨ Bounding box few-shot preparation complete!")
print("="*60)
print(f"\nFiles created in {CONFIG['output_dir']}:")
print(f"  1. Balanced test set: {balanced_file.name}")
print(f"     - {CONFIG['n_test_samples']} samples for evaluation")
print(f"     - Same seed ensures same test set as point version")
print(f"  2. Bbox few-shot plan: {bbox_plan_file.name}")
print(f"     - Generated with seed {CONFIG['seed'] + 2}")
print(f"     - Same seed as point version ensures same examples selected")
print(f"     - Up to {CONFIG['max_bboxes']} boxes per organ for disconnected segments")
print(f"\n📌 Key Features:")
print("  • Standalone script (no dependency on point-based files)")
print("  • Uses same seed → selects same examples as point version")
print("  • Multiple bboxes per organ for disconnected segments")
print("  • Each organ can have 1-3 bounding boxes")
print("  • Bboxes extracted from actual organ masks")
print("  • Compatible with IoU-based evaluation metrics")
print(f"\n📊 Statistics:")
print(f"  • Total organs: {len(plan_bbox['plan'])}")
print(f"  • Examples per organ: {CONFIG['n_pos_examples']} positive, {CONFIG['n_neg_easy']} easy negative, {CONFIG['n_neg_hard']} hard negative")
print(f"  • Min pixels for valid segment: {CONFIG['min_pixels']}")
print(f"  • Min bbox size: {CONFIG['min_bbox_size']}px")
print("\nNext step: Run bbox-based evaluation with IoU metrics.")