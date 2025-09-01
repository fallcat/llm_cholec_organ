"""
Few-shot example selection for organ detection in CholecSeg8k.
Includes balanced sampling, positive/negative selection, and hard negative mining.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage as ndi
import torch.nn.functional as F

from cholecseg8k_utils import (
    ID2LABEL, LABEL2ID, LABEL_IDS,
    example_to_tensors, labels_to_presence_vector
)


def _select_diverse_indices(candidates: List[int], n_select: int, 
                           min_spacing: int = 100, 
                           avoid_indices: Optional[List[int]] = None,
                           rng: Optional[np.random.Generator] = None) -> List[int]:
    """
    Select indices that are spaced apart to ensure diversity.
    Videos often have consecutive frames that look similar.
    
    Args:
        candidates: List of candidate indices
        n_select: Number to select
        min_spacing: Minimum index distance between selected samples
        avoid_indices: Indices to avoid being close to
        rng: Random number generator
        
    Returns:
        List of selected indices
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if len(candidates) <= n_select:
        return candidates
    
    # Sort candidates for efficient selection
    sorted_candidates = sorted(candidates)
    selected = []
    avoid_set = set(avoid_indices) if avoid_indices else set()
    
    # Shuffle for randomness but maintain spacing
    shuffled_indices = list(range(len(sorted_candidates)))
    rng.shuffle(shuffled_indices)
    
    for idx in shuffled_indices:
        candidate = sorted_candidates[idx]
        
        # Check spacing from already selected
        too_close = False
        for sel in selected:
            if abs(candidate - sel) < min_spacing:
                too_close = True
                break
        
        # Check spacing from avoid list
        if not too_close and avoid_set:
            for avoid in avoid_set:
                if abs(candidate - avoid) < min_spacing:
                    too_close = True
                    break
        
        if not too_close:
            selected.append(candidate)
            if len(selected) >= n_select:
                break
    
    # If we couldn't get enough with spacing, gradually reduce requirement
    if len(selected) < n_select:
        reduced_spacing = min_spacing // 2
        while len(selected) < n_select and reduced_spacing > 10:
            for candidate in sorted_candidates:
                if candidate in selected:
                    continue
                    
                too_close = False
                for sel in selected:
                    if abs(candidate - sel) < reduced_spacing:
                        too_close = True
                        break
                
                if not too_close:
                    selected.append(candidate)
                    if len(selected) >= n_select:
                        break
            
            reduced_spacing = reduced_spacing // 2
    
    # Final fallback: random selection if still not enough
    if len(selected) < n_select:
        remaining = [c for c in candidates if c not in selected]
        if remaining:
            additional = rng.choice(remaining, min(n_select - len(selected), len(remaining)), replace=False)
            selected.extend(additional.tolist())
    
    return selected[:n_select]


def build_presence_matrix(dataset, split: str = "train", indices: Optional[List[int]] = None, 
                         min_pixels: int = 1) -> Tuple[np.ndarray, List[int]]:
    """
    Build presence matrix for organ detection.
    
    Args:
        dataset: HuggingFace dataset
        split: Dataset split to use
        indices: Specific indices to use (if None, use all)
        min_pixels: Minimum pixels for an organ to be considered present
        
    Returns:
        Y: Binary presence matrix [N, 12] for classes 1-12
        indices: List of dataset indices used
    """
    if indices is None:
        indices = list(range(len(dataset[split])))
    
    N = len(indices)
    K = 12  # Number of organ classes (excluding background)
    Y = np.zeros((N, K), dtype=np.uint8)
    
    for row_i, idx in enumerate(tqdm(indices, desc="Building presence matrix")):
        example = dataset[split][idx]
        _, lab_t = example_to_tensors(example)
        
        # Check presence for each class
        for ci, class_id in enumerate(LABEL_IDS):  # LABEL_IDS = [1,2,...,12]
            mask = (lab_t == class_id)
            if mask.sum() >= min_pixels:
                Y[row_i, ci] = 1
    
    return Y, indices


def select_balanced_indices(Y: np.ndarray, all_indices: List[int], n_select: int,
                           seed: int = 42) -> List[int]:
    """
    Greedy selection for balanced organ representation.
    
    Args:
        Y: Presence matrix [N, 12]
        all_indices: Dataset indices corresponding to Y rows
        n_select: Number of samples to select
        seed: Random seed
        
    Returns:
        List of selected dataset indices
    """
    rng = np.random.default_rng(seed)
    N, K = Y.shape
    
    if n_select > N:
        raise ValueError(f"Cannot select {n_select} from {N} samples")
    
    # Track selected indices and organ counts
    selected = []
    organ_counts = np.zeros(K, dtype=int)
    
    # Available pool
    available = set(range(N))
    
    for _ in range(n_select):
        if not available:
            break
        
        # Find organs with minimum representation
        min_count = organ_counts.min()
        rare_organs = np.where(organ_counts == min_count)[0]
        
        # Find samples that have these rare organs
        candidates = []
        for idx in available:
            score = Y[idx, rare_organs].sum()  # How many rare organs in this sample
            if score > 0:
                candidates.append((idx, score))
        
        if not candidates:
            # No samples with rare organs, pick randomly
            idx = rng.choice(list(available))
        else:
            # Sort by score (descending) and pick from top candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_score = candidates[0][1]
            top_candidates = [c[0] for c in candidates if c[1] == top_score]
            idx = rng.choice(top_candidates)
        
        # Update tracking
        selected.append(all_indices[idx])
        organ_counts += Y[idx]
        available.remove(idx)
    
    return selected


def sample_point_in_mask(lab_t: torch.Tensor, class_id: int, 
                        strategy: str = "centroid") -> Optional[Tuple[int, int]]:
    """
    Sample a point within the mask for a given organ class.
    
    Args:
        lab_t: Label tensor [H, W]
        class_id: Organ class ID
        strategy: "centroid", "random", or "interior"
        
    Returns:
        (x, y) coordinates or None if organ not present
    """
    mask = (lab_t == class_id).cpu().numpy()
    if mask.sum() == 0:
        return None
    
    if strategy == "centroid":
        # Use center of mass
        y_coords, x_coords = np.where(mask)
        cx = int(x_coords.mean())
        cy = int(y_coords.mean())
        # Ensure point is in mask
        if mask[cy, cx]:
            return (int(cx), int(cy))
        # Find nearest point in mask
        dists = (x_coords - cx)**2 + (y_coords - cy)**2
        min_idx = dists.argmin()
        return (int(x_coords[min_idx]), int(y_coords[min_idx]))
    
    elif strategy == "random":
        y_coords, x_coords = np.where(mask)
        idx = np.random.randint(len(x_coords))
        return (int(x_coords[idx]), int(y_coords[idx]))
    
    elif strategy == "interior":
        # Use distance transform to find interior point
        dt = ndi.distance_transform_edt(mask)
        if dt.max() > 0:
            y, x = np.unravel_index(dt.argmax(), dt.shape)
            return (int(x), int(y))
        # Fallback to centroid
        return sample_point_in_mask(lab_t, class_id, "centroid")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def build_fewshot_plan(dataset, split: str, balanced_indices: List[int],
                       n_pos: int = 1, n_neg_easy: int = 1, n_neg_hard: int = 0,
                       min_pixels: int = 50, seed: int = 123,
                       cache_dir: Optional[Path] = None) -> Dict:
    """
    Build few-shot example plan with positives and negatives for each organ.
    
    Args:
        dataset: HuggingFace dataset
        split: Dataset split
        balanced_indices: Indices to exclude (test set)
        n_pos: Number of positive examples per organ
        n_neg_easy: Number of easy negative examples per organ
        n_neg_hard: Number of hard negative examples per organ
        min_pixels: Minimum pixels for presence
        seed: Random seed
        cache_dir: Directory to cache presence matrix
        
    Returns:
        Dictionary with few-shot plan
    """
    rng = np.random.default_rng(seed)
    total = len(dataset[split])
    
    # Exclude balanced indices (test set)
    balanced_set = set(int(i) for i in balanced_indices)
    base_pool = [i for i in range(total) if i not in balanced_set]
    
    if len(base_pool) == 0:
        raise ValueError("Base pool is empty after excluding balanced indices")
    
    # Build presence matrix for pool
    Y, pool_indices = build_presence_matrix(dataset, split, base_pool, min_pixels)
    M, K = Y.shape
    assert K == 12
    
    # Build candidate lists per organ
    pos_candidates = {}
    neg_candidates = {}
    
    for ci, class_id in enumerate(LABEL_IDS):
        pos_rows = np.where(Y[:, ci] == 1)[0]
        neg_rows = np.where(Y[:, ci] == 0)[0]
        pos_candidates[class_id] = [base_pool[r] for r in pos_rows]
        neg_candidates[class_id] = [base_pool[r] for r in neg_rows]
    
    # Build plan
    plan = {
        "dataset": "CholecSeg8k",
        "split": split,
        "min_pixels": int(min_pixels),
        "seed": int(seed),
        "n_pos": int(n_pos),
        "n_neg_easy": int(n_neg_easy),
        "n_neg_hard": int(n_neg_hard),
        "exclude_balanced": sorted(list(balanced_set)),
        "base_pool_size": len(base_pool),
        "plan": {}  # Per organ
    }
    
    # Define confuser organs for hard negatives
    confuser_map = get_confuser_map()
    
    for class_id in LABEL_IDS:
        organ_name = ID2LABEL[class_id]
        
        # Select positive examples
        pos_avail = pos_candidates[class_id]
        pos_pick = []
        if n_pos > 0 and pos_avail:
            pos_pick = rng.choice(pos_avail, min(n_pos, len(pos_avail)), replace=False).tolist()
        
        # Select easy negative examples (random negatives)
        neg_avail = neg_candidates[class_id]
        neg_easy_pick = []
        if n_neg_easy > 0 and neg_avail:
            # Ensure diversity by spacing out selected indices
            neg_easy_pick = _select_diverse_indices(neg_avail, n_neg_easy, min_spacing=100, rng=rng)
        
        # Select hard negative examples (contain confuser organs)
        neg_hard_pick = []
        if n_neg_hard > 0 and neg_avail:
            confusers = confuser_map.get(class_id, [])
            if confusers:
                # Find negatives that contain confuser organs
                hard_neg_candidates = []
                for neg_idx in neg_avail:
                    row = base_pool.index(neg_idx)
                    for conf_id in confusers:
                        conf_ci = LABEL_IDS.index(conf_id)
                        if Y[row, conf_ci] == 1:
                            hard_neg_candidates.append(neg_idx)
                            break
                
                if hard_neg_candidates:
                    # Ensure hard negatives are diverse and not too close to easy negatives
                    hard_neg_pick = _select_diverse_indices(
                        hard_neg_candidates, n_neg_hard, 
                        min_spacing=100, avoid_indices=neg_easy_pick, rng=rng
                    )
        
        # Add points for positive examples
        pos_entries = []
        for idx in pos_pick:
            example = dataset[split][idx]
            _, lab_t = example_to_tensors(example)
            point = sample_point_in_mask(lab_t, class_id, strategy="interior")
            pos_entries.append({"idx": int(idx), "point": point})
        
        plan["plan"][str(class_id)] = {
            "name": organ_name,
            "positives": pos_entries,
            "negatives_easy": [int(i) for i in neg_easy_pick],
            "negatives_hard": [int(i) for i in neg_hard_pick],
            "pos_available": len(pos_avail),
            "neg_available": len(neg_avail)
        }
    
    return plan


# ========= Near-miss Hard Negative Sampling Helpers =========

def _binary_dilate(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Fast binary dilation via conv2d. mask: [H,W] bool/0-1 -> returns dilated bool.
    """
    t = torch.from_numpy(mask.astype(np.float32))[None, None, ...]  # [1,1,H,W]
    k = torch.ones((1, 1, 2*radius+1, 2*radius+1), dtype=torch.float32)
    t = F.conv2d(t, k, padding=radius)
    return (t[0, 0].numpy() > 0)


def _outer_band(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Outer band: dilate(mask) - mask. Returns bool [H,W] just outside the organ.
    """
    dil = _binary_dilate(mask, radius=radius)
    band = np.logical_and(dil, ~mask)
    return band


def _choose_band_candidate_by_color(
    img: np.ndarray,           # [C,H,W], float in [0,1]
    band: np.ndarray,          # [H,W] bool
    organ_mask: np.ndarray,    # [H,W] bool for the class
    k_rand: int = 400
) -> Optional[Tuple[int, int]]:
    """
    Prefer candidate pixels in the outer band whose color is close to
    the mean color of the organ region (looks similar).
    """
    C, H, W = img.shape
    ys, xs = np.where(band)
    if ys.size == 0:
        return None
    
    # Subsample to limit compute
    if ys.size > k_rand:
        idxs = np.random.choice(ys.size, size=k_rand, replace=False)
        ys, xs = ys[idxs], xs[idxs]
    
    # Mean organ color
    oy, ox = np.where(organ_mask)
    if oy.size == 0:
        # Fallback: just pick a random band pixel
        j = np.random.randint(0, ys.size)
        return (int(xs[j]), int(ys[j]))
    
    organ_colors = img[:, oy, ox]  # [C, N]
    mean_color = organ_colors.mean(axis=1, keepdims=True)  # [C,1]
    
    # Candidate colors
    cand_colors = img[:, ys, xs]  # [C, M]
    # L2 distance to organ mean color
    diffs = np.linalg.norm(cand_colors - mean_color, axis=0)  # [M]
    j = int(np.argmin(diffs))
    return (int(xs[j]), int(ys[j]))


def sample_hard_negative_point(
    img_t: torch.Tensor,   # [C,H,W], 0..1 floats
    lab_t: torch.Tensor,   # [H,W] long
    class_id: int,
    radii: Tuple[int, ...] = (10, 15, 20, 25, 30),  # Increased from (3,5,7,9,12) for clearer separation
    min_distance: int = 10,  # Minimum pixels away from organ boundary
    k_rand: int = 400
) -> Optional[Tuple[int, int]]:
    """
    Pick a 'near-miss' negative point: outside the organ but close to its boundary
    and color-similar to the organ region. Ensures minimum distance from organ.
    Returns (x,y) in ORIGINAL coords or None if not possible.
    """
    lab = lab_t.cpu().numpy()
    mask = (lab == class_id)
    if mask.sum() == 0:
        return None
    img = img_t.cpu().numpy()
    
    # Start with minimum distance to ensure clear separation
    effective_radii = [max(r, min_distance) for r in radii]
    
    for r in effective_radii:
        band = _outer_band(mask, radius=r)
        if band.any():
            # Additional check: ensure the point is at least min_distance pixels from organ
            # by eroding the band from the inner edge
            if r > min_distance:
                inner_band = _outer_band(mask, radius=min_distance-1)
                band = np.logical_and(band, ~inner_band)  # Remove points too close
            
            if band.any():
                pt = _choose_band_candidate_by_color(img, band, mask, k_rand=k_rand)
                if pt is not None:
                    # Verify the point is truly outside and at minimum distance
                    x, y = pt
                    if not mask[y, x]:  # Double-check it's not on the organ
                        # Check minimum distance using distance transform
                        from scipy.ndimage import distance_transform_edt
                        dist_from_organ = distance_transform_edt(~mask)
                        if dist_from_organ[y, x] >= min_distance:
                            return pt
    
    # Fallback: try larger radii before giving up
    extended_radii = [40, 50, 60, 80, 100]
    for r in extended_radii:
        band = _outer_band(mask, radius=r)
        if band.any():
            # Still maintain minimum distance
            if r > min_distance:
                inner_band = _outer_band(mask, radius=min_distance-1)
                band = np.logical_and(band, ~inner_band)
            
            if band.any():
                # Just pick random point from band if color matching fails
                ys, xs = np.where(band)
                if ys.size > 0:
                    j = np.random.randint(0, ys.size)
                    return (int(xs[j]), int(ys[j]))
    
    # Final fallback: return None to indicate no suitable near-miss point
    # This is better than returning a completely random point
    return None


def get_confuser_map() -> Dict[int, List[int]]:
    """
    Define which organs are easily confused with each other.
    Used for hard negative selection.
    """
    return {
        2: [4, 6],      # Liver confused with Fat, Connective Tissue
        3: [1, 10],     # GI Tract confused with Abdominal Wall, Gallbladder
        4: [2, 6],      # Fat confused with Liver, Connective Tissue
        5: [9],         # Grasper confused with L-hook
        6: [2, 4],      # Connective Tissue confused with Liver, Fat
        7: [8],         # Blood confused with Cystic Duct
        8: [7, 11],     # Cystic Duct confused with Blood, Hepatic Vein
        9: [5],         # L-hook confused with Grasper
        10: [3],        # Gallbladder confused with GI Tract
        11: [8],        # Hepatic Vein confused with Cystic Duct
        12: [6],        # Liver Ligament confused with Connective Tissue
    }


def save_balanced_indices(indices: List[int], split: str, n_select: int, 
                         output_dir: Path = Path("../data_info/cholecseg8k")):
    """Save balanced indices to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"balanced_indices_{split}_{n_select}.json"
    output_path = output_dir / filename
    
    payload = {
        "split": split,
        "n": len(indices),
        "indices": indices
    }
    
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"Saved balanced indices to: {output_path}")
    return output_path


def save_fewshot_plan(plan: Dict, output_dir: Path = Path("../data_info/cholecseg8k")):
    """Save few-shot plan to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename from plan parameters
    n_pos = plan["n_pos"]
    n_neg_easy = plan["n_neg_easy"]
    n_neg_hard = plan["n_neg_hard"]
    seed = plan["seed"]
    n_excluded = len(plan["exclude_balanced"])
    
    if n_neg_hard > 0:
        filename = f"fewshot_plan_{plan['split']}_pos{n_pos}_nege{n_neg_easy}_negh{n_neg_hard}_seed{seed}_excl{n_excluded}.json"
    else:
        filename = f"fewshot_plan_{plan['split']}_pos{n_pos}_neg{n_neg_easy}_seed{seed}_excl{n_excluded}.json"
    
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2)
    
    print(f"Saved few-shot plan to: {output_path}")
    
    # Print summary
    print("\nFew-shot plan summary:")
    for class_id_str, info in plan["plan"].items():
        name = info["name"]
        n_pos = len(info["positives"])
        n_neg_e = len(info["negatives_easy"])
        n_neg_h = len(info["negatives_hard"])
        print(f"  {name:25} pos={n_pos}, neg_easy={n_neg_e}, neg_hard={n_neg_h}")
    
    return output_path


def load_balanced_indices(filepath: Path) -> List[int]:
    """Load balanced indices from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["indices"]


def load_fewshot_plan(filepath: Path) -> Dict:
    """Load few-shot plan from JSON file."""
    with open(filepath, "r") as f:
        plan = json.load(f)
    return plan


def build_fewshot_plan_with_nearmiss(
    dataset, split: str, balanced_indices: List[int],
    n_pos: int = 1, n_neg_easy: int = 1, n_near_miss: int = 1,
    min_pixels: int = 50, seed: int = 123,
    cache_dir: Optional[Path] = None) -> Dict:
    """
    Build few-shot example plan with positives, easy negatives, and near-miss hard negatives.
    
    Near-miss hard negatives: Images that DO contain the target organ, but the point 
    is placed just outside the organ boundary in a visually similar area.
    
    Args:
        dataset: HuggingFace dataset
        split: Dataset split
        balanced_indices: Indices to exclude (test set)
        n_pos: Number of positive examples per organ
        n_neg_easy: Number of easy negative examples per organ
        n_near_miss: Number of near-miss hard negative examples per organ
        min_pixels: Minimum pixels for presence
        seed: Random seed
        cache_dir: Directory to cache presence matrix
        
    Returns:
        Dictionary with few-shot plan including near-miss points
    """
    rng = np.random.default_rng(seed)
    total = len(dataset[split])
    
    # Exclude balanced indices (test set)
    balanced_set = set(int(i) for i in balanced_indices)
    base_pool = [i for i in range(total) if i not in balanced_set]
    
    if len(base_pool) == 0:
        raise ValueError("Base pool is empty after excluding balanced indices")
    
    # Build presence matrix for pool
    Y, pool_indices = build_presence_matrix(dataset, split, base_pool, min_pixels)
    M, K = Y.shape
    assert K == 12
    
    # Build candidate lists per organ
    pos_candidates = {}
    neg_candidates = {}
    
    for ci, class_id in enumerate(LABEL_IDS):
        pos_rows = np.where(Y[:, ci] == 1)[0]
        neg_rows = np.where(Y[:, ci] == 0)[0]
        pos_candidates[class_id] = [base_pool[r] for r in pos_rows]
        neg_candidates[class_id] = [base_pool[r] for r in neg_rows]
    
    # Build plan
    plan = {
        "dataset": "CholecSeg8k",
        "split": split,
        "min_pixels": int(min_pixels),
        "seed": int(seed),
        "n_pos": int(n_pos),
        "n_neg_easy": int(n_neg_easy),
        "n_near_miss": int(n_near_miss),
        "exclude_balanced": sorted(list(balanced_set)),
        "base_pool_size": len(base_pool),
        "plan": {}  # Per organ
    }
    
    for class_id in LABEL_IDS:
        organ_name = ID2LABEL[class_id]
        
        # Select positive examples
        pos_avail = pos_candidates[class_id]
        pos_pick = []
        if n_pos > 0 and pos_avail:
            pos_pick = _select_diverse_indices(pos_avail, n_pos, min_spacing=100, rng=rng)
        
        # Select easy negative examples (random negatives)
        neg_avail = neg_candidates[class_id]
        neg_easy_pick = []
        if n_neg_easy > 0 and neg_avail:
            neg_easy_pick = _select_diverse_indices(neg_avail, n_neg_easy, min_spacing=100, rng=rng)
        
        # Add points for positive examples (correct points on organ)
        pos_entries = []
        for idx in pos_pick:
            example = dataset[split][idx]
            img_t, lab_t = example_to_tensors(example)
            point = sample_point_in_mask(lab_t, class_id, strategy="interior")
            pos_entries.append({
                "idx": int(idx), 
                "point": point,
                "point_type": "positive"
            })
        
        # Add near-miss hard negative points
        # Use different positive examples for near-miss (or reuse if not enough)
        near_miss_entries = []
        if n_near_miss > 0 and pos_avail:
            # Try to use different images than the positive examples
            remaining_pos = [x for x in pos_avail if x not in pos_pick]
            if len(remaining_pos) >= n_near_miss:
                near_miss_pick = rng.choice(remaining_pos, n_near_miss, replace=False).tolist()
            else:
                # Not enough different images, may reuse some
                near_miss_pick = rng.choice(pos_avail, min(n_near_miss, len(pos_avail)), replace=False).tolist()
            
            for idx in near_miss_pick:
                example = dataset[split][idx]
                img_t, lab_t = example_to_tensors(example)
                # Get a near-miss point (clearly outside the organ with min 10 pixel distance)
                near_miss_point = sample_hard_negative_point(
                    img_t, lab_t, class_id, 
                    min_distance=10  # Ensure at least 10 pixels from organ boundary
                )
                near_miss_entries.append({
                    "idx": int(idx),
                    "point": near_miss_point,
                    "point_type": "near_miss"
                })
        
        plan["plan"][str(class_id)] = {
            "name": organ_name,
            "positives": pos_entries,
            "negatives_easy": [int(i) for i in neg_easy_pick],
            "near_miss": near_miss_entries,  # Near-miss hard negatives
            "pos_available": len(pos_avail),
            "neg_available": len(neg_avail)
        }
    
    return plan


if __name__ == "__main__":
    # Example usage
    from datasets import load_dataset
    
    print("Loading CholecSeg8k dataset...")
    dataset = load_dataset("minwoosun/CholecSeg8k")
    
    # Step 1: Build presence matrix and select balanced indices
    print("\nBuilding presence matrix...")
    Y, all_indices = build_presence_matrix(dataset, "train", indices=None, min_pixels=50)
    print(f"Presence matrix shape: {Y.shape}")
    
    # Step 2: Select balanced subset
    N_SELECT = 100
    print(f"\nSelecting {N_SELECT} balanced samples...")
    balanced_indices = select_balanced_indices(Y, all_indices, n_select=N_SELECT, seed=42)
    
    # Step 3: Save balanced indices
    output_dir = Path("../data_info/cholecseg8k")
    save_balanced_indices(balanced_indices, "train", N_SELECT, output_dir)
    
    # Step 4: Build few-shot plan
    print("\nBuilding few-shot plan...")
    plan = build_fewshot_plan(
        dataset, "train", balanced_indices,
        n_pos=1, n_neg_easy=1, n_neg_hard=1,
        min_pixels=50, seed=123
    )
    
    # Step 5: Save few-shot plan
    save_fewshot_plan(plan, output_dir)
    
    print("\nDone!")