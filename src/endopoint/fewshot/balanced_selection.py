"""Balanced selection algorithms for few-shot examples."""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


def select_balanced_simple(
    Y: np.ndarray,
    pool_indices: list,
    n_select: int,
    seed: int = 42
) -> List[int]:
    """Simple balanced selection that maximizes representation of all classes.
    
    Args:
        Y: Presence matrix [N, K] where K is number of classes
        pool_indices: List of indices available for selection
        n_select: Number of samples to select
        seed: Random seed
    
    Returns:
        List of selected indices from pool_indices
    """
    np.random.seed(seed)
    n_samples, n_classes = Y.shape
    
    # Start with samples that have rare classes
    class_counts = Y.sum(axis=0)
    selected_indices = set()
    
    # Greedy selection: prioritize underrepresented classes
    while len(selected_indices) < n_select:
        # Calculate current representation
        if len(selected_indices) > 0:
            current_counts = Y[list(selected_indices)].sum(axis=0)
        else:
            current_counts = np.zeros(n_classes)
        
        # Calculate representation ratio
        target_ratio = n_select / n_samples
        expected_counts = class_counts * target_ratio
        deficit = expected_counts - current_counts
        
        # Find samples that help most with underrepresented classes
        scores = np.zeros(n_samples)
        for i in range(n_samples):
            if i not in selected_indices:
                # Score based on how much this sample helps underrepresented classes
                contribution = Y[i] * deficit
                scores[i] = contribution.sum()
        
        # Select sample with highest score
        scores[list(selected_indices)] = -np.inf
        top_k = min(10, n_samples - len(selected_indices))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        chosen_idx = np.random.choice(top_indices)
        selected_indices.add(chosen_idx)
    
    # Convert to original indices
    selected = sorted(list(selected_indices))
    return [int(pool_indices[i]) for i in selected]


def select_balanced_with_caps(
    Y: np.ndarray,
    pool_indices: list,
    *,
    n_select: int,
    rare_top_k: int = 4,
    min_quota_rare: int = 20,
    max_cap_frac: float = 0.70,
    extra_min_quota: Optional[Dict[int, int]] = None,
    seed: int = 0,
) -> Tuple[List[int], Dict[str, Any]]:
    """Advanced balanced selection with rare boosting and caps for abundant classes.
    
    This algorithm:
    1. Identifies rare classes and ensures minimum representation
    2. Caps abundant classes to prevent over-representation
    3. Uses two-phase greedy selection for optimal balance
    
    Args:
        Y: Presence matrix [N, K] where K is number of classes
        pool_indices: List of indices available for selection
        n_select: Number of samples to select
        rare_top_k: Number of rarest classes to boost
        min_quota_rare: Minimum samples for rare classes
        max_cap_frac: Maximum fraction for classes that are sometimes absent
        extra_min_quota: Additional minimum quotas per class {class_id: min_count}
        seed: Random seed
    
    Returns:
        Tuple of:
        - selected_indices: List of selected indices from pool_indices
        - info: Dictionary with selection statistics
    """
    rng = np.random.default_rng(seed)
    
    M, K = Y.shape  # M samples, K classes
    
    # Pool prevalence and rare classes
    preval = Y.mean(axis=0)  # Fraction present in pool
    present_pool = Y.sum(axis=0)  # Count present
    absent_pool = M - present_pool  # Count absent
    rare_order = np.argsort(preval)[:rare_top_k]
    rare_set = set(int(i) for i in rare_order)
    
    # Build per-class target counts
    T_total = np.rint(preval * n_select).astype(int)
    
    # Apply caps for classes that are sometimes absent
    cap_counts = np.full(K, n_select, dtype=int)
    for k in range(K):
        if absent_pool[k] > 0:  # Sometimes absent
            cap_counts[k] = int(np.floor(max_cap_frac * n_select))
            T_total[k] = min(T_total[k], cap_counts[k])
    
    # Apply rare quotas
    for k in rare_set:
        T_total[k] = max(T_total[k], min_quota_rare)
    
    # Apply extra quotas
    if extra_min_quota:
        for k, v in extra_min_quota.items():
            if 0 <= int(k) < K:
                T_total[int(k)] = max(T_total[int(k)], int(v))
    
    # Cannot exceed available
    for k in range(K):
        T_total[k] = min(T_total[k], int(present_pool[k]))
    
    # Two-phase greedy selection
    selected_rows = []
    remaining = set(range(M))
    pos_counts = np.zeros(K, dtype=int)
    selected_so_far = 0
    
    # Phase 1: Meet rare quotas
    def unmet_quota():
        need = np.zeros(K, dtype=int)
        for k in rare_set:
            need[k] = max(0, T_total[k] - pos_counts[k])
        return need
    
    while selected_so_far < n_select and np.any(unmet_quota() > 0):
        need = unmet_quota()
        if need.sum() == 0:
            break
        
        best_rows, best_help, best_gain = [], -1, -1.0
        for r in list(remaining):
            y = Y[r]
            help_rare = int(np.sum((y == 1) & (need > 0)))
            if help_rare <= 0:
                continue
            
            # Target-aware L1 gain
            before = np.abs(pos_counts - T_total).sum()
            after = np.abs((pos_counts + y) - T_total).sum()
            gain = before - after
            
            if (help_rare > best_help) or (help_rare == best_help and gain > best_gain):
                best_help, best_gain = help_rare, gain
                best_rows = [r]
            elif help_rare == best_help and abs(gain - best_gain) < 1e-9:
                best_rows.append(r)
        
        if not best_rows:
            break
        
        r_pick = int(rng.choice(best_rows))
        selected_rows.append(r_pick)
        remaining.remove(r_pick)
        pos_counts += Y[r_pick]
        selected_so_far += 1
    
    # Phase 2: Drive toward targets
    while selected_so_far < n_select and len(remaining) > 0:
        best_rows, best_gain = [], -1.0
        
        for r in list(remaining):
            y = Y[r]
            
            # Target-aware L1 reduction with penalty for exceeding targets
            before = np.abs(pos_counts - T_total).sum()
            penalty = float(np.sum((pos_counts >= T_total) & (y == 1))) * 0.1
            after = np.abs((pos_counts + y) - T_total).sum() + penalty
            gain = before - after
            
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_rows = [r]
            elif abs(gain - best_gain) <= 1e-12:
                best_rows.append(r)
        
        if not best_rows:
            break
        
        r_pick = int(rng.choice(best_rows))
        selected_rows.append(r_pick)
        remaining.remove(r_pick)
        pos_counts += Y[r_pick]
        selected_so_far += 1
    
    # Convert to original indices
    selected_indices = [int(pool_indices[r]) for r in selected_rows]
    
    # Compute statistics
    sel_Y = Y[selected_rows] if selected_rows else np.zeros((0, K), dtype=np.uint8)
    sel_present = sel_Y.sum(axis=0)
    sel_absent = len(selected_rows) - sel_present
    
    info = {
        "pool_size": int(M),
        "selected_n": int(len(selected_rows)),
        "pool_prevalence": preval.tolist(),
        "rare_order_cols": [int(i) for i in rare_order],
        "cap_frac": float(max_cap_frac),
        "cap_counts": cap_counts.tolist(),
        "T_total_targets": T_total.tolist(),
        "selected_present": sel_present.tolist(),
        "selected_absent": sel_absent.tolist() if isinstance(sel_absent, np.ndarray) else int(sel_absent),
    }
    
    return selected_indices, info