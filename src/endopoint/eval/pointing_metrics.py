"""Enhanced pointing metrics calculation following the notebook approach."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ..datasets.cholecseg8k import ID2LABEL, LABEL_IDS


def calculate_comprehensive_metrics(
    records: List[Dict],
    organ_names: Optional[List[str]] = None,
) -> Tuple[List[Dict], Dict, int]:
    """Calculate comprehensive pointing metrics as per the notebook.
    
    Args:
        records: List of evaluation records with y_true, y_pred, hits fields
        organ_names: Optional list of organ names (uses default if not provided)
        
    Returns:
        Tuple of:
            - List of per-organ metric rows
            - Dictionary of totals across organs
            - Number of examples processed
    """
    if organ_names is None:
        organ_names = [ID2LABEL[i] for i in LABEL_IDS]
    
    K = len(organ_names)
    
    # Initialize counters
    tp = np.zeros(K, dtype=np.int64)
    tn = np.zeros(K, dtype=np.int64)
    fp = np.zeros(K, dtype=np.int64)
    fn = np.zeros(K, dtype=np.int64)
    
    # Gated metrics (pred=1 only if point hits)
    tpg = np.zeros(K, dtype=np.int64)
    tng = np.zeros(K, dtype=np.int64)
    fpg = np.zeros(K, dtype=np.int64)
    fng = np.zeros(K, dtype=np.int64)
    
    hit_present = np.zeros(K, dtype=np.int64)
    present_n = np.zeros(K, dtype=np.int64)
    n_examples = 0
    
    for obj in records:
        y_true = np.asarray(obj["y_true"], dtype=int)
        y_pred = np.asarray(obj["y_pred"], dtype=int)
        hits = np.asarray(obj.get("hits", np.zeros(K)), dtype=int)
        
        # Standard confusion matrix
        tp += ((y_true == 1) & (y_pred == 1))
        tn += ((y_true == 0) & (y_pred == 0))
        fp += ((y_true == 0) & (y_pred == 1))
        fn += ((y_true == 1) & (y_pred == 0))
        
        # Gated predictions (only predict 1 if point hits)
        pred_g = ((y_pred == 1) & (hits == 1))
        tpg += ((y_true == 1) & (pred_g == 1))
        tng += ((y_true == 0) & (pred_g == 0))
        fpg += ((y_true == 0) & (pred_g == 1))
        fng += ((y_true == 1) & (pred_g == 0))
        
        # Hitting statistics
        hit_present += ((hits == 1) & (y_true == 1))
        present_n += (y_true == 1)
        
        n_examples += 1
    
    absent_n = (tn + fp)
    total_n = present_n + absent_n
    
    def safe_div(a, b):
        return float(a / b) if b > 0 else None
    
    # Build per-organ metric rows
    rows = []
    for i, organ_name in enumerate(organ_names):
        label_id = LABEL_IDS[i] if i < len(LABEL_IDS) else i + 1
        
        rows.append({
            "label_id": label_id,
            "label": organ_name,
            "TP": int(tp[i]),
            "FN": int(fn[i]),
            "TN": int(tn[i]),
            "FP": int(fp[i]),
            "Present_n": int(present_n[i]),
            "Absent_n": int(absent_n[i]),
            "Total": int(total_n[i]),
            "PresenceAcc": safe_div(tp[i] + tn[i], total_n[i]),
            "Hit@Point|Present": safe_div(hit_present[i], present_n[i]),
            "Gated_TP": int(tpg[i]),
            "Gated_FN": int(fng[i]),
            "Gated_TN": int(tng[i]),
            "Gated_FP": int(fpg[i]),
            "GatedAcc": safe_div(tpg[i] + tng[i], total_n[i]),
            "Precision": safe_div(tp[i], tp[i] + fp[i]),
            "Recall": safe_div(tp[i], tp[i] + fn[i]),
            "F1": safe_div(2 * tp[i], 2 * tp[i] + fp[i] + fn[i]),
        })
    
    # Calculate totals
    totals = {
        "TP": int(tp.sum()),
        "FN": int(fn.sum()),
        "TN": int(tn.sum()),
        "FP": int(fp.sum()),
        "Present_n": int(present_n.sum()),
        "Absent_n": int(absent_n.sum()),
        "Total": int(total_n.sum()),
        "Gated_TP": int(tpg.sum()),
        "Gated_FN": int(fng.sum()),
        "Gated_TN": int(tng.sum()),
        "Gated_FP": int(fpg.sum()),
    }
    
    return rows, totals, n_examples


def format_percentage(x: Optional[float]) -> str:
    """Format a percentage value."""
    return f"{x*100:6.2f}%" if x is not None else "   --  "


def print_metrics_table(
    rows: List[Dict],
    totals: Dict,
    n_examples: int,
    model_name: str,
    prompt_name: str,
    split: str = "train",
) -> str:
    """Print a formatted metrics table.
    
    Args:
        rows: Per-organ metric rows
        totals: Totals dictionary
        n_examples: Number of examples
        model_name: Model name
        prompt_name: Prompt/evaluation name
        split: Dataset split
        
    Returns:
        Formatted string output
    """
    lines = []
    
    lines.append(f"\nModel: {model_name} | Prompt: {prompt_name} | Split: {split} | Examples used: {n_examples}")
    
    header = (
        "ID  Label                     TP   FN   TN   FP   "
        "Pres  Abs   Tot   PresenceAcc   Hit@Pt|Pres   gTP  gFN  gTN  gFP   GatedAcc"
    )
    lines.append(header)
    
    for r in rows:
        line = (
            f"{r['label_id']:>2}  {r['label']:<24}  "
            f"{r['TP']:>3}  {r['FN']:>3}  {r['TN']:>3}  {r['FP']:>3}   "
            f"{r['Present_n']:>4} {r['Absent_n']:>4} {r['Total']:>5}   "
            f"{format_percentage(r['PresenceAcc'])}    {format_percentage(r['Hit@Point|Present'])}   "
            f"{r['Gated_TP']:>3}  {r['Gated_FN']:>3}  {r['Gated_TN']:>3}  {r['Gated_FP']:>3}   "
            f"{format_percentage(r['GatedAcc'])}"
        )
        lines.append(line)
    
    # Calculate macro averages
    pres_accs = [r["PresenceAcc"] for r in rows if r["PresenceAcc"] is not None]
    gated_accs = [r["GatedAcc"] for r in rows if r["GatedAcc"] is not None]
    hit_rates = [r["Hit@Point|Present"] for r in rows if r["Hit@Point|Present"] is not None]
    f1_scores = [r["F1"] for r in rows if r["F1"] is not None]
    
    macro_presence = np.mean(pres_accs) if pres_accs else 0.0
    macro_gated = np.mean(gated_accs) if gated_accs else 0.0
    macro_hit = np.mean(hit_rates) if hit_rates else 0.0
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    lines.append("\nTotals across organs:")
    lines.append(
        f"TP={totals['TP']}  FN={totals['FN']}  TN={totals['TN']}  FP={totals['FP']}  "
        f"Present={totals['Present_n']}  Absent={totals['Absent_n']}  Total={totals['Total']}"
    )
    lines.append(
        f"Macro PresenceAcc={format_percentage(macro_presence)}   "
        f"Macro Hit@Point|Present={format_percentage(macro_hit)}   "
        f"Macro GatedAcc={format_percentage(macro_gated)}   "
        f"Macro F1={format_percentage(macro_f1)}"
    )
    
    return "\n".join(lines)


def save_metrics_json(
    rows: List[Dict],
    totals: Dict,
    n_examples: int,
    model_name: str,
    prompt_name: str,
    split: str,
    output_path: Path,
) -> None:
    """Save metrics to JSON file.
    
    Args:
        rows: Per-organ metric rows
        totals: Totals dictionary
        n_examples: Number of examples
        model_name: Model name
        prompt_name: Prompt/evaluation name
        split: Dataset split
        output_path: Path to save JSON file
    """
    # Calculate macro metrics
    pres_accs = [r["PresenceAcc"] for r in rows if r["PresenceAcc"] is not None]
    gated_accs = [r["GatedAcc"] for r in rows if r["GatedAcc"] is not None]
    hit_rates = [r["Hit@Point|Present"] for r in rows if r["Hit@Point|Present"] is not None]
    f1_scores = [r["F1"] for r in rows if r["F1"] is not None]
    
    macro_metrics = {
        "macro_presence_acc": float(np.mean(pres_accs)) if pres_accs else None,
        "macro_gated_acc": float(np.mean(gated_accs)) if gated_accs else None,
        "macro_hit_rate": float(np.mean(hit_rates)) if hit_rates else None,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else None,
    }
    
    # Micro metrics (overall)
    micro_metrics = {
        "micro_presence_acc": float((totals["TP"] + totals["TN"]) / totals["Total"]) if totals["Total"] > 0 else None,
        "micro_gated_acc": float((totals["Gated_TP"] + totals["Gated_TN"]) / totals["Total"]) if totals["Total"] > 0 else None,
    }
    
    output_data = {
        "model": model_name,
        "prompt": prompt_name,
        "split": split,
        "n_examples": n_examples,
        "per_organ_metrics": rows,
        "totals": totals,
        "macro_metrics": macro_metrics,
        "micro_metrics": micro_metrics,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def check_point_hit(
    point: Optional[Tuple[int, int]],
    mask: np.ndarray,
    canvas_width: int,
    canvas_height: int,
) -> bool:
    """Check if a predicted point hits the organ mask.
    
    Args:
        point: Predicted point (x, y) in canvas coordinates
        mask: Binary mask for the organ [H, W]
        canvas_width: Canvas width
        canvas_height: Canvas height
        
    Returns:
        True if point hits the organ, False otherwise
    """
    if point is None:
        return False
    
    x, y = point
    
    # Check bounds
    if x < 0 or x >= canvas_width or y < 0 or y >= canvas_height:
        return False
    
    # Convert canvas coordinates to mask coordinates
    mask_h, mask_w = mask.shape
    mask_x = int(x * mask_w / canvas_width)
    mask_y = int(y * mask_h / canvas_height)
    
    # Check if point hits the mask
    return mask[mask_y, mask_x] > 0 if 0 <= mask_y < mask_h and 0 <= mask_x < mask_w else False