"""Cell selection evaluation utilities for grid-based organ localization."""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import torch


def get_cell_labels(grid_size: int) -> List[str]:
    """Generate cell labels for a grid (e.g., A1, A2, B1, B2 for 2x2).
    
    Args:
        grid_size: Size of the grid (e.g., 3 for 3x3)
        
    Returns:
        List of cell labels in row-major order
    """
    labels = []
    for row in range(grid_size):
        for col in range(grid_size):
            label = chr(65 + row) + str(col + 1)  # A1, A2, ..., B1, B2, ...
            labels.append(label)
    return labels


def get_cell_from_label(label: str, grid_size: int) -> Optional[Tuple[int, int]]:
    """Convert cell label to (row, col) coordinates.
    
    Args:
        label: Cell label (e.g., "B2")
        grid_size: Size of the grid
        
    Returns:
        (row, col) tuple or None if invalid
    """
    if not label or len(label) < 2:
        return None
    
    row_char = label[0]
    col_str = label[1:]
    
    # Validate row
    if not row_char.isalpha():
        return None
    row = ord(row_char.upper()) - 65
    if row < 0 or row >= grid_size:
        return None
    
    # Validate column
    try:
        col = int(col_str) - 1
        if col < 0 or col >= grid_size:
            return None
    except ValueError:
        return None
    
    return (row, col)


def compute_cell_ground_truth(
    mask: np.ndarray,
    grid_size: int,
    min_pixels: int = 50,
    return_counts: bool = False
) -> Dict[str, any]:
    """Compute ground truth cell set for an organ mask.
    
    Args:
        mask: Binary mask (H, W) with 1 where organ is present
        grid_size: Grid size G (results in G×G grid)
        min_pixels: Minimum pixels for organ to be considered present
        return_counts: If True, also return pixel counts per cell
        
    Returns:
        Dictionary with:
            - present: bool, whether organ is present
            - cells: Set of cell labels where organ is present
            - dominant_cell: Cell with most pixels (if present)
            - pixel_counts: Dict[str, int] of pixels per cell (if return_counts=True)
    """
    H, W = mask.shape
    cell_height = H // grid_size
    cell_width = W // grid_size
    
    # Check if organ is present
    total_pixels = np.sum(mask)
    present = total_pixels >= min_pixels
    
    if not present:
        result = {
            'present': False,
            'cells': set(),
            'dominant_cell': None
        }
        if return_counts:
            result['pixel_counts'] = {}
        return result
    
    # Count pixels per cell
    cell_pixels = {}
    cells_with_organ = set()
    
    for row in range(grid_size):
        for col in range(grid_size):
            # Cell boundaries (handle edge cells that might be slightly larger)
            y_start = row * cell_height
            y_end = (row + 1) * cell_height if row < grid_size - 1 else H
            x_start = col * cell_width
            x_end = (col + 1) * cell_width if col < grid_size - 1 else W
            
            # Count pixels in this cell
            cell_mask = mask[y_start:y_end, x_start:x_end]
            pixel_count = np.sum(cell_mask)
            
            # Generate cell label
            cell_label = chr(65 + row) + str(col + 1)
            
            if pixel_count > 0:
                cells_with_organ.add(cell_label)
                cell_pixels[cell_label] = int(pixel_count)
    
    # Find dominant cell (with most pixels)
    dominant_cell = None
    if cell_pixels:
        dominant_cell = max(cell_pixels.items(), key=lambda x: x[1])[0]
    
    result = {
        'present': True,
        'cells': cells_with_organ,
        'dominant_cell': dominant_cell
    }
    
    if return_counts:
        result['pixel_counts'] = cell_pixels
    
    return result


def compute_cell_metrics(
    pred_cells: List[str],
    gt_cells: Set[str],
    gt_present: bool,
    pred_present: bool,
    top_k: int = 1
) -> Dict[str, float]:
    """Compute cell selection metrics.
    
    Args:
        pred_cells: Predicted cell labels
        gt_cells: Ground truth cell set
        gt_present: Whether organ is present in GT
        pred_present: Whether organ is predicted as present
        top_k: Consider prediction correct if any of top-k cells hit
        
    Returns:
        Dictionary with metrics:
            - cell_hit: 1 if any predicted cell is in GT cells, 0 otherwise
            - cell_precision: |pred ∩ gt| / |pred| if pred non-empty
            - cell_recall: |pred ∩ gt| / |gt| if gt non-empty
            - cell_f1: F1 score from precision and recall
            - false_positive_cells: Number of cells predicted when organ absent
    """
    metrics = {}
    
    # Convert to sets for intersection
    pred_set = set(pred_cells[:top_k]) if pred_cells else set()
    
    if gt_present and gt_cells:
        # Organ is present in ground truth
        intersection = pred_set & gt_cells
        
        # Cell hit: at least one predicted cell is correct
        metrics['cell_hit'] = 1.0 if intersection else 0.0
        
        # Precision: fraction of predicted cells that are correct
        if pred_set:
            metrics['cell_precision'] = len(intersection) / len(pred_set)
        else:
            metrics['cell_precision'] = 0.0
        
        # Recall: fraction of GT cells that were found
        metrics['cell_recall'] = len(intersection) / len(gt_cells)
        
        # F1 score
        if metrics['cell_precision'] + metrics['cell_recall'] > 0:
            metrics['cell_f1'] = (
                2 * metrics['cell_precision'] * metrics['cell_recall'] / 
                (metrics['cell_precision'] + metrics['cell_recall'])
            )
        else:
            metrics['cell_f1'] = 0.0
            
        metrics['false_positive_cells'] = 0
        
    else:
        # Organ is absent in ground truth
        # Any predicted cells are false positives
        metrics['cell_hit'] = 0.0
        metrics['cell_precision'] = 0.0
        metrics['cell_recall'] = 0.0 if gt_cells else 1.0  # Perfect recall if no GT cells
        metrics['cell_f1'] = 0.0
        metrics['false_positive_cells'] = len(pred_set)
    
    return metrics


def visualize_cell_grid(
    image: np.ndarray,
    grid_size: int,
    highlighted_cells: Optional[List[str]] = None,
    cell_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """Visualize cell grid overlay on image.
    
    Args:
        image: Image array (H, W, 3) or (H, W)
        grid_size: Grid size
        highlighted_cells: List of cells to highlight
        cell_colors: Dict mapping cell labels to RGB colors
        
    Returns:
        Image with grid overlay
    """
    import cv2
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Make a copy to draw on
    vis_image = image.copy()
    H, W = image.shape[:2]
    
    cell_height = H // grid_size
    cell_width = W // grid_size
    
    # Draw grid lines
    for i in range(1, grid_size):
        # Horizontal lines
        y = i * cell_height
        cv2.line(vis_image, (0, y), (W, y), (128, 128, 128), 1)
        
        # Vertical lines
        x = i * cell_width
        cv2.line(vis_image, (x, 0), (x, H), (128, 128, 128), 1)
    
    # Draw border
    cv2.rectangle(vis_image, (0, 0), (W-1, H-1), (128, 128, 128), 1)
    
    # Label cells
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    for row in range(grid_size):
        for col in range(grid_size):
            label = chr(65 + row) + str(col + 1)
            
            # Cell center
            y_start = row * cell_height
            y_end = (row + 1) * cell_height if row < grid_size - 1 else H
            x_start = col * cell_width
            x_end = (col + 1) * cell_width if col < grid_size - 1 else W
            
            cx = (x_start + x_end) // 2
            cy = (y_start + y_end) // 2
            
            # Highlight if requested
            if highlighted_cells and label in highlighted_cells:
                if cell_colors and label in cell_colors:
                    color = cell_colors[label]
                else:
                    color = (0, 255, 0)  # Default green
                
                # Draw semi-transparent rectangle
                overlay = vis_image.copy()
                cv2.rectangle(overlay, (x_start, y_start), (x_end-1, y_end-1), color, -1)
                vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
            
            # Draw label
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            
            # White background for text
            cv2.rectangle(vis_image, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (255, 255, 255), -1)
            
            # Draw text
            cv2.putText(vis_image, label, (text_x, text_y), 
                       font, font_scale, (0, 0, 0), font_thickness)
    
    return vis_image


def point_to_cell(x: int, y: int, canvas_width: int, canvas_height: int, grid_size: int) -> str:
    """Convert (x, y) point to cell label.
    
    Args:
        x: X coordinate
        y: Y coordinate  
        canvas_width: Canvas width
        canvas_height: Canvas height
        grid_size: Grid size
        
    Returns:
        Cell label (e.g., "B2")
    """
    cell_width = canvas_width // grid_size
    cell_height = canvas_height // grid_size
    
    # Clamp to canvas bounds
    x = max(0, min(x, canvas_width - 1))
    y = max(0, min(y, canvas_height - 1))
    
    # Compute cell indices
    col = min(x // cell_width, grid_size - 1)
    row = min(y // cell_height, grid_size - 1)
    
    return chr(65 + row) + str(col + 1)


def cells_to_points(
    cells: List[str], 
    canvas_width: int, 
    canvas_height: int, 
    grid_size: int,
    position: str = 'center'
) -> List[Tuple[int, int]]:
    """Convert cell labels to (x, y) points.
    
    Args:
        cells: List of cell labels
        canvas_width: Canvas width
        canvas_height: Canvas height
        grid_size: Grid size
        position: Where in cell to place point ('center', 'top-left', 'random')
        
    Returns:
        List of (x, y) tuples
    """
    cell_width = canvas_width // grid_size
    cell_height = canvas_height // grid_size
    
    points = []
    for cell_label in cells:
        cell_coords = get_cell_from_label(cell_label, grid_size)
        if cell_coords is None:
            continue
            
        row, col = cell_coords
        
        # Cell boundaries
        y_start = row * cell_height
        y_end = (row + 1) * cell_height if row < grid_size - 1 else canvas_height
        x_start = col * cell_width
        x_end = (col + 1) * cell_width if col < grid_size - 1 else canvas_width
        
        if position == 'center':
            x = (x_start + x_end) // 2
            y = (y_start + y_end) // 2
        elif position == 'top-left':
            x = x_start
            y = y_start
        elif position == 'random':
            import random
            x = random.randint(x_start, x_end - 1)
            y = random.randint(y_start, y_end - 1)
        else:
            # Default to center
            x = (x_start + x_end) // 2
            y = (y_start + y_end) // 2
            
        points.append((x, y))
    
    return points