"""JSON parsing utilities for model outputs."""

import json
import re
from typing import Dict, Optional, Tuple, List, Set


# Regular expression to find coordinates in format [x,y]
COORD_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

# Regular expression to find cell labels (e.g., A1, B2, C3)
CELL_RE = re.compile(r"\b([A-Z]\d+)\b")


def parse_pointing_json(
    text: str, 
    canvas_width: int, 
    canvas_height: int
) -> Dict[str, any]:
    """Parse pointing model output with fallbacks.
    
    Attempts to parse JSON first, then falls back to regex extraction.
    
    Args:
        text: Raw model output text
        canvas_width: Canvas width for bounds checking
        canvas_height: Canvas height for bounds checking
        
    Returns:
        Dictionary with:
            - present: 0 or 1
            - point_canvas: (x, y) tuple or None
            - raw: Original text
    """
    out = {"present": 0, "point_canvas": None, "raw": text}
    
    # Try JSON parsing first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # Parse present field
            pres = obj.get("present", 0)
            if isinstance(pres, bool):
                out["present"] = 1 if pres else 0
            elif isinstance(pres, (int, float)):
                out["present"] = 1 if int(pres) == 1 else 0
            elif isinstance(pres, str):
                out["present"] = 1 if pres.strip().lower() in ("1", "yes", "true") else 0
            
            # Parse point_canvas field
            pt = obj.get("point_canvas", None)
            if isinstance(pt, list) and len(pt) == 2:
                try:
                    x = int(round(pt[0]))
                    y = int(round(pt[1]))
                    if 0 <= x < canvas_width and 0 <= y < canvas_height:
                        out["point_canvas"] = (x, y)
                except (ValueError, TypeError):
                    pass
            
            return out
    except (json.JSONDecodeError, Exception):
        pass
    
    # Fallback: if coordinates appear in text, assume present=1
    if isinstance(text, str):
        match = COORD_RE.search(text)
        if match:
            try:
                x = int(match.group(1))
                y = int(match.group(2))
                if 0 <= x < canvas_width and 0 <= y < canvas_height:
                    out["present"] = 1
                    out["point_canvas"] = (x, y)
            except (ValueError, TypeError):
                pass
    
    return out


def parse_existence_response(text: str) -> int:
    """Parse existence detection response.
    
    Args:
        text: Raw model output
        
    Returns:
        1 for yes/present, 0 for no/absent
    """
    if not isinstance(text, str):
        return 0
    
    text_lower = text.strip().lower()
    
    # Check for explicit yes/no
    if text_lower in ("yes", "1", "true", "present"):
        return 1
    elif text_lower in ("no", "0", "false", "absent", "not present"):
        return 0
    
    # Check for yes/no anywhere in text
    yes_re = re.compile(r"\b(yes|present)\b", re.IGNORECASE)
    no_re = re.compile(r"\b(no|absent|not\s+present)\b", re.IGNORECASE)
    
    if yes_re.search(text) and not no_re.search(text):
        return 1
    
    return 0


def parse_cell_selection_json(
    text: str,
    grid_size: int,
    top_k: int = 1
) -> Dict[str, any]:
    """Parse cell selection model output.
    
    Args:
        text: Raw model output text
        grid_size: Grid size for validation (e.g., 3 for 3x3)
        top_k: Maximum number of cells allowed
        
    Returns:
        Dictionary with:
            - present: 0 or 1
            - cells: List of cell labels (e.g., ["A1", "B2"])
            - raw: Original text
    """
    out = {"present": 0, "cells": [], "raw": text}
    
    # Generate valid cell labels for this grid size
    valid_cells = set()
    for row in range(grid_size):
        for col in range(grid_size):
            valid_cells.add(chr(65 + row) + str(col + 1))
    
    # Try JSON parsing first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # Parse present field
            pres = obj.get("present", 0)
            if isinstance(pres, bool):
                out["present"] = 1 if pres else 0
            elif isinstance(pres, (int, float)):
                out["present"] = 1 if int(pres) == 1 else 0
            elif isinstance(pres, str):
                out["present"] = 1 if pres.strip().lower() in ("1", "yes", "true") else 0
            
            # Parse cells field
            cells = obj.get("cells", [])
            if isinstance(cells, list):
                validated_cells = []
                for cell in cells:
                    if isinstance(cell, str):
                        cell = cell.strip().upper()
                        if cell in valid_cells:
                            validated_cells.append(cell)
                
                # Enforce consistency: if present=0, cells must be empty
                if out["present"] == 0:
                    out["cells"] = []
                else:
                    # Keep only unique cells up to top_k
                    seen = set()
                    unique_cells = []
                    for cell in validated_cells:
                        if cell not in seen:
                            seen.add(cell)
                            unique_cells.append(cell)
                            if len(unique_cells) >= top_k:
                                break
                    out["cells"] = unique_cells
            
            # Additional validation: if present=1 but no valid cells, try fallback
            if out["present"] == 1 and not out["cells"]:
                # Try to find cells in the raw text
                cells_from_text = extract_cells_from_text(text, valid_cells, top_k)
                if cells_from_text:
                    out["cells"] = cells_from_text
            
            return out
            
    except (json.JSONDecodeError, Exception):
        pass
    
    # Fallback: try to extract cells from text using regex
    cells_found = extract_cells_from_text(text, valid_cells, top_k)
    if cells_found:
        out["present"] = 1
        out["cells"] = cells_found
    
    return out


def extract_cells_from_text(text: str, valid_cells: Set[str], top_k: int) -> List[str]:
    """Extract cell labels from text using regex.
    
    Args:
        text: Text to search
        valid_cells: Set of valid cell labels
        top_k: Maximum number of cells to return
        
    Returns:
        List of valid cell labels found
    """
    if not isinstance(text, str):
        return []
    
    matches = CELL_RE.findall(text)
    seen = set()
    result = []
    
    for match in matches:
        cell = match.upper()
        if cell in valid_cells and cell not in seen:
            seen.add(cell)
            result.append(cell)
            if len(result) >= top_k:
                break
    
    return result


def validate_cell_selection_response(
    response: Dict[str, any],
    grid_size: int,
    top_k: int = 1
) -> Dict[str, any]:
    """Validate and clean a cell selection response.
    
    Args:
        response: Parsed response dictionary
        grid_size: Grid size for validation
        top_k: Maximum number of cells allowed
        
    Returns:
        Cleaned response dictionary
    """
    # Generate valid cells
    valid_cells = set()
    for row in range(grid_size):
        for col in range(grid_size):
            valid_cells.add(chr(65 + row) + str(col + 1))
    
    # Ensure present is 0 or 1
    present = response.get("present", 0)
    if not isinstance(present, int) or present not in (0, 1):
        present = 1 if present else 0
    
    # Validate cells
    cells = response.get("cells", [])
    if not isinstance(cells, list):
        cells = []
    
    # Filter to valid cells only
    valid_selected = []
    seen = set()
    for cell in cells:
        if isinstance(cell, str):
            cell = cell.strip().upper()
            if cell in valid_cells and cell not in seen:
                seen.add(cell)
                valid_selected.append(cell)
                if len(valid_selected) >= top_k:
                    break
    
    # Enforce consistency
    if present == 0:
        valid_selected = []
    elif present == 1 and not valid_selected:
        # If marked present but no valid cells, mark as absent
        present = 0
    
    return {
        "present": present,
        "cells": valid_selected,
        "raw": response.get("raw", "")
    }