"""JSON parsing utilities for model outputs."""

import json
import re
from typing import Dict, Optional, Tuple


# Regular expression to find coordinates in format [x,y]
COORD_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


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