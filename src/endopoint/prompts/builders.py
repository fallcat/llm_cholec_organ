"""Prompt builders for different strategies."""

from typing import Optional


def build_pointing_system_prompt(canvas_width: int, canvas_height: int) -> str:
    """Build base system prompt for pointing task.
    
    Args:
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        
    Returns:
        System prompt string
    """
    return (
        "You are a surgical vision validator looking at ONE image on a fixed canvas.\n"
        'Return STRICT JSON only: {"name":"<organ>", "present":0|1, "point_canvas":[x,y] or null}\n'
        f"- Coordinates: origin=(0,0) is top-left of the CANVAS, x∈[0,{canvas_width-1}], y∈[0,{canvas_height-1}], integers only.\n"
        "- present=1 ONLY if any visible part of the named structure is in view.\n"
        "- If present=1, point_canvas MUST be inside the structure; else use null.\n"
        "- No extra text or markdown."
    )


def build_pointing_system_prompt_strict(canvas_width: int, canvas_height: int) -> str:
    """Build strict system prompt for pointing task.
    
    More explicit instructions for better compliance.
    
    Args:
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        
    Returns:
        System prompt string
    """
    return (
        "You are a surgical vision validator analyzing ONE laparoscopic image.\n"
        "\n"
        "CRITICAL INSTRUCTIONS:\n"
        '1. Return ONLY valid JSON: {"name":"<organ>", "present":0|1, "point_canvas":[x,y] or null}\n'
        "2. NO markdown blocks, NO extra text, NO explanations\n"
        "3. The JSON MUST parse successfully\n"
        "\n"
        "COORDINATE SYSTEM:\n"
        f"- Canvas dimensions: {canvas_width}×{canvas_height} pixels\n"
        "- Origin (0,0) is TOP-LEFT corner\n"
        f"- Valid x-coordinates: 0 to {canvas_width-1}\n"
        f"- Valid y-coordinates: 0 to {canvas_height-1}\n"
        "- All coordinates MUST be integers\n"
        "\n"
        "PRESENCE RULES:\n"
        "- present=1 if ANY visible part of the organ/tool is in the image\n"
        "- present=0 if the organ/tool is NOT visible or you're uncertain\n"
        "\n"
        "POINTING RULES:\n"
        "- If present=1: point_canvas MUST be [x,y] inside the visible organ/tool\n"
        "- If present=0: point_canvas MUST be null\n"
        "- The point should be well within the structure, not on edges"
    )


def build_pointing_system_prompt_qna(canvas_width: int, canvas_height: int) -> str:
    """Build Q&A style system prompt for pointing task.
    
    Uses question-answer format for clarity.
    
    Args:
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        
    Returns:
        System prompt string
    """
    return (
        "You are analyzing laparoscopic surgery images. Answer these questions:\n"
        "\n"
        "Q1: Is the specified organ/tool visible in the image?\n"
        "A1: Answer with 1 (yes) or 0 (no)\n"
        "\n"
        "Q2: If visible, where is it located?\n"
        "A2: Provide [x,y] coordinates or null if not visible\n"
        "\n"
        f"Canvas: {canvas_width}×{canvas_height} pixels, origin at top-left\n"
        f"Valid coordinates: x∈[0,{canvas_width-1}], y∈[0,{canvas_height-1}]\n"
        "\n"
        'Format your response as: {"name":"<organ>", "present":<0|1>, "point_canvas":<[x,y]|null>}\n'
        "Return ONLY the JSON, no other text."
    )


def build_pointing_user_prompt(organ_name: str) -> str:
    """Build user prompt for pointing task.
    
    Args:
        organ_name: Name of organ/tool to detect
        
    Returns:
        User prompt string
    """
    return (
        f'Organ: "{organ_name}". '
        f'Return exactly: {{"name":"{organ_name}", "present":0|1, "point_canvas":[x,y] or null}}'
    )


def build_existence_system_prompt() -> str:
    """Build system prompt for existence detection only.
    
    Returns:
        System prompt string
    """
    return (
        "You are a surgical vision validator. You will be shown one laparoscopic image.\n"
        "Answer STRICTLY with a single word: Yes or No.\n"
        "Rules:\n"
        "- 'Yes' only if ANY visible part of the named structure is present in the image.\n"
        "- If uncertain/occluded/blurred, answer 'No'.\n"
        "- Do not include punctuation, explanation, JSON, or extra words.\n"
    )


def build_existence_user_prompt(organ_name: str) -> str:
    """Build user prompt for existence detection.
    
    Args:
        organ_name: Name of organ/tool to detect
        
    Returns:
        User prompt string
    """
    return f"Question: Is {organ_name} visible in the image?\nAnswer:"


def build_cell_selection_system_prompt(
    canvas_width: int, 
    canvas_height: int, 
    grid_size: int, 
    top_k: int = 1
) -> str:
    """Build system prompt for cell selection task.
    
    Args:
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        grid_size: Grid size (e.g., 3 for 3x3 grid)
        top_k: Maximum number of cells to return
        
    Returns:
        System prompt string
    """
    # Generate cell labels
    cell_labels = []
    for row in range(grid_size):
        row_labels = [chr(65 + row) + str(col + 1) for col in range(grid_size)]
        cell_labels.append(row_labels)
    
    # Format as grid
    grid_str = ""
    for row_labels in cell_labels:
        grid_str += "  " + " | ".join(row_labels) + "\n"
    
    return (
        f"You are a surgical vision validator analyzing ONE laparoscopic image divided into a {grid_size}×{grid_size} grid.\n"
        "\n"
        "GRID LAYOUT:\n"
        f"{grid_str}"
        "\n"
        "INSTRUCTIONS:\n"
        '1. Return ONLY valid JSON: {"name":"<organ>", "present":0|1, "cells":[]}\n'
        "2. NO markdown blocks, NO extra text, NO explanations\n"
        "\n"
        "RULES:\n"
        f"- The image is divided into {grid_size}×{grid_size} cells labeled as shown above\n"
        "- present=1 if ANY part of the organ/tool is visible\n"
        "- present=0 if not visible or uncertain\n"
        f"- If present=1: cells must contain 1 to {top_k} cell label(s) where the organ is located\n"
        "- If present=0: cells must be an empty list []\n"
        "- Choose cells that contain the most significant portion of the organ\n"
        f"- Valid cell labels: {', '.join([label for row in cell_labels for label in row])}"
    )


def build_cell_selection_system_prompt_strict(
    canvas_width: int, 
    canvas_height: int, 
    grid_size: int, 
    top_k: int = 1
) -> str:
    """Build strict system prompt for cell selection with more explicit instructions.
    
    Args:
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels  
        grid_size: Grid size (e.g., 3 for 3x3 grid)
        top_k: Maximum number of cells to return
        
    Returns:
        System prompt string
    """
    # Generate all valid cell labels
    valid_cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            valid_cells.append(chr(65 + row) + str(col + 1))
    
    # Create visual grid
    grid_visual = "```\n"
    for row in range(grid_size):
        row_cells = [chr(65 + row) + str(col + 1) for col in range(grid_size)]
        grid_visual += " | ".join(f"[{cell}]" for cell in row_cells) + "\n"
        if row < grid_size - 1:
            grid_visual += "-" * (7 * grid_size - 1) + "\n"
    grid_visual += "```"
    
    return (
        f"You are analyzing a surgical image divided into a {grid_size}×{grid_size} grid of cells.\n"
        "\n"
        f"GRID VISUALIZATION:\n"
        f"{grid_visual}\n"
        "\n"
        "CRITICAL REQUIREMENTS:\n"
        '1. Output MUST be valid JSON: {"name":"<organ>", "present":0|1, "cells":[...]}\n'
        "2. NO markdown, NO backticks, NO explanations - ONLY the JSON\n"
        "3. Cell labels are CASE-SENSITIVE (use uppercase letters)\n"
        "\n"
        "DETECTION RULES:\n"
        "- Set present=1 if you can see ANY part of the specified organ/tool\n"
        "- Set present=0 if you cannot see it or are uncertain\n"
        "\n"
        "CELL SELECTION RULES:\n"
        f"- If present=1: Select 1 to {top_k} cell(s) that contain the organ\n"
        "- If present=0: cells MUST be empty list []\n"
        "- Choose cells with the largest/clearest portion of the organ\n"
        f"- ONLY use these valid cells: {', '.join(valid_cells)}\n"
        "- Order cells by relevance (most significant portion first)"
    )


def build_cell_selection_user_prompt(organ_name: str, grid_size: int = 3) -> str:
    """Build user prompt for cell selection task.
    
    Args:
        organ_name: Name of organ/tool to detect
        grid_size: Grid size for reference
        
    Returns:
        User prompt string
    """
    return (
        f'Detect "{organ_name}" in the {grid_size}×{grid_size} grid.\n'
        f'Return: {{"name":"{organ_name}", "present":0|1, "cells":[]}}'
    )