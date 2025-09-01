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