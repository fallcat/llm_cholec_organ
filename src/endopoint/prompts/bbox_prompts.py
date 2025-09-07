"""
Bounding box prompt templates for organ detection.
"""

from typing import Dict, List, Optional


def get_combined_bbox_prompt(
    organ_names: List[str],
    prompt_type: str = "standard",
    use_fewshot: bool = False,
    examples: Optional[List[Dict]] = None,
    canvas_width: int = 768,
    canvas_height: int = 768
) -> str:
    """Generate a combined bounding box detection prompt for all organs.
    
    Args:
        organ_names: List of organ names to detect
        prompt_type: Type of prompt ("standard" or "strict")
        use_fewshot: Whether to include few-shot examples
        examples: Few-shot examples if use_fewshot is True
        canvas_width: Width of the canvas
        canvas_height: Height of the canvas
        
    Returns:
        Formatted prompt string for all organs
    """
    
    organ_list = "\n".join([f"- {name}" for name in organ_names])
    
    if use_fewshot and examples:
        # Format few-shot examples
        examples_text = "Here are examples of correct organ detection:\n\n"
        for i, ex in enumerate(examples[:3], 1):  # Use up to 3 examples
            examples_text += f"Example {i}:\n"
            examples_text += "{\n"
            
            # Handle both formats: 'detections' or direct 'bboxes' format
            if 'detections' in ex:
                # Old format
                for organ, data in ex['detections'].items():
                    present = data.get('present', False)
                    bbox = data.get('bbox', [])
                    examples_text += f'  "{organ}": {{\n'
                    examples_text += f'    "present": {str(present).lower()},\n'
                    if present and bbox:
                        examples_text += f'    "bbox": [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n'
                    examples_text += f'  }},\n'
            elif 'bboxes' in ex:
                # New combined format
                for organ in organ_names:
                    if organ in ex.get('organs_present', []):
                        # Organ is present - get bboxes
                        organ_bboxes = ex['bboxes'].get(organ, {}).get('bboxes', [])
                        examples_text += f'  "{organ}": {{\n'
                        examples_text += f'    "present": true'
                        if organ_bboxes:
                            # Use first bbox if multiple
                            bbox = organ_bboxes[0]
                            examples_text += f',\n    "bbox": [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]'
                        examples_text += f'\n  }},\n'
                    else:
                        # Organ is not present
                        examples_text += f'  "{organ}": {{\n'
                        examples_text += f'    "present": false\n'
                        examples_text += f'  }},\n'
            
            examples_text = examples_text.rstrip(',\n') + '\n}\n\n'
        
        prompt = f"""{examples_text}
Now analyze the current image and detect these organs:

{organ_list}

Return your response as a JSON object with this format:
{{
  "Organ Name": {{
    "present": true/false,
    "bbox": [x1, y1, x2, y2]  // only if present is true
  }},
  ...
}}

IMPORTANT CONSTRAINTS:
- x1, x2 must be integers between 0 and {canvas_width - 1}
- y1, y2 must be integers between 0 and {canvas_height - 1}
- x2 > x1 and y2 > y1 (bottom-right must be greater than top-left)

Canvas dimensions: {canvas_width}x{canvas_height}"""
        
    elif prompt_type == "strict":
        prompt = f"""You are an expert medical image analyst specializing in laparoscopic surgery.

TASK: Detect and localize ALL visible organs in this surgical image.

Organs to detect:
{organ_list}

CRITICAL INSTRUCTIONS:
1. For each organ, determine if it is present (visible with at least 50 pixels)
2. If present, provide bounding box in format [x1, y1, x2, y2]
3. COORDINATES MUST BE:
   - x1, x2: integers between 0 and {canvas_width - 1}
   - y1, y2: integers between 0 and {canvas_height - 1}
   - x2 > x1 and y2 > y1 (bottom-right must be greater than top-left)
4. Return ONLY valid JSON, no additional text

OUTPUT FORMAT (strict JSON):
{{
  "Organ Name": {{
    "present": true/false,
    "bbox": [x1, y1, x2, y2]  // only if present is true
  }},
  ...
}}"""
    else:  # standard
        prompt = f"""Please analyze this surgical image and detect the following organs:

{organ_list}

For each organ that is present (visible with at least 50 pixels), provide the bounding box in format [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right.

COORDINATE REQUIREMENTS:
- x1, x2 must be integers between 0 and {canvas_width - 1}
- y1, y2 must be integers between 0 and {canvas_height - 1}
- x2 > x1 and y2 > y1 (bottom-right must be greater than top-left)

Return your response as a JSON object with this format:
{{
  "Organ Name": {{
    "present": true/false,
    "bbox": [x1, y1, x2, y2]  // only if present is true
  }},
  ...
}}

IMPORTANT CONSTRAINTS:
- x1, x2 must be integers between 0 and {canvas_width - 1}
- y1, y2 must be integers between 0 and {canvas_height - 1}
- x2 > x1 and y2 > y1 (bottom-right must be greater than top-left)

Canvas dimensions: {canvas_width}x{canvas_height}"""
    
    return prompt


def get_bbox_prompt(
    organ_name: str,
    prompt_type: str = "standard",
    use_fewshot: bool = False,
    examples: Optional[Dict] = None,
    canvas_width: int = 768,
    canvas_height: int = 768
) -> str:
    """Generate a bounding box detection prompt.
    
    Args:
        organ_name: Name of the organ to detect
        prompt_type: Type of prompt ("standard" or "strict")
        use_fewshot: Whether to include few-shot examples
        examples: Few-shot examples if use_fewshot is True
        canvas_width: Width of the canvas
        canvas_height: Height of the canvas
        
    Returns:
        Formatted prompt string
    """
    
    # Base instruction
    if prompt_type == "strict":
        base_prompt = f"""You are an expert medical image analyst specializing in laparoscopic surgery.

TASK: Detect and localize the organ "{organ_name}" in this surgical image.

CRITICAL INSTRUCTIONS:
1. Determine if the organ is present (visible with at least 50 pixels)
2. If present, provide bounding box(es) in format [x1, y1, x2, y2]
3. COORDINATES MUST BE:
   - x1, x2: integers between 0 and {canvas_width - 1}
   - y1, y2: integers between 0 and {canvas_height - 1}
   - x2 > x1 and y2 > y1 (bottom-right must be greater than top-left)
4. Support up to 3 bounding boxes for disconnected segments
5. Return ONLY valid JSON, no additional text

OUTPUT FORMAT (strict JSON):
{{
  "name": "{organ_name}",
  "present": 0 or 1,
  "bboxes": [[x1, y1, x2, y2], ...] or []
}}

COORDINATE SYSTEM:
- Origin (0,0) is at top-left corner
- x increases rightward (valid range: 0 to {canvas_width - 1})
- y increases downward (valid range: 0 to {canvas_height - 1})
- Bounding box format: [left, top, right, bottom]
- All coordinates must be integers
- Constraint: x2 > x1 and y2 > y1

EXAMPLES:
- Present with single box: {{"name": "{organ_name}", "present": 1, "bboxes": [[100, 150, 300, 350]]}}
- Present with multiple boxes: {{"name": "{organ_name}", "present": 1, "bboxes": [[100, 150, 300, 350], [400, 200, 500, 400]]}}
- Absent: {{"name": "{organ_name}", "present": 0, "bboxes": []}}
"""
    else:  # standard
        base_prompt = f"""Detect the organ "{organ_name}" in this surgical image.

Canvas size: {canvas_width}x{canvas_height} pixels (origin at top-left)

If the organ is present (≥50 pixels visible), provide bounding box(es).
Support up to 3 boxes for disconnected segments.

COORDINATE REQUIREMENTS:
- x1, x2: integers between 0 and {canvas_width - 1}
- y1, y2: integers between 0 and {canvas_height - 1}
- x2 > x1 and y2 > y1 (bottom-right must be greater than top-left)

Return JSON:
{{
  "name": "{organ_name}",
  "present": 0 or 1,
  "bboxes": [[x1, y1, x2, y2], ...] or []
}}

Bounding box format: [left, top, right, bottom]"""
    
    # Add few-shot examples if provided
    if use_fewshot and examples:
        examples_text = "\n\nFEW-SHOT EXAMPLES:\n"
        
        # Add positive examples
        if 'positives' in examples:
            for pos_ex in examples['positives']:
                if 'bboxes' in pos_ex:
                    examples_text += f"\nExample (present):\n"
                    examples_text += f'Output: {{"name": "{organ_name}", "present": 1, "bboxes": {pos_ex["bboxes"]}}}\n'
        
        # Add negative examples (absent)
        if 'negatives_absent' in examples:
            for neg_idx in examples['negatives_absent'][:1]:  # Just show one
                examples_text += f"\nExample (absent):\n"
                examples_text += f'Output: {{"name": "{organ_name}", "present": 0, "bboxes": []}}\n'
        
        # Add wrong bbox examples if available
        if 'negatives_wrong_bbox' in examples:
            for wrong_ex in examples['negatives_wrong_bbox'][:1]:
                if 'wrong_bbox' in wrong_ex:
                    examples_text += f"\nExample (incorrect - do NOT predict this):\n"
                    examples_text += f'Wrong: {{"name": "{organ_name}", "present": 1, "bboxes": [{wrong_ex["wrong_bbox"]}]}}\n'
                    examples_text += f'Correct: {{"name": "{organ_name}", "present": 0, "bboxes": []}}\n'
        
        base_prompt += examples_text
    
    # Add final reminder
    if prompt_type == "strict":
        base_prompt += "\n\nREMEMBER: Output ONLY valid JSON for the organ detection task."
    else:
        base_prompt += f"\n\nNow detect '{organ_name}' in the image and return the JSON result."
    
    return base_prompt