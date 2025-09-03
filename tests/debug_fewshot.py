#!/usr/bin/env python
"""Debug script to trace few-shot example handling."""

import sys
import os
import json
import torch
import numpy as np
from PIL import Image

sys.path.append('src')

from endopoint.models.vllm import QwenVLModel
from endopoint.eval.pointing import run_cell_selection_on_canvas

def create_dummy_examples():
    """Create dummy few-shot examples."""
    # Create dummy images and labels
    img1 = torch.rand(3, 224, 224)
    lab1 = torch.zeros(224, 224)
    lab1[50:100, 50:100] = 1  # Object in top-left region
    
    img2 = torch.rand(3, 224, 224)
    lab2 = torch.zeros(224, 224)
    # No object (negative example)
    
    return [
        (img1, {'name': 'TestOrgan', 'present': 1, 'cells': ['A1']}),
        (img2, {'name': 'TestOrgan', 'present': 0, 'cells': []})
    ]

def test_with_debug():
    """Test with debug output."""
    
    print("Loading Qwen model...")
    model = QwenVLModel(
        use_vllm=True,
        verbose=True,
        max_tokens=100,
        temperature=0.1,
        use_cache=False  # Disable cache
    )
    
    # Test image and label
    test_img = torch.rand(3, 224, 224)
    test_lab = torch.zeros(224, 224)
    test_lab[100:150, 100:150] = 1  # Object in center
    
    # Get few-shot examples
    few_shot_examples = create_dummy_examples()
    
    print("\n=== Testing WITH few-shot examples ===")
    print(f"Number of few-shot examples: {len(few_shot_examples)}")
    
    # Add debug wrapper to model
    original_call = model.__call__
    
    def debug_wrapper(prompts, system_prompt=None):
        print("\n--- DEBUG: Model called with ---")
        if isinstance(prompts, list) and len(prompts) > 0:
            prompt = prompts[0]
            if isinstance(prompt, tuple):
                print(f"Prompt is tuple with {len(prompt)} elements:")
                for i, elem in enumerate(prompt):
                    if isinstance(elem, str):
                        print(f"  [{i}] Text: {elem[:100]}..." if len(elem) > 100 else f"  [{i}] Text: {elem}")
                    elif isinstance(elem, Image.Image):
                        print(f"  [{i}] PIL Image: {elem.size}")
                    else:
                        print(f"  [{i}] {type(elem)}")
            else:
                print(f"Prompt type: {type(prompt)}")
        print(f"System prompt: {system_prompt[:100] if system_prompt else None}")
        print("--- END DEBUG ---\n")
        return original_call(prompts, system_prompt)
    
    model.__call__ = debug_wrapper
    
    # Test with few-shot
    result_with = run_cell_selection_on_canvas(
        model=model,
        img_t=test_img,
        lab_t=test_lab,
        organ_name="TestOrgan",
        grid_size=3,
        top_k=1,
        canvas_width=224,
        canvas_height=224,
        prompt_style="standard",
        few_shot_examples=few_shot_examples,
        min_pixels=50
    )
    
    print(f"\nResult WITH few-shot: {result_with}")
    
    # Test without few-shot
    print("\n=== Testing WITHOUT few-shot examples ===")
    
    result_without = run_cell_selection_on_canvas(
        model=model,
        img_t=test_img,
        lab_t=test_lab,
        organ_name="TestOrgan",
        grid_size=3,
        top_k=1,
        canvas_width=224,
        canvas_height=224,
        prompt_style="standard",
        few_shot_examples=None,
        min_pixels=50
    )
    
    print(f"\nResult WITHOUT few-shot: {result_without}")
    
    # Compare
    print("\n=== COMPARISON ===")
    print(f"Predictions identical: {result_with['present'] == result_without['present'] and result_with['cells'] == result_without['cells']}")
    print(f"Raw responses identical: {result_with['raw'] == result_without['raw']}")

if __name__ == "__main__":
    test_with_debug()