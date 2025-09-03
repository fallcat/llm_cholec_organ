#!/usr/bin/env python
"""Test to diagnose why LLaVA few-shot produces identical results to zero-shot."""

import sys
import os
import json
import numpy as np
from PIL import Image

sys.path.append('src')

from endopoint.models.vllm import LLaVAModel

def test_llava_fewshot():
    """Test LLaVA with and without few-shot examples."""
    
    print("Loading LLaVA model...")
    model = LLaVAModel(
        use_vllm=True,
        verbose=True,
        max_tokens=100,
        temperature=0.1,
        use_cache=False  # Disable cache for testing
    )
    
    # Create test images
    img1 = Image.new('RGB', (224, 224), color='red')
    img2 = Image.new('RGB', (224, 224), color='blue')
    query_img = Image.new('RGB', (224, 224), color='green')
    
    # System prompt
    system_prompt = """You are analyzing images divided into a 3x3 grid.
Grid layout:
  A1 | A2 | A3
  B1 | B2 | B3
  C1 | C2 | C3

Return JSON: {"name":"<object>", "present":0|1, "cells":[]}"""
    
    # Test 1: Zero-shot
    print("\n=== Test 1: Zero-shot ===")
    zero_shot_prompt = ("Detect 'red object' in the 3x3 grid.\nReturn: {\"name\":\"red object\", \"present\":0|1, \"cells\":[]}", query_img)
    
    zero_shot_response = model([zero_shot_prompt], system_prompt=system_prompt)[0]
    print(f"Zero-shot response: {zero_shot_response}")
    
    # Test 2: Few-shot with examples
    print("\n=== Test 2: Few-shot ===")
    few_shot_prompt = (
        "Here are some examples:\n",
        "\nExample 1: Detect 'red object' in the 3x3 grid.\nReturn: {\"name\":\"red object\", \"present\":0|1, \"cells\":[]}",
        img1,
        "\nResponse: {\"name\":\"red object\", \"present\":1, \"cells\":[\"B2\"]}\n",
        "\nExample 2: Detect 'red object' in the 3x3 grid.\nReturn: {\"name\":\"red object\", \"present\":0|1, \"cells\":[]}",
        img2,
        "\nResponse: {\"name\":\"red object\", \"present\":0, \"cells\":[]}\n",
        "\nNow for the actual query: Detect 'red object' in the 3x3 grid.\nReturn: {\"name\":\"red object\", \"present\":0|1, \"cells\":[]}",
        query_img
    )
    
    few_shot_response = model([few_shot_prompt], system_prompt=system_prompt)[0]
    print(f"Few-shot response: {few_shot_response}")
    
    # Test 3: Check if prompt is being built correctly
    print("\n=== Test 3: Prompt inspection ===")
    
    # Try to see what the actual prompt looks like by temporarily modifying verbose output
    # This would require checking the internal prompt construction
    
    # Compare responses
    print("\n=== Comparison ===")
    print(f"Zero-shot == Few-shot: {zero_shot_response == few_shot_response}")
    
    if zero_shot_response == few_shot_response:
        print("WARNING: Responses are identical! Few-shot examples may not be working.")
        
        # Additional diagnostic: Try with a simpler prompt
        print("\n=== Test 4: Simple text-only few-shot ===")
        simple_few_shot = (
            "Example: What color? Answer: red\n",
            "Example: What color? Answer: blue\n", 
            "Now: What color?"
        )
        simple_response = model(simple_few_shot)
        print(f"Simple few-shot response: {simple_response}")
    
    return zero_shot_response != few_shot_response

if __name__ == "__main__":
    success = test_llava_fewshot()
    if success:
        print("\n✓ LLaVA few-shot is working differently from zero-shot")
    else:
        print("\n✗ LLaVA few-shot produces identical results to zero-shot")
    sys.exit(0 if success else 1)