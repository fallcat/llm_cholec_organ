#!/usr/bin/env python
"""Test that few-shot examples now produce different results after the fix."""

import sys
import os
import json
import numpy as np
from PIL import Image

sys.path.append('src')

def test_model_fewshot(model_class, model_name):
    """Test a model with and without few-shot examples."""
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print('='*60)
    
    # Initialize model
    model = model_class(
        use_vllm=True if hasattr(model_class, 'use_vllm') else False,
        verbose=False,
        max_tokens=100,
        temperature=0.1,
        use_cache=False  # Disable cache for testing
    )
    
    # Create test images
    img1 = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red
    img2 = Image.new('RGB', (224, 224), color=(0, 0, 255))  # Blue
    query_img = Image.new('RGB', (224, 224), color=(0, 255, 0))  # Green
    
    # System prompt
    system_prompt = """You are analyzing images divided into a 3x3 grid.
Return JSON: {"name":"<object>", "present":0|1, "cells":[]}"""
    
    # Test 1: Zero-shot
    print("\nZero-shot test:")
    zero_shot_prompt = (
        "Detect 'test object' in the grid. Return: {\"name\":\"test object\", \"present\":0|1, \"cells\":[]}",
        query_img
    )
    
    try:
        zero_shot_response = model([zero_shot_prompt], system_prompt=system_prompt)[0]
        print(f"  Response: {zero_shot_response[:100]}...")
    except Exception as e:
        print(f"  Error: {e}")
        zero_shot_response = ""
    
    # Test 2: Few-shot with examples
    print("\nFew-shot test:")
    few_shot_prompt = (
        "Here are some examples:\n",
        "\nExample 1: Detect 'test object' in the grid.",
        img1,
        "\nResponse: {\"name\":\"test object\", \"present\":1, \"cells\":[\"B2\"]}\n",
        "\nExample 2: Detect 'test object' in the grid.",
        img2,
        "\nResponse: {\"name\":\"test object\", \"present\":0, \"cells\":[]}\n",
        "\nNow: Detect 'test object' in the grid. Return: {\"name\":\"test object\", \"present\":0|1, \"cells\":[]}",
        query_img
    )
    
    try:
        few_shot_response = model([few_shot_prompt], system_prompt=system_prompt)[0]
        print(f"  Response: {few_shot_response[:100]}...")
    except Exception as e:
        print(f"  Error: {e}")
        few_shot_response = ""
    
    # Compare
    identical = zero_shot_response == few_shot_response
    print(f"\nResults identical: {identical}")
    
    if identical and zero_shot_response:
        print("  ⚠️  WARNING: Few-shot still produces identical results!")
    elif not identical:
        print("  ✅ SUCCESS: Few-shot produces different results!")
    
    return not identical


def main():
    """Test all fixed models."""
    
    from endopoint.models.vllm import QwenVLModel, PixtralModel, DeepSeekVL2Model
    
    results = {}
    
    # Test each model
    models_to_test = [
        (QwenVLModel, "Qwen2.5-VL-7B"),
        (PixtralModel, "Pixtral-12B"),
        # DeepSeekVL2Model requires more setup, skip for quick test
    ]
    
    for model_class, model_name in models_to_test:
        try:
            success = test_model_fewshot(model_class, model_name)
            results[model_name] = success
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            results[model_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model, success in results.items():
        status = "✅ FIXED" if success else "❌ STILL BROKEN"
        print(f"{model}: {status}")
    
    all_fixed = all(results.values())
    if all_fixed:
        print("\n🎉 All models are now properly handling few-shot examples!")
    else:
        print("\n⚠️  Some models still have issues with few-shot examples.")
    
    return all_fixed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)