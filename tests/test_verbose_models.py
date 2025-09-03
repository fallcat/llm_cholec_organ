#!/usr/bin/env python
"""
Test script to verify verbose logging for OpenAI GPT and Gemini models.
Tests a single zero-shot example to check error reporting.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
ROOT_DIR = "."
sys.path.append(os.path.join(ROOT_DIR, 'src'))

# Load API keys
API_KEYS_FILE = os.path.join(ROOT_DIR, "API_KEYS2.json")
if os.path.exists(API_KEYS_FILE):
    with open(API_KEYS_FILE, 'r') as f:
        API_KEYS = json.load(f)
        os.environ["OPENAI_API_KEY"] = API_KEYS.get("OPENAI_API_KEY", "")
        os.environ["GOOGLE_API_KEY"] = API_KEYS.get("GOOGLE_API_KEY", "")

# Import modules
from endopoint.models import create_model
from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter, ID2LABEL
from endopoint.prompts.builders import build_pointing_system_prompt, build_pointing_user_prompt
from datasets import load_dataset

def test_single_example():
    """Test a single example with verbose logging."""
    
    print("="*60)
    print("Testing Verbose Logging for Models")
    print("="*60)
    
    # Load dataset
    print("\n📊 Loading dataset...")
    dataset = load_dataset("minwoosun/CholecSeg8k")
    adapter = CholecSeg8kAdapter()
    
    # Get a single example
    example_idx = 2514  # First test sample
    example = dataset['train'][example_idx]
    img_tensor, lab_tensor = adapter.example_to_tensors(example)
    
    print(f"✅ Loaded example {example_idx}")
    print(f"   Image shape: {img_tensor.size}")
    print(f"   Label shape: {lab_tensor.shape}")
    
    # Test models
    models_to_test = [
        "gpt-5-mini",
        "gemini-2.5-flash"
    ]
    
    # Pick one organ to test
    test_organ = "liver"
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Create model with verbose=True (default now)
            print(f"\n📌 Creating model adapter...")
            model = create_model(model_name, use_cache=False, verbose=True)
            print(f"✅ Model created: {model_name}")
            
            # Build prompts
            system_prompt = build_pointing_system_prompt(
                canvas_width=img_tensor.width,
                canvas_height=img_tensor.height,
                strict=False
            )
            
            user_prompt = build_pointing_user_prompt(
                organ_name=test_organ,
                few_shot_examples=None  # Zero-shot
            )
            
            print(f"\n📝 Prompt details:")
            print(f"   System prompt length: {len(system_prompt)} chars")
            print(f"   User prompt length: {len(user_prompt)} chars")
            print(f"   Testing organ: {test_organ}")
            
            # Create the query
            query = (img_tensor, user_prompt)
            
            # Call the model
            print(f"\n🚀 Calling {model_name}...")
            response = model([query], system_prompt=system_prompt)[0]
            
            if response:
                print(f"\n✅ Response received!")
                print(f"   Response length: {len(response)} chars")
                print(f"   First 100 chars: {response[:100]}...")
            else:
                print(f"\n⚠️ Empty response received")
                
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_single_example()