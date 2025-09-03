#!/usr/bin/env python3
"""Test Gemini fix."""

import os
import sys
import json

# Add src to path
sys.path.append('./src')

# Load API keys
with open("API_KEYS2.json", 'r') as f:
    API_KEYS = json.load(f)
    os.environ["GOOGLE_API_KEY"] = API_KEYS.get("GOOGLE_API_KEY", "")

from endopoint.models import create_model
from PIL import Image
import torch

# Test Gemini
print("Testing Gemini model fix...")
print("="*60)

# Create model
model = create_model("gemini-1.5-flash", use_cache=False, verbose=True)
print(f"✅ Model created: gemini-1.5-flash")

# Test with text only first
print("\n1. Testing text-only prompt:")
response = model([("What is 2+2?",)], system_prompt="Answer concisely")[0]
if response:
    print(f"   ✅ Response: {response}")
else:
    print(f"   ⚠️ Empty response")

# Test with image
print("\n2. Testing with image:")
# Create a simple test image
test_image = Image.new('RGB', (100, 100), color='red')
response = model([(test_image, "What color is this image?")], system_prompt="Answer concisely")[0]
if response:
    print(f"   ✅ Response: {response}")
else:
    print(f"   ⚠️ Empty response")

print("\nDone!")