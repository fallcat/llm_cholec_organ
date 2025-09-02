#!/usr/bin/env python3
"""Test the fixed vLLM implementation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from PIL import Image
from endopoint.models import LLaVAModel

print("Testing vLLM fix...")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("⚠️  CUDA not available, cannot test vLLM")
    exit(1)

# Create test image
img = Image.new('RGB', (224, 224), color='red')

try:
    # Load model
    print("\nLoading LLaVA with vLLM...")
    model = LLaVAModel(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        use_vllm=True,
        use_cache=False,
        max_tokens=50,
        verbose=True
    )
    print("✅ Model loaded")
    
    # Test simple inference
    print("\nTesting inference...")
    response = model((img, "What color is this image?"))
    print(f"Response: {response}")
    
    if response:
        print("\n✅ vLLM fix successful!")
    else:
        print("\n⚠️  Empty response")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    print("\nCleanup complete")