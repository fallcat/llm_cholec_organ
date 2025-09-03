#!/usr/bin/env python
"""Quick test to verify DeepSeek-VL2 vLLM fix."""

import sys
import os
sys.path.append('src')

from PIL import Image
import numpy as np

# Test the fix
def test_deepseek_vllm():
    """Test DeepSeek-VL2 with vLLM after fixing the generate call."""
    
    from endopoint.models.vllm import DeepSeekVL2Model
    
    print("Loading DeepSeek-VL2 model...")
    model = DeepSeekVL2Model(
        use_vllm=True,
        verbose=True,
        max_tokens=100,
        temperature=0.1
    )
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Test with simple prompt
    prompt = ("What color is this image?", dummy_image)
    
    print("\nTesting model with prompt:", prompt[0])
    try:
        response = model(prompt)
        print(f"Success! Response: {response}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_deepseek_vllm()
    if success:
        print("\n✓ DeepSeek-VL2 vLLM integration is working!")
    else:
        print("\n✗ DeepSeek-VL2 vLLM integration still has issues.")
    sys.exit(0 if success else 1)