#!/usr/bin/env python3
"""Test script for VLM model implementations."""

import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from endopoint.models.vllm import (
    LLaVAModel,
    QwenVLModel, 
    PixtralModel,
    DeepSeekVL2Model,
    load_vlm_model
)


def create_test_image(width=224, height=224):
    """Create a simple test image with colored rectangles."""
    img = Image.new('RGB', (width, height), color='white')
    pixels = img.load()
    
    # Add colored rectangles
    # Red rectangle in top-left
    for i in range(50):
        for j in range(50):
            pixels[i, j] = (255, 0, 0)
    
    # Green rectangle in top-right
    for i in range(width-50, width):
        for j in range(50):
            pixels[i, j] = (0, 255, 0)
    
    # Blue rectangle in bottom-left
    for i in range(50):
        for j in range(height-50, height):
            pixels[i, j] = (0, 0, 255)
    
    return img


def test_model(model_class, model_name, use_vllm=True):
    """Test a single VLM model."""
    print(f"\n{'='*60}")
    print(f"Testing {model_class.__name__} with {model_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize model
        print("Initializing model...")
        model = model_class(
            model_name=model_name,
            use_vllm=use_vllm,
            verbose=True,
            use_cache=False,  # Disable cache for testing
            max_tokens=100,
            temperature=0.1
        )
        print("✓ Model initialized successfully")
        
        # Create test image
        test_image = create_test_image()
        
        # Test 1: Text-only prompt
        print("\nTest 1: Text-only prompt")
        response = model("What is 2+2?")
        print(f"Response: {response[:100]}...")
        assert response, "No response for text-only prompt"
        print("✓ Text-only prompt successful")
        
        # Test 2: Image + text prompt
        print("\nTest 2: Image + text prompt")
        response = model((test_image, "Describe what you see in this image."))
        print(f"Response: {response[:100]}...")
        assert response, "No response for image+text prompt"
        print("✓ Image+text prompt successful")
        
        # Test 3: System prompt
        print("\nTest 3: System prompt")
        response = model(
            "What colors do you see?",
            system_prompt="You are a helpful assistant that identifies colors in images."
        )
        print(f"Response: {response[:100]}...")
        assert response, "No response with system prompt"
        print("✓ System prompt successful")
        
        # Test 4: Multiple images (if supported)
        print("\nTest 4: Multiple images")
        try:
            test_image2 = create_test_image(300, 300)
            response = model((test_image, test_image2, "How many images do you see?"))
            print(f"Response: {response[:100]}...")
            print("✓ Multiple images successful")
        except Exception as e:
            print(f"⚠ Multiple images not supported or failed: {e}")
        
        print(f"\n✅ All tests passed for {model_class.__name__}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing {model_class.__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for all VLM models."""
    print("VLM Model Test Suite")
    print("=" * 60)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Define models to test
    models_to_test = [
        # LLaVA models
        (LLaVAModel, "llava-hf/llava-v1.6-mistral-7b-hf", True),
        # (LLaVAModel, "llava-hf/llava-1.5-7b-hf", True),
        
        # Qwen-VL models
        # (QwenVLModel, "Qwen/Qwen2.5-VL-7B-Instruct", True),
        
        # Pixtral model
        # (PixtralModel, "mistralai/Pixtral-12B-2409", True),
        
        # DeepSeek-VL2 model (doesn't use vLLM)
        # (DeepSeekVL2Model, "deepseek-ai/deepseek-vl2", False),
    ]
    
    results = []
    
    # Test each model
    for model_class, model_name, use_vllm in models_to_test:
        # Skip vLLM models if CUDA not available
        if use_vllm and not cuda_available:
            print(f"\nSkipping {model_name} (requires CUDA for vLLM)")
            continue
        
        success = test_model(model_class, model_name, use_vllm)
        results.append((model_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for model_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_name}: {status}")
    
    # Test factory function
    print("\n" + "=" * 60)
    print("Testing factory function")
    print("=" * 60)
    
    try:
        # Test loading by name
        print("Loading LLaVA model via factory...")
        model = load_vlm_model("llava-hf/llava-v1.6-mistral-7b-hf", use_cache=False)
        print("✓ Factory function successful")
    except Exception as e:
        print(f"❌ Factory function failed: {e}")
    
    print("\n✨ Test suite completed!")


if __name__ == "__main__":
    main()