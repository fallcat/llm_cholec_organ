#!/usr/bin/env python3
"""Simple test of LLaVA without vLLM to avoid multiprocessing issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from PIL import Image
from endopoint.models import LLaVAModel

def main():
    print("Testing LLaVA with transformers (no vLLM)...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create test image
    img = Image.new('RGB', (224, 224), color='blue')
    
    try:
        # Load model WITHOUT vLLM to avoid multiprocessing issues
        print("\nLoading LLaVA with transformers backend...")
        model = LLaVAModel(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            use_vllm=False,  # Use transformers instead of vLLM
            use_cache=False,
            max_tokens=50,
            temperature=0.1,
            verbose=True
        )
        print("✅ Model loaded")
        
        # Test 1: Simple inference
        print("\nTest 1: Color identification")
        response = model((img, "What color is this image?"))
        print(f"Response: {response}")
        
        # Test 2: JSON response
        print("\nTest 2: JSON format response")
        prompt = 'Is this a medical image? Respond with JSON: {"medical": true/false}'
        response = model((img, prompt))
        print(f"Response: {response}")
        
        # Test 3: With system prompt
        print("\nTest 3: With system prompt")
        response = model(
            (img, "Describe this image."),
            system_prompt="You are a medical image analyst. Be very brief."
        )
        print(f"Response: {response}")
        
        if response:
            print("\n✅ All tests successful!")
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nCleanup complete")

if __name__ == "__main__":
    main()