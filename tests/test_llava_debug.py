#!/usr/bin/env python3
"""Debug script for LLaVA model issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_llava_direct():
    """Test LLaVA model directly."""
    print("Testing LLaVA model...")
    
    # Import from endopoint
    from endopoint.models.vllm import LLaVAModel
    from endopoint.datasets import build_dataset
    
    # Try with verbose mode to see what's happening
    print("\n1. Loading LLaVA with verbose=True")
    model = LLaVAModel(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        use_vllm=True,  # Try vLLM first
        max_tokens=100,
        temperature=0.0,
        verbose=True,  # Enable verbose for debugging
        use_cache=False  # Disable cache for fresh responses
    )
    print("Model loaded successfully")
    
    # Load dataset for real image
    print("\n2. Loading dataset...")
    dataset = build_dataset("cholecseg8k")
    example = dataset.get_example("train", 0)
    image = example['image']
    
    # Test simple query
    print("\n3. Testing simple query...")
    prompt = "Is this a medical image? Answer with JSON: {\"medical\": true/false}"
    
    # Test the model call directly
    response = model.one_call((prompt, image))
    print(f"Response: '{response}'")
    print(f"Response length: {len(response)}")
    print(f"Response type: {type(response)}")
    
    # Test with system prompt
    print("\n4. Testing with system prompt...")
    system_prompt = "You are a medical image analyst. Respond only with valid JSON."
    response2 = model.one_call((prompt, image), system_prompt=system_prompt)
    print(f"Response with system: '{response2}'")
    
    # Test batch call
    print("\n5. Testing batch call (as used in pointing.py)...")
    response3 = model([(prompt, image)], system_prompt=system_prompt)
    print(f"Batch response: {response3}")
    print(f"Batch response type: {type(response3)}")
    if response3:
        print(f"First element: '{response3[0]}'")
    
    return True

if __name__ == "__main__":
    try:
        test_llava_direct()
        print("\n✅ Test completed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()