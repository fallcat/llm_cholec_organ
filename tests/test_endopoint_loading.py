#!/usr/bin/env python3
"""Test script to verify loading models from endopoint package."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llms import load_model
from PIL import Image


def test_loading_from_endopoint():
    """Test loading various models from endopoint package."""
    
    print("=" * 60)
    print("Testing Model Loading from Endopoint Package")
    print("=" * 60)
    
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    test_cases = [
        # Test loading API models from endopoint
        {
            "name": "GPT-4 from endopoint",
            "model": "gpt-4o-mini",
            "use_endopoint": True,
            "prompt": "What is 2+2?",
            "expect_success": True
        },
        {
            "name": "GPT-4 from legacy",
            "model": "gpt-4o-mini",
            "use_endopoint": False,
            "prompt": "What is 2+2?",
            "expect_success": True
        },
        {
            "name": "Claude from endopoint",
            "model": "claude-3-haiku-20240307",
            "use_endopoint": True,
            "prompt": "What is 2+2?",
            "expect_success": True
        },
        {
            "name": "Gemini from endopoint",
            "model": "gemini-1.5-flash",
            "use_endopoint": True,
            "prompt": "What is 2+2?",
            "expect_success": True
        },
        # Test loading VLM models (only from endopoint)
        {
            "name": "LLaVA from endopoint",
            "model": "llava-hf/llava-v1.6-mistral-7b-hf",
            "use_endopoint": True,
            "prompt": (test_image, "Describe this image"),
            "expect_success": True,
            "skip_if_no_cuda": True
        },
    ]
    
    import torch
    cuda_available = torch.cuda.is_available()
    
    results = []
    
    for test in test_cases:
        # Skip CUDA-dependent tests if no GPU
        if test.get("skip_if_no_cuda") and not cuda_available:
            print(f"\n⏭ Skipping {test['name']} (requires CUDA)")
            continue
        
        print(f"\n📝 Testing: {test['name']}")
        print(f"   Model: {test['model']}")
        print(f"   Use endopoint: {test['use_endopoint']}")
        
        try:
            # Try to load the model
            model = load_model(
                test['model'],
                use_endopoint=test['use_endopoint'],
                use_cache=False,
                max_tokens=50,
                verbose=True
            )
            print(f"   ✓ Model loaded successfully")
            
            # Check if model has expected attributes
            assert hasattr(model, '__call__'), "Model missing __call__ method"
            assert hasattr(model, 'model_name'), "Model missing model_name attribute"
            print(f"   ✓ Model has required attributes")
            
            # Try a simple inference (if API key is available)
            try:
                if isinstance(test['prompt'], str):
                    response = model(test['prompt'], system_prompt="Be concise")
                else:
                    response = model(test['prompt'])
                    
                if response:
                    print(f"   ✓ Inference successful: {response[:50]}...")
                else:
                    print(f"   ⚠ Empty response (may need API key)")
            except Exception as e:
                print(f"   ⚠ Inference failed (expected if no API key): {e}")
            
            results.append((test['name'], True))
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append((test['name'], False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    # Test that both loading methods produce compatible objects
    print("\n" + "=" * 60)
    print("Testing Compatibility")
    print("=" * 60)
    
    try:
        # Load same model both ways
        model_endopoint = load_model("gpt-4o-mini", use_endopoint=True, use_cache=False)
        model_legacy = load_model("gpt-4o-mini", use_endopoint=False, use_cache=False)
        
        # Check both have same interface
        assert hasattr(model_endopoint, '__call__')
        assert hasattr(model_legacy, '__call__')
        assert hasattr(model_endopoint, 'model_name')
        assert hasattr(model_legacy, 'model_name')
        
        print("✅ Both loading methods produce compatible interfaces")
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
    
    print("\n✨ Testing completed!")


if __name__ == "__main__":
    test_loading_from_endopoint()