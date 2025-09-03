#!/usr/bin/env python3
"""Test Claude API with CholecSeg8k data."""

import os
import sys
import json
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test if required modules can be imported."""
    print("Testing imports...")
    
    try:
        import anthropic
        print("✓ anthropic module imported successfully")
        print(f"  anthropic version: {anthropic.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import anthropic: {e}")
        print("  Install with: pip install anthropic")
        return False

def load_api_keys():
    """Load API keys from API_KEYS2.json."""
    print("\nLoading API keys...")
    
    api_keys_path = Path(__file__).parent.parent / "API_KEYS2.json"
    if not api_keys_path.exists():
        print(f"✗ API keys file not found: {api_keys_path}")
        print("  Create API_KEYS2.json with your ANTHROPIC_API_KEY")
        return None
    
    try:
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
        
        anthropic_key = api_keys.get('ANTHROPIC_API_KEY')
        if not anthropic_key:
            print("✗ ANTHROPIC_API_KEY not found in API_KEYS2.json")
            return None
        
        print("✓ API keys loaded successfully")
        return anthropic_key
    except Exception as e:
        print(f"✗ Error loading API keys: {e}")
        return None

def test_claude_models(api_key):
    """Test different Claude model names."""
    print("\nTesting Claude model names...")
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        print(f"✗ Failed to create Anthropic client: {e}")
        return []
    
    # Test different model names
    model_names = [
        "claude-sonnet-4-20250514",  # The correct name to use
        "claude-3-5-sonnet-latest",  # Current latest
        "claude-3-5-sonnet-20241022",  # Specific version
        "claude-3-sonnet-20240229",  # Claude 3 Sonnet
        "claude-3-haiku-20240307",  # Claude 3 Haiku (faster)
    ]
    
    working_models = []
    
    for model_name in model_names:
        print(f"\n  Testing model: {model_name}")
        try:
            response = client.messages.create(
                model=model_name,
                messages=[{"role": "user", "content": "Respond with just 'OK' in one word"}],
                max_tokens=10,
                temperature=0
            )
            response_text = response.content[0].text.strip()
            print(f"    ✓ Success: '{response_text}'")
            working_models.append(model_name)
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    return working_models

def test_cholecseg8k_data(api_key, model_name):
    """Test Claude with actual CholecSeg8k data."""
    print(f"\nTesting with CholecSeg8k data using {model_name}...")
    
    try:
        # Import dataset utilities
        from endopoint.datasets.cholecseg8k import CholecSeg8kAdapter
        print("✓ Successfully imported CholecSeg8k adapter")
    except Exception as e:
        print(f"✗ Failed to import CholecSeg8k adapter: {e}")
        print("  Make sure the endopoint package is installed")
        return False
    
    try:
        # Initialize dataset
        dataset = CholecSeg8kAdapter()
        print("✓ CholecSeg8k dataset initialized")
        
        # Get a test example
        example = dataset.get_example('train', 0)
        print(f"✓ Retrieved test example: {example.sample_id}")
        
        # Convert to tensor format
        img_tensor, lab_tensor = dataset.example_to_tensors(example)
        print(f"✓ Converted to tensors: image {img_tensor.shape}, labels {lab_tensor.shape}")
        
    except Exception as e:
        print(f"✗ Error with CholecSeg8k data: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test with Claude model adapter
        from endopoint.models import create_model
        model = create_model(model_name, use_cache=False)
        print(f"✓ Created model adapter for {model_name}")
        
        # Simple test prompt
        test_prompt = [
            "Look at this medical image and tell me if you can see it clearly. Respond with just 'YES' or 'NO'.",
            img_tensor
        ]
        
        system_prompt = "You are a medical image analysis assistant."
        
        # Make API call
        response = model._one_call((test_prompt,), system_prompt)
        print(f"✓ API call successful")
        print(f"  Response: '{response[:100]}{'...' if len(response) > 100 else ''}'")
        
        if response.strip():
            print("✓ Received non-empty response")
            return True
        else:
            print("✗ Received empty response")
            return False
            
    except Exception as e:
        print(f"✗ Error testing with model: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation with problematic model name."""
    print("\nTesting model creation...")
    
    try:
        from endopoint.models import create_model
        
        # Test the corrected model name
        print("  Testing 'claude-sonnet-4-20250514' model creation...")
        model = create_model("claude-sonnet-4-20250514", use_cache=False)
        print(f"    ✓ Model created: {type(model).__name__}")
        print(f"    Model name passed to adapter: {model.model_name}")
        
        return model
        
    except Exception as e:
        print(f"    ✗ Error creating model: {e}")
        traceback.print_exc()
        return None

def main():
    """Run all tests."""
    print("=" * 60)
    print("Claude API Test Suite for CholecSeg8k")
    print("=" * 60)
    
    # Test 1: Check imports
    if not test_imports():
        print("\n❌ Import test failed - cannot proceed with API tests")
        return
    
    # Test 2: Load API keys
    api_key = load_api_keys()
    if not api_key:
        print("\n❌ API key test failed - cannot proceed with API tests")
        return
    
    # Test 3: Test model creation
    model = test_model_creation()
    if not model:
        print("\n❌ Model creation failed")
        return
    
    # Test 4: Test different model names
    working_models = test_claude_models(api_key)
    
    if not working_models:
        print("\n❌ No working Claude models found")
        return
    
    print(f"\n✓ Found {len(working_models)} working model(s):")
    for model_name in working_models:
        print(f"  - {model_name}")
    
    # Test 5: Test with CholecSeg8k data
    success = False
    for model_name in working_models[:1]:  # Test with first working model
        if test_cholecseg8k_data(api_key, model_name):
            success = True
            break
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
        print(f"\nRecommendation: Use '{working_models[0]}' as the correct model name")
    else:
        print("❌ Some tests failed")
        
    print(f"Working models: {working_models}")

if __name__ == "__main__":
    main()