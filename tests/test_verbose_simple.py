#!/usr/bin/env python3
"""
Simple test to verify verbose logging is working.
"""

import os
import sys
import json

# Add src to path
sys.path.append('./src')

# Load API keys
with open("API_KEYS2.json", 'r') as f:
    API_KEYS = json.load(f)
    os.environ["OPENAI_API_KEY"] = API_KEYS.get("OPENAI_API_KEY", "")
    os.environ["GOOGLE_API_KEY"] = API_KEYS.get("GOOGLE_API_KEY", "")

print("="*60)
print("Testing Verbose Logging")
print("="*60)

# Test OpenAI
print("\n1. Testing OpenAI GPT model:")
print("-"*40)
try:
    from endopoint.models.openai_gpt import OpenAIAdapter
    
    # Create with verbose=True (now default)
    model = OpenAIAdapter(model_name="gpt-5-mini", use_cache=False)
    print(f"✅ OpenAI adapter created")
    print(f"   Verbose mode: {model.verbose}")
    print(f"   Model: {model.model_name}")
    
    # Test with an intentionally problematic prompt to trigger error handling
    # Using a simple text-only prompt for testing
    test_prompt = [("Describe what you see in this image.",)]
    system_prompt = "You are a helpful assistant."
    
    print(f"\n   Testing with text-only prompt (should fail or return empty)...")
    response = model(test_prompt, system_prompt=system_prompt)[0]
    
    if response:
        print(f"   Response received: {response[:50]}...")
    else:
        print(f"   Empty response (verbose warning should appear above)")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test Google Gemini
print("\n2. Testing Google Gemini model:")
print("-"*40)
try:
    from endopoint.models.google_gemini import GoogleAdapter
    
    # Create with verbose=True (now default)
    model = GoogleAdapter(model_name="gemini-2.5-flash", use_cache=False)
    print(f"✅ Gemini adapter created")
    print(f"   Verbose mode: {model.verbose}")
    print(f"   Model: {model.model_name}")
    
    # Test with a simple prompt
    test_prompt = [("What is 2+2?",)]
    system_prompt = "You are a helpful math assistant. Answer concisely."
    
    print(f"\n   Testing with simple math prompt...")
    response = model(test_prompt, system_prompt=system_prompt)[0]
    
    if response:
        print(f"   Response received: {response}")
    else:
        print(f"   Empty response (verbose warning should appear above)")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("Verbose logging test complete!")
print("Check above for any error messages or warnings.")
print("="*60)