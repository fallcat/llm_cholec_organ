#!/usr/bin/env python3
"""Test Claude API connection and response."""

import os
import json
import anthropic

# Load API key
api_keys_path = "/shared_data0/weiqiuy/llm_cholec_organ/API_KEYS2.json"
if os.path.exists(api_keys_path):
    with open(api_keys_path, 'r') as f:
        api_keys = json.load(f)
        os.environ['ANTHROPIC_API_KEY'] = api_keys.get('ANTHROPIC_API_KEY', '')

# Test with different model names
model_names = [
    "claude-sonnet-4-20250514",  # The name being used
    "claude-3-5-sonnet-latest",  # Correct current model name
    "claude-3-5-sonnet-20241022",  # Specific version
    "claude-3-sonnet-20240229",  # Claude 3 Sonnet
]

client = anthropic.Anthropic()

for model_name in model_names:
    print(f"\nTesting model: {model_name}")
    try:
        response = client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say 'hello' in one word"}],
            max_tokens=10,
            temperature=0
        )
        print(f"  ✓ Success: {response.content[0].text}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n\nRecommendation:")
print("Using the correct model name 'claude-sonnet-4-20250514'.")
print("Use 'claude-3-5-sonnet-latest' or 'claude-3-5-sonnet-20241022' instead.")