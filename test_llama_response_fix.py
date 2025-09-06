#!/usr/bin/env python3
"""Test the Llama response extraction fix."""

import sys
sys.path.append('src')

from endopoint.models import LlamaAdapter
from endopoint.datasets import build_dataset

def test_response_extraction():
    """Test that the response extraction properly removes the prompt."""
    
    print("Testing Llama response extraction fix...")
    print("=" * 60)
    
    # Initialize model with verbose to see what's happening
    model = LlamaAdapter(
        model_name="nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature=0.0,
        max_tokens=100,
        use_cache=False,
        verbose=True
    )
    
    # Load dataset
    dataset = build_dataset("cholecseg8k")
    example = dataset.get_example("train", 0)
    image = example['image']
    
    # Test with the exact prompt format used in evaluation
    width, height = image.size
    
    system_prompt = (
        "You are a surgical vision validator looking at ONE image on a fixed canvas.\n"
        f'Return STRICT JSON only: {{"name":"<organ>", "present":0|1, "point_canvas":[x,y] or null}}\n'
        f"- Coordinates: origin=(0,0) is top-left of the CANVAS, x∈[0,{width-1}], y∈[0,{height-1}], integers only.\n"
        "- present=1 ONLY if any visible part of the named structure is in view.\n"
        "- If present=1, point_canvas MUST be inside the structure; else use null.\n"
        "- No extra text or markdown."
    )
    
    user_prompt = (
        'Organ: "Liver". '
        'Return exactly: {"name":"Liver", "present":0|1, "point_canvas":[x,y] or null}'
    )
    
    print(f"System prompt:\n{system_prompt}\n")
    print(f"User prompt:\n{user_prompt}\n")
    print("=" * 60)
    
    # Query model
    query = (image, user_prompt)
    response = model([query], system_prompt=system_prompt)[0]
    
    print(f"Response:\n{response}\n")
    print("=" * 60)
    
    # Check if the response still contains prompt markers
    if "user\n" in response:
        print("❌ ERROR: Response still contains 'user' marker")
    else:
        print("✓ 'user' marker removed")
    
    if "assistant\n" in response:
        print("❌ ERROR: Response still contains 'assistant' marker")
    else:
        print("✓ 'assistant' marker removed")
    
    if system_prompt in response:
        print("❌ ERROR: Response still contains system prompt")
    else:
        print("✓ System prompt removed")
    
    if user_prompt in response:
        print("❌ ERROR: Response still contains user prompt")
    else:
        print("✓ User prompt removed")
    
    # Try to parse as JSON
    print("\nTrying to parse response as JSON...")
    try:
        import json
        # Look for JSON in the response
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            print(f"✓ Successfully parsed JSON: {parsed}")
            
            # Check if it has the expected fields
            expected_fields = ["name", "present", "point_canvas"]
            for field in expected_fields:
                if field in parsed:
                    print(f"  ✓ Has field '{field}'")
                else:
                    print(f"  ❌ Missing field '{field}'")
        else:
            print("❌ No JSON found in response")
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
    
    print("\n" + "=" * 60)
    print("Note: The surgical-specific model may not follow the general")
    print("organ detection format. It might be trained for different tasks.")
    print("Consider using a more general Llama vision model if available.")
    print("=" * 60)


if __name__ == "__main__":
    test_response_extraction()