#!/usr/bin/env python3
"""
Test RASO response format to debug the evaluation accuracy issue.
"""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.models import create_model
from PIL import Image
import json

def test_raso_response(dataset_name, organ_list, image_path):
    """Test RASO response for a specific dataset."""
    
    print(f"\n{'='*60}")
    print(f"Testing RASO for {dataset_name}")
    print(f"{'='*60}")
    
    # Create RASO model for the dataset
    print(f"Creating RASO model for {dataset_name}...")
    model = create_model('raso', use_cache=False, verbose=True, dataset=dataset_name)
    
    # Create a test prompt with the exact organ names
    prompt = f'''Detect the following organs in the surgical image:
{{
    "organs": {json.dumps(organ_list)}
}}

Return JSON with format:
{{
    "organ_name": {{
        "present": true/false,
        "bbox": null
    }}
}}'''
    
    print(f"\nPrompt organ list: {organ_list}")
    
    # Load the test image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    
    # Run detection
    batch = [(prompt, image)]
    responses = model(batch, system_prompt='You are an expert medical image analyst.')
    
    print(f"\n--- RAW RASO Response ---")
    print(responses[0])
    
    # Try to parse the response
    try:
        parsed = json.loads(responses[0])
        print(f"\n--- Parsed Response ---")
        for organ, data in parsed.items():
            print(f"  {organ}: present={data.get('present', False)}, bbox={data.get('bbox')}")
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    return responses[0]


def main():
    """Test RASO response for all three datasets."""
    
    print("="*80)
    print("RASO RESPONSE FORMAT TEST")
    print("="*80)
    
    # Use an existing image
    test_image = "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib/images/AdnanSet_LC_100_001.png"
    
    # Test 1: CholecSeg8k with exact label names
    print("\n1. CholecSeg8k with label file names:")
    test_raso_response(
        "cholecseg8k",
        ["black background", "abdominal wall", "liver", "gastrointestinal tract", 
         "fat", "grasper", "connective tissue", "blood", "cystic duct",
         "l-hook electrocautery", "gallbladder", "hepatic vein", "liver ligament"],
        test_image
    )
    
    # Test 2: CholecOrgans with capitalized names (as used in evaluation)
    print("\n2. CholecOrgans with CAPITALIZED names (as in evaluation):")
    test_raso_response(
        "cholec_organs",
        ["Liver", "Gallbladder", "Hepatocystic Triangle"],
        test_image
    )
    
    # Test 3: CholecOrgans with lowercase names (as in label file)
    print("\n3. CholecOrgans with lowercase names (as in label file):")
    test_raso_response(
        "cholec_organs",
        ["background", "liver", "gallbladder", "hepatocystic triangle"],
        test_image
    )
    
    # Test 4: CholecGoNoGo
    print("\n4. CholecGoNoGo with exact label names:")
    test_raso_response(
        "cholec_gonogo",
        ["background", "go (safe to incise)", "nogo (unsafe to incise)"],
        test_image
    )
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nKey observations:")
    print("1. Check if RASO preserves the exact organ names from the prompt")
    print("2. Check if capitalization matters")
    print("3. Check if the response keys match what the evaluation expects")


if __name__ == "__main__":
    main()