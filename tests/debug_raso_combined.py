#!/usr/bin/env python3
"""
Debug RASO evaluation issue with combined detection.
"""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

import json
from pathlib import Path
from PIL import Image
from endopoint.datasets.cholec_organs import CholecOrgansAdapter
from endopoint.models.raso_adapter import RASOAdapter
from endopoint.eval.bbox_evaluator import BBoxPrediction
from endopoint.prompts.bbox_prompts import get_combined_bbox_prompt

def test_raso_pipeline():
    """Test the full RASO pipeline to debug the issue."""
    
    print("="*80)
    print("DEBUGGING RASO COMBINED DETECTION PIPELINE")
    print("="*80)
    
    # Load the CholecOrgans adapter
    print("\n1. Loading CholecOrgans adapter...")
    adapter = CholecOrgansAdapter()
    print(f"   Organ classes: {adapter.id2label}")
    
    # Get organ classes (excluding background)
    organ_classes = {id: adapter.id2label[id] for id in adapter.label_ids}
    print(f"   Evaluation organ classes: {organ_classes}")
    
    # Build the prompt
    print("\n2. Building combined prompt...")
    organ_names = list(organ_classes.values())
    print(f"   Organ names for prompt: {organ_names}")
    
    prompt = get_combined_bbox_prompt(
        organ_names=organ_names,
        prompt_type="standard",
        use_fewshot=False,
        canvas_width=640,
        canvas_height=384
    )
    print(f"   Prompt preview:\n{prompt[:500]}...")
    
    # Extract what organs are in the prompt JSON structure
    print("\n3. Extracting organs from prompt...")
    import re
    json_match = re.search(r'\{"organs":\s*\[([^\]]+)\]', prompt, re.DOTALL)
    if json_match:
        organs_str = json_match.group(1)
        organs_in_prompt = re.findall(r'"([^"]+)"', organs_str)
        print(f"   Organs in prompt JSON: {organs_in_prompt}")
    
    # Load RASO adapter
    print("\n4. Loading RASO adapter...")
    raso = RASOAdapter(
        model_name="raso",
        use_cache=False,
        verbose=True,
        dataset="cholec_organs"
    )
    
    # Get a test image
    print("\n5. Getting test image...")
    example = adapter.get_example_by_global_index(63)
    image = example['image']
    print(f"   Image size: {image.size}")
    
    # Run RASO
    print("\n6. Running RASO...")
    response = raso([(image, prompt)], system_prompt="")[0]
    
    print("\n7. RASO Response:")
    print(response)
    
    # Parse response for each organ
    print("\n8. Parsing response for each organ:")
    print("-"*40)
    
    response_data = json.loads(response)
    print(f"Response keys: {list(response_data.keys())}")
    
    for organ_id, organ_name in organ_classes.items():
        print(f"\nOrgan ID {organ_id}: '{organ_name}'")
        print(f"  Looking for key: '{organ_name}'")
        print(f"  Key in response? {organ_name in response_data}")
        
        pred = BBoxPrediction.from_json(response, organ_name)
        print(f"  BBoxPrediction.present: {pred.present}")
        
        # Also try manual parsing
        if organ_name in response_data:
            organ_data = response_data[organ_name]
            present = organ_data.get('present', False)
            print(f"  Manual parse - present: {present}")

if __name__ == "__main__":
    test_raso_pipeline()