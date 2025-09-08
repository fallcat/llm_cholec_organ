#!/usr/bin/env python3
"""
Test that RASO now correctly handles capitalized organ names from adapters.
"""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.models import create_model
from PIL import Image
import json

def test_raso_with_adapter_names():
    """Test RASO with the exact capitalized names from adapters."""
    
    print("="*80)
    print("TESTING RASO FIX FOR CAPITALIZATION")
    print("="*80)
    
    # Use an existing test image
    test_image = "/shared_data0/weiqiuy/real_drs/data/abdomen_exlib/images/AdnanSet_LC_100_001.png"
    image = Image.open(test_image)
    
    # Test 1: CholecOrgans with adapter's capitalized names
    print("\n1. CholecOrgans - Testing with Capitalized Names")
    print("-"*60)
    
    model = create_model('raso', use_cache=False, verbose=False, dataset='cholec_organs')
    
    # Use the exact organ names from the CholecOrgans adapter
    prompt = '''Detect the following organs in the surgical image:
{
    "organs": ["Liver", "Gallbladder", "Hepatocystic Triangle"]
}'''
    
    batch = [(prompt, image)]
    responses = model(batch, system_prompt='You are an expert.')
    
    print("Prompt organs: ['Liver', 'Gallbladder', 'Hepatocystic Triangle']")
    print("\nRASO Response:")
    response = json.loads(responses[0])
    for organ, data in response.items():
        status = "✓ Present" if data['present'] else "✗ Not detected"
        print(f"  {organ}: {status}")
    
    # Check if keys match exactly
    expected_keys = ["Liver", "Gallbladder", "Hepatocystic Triangle"]
    actual_keys = list(response.keys())
    
    print(f"\nKey matching test:")
    print(f"  Expected keys: {expected_keys}")
    print(f"  Actual keys: {actual_keys}")
    print(f"  Keys match: {expected_keys == actual_keys}")
    
    # Test 2: CholecSeg8k with adapter's capitalized names
    print("\n2. CholecSeg8k - Testing with Title Case Names")
    print("-"*60)
    
    model = create_model('raso', use_cache=False, verbose=False, dataset='cholecseg8k')
    
    # Use title case names like the adapter
    prompt = '''Detect the following organs in the surgical image:
{
    "organs": ["Liver", "Gallbladder", "Grasper", "Fat", "Blood"]
}'''
    
    batch = [(prompt, image)]
    responses = model(batch, system_prompt='You are an expert.')
    
    print("Prompt organs: ['Liver', 'Gallbladder', 'Grasper', 'Fat', 'Blood']")
    print("\nRASO Response:")
    response = json.loads(responses[0])
    for organ, data in response.items():
        status = "✓ Present" if data['present'] else "✗ Not detected"
        print(f"  {organ}: {status}")
    
    # Test 3: CholecGoNoGo with adapter's format
    print("\n3. CholecGoNoGo - Testing with Adapter Format")
    print("-"*60)
    
    model = create_model('raso', use_cache=False, verbose=False, dataset='cholec_gonogo')
    
    # Use the exact format from the adapter
    prompt = '''Detect the following organs in the surgical image:
{
    "organs": ["Background", "Go (Safe to Incise)", "NoGo (Unsafe to Incise)"]
}'''
    
    batch = [(prompt, image)]
    responses = model(batch, system_prompt='You are an expert.')
    
    print("Prompt organs: ['Background', 'Go (Safe to Incise)', 'NoGo (Unsafe to Incise)']")
    print("\nRASO Response:")
    response = json.loads(responses[0])
    for organ, data in response.items():
        status = "✓ Present" if data['present'] else "✗ Not detected"
        print(f"  {organ}: {status}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nThe RASO adapter should now:")
    print("1. Accept organ names with any capitalization")
    print("2. Return response keys that match the input capitalization")
    print("3. Correctly detect organs regardless of case")
    print("\nThis should fix the 0% accuracy issue in the evaluation.")

if __name__ == "__main__":
    test_raso_with_adapter_names()