#!/usr/bin/env python3
"""Test the per-organ analysis fix."""

import json
from pathlib import Path

# Test with a sample file
test_file = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholecseg8k_local_quick/zeroshot_combined/cholenet/test_00000.json")

if test_file.exists():
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    print("File structure:")
    print(f"Keys: {list(data.keys())}")
    
    if 'organs' in data:
        print(f"Number of organs: {len(data['organs'])}")
        if data['organs']:
            print(f"First organ keys: {list(data['organs'][0].keys())}")
            print(f"First organ name: {data['organs'][0].get('organ_name', 'NOT FOUND')}")
    
    # Test the organ name extraction logic
    for organ in data.get('organs', []):
        # Get organ name - handle different possible field names
        organ_name = None
        for name_field in ['organ_name', 'name', 'organ']:
            if name_field in organ:
                organ_name = organ[name_field]
                break
        
        if not organ_name:
            print(f"Warning: No organ name found, organ keys: {list(organ.keys())}")
        else:
            print(f"Found organ: {organ_name}")
            break  # Just test the first one
else:
    print(f"Test file not found: {test_file}")