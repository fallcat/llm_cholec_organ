#!/usr/bin/env python3
"""Test script to verify mask saving functionality for GoNoGoNet and CholeNet."""

import os
import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

# Setup API keys
api_keys_file = Path("/shared_data0/weiqiuy/llm_cholec_organ/API_KEYS2.json")
if api_keys_file.exists():
    with open(api_keys_file, "r") as f:
        api_keys = json.load(f)
    os.environ['OPENAI_API_KEY'] = api_keys.get('OPENAI_API_KEY', '')
    os.environ['ANTHROPIC_API_KEY'] = api_keys.get('ANTHROPIC_API_KEY', '')
    os.environ['GOOGLE_API_KEY'] = api_keys.get('GOOGLE_API_KEY', '')


def test_mask_saving():
    """Test mask saving with GoNoGoNet and CholeNet on a few samples."""
    
    print("=" * 80)
    print("TESTING MASK SAVING FUNCTIONALITY")
    print("=" * 80)
    
    # Test GoNoGoNet on CholecGoNoGo
    print("\n1. Testing GoNoGoNet on CholecGoNoGo...")
    print("-" * 60)
    
    # Run evaluation with just 3 samples
    os.system("""
    cd /shared_data0/weiqiuy/llm_cholec_organ/notebooks_py && \
    EVAL_MODEL=gonogonet \
    EVAL_DATASET=cholec_gonogo \
    EVAL_NUM_SAMPLES=3 \
    EVAL_DETECTION_MODE=combined \
    EVAL_FEWSHOT=0 \
    python3 eval_bbox_unified.py
    """)
    
    # Check if masks were saved
    gonogo_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_gonogo_quick/zeroshot_combined/gonogonet")
    mask_dir = gonogo_dir / "masks"
    
    if mask_dir.exists():
        mask_files = list(mask_dir.glob("*.npy"))
        print(f"\n✓ Found {len(mask_files)} mask files in {mask_dir}")
        
        # Load and check a mask
        if mask_files:
            test_mask = np.load(mask_files[0])
            print(f"  Sample mask shape: {test_mask.shape}")
            print(f"  Unique values: {np.unique(test_mask)}")
    else:
        print(f"\n✗ No masks directory found at {mask_dir}")
    
    # Test CholeNet on CholecOrgans
    print("\n2. Testing CholeNet on CholecOrgans...")
    print("-" * 60)
    
    os.system("""
    cd /shared_data0/weiqiuy/llm_cholec_organ/notebooks_py && \
    EVAL_MODEL=cholenet \
    EVAL_DATASET=cholec_organs \
    EVAL_NUM_SAMPLES=3 \
    EVAL_DETECTION_MODE=combined \
    EVAL_FEWSHOT=0 \
    python3 eval_bbox_unified.py
    """)
    
    # Check if masks were saved
    cholenet_dir = Path("/shared_data0/weiqiuy/llm_cholec_organ/results/bbox_cholec_organs_quick/zeroshot_combined/cholenet")
    mask_dir = cholenet_dir / "masks"
    
    if mask_dir.exists():
        mask_files = list(mask_dir.glob("*.npy"))
        print(f"\n✓ Found {len(mask_files)} mask files in {mask_dir}")
        
        # Load and check a mask
        if mask_files:
            test_mask = np.load(mask_files[0])
            print(f"  Sample mask shape: {test_mask.shape}")
            print(f"  Unique values: {np.unique(test_mask)}")
    else:
        print(f"\n✗ No masks directory found at {mask_dir}")
    
    # Check JSON files for mask references
    print("\n3. Checking JSON files for mask references...")
    print("-" * 60)
    
    # Check GoNoGoNet JSON
    gonogo_json_files = list(gonogo_dir.glob("test_*.json"))[:3]
    for json_file in gonogo_json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n{json_file.name}:")
        for organ in data.get('organs', []):
            organ_name = organ.get('organ_name', 'Unknown')
            has_mask = organ.get('has_mask', False)
            mask_file = organ.get('mask_file', None)
            print(f"  {organ_name}: has_mask={has_mask}, mask_file={mask_file}")
    
    print("\n" + "=" * 80)
    print("MASK SAVING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_mask_saving()