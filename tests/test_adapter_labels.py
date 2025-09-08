#!/usr/bin/env python3
"""
Check what labels the adapters are using vs what RASO expects.
"""

import sys
sys.path.insert(0, '/shared_data0/weiqiuy/llm_cholec_organ/src')

from endopoint.datasets.cholecseg8k_local import CholecSeg8kLocalAdapter
from endopoint.datasets.cholec_organs import CholecOrgansAdapter
from endopoint.datasets.cholec_gonogo import CholecGoNoGoAdapter

def check_adapter_labels(name, adapter):
    """Check labels from an adapter."""
    print(f"\n{'='*60}")
    print(f"{name} Adapter Labels")
    print(f"{'='*60}")
    
    if hasattr(adapter, 'id2label'):
        print("id2label mapping:")
        for id, label in adapter.id2label.items():
            print(f"  {id}: '{label}'")
    else:
        print("  No id2label attribute")
    
    if hasattr(adapter, 'label_ids'):
        print(f"\nlabel_ids (excluding background): {adapter.label_ids}")
    else:
        print("  No label_ids attribute")
    
    if hasattr(adapter, 'label2id'):
        print("\nlabel2id mapping:")
        for label, id in adapter.label2id.items():
            print(f"  '{label}': {id}")
    else:
        print("  No label2id attribute")

def main():
    print("="*80)
    print("ADAPTER LABEL COMPARISON")
    print("="*80)
    
    # Check CholecSeg8k
    print("\n1. CholecSeg8k Dataset")
    try:
        adapter = CholecSeg8kLocalAdapter('/shared_data0/weiqiuy/datasets/cholecseg8k')
        check_adapter_labels("CholecSeg8k", adapter)
    except Exception as e:
        print(f"Error loading CholecSeg8k: {e}")
    
    # Check CholecOrgans
    print("\n2. CholecOrgans Dataset")
    try:
        adapter = CholecOrgansAdapter()
        check_adapter_labels("CholecOrgans", adapter)
    except Exception as e:
        print(f"Error loading CholecOrgans: {e}")
    
    # Check CholecGoNoGo
    print("\n3. CholecGoNoGo Dataset")
    try:
        adapter = CholecGoNoGoAdapter()
        check_adapter_labels("CholecGoNoGo", adapter)
    except Exception as e:
        print(f"Error loading CholecGoNoGo: {e}")
    
    # Now compare with RASO label files
    print("\n" + "="*80)
    print("RASO LABEL FILES")
    print("="*80)
    
    import os
    
    raso_labels = {
        'cholecseg8k': '/shared_data0/weiqiuy/github/raso/raso/labels_cholecseg8k.txt',
        'cholec_organs': '/shared_data0/weiqiuy/github/raso/raso/labels_cholec_organs.txt',
        'cholec_gonogo': '/shared_data0/weiqiuy/github/raso/raso/labels_cholec_gonogo.txt'
    }
    
    for dataset, label_file in raso_labels.items():
        print(f"\n{dataset} RASO labels ({label_file}):")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            for i, label in enumerate(labels):
                print(f"  {i}: '{label}'")
        else:
            print(f"  File not found: {label_file}")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)
    print("\n1. Check if adapter labels match RASO label file names")
    print("2. Check capitalization differences")
    print("3. Check if background/black background is handled correctly")
    print("4. Note any naming mismatches that could cause evaluation issues")

if __name__ == "__main__":
    main()