#!/usr/bin/env python3
"""
Simple test for RASO on all three datasets.
"""

import subprocess
import sys
import os

def test_dataset(dataset_name, num_samples=2):
    """Test RASO on a specific dataset."""
    print(f"\n{'='*60}")
    print(f"Testing RASO on {dataset_name}")
    print('='*60)
    
    # Clean previous results
    cmd_clean = f"rm -rf /shared_data0/weiqiuy/llm_cholec_organ/results/bbox_{dataset_name}_quick/zeroshot_combined/raso 2>/dev/null"
    subprocess.run(cmd_clean, shell=True)
    
    # Set environment variables
    env = os.environ.copy()
    env['EVAL_DATASET'] = dataset_name
    env['EVAL_MODEL'] = 'raso'
    env['EVAL_NUM_SAMPLES'] = str(num_samples)
    env['EVAL_USE_CACHE'] = 'false'
    
    cmd = ['python3', 'notebooks_py/eval_bbox_unified.py']
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # Extract key metrics from output
        output = result.stdout
        presence_acc = None
        
        for line in output.split('\n'):
            if 'Presence Accuracy:' in line:
                # Extract percentage
                try:
                    acc_str = line.split(':')[1].strip().replace('%', '')
                    presence_acc = float(acc_str)
                    print(f"  ✓ Presence Accuracy: {presence_acc:.1f}%")
                except:
                    print(f"  {line.strip()}")
                    
        if result.returncode != 0:
            print(f"  ✗ Error occurred")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
                
        return presence_acc
            
    except Exception as e:
        print(f"  ✗ Failed to run: {e}")
        return None

def main():
    """Test RASO on all datasets."""
    print("Testing RASO adapter on all datasets")
    print("="*60)
    
    # Test each dataset with 2 samples for quick testing
    results = {}
    datasets = [
        'cholec_organs',
        'cholec_gonogo', 
        'cholecseg8k'
    ]
    
    for dataset in datasets:
        acc = test_dataset(dataset, num_samples=2)
        results[dataset] = acc
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for dataset, acc in results.items():
        if acc is not None:
            print(f"  {dataset:20s}: {acc:5.1f}% presence accuracy")
        else:
            print(f"  {dataset:20s}: FAILED")
    
    print("="*60)
    print("All tests completed!")

if __name__ == "__main__":
    main()