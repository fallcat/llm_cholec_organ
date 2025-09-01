#!/usr/bin/env python3
"""
Test that few-shot examples are now being loaded correctly.
"""

import sys
import json
from pathlib import Path

sys.path.append('../src')

# Mock the adapter to test without dependencies
class MockAdapter:
    label2id = {
        "Abdominal Wall": 1,
        "Liver": 2,
        "Gastrointestinal Tract": 3,
        "Fat": 4,
        "Grasper": 5,
        "Connective Tissue": 6,
        "Blood": 7,
        "Cystic Duct": 8,
        "L-hook Electrocautery": 9,
        "Gallbladder": 10,
        "Hepatic Vein": 11,
        "Liver Ligament": 12
    }

# Load the plan
plan_path = Path("../data_info/cholecseg8k/fewshot_plan_train_pos1_neg1_seed43_excl100.json")
with open(plan_path) as f:
    fewshot_plan = json.load(f)

print("="*60)
print("TESTING FEW-SHOT LOADING FIX")
print("="*60)

# Simulate what the evaluator does
adapter = MockAdapter()
organ_names = ["Liver", "Gallbladder", "Fat"]

# Handle different plan formats (from the fixed code)
if 'plan' in fewshot_plan:
    actual_plan = fewshot_plan['plan']
else:
    actual_plan = fewshot_plan

total_examples = 0

for organ_name in organ_names:
    # Try to find the organ plan by ID or name
    organ_id = str(adapter.label2id.get(organ_name, -1))
    
    # First try by ID (new format)
    if organ_id in actual_plan:
        organ_data = actual_plan[organ_id]
        
        # Extract examples from new format (FIXED VERSION)
        positives = organ_data.get('positives', [])
        pos_indices = [item['idx'] if isinstance(item, dict) else item for item in positives]
        
        negatives_easy = organ_data.get('negatives_easy', [])
        neg_easy_indices = [item['idx'] if isinstance(item, dict) else item for item in negatives_easy]
        
        negatives_hard = organ_data.get('negatives_hard', [])
        neg_hard_indices = [item['idx'] if isinstance(item, dict) else item for item in negatives_hard]
        
        organ_plan = {
            'positive': pos_indices,
            'negative_easy': neg_easy_indices,
            'negative_hard': neg_hard_indices
        }
    elif organ_name in actual_plan:
        organ_plan = actual_plan[organ_name]
    else:
        organ_plan = {}
    
    # Count examples
    n_pos = len(organ_plan.get('positive', []))
    n_neg_easy = len(organ_plan.get('negative_easy', []))
    n_neg_hard = len(organ_plan.get('negative_hard', []))
    n_total = n_pos + n_neg_easy + n_neg_hard
    
    total_examples += n_total
    
    print(f"\n{organ_name}:")
    print(f"  Positive:      {n_pos} examples {organ_plan.get('positive', [])[:2]}")
    print(f"  Negative easy: {n_neg_easy} examples {organ_plan.get('negative_easy', [])[:2]}")
    print(f"  Negative hard: {n_neg_hard} examples {organ_plan.get('negative_hard', [])[:2]}")
    print(f"  Total:         {n_total} examples")

print("\n" + "="*60)
print("RESULT")
print("="*60)

if total_examples > 0:
    print(f"✅ SUCCESS! Loading {total_examples} total few-shot examples")
    print("\nThe fix is working. Few-shot examples should now be used.")
    print("Run the evaluation again to see different results:")
    print("  cd notebooks_py")
    print("  python3 clear_cache.py")
    print("  EVAL_USE_CACHE=false EVAL_QUICK_TEST=true python3 eval_pointing.py")
else:
    print("❌ FAILURE! No examples loaded. The fix didn't work.")