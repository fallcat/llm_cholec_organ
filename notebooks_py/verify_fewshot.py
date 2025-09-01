#!/usr/bin/env python3
"""
Standalone verification that few-shot plans exist and are valid.
No external dependencies required.
"""

import json
from pathlib import Path

print("="*60)
print("FEW-SHOT VERIFICATION")
print("="*60)

# Check for few-shot plan files
data_dir = Path("../data_info/cholecseg8k")

plan_files = {
    "standard": "fewshot_plan_train_pos1_neg1_seed43_excl100.json",
    "hard_negatives": "fewshot_plan_train_pos1_neg1_nearmiss1_seed45_excl100.json",
}

organs_to_check = ["Liver", "Gallbladder", "Fat", "Grasper"]

for plan_name, filename in plan_files.items():
    plan_path = data_dir / filename
    
    print(f"\n{plan_name.upper()} PLAN:")
    print("-" * 40)
    
    if not plan_path.exists():
        print(f"❌ File not found: {plan_path}")
        print("   THIS IS THE PROBLEM! No few-shot examples can be loaded.")
        continue
    
    print(f"✓ File exists: {filename}")
    
    # Load and check content
    try:
        with open(plan_path) as f:
            plan = json.load(f)
        
        print(f"  Total organs in plan: {len(plan)}")
        
        # Check specific organs
        for organ in organs_to_check:
            if organ in plan:
                organ_data = plan[organ]
                pos = len(organ_data.get("positive", []))
                neg_easy = len(organ_data.get("negative_easy", []))
                neg_hard = len(organ_data.get("negative_hard", []))
                total = pos + neg_easy + neg_hard
                
                print(f"  {organ:20} -> pos:{pos}, neg_easy:{neg_easy}, neg_hard:{neg_hard}, total:{total}")
                
                if total == 0:
                    print(f"    ⚠️  WARNING: No examples for {organ}!")
            else:
                print(f"  {organ:20} -> ❌ NOT IN PLAN")
    
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
    except Exception as e:
        print(f"❌ Error loading plan: {e}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

print("""
If the files don't exist or have no examples:
  → Run: python3 prepare_fewshot_examples.py
  
If the files exist and have examples:
  → The issue is in the evaluator code
  → Check that organ_plan.get() keys match exactly
  → Add more debug prints to see what's happening

To manually test if the model responds differently to few-shot:
  1. Use a Jupyter notebook
  2. Send the same query with and without examples
  3. Compare the responses
  
The debug output should show:
  [DEBUG FEW-SHOT] Total examples prepared: N
  [DEBUG] Organ: N examples
  [DEBUG POINTING] Organ: Using N few-shot examples
  
If N is always 0, examples aren't being prepared.
""")